"""v1 — EMNIST letters loader + synthetic face proxy.

Defines:
 - LetterBank: holds (N, 32, 32) tensor per class; supports per-class sampling +
   random affine augmentation (= Vp positive sampler).
 - sample_letter_strings: generator of (labels [B,K], letter_imgs [B,K,1,32,32]).
 - synth_face_batch: cheap gray-ellipse proxy face (B,1,128,128) in [0,1].
If EMNIST download fails, falls back to PIL-drawn letters.
"""
from __future__ import annotations
import os
import string
import math
import random
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_LETTERS = string.ascii_uppercase  # 26 classes


def _fallback_synth_letters(size: int = 32, per_class: int = 200) -> torch.Tensor:
    """Hand-render letters with PIL if EMNIST missing. Returns [26, per_class, size, size]."""
    from PIL import Image, ImageDraw, ImageFont
    out = torch.zeros(26, per_class, size, size, dtype=torch.float32)
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    font_path = next((p for p in font_paths if os.path.exists(p)), None)
    for c in range(26):
        for i in range(per_class):
            img = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(img)
            # slight jitter: size, position, stroke
            fs = size - random.randint(2, 8)
            try:
                font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default()
            except Exception:
                font = ImageFont.load_default()
            ch = _LETTERS[c]
            try:
                bbox = draw.textbbox((0, 0), ch, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                ox = (size - tw) / 2 - bbox[0] + random.uniform(-2, 2)
                oy = (size - th) / 2 - bbox[1] + random.uniform(-2, 2)
            except Exception:
                ox = oy = 2.0
            draw.text((ox, oy), ch, fill=255, font=font)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            out[c, i] = torch.from_numpy(arr)
    return out


def _try_load_emnist(root: str, size: int = 32) -> Optional[torch.Tensor]:
    """Return [26, N_per_class, size, size] or None on failure."""
    try:
        from torchvision import datasets, transforms
        tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        ds = datasets.EMNIST(root, split="letters", train=True, download=True, transform=tf)
    except Exception as e:
        print(f"[data] EMNIST download/load failed: {e}", flush=True)
        return None
    # EMNIST 'letters' labels are 1..26; orientation is transposed+flipped
    buckets = [[] for _ in range(26)]
    # limit per class to avoid huge RAM
    MAX_PER = 400
    for img, lbl in ds:
        c = int(lbl) - 1
        if 0 <= c < 26 and len(buckets[c]) < MAX_PER:
            # EMNIST image convention: transpose to fix orientation
            a = img.squeeze(0)
            a = a.transpose(0, 1)  # fix EMNIST orientation
            buckets[c].append(a)
        if all(len(b) >= MAX_PER for b in buckets):
            break
    out = torch.stack([torch.stack(b[:MAX_PER]) for b in buckets])  # [26, MAX_PER, size, size]
    return out


class LetterBank:
    """Holds [26, N, 32, 32] letter images. Samples class-conditioned with affine aug."""

    def __init__(self, data: torch.Tensor, device: torch.device):
        assert data.ndim == 4 and data.shape[0] == 26
        self.data = data.to(device)
        self.device = device
        self.size = data.shape[-1]

    @classmethod
    def build(cls, root: str, size: int, device: torch.device) -> "LetterBank":
        arr = _try_load_emnist(root, size)
        if arr is None:
            print("[data] Falling back to PIL synthetic letters.", flush=True)
            arr = _fallback_synth_letters(size, per_class=200)
        print(f"[data] LetterBank: shape {tuple(arr.shape)}", flush=True)
        return cls(arr, device)

    def sample(self, labels: torch.Tensor) -> torch.Tensor:
        """labels: [N] long in [0,26). Return [N,1,size,size]."""
        n = labels.shape[0]
        idx = torch.randint(0, self.data.shape[1], (n,), device=self.device)
        imgs = self.data[labels, idx]  # [N, size, size]
        return imgs.unsqueeze(1)  # [N,1,size,size]


def _affine_matrix(theta_scale: torch.Tensor, theta_tx: torch.Tensor,
                   theta_ty: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """Build affine grid matrices (N,2,3) for STN. PyTorch STN uses inverse mapping:
    sample-coord = M @ out-coord, so scale<1 zooms out the patch (makes letter smaller in canvas).
    We want scale in canvas-space (how big letter appears); conv: M scale = 1/scale_canvas.
    Here theta_scale is INVERSE (directly fed to grid) — STN conventional.
    """
    N = theta_scale.shape[0]
    cos = torch.cos(rot)
    sin = torch.sin(rot)
    # M = [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
    M = torch.zeros(N, 2, 3, device=theta_scale.device, dtype=theta_scale.dtype)
    M[:, 0, 0] = theta_scale * cos
    M[:, 0, 1] = -theta_scale * sin
    M[:, 0, 2] = theta_tx
    M[:, 1, 0] = theta_scale * sin
    M[:, 1, 1] = theta_scale * cos
    M[:, 1, 2] = theta_ty
    return M


def stn_place(letter_imgs: torch.Tensor, theta_scale: torch.Tensor,
              theta_tx: torch.Tensor, theta_ty: torch.Tensor,
              canvas: int, rot: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Place letter_imgs (N,1,h,w) onto canvas×canvas via affine STN.
    theta_scale is the 'zoom factor in sampling-grid space': value 1/s means letter occupies 's' of canvas.
    To make code intuitive we define: letter visual size ratio = r  =>  grid-scale = 1/r (inverse).

    Here input args are ALREADY the inverse scale (i.e. sample at output coord*inv_scale).
    For letter of canvas-ratio r=0.5, caller passes theta_scale = 1/0.5 = 2.0.
    """
    N = letter_imgs.shape[0]
    if rot is None:
        rot = torch.zeros(N, device=letter_imgs.device, dtype=letter_imgs.dtype)
    M = _affine_matrix(theta_scale, theta_tx, theta_ty, rot)  # [N,2,3]
    grid = F.affine_grid(M, size=(N, 1, canvas, canvas), align_corners=False)
    out = F.grid_sample(letter_imgs, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out  # [N,1,canvas,canvas]


def sample_positive_placements(letter_imgs: torch.Tensor, canvas: int,
                               scale_range: Tuple[float, float],
                               trans_range: Tuple[float, float],
                               n_samples: int) -> torch.Tensor:
    """Vp positive sampler: random affine transforms of the SAME letter image.

    letter_imgs: [N, 1, h, w]. Return [N, n_samples, 1, canvas, canvas].
    Each call produces fresh random affines."""
    N = letter_imgs.shape[0]
    P = n_samples
    device = letter_imgs.device
    # tile: (N*P, 1, h, w)
    tiled = letter_imgs.unsqueeze(1).expand(N, P, -1, -1, -1).contiguous().view(N * P, *letter_imgs.shape[1:])
    rmin, rmax = scale_range
    tmin, tmax = trans_range
    ratio = torch.rand(N * P, device=device) * (rmax - rmin) + rmin  # letter canvas-ratio r
    inv_scale = 1.0 / ratio  # STN grid-scale
    tx = torch.rand(N * P, device=device) * (tmax - tmin) + tmin
    ty = torch.rand(N * P, device=device) * (tmax - tmin) + tmin
    placed = stn_place(tiled, inv_scale, tx, ty, canvas)  # [N*P,1,C,C]
    placed = placed.view(N, P, 1, canvas, canvas)
    return placed


def synth_face_batch(B: int, canvas: int, device: torch.device) -> torch.Tensor:
    """Cheap synthetic face: gray ellipse + 2 eye dots + mouth. [B,1,canvas,canvas] in [0,1]."""
    ys = torch.linspace(-1, 1, canvas, device=device)
    xs = torch.linspace(-1, 1, canvas, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    imgs = torch.zeros(B, 1, canvas, canvas, device=device)
    for b in range(B):
        cx = float(torch.empty(1).uniform_(-0.08, 0.08))
        cy = float(torch.empty(1).uniform_(-0.05, 0.05))
        a = float(torch.empty(1).uniform_(0.55, 0.72))  # horiz radius
        b_ = float(torch.empty(1).uniform_(0.72, 0.88))  # vert radius
        face = torch.exp(-((xx - cx) ** 2 / a ** 2 + (yy - cy) ** 2 / b_ ** 2) * 3.0)
        # eyes
        ex = 0.22; ey = -0.18
        eye_sigma = 0.06
        left = torch.exp(-((xx - (cx - ex)) ** 2 + (yy - (cy + ey)) ** 2) / eye_sigma ** 2)
        right = torch.exp(-((xx - (cx + ex)) ** 2 + (yy - (cy + ey)) ** 2) / eye_sigma ** 2)
        # mouth
        mouth = torch.exp(-((xx - cx) ** 2 / 0.25 ** 2 + (yy - (cy + 0.30)) ** 2 / 0.05 ** 2))
        img = face * 0.7 - (left + right) * 0.55 - mouth * 0.45
        img = torch.clamp(img, 0, 1)
        imgs[b, 0] = img
    return imgs


def make_string_batch(bank: LetterBank, B: int, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Random strings: labels [B,K] long, letter_imgs [B,K,1,size,size]."""
    labels = torch.randint(0, 26, (B, K), device=bank.device)
    imgs = bank.sample(labels.view(-1)).view(B, K, 1, bank.size, bank.size)
    return labels, imgs


def string_to_labels(s: str) -> torch.Tensor:
    return torch.tensor([ord(c.upper()) - ord("A") for c in s if c.isalpha()], dtype=torch.long)
