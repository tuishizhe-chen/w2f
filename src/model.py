"""Generator: per-letter MLP + eps -> Transformer -> affine θ -> STN -> max-compose.

v1: eps appended as an extra token; theta_head zero-init.
v2: (unchanged from v1 model)
v3: eps injected via FiLM (scale+shift on every letter token) so attention cannot ignore
    it. theta_head uses small random init to preserve gradient flow at step 0.
v4: FiLM init non-zero (~N(0,0.3)) so eps has immediate effect; pos embedding scaled up 2x
    to differentiate repeated letters; larger d_model (256→320) for capacity.
v5: per-slot theta projection head (one Linear per position) so repeated same-letter tokens
    CANNOT collapse to the same (tx,ty,r) — the per-slot head breaks symmetry directly.
v6: roll back to v4 single shared head. Per-slot heads hurt letter_acc more than helped.
    Duplicate-letter issue instead addressed by same-class repulsion in train.py.
v7: small per-slot translation bias added AFTER tanh (registered buffer), just enough to
    break the permutation-equivariance for duplicates without overriding what cls loss wants.
"""
from __future__ import annotations
from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import stn_place  # [N*K,1,C,C]


class LetterEncoder(nn.Module):
    """Flatten letter image (1, h, w) + class-id embed -> d_model token."""

    def __init__(self, d_model: int, n_classes: int, letter_size: int):
        super().__init__()
        self.pix = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),  # 16
            nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8
            nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 4
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),  # [B,128,1,1]
            nn.Flatten(1),            # [B,128]
        )
        self.cls_emb = nn.Embedding(n_classes, d_model)
        self.pix_proj = nn.Linear(128, d_model)
        self.pos_proj = nn.Embedding(64, d_model)  # up to 64 letter positions

    def forward(self, imgs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # imgs: [B,K,1,h,w]; labels: [B,K]
        B, K = labels.shape
        flat = imgs.view(B * K, *imgs.shape[2:])  # [B*K,1,h,w]
        pix = self.pix(flat)                       # [B*K,128]
        pix = self.pix_proj(pix).view(B, K, -1)    # [B,K,d]
        cls = self.cls_emb(labels)                 # [B,K,d]
        pos_ids = torch.arange(K, device=imgs.device).unsqueeze(0).expand(B, K)
        pos = self.pos_proj(pos_ids)
        # v4: scale pos embedding up to differentiate repeated letters (avoid HIHI collapse)
        return pix + cls + 2.0 * pos


class TransformerBlock(nn.Module):
    def __init__(self, d: int, h: int, mlp_ratio: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        hidden = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.enc = LetterEncoder(cfg.d_model, cfg.n_letter_classes, cfg.letter_size)
        # v3: FiLM eps — project eps to (scale, shift) ∈ R^{2*d}
        self.eps_film = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, 2 * cfg.d_model),
        )
        # v4: non-zero init → eps immediately modulates every letter token
        nn.init.normal_(self.eps_film[-1].weight, std=0.02)
        nn.init.zeros_(self.eps_film[-1].bias)

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        # v6: single shared theta head (rolled back from v5)
        self.theta_head = nn.Linear(cfg.d_model, 3)
        nn.init.xavier_normal_(self.theta_head.weight, gain=0.01)
        self.theta_head.bias.data.zero_()
        # v7: small per-slot translation bias (breaks symmetry for duplicate letters).
        # Small enough (~15% canvas) that cls loss can still steer; fixed (not trainable).
        MAX_K = 8
        slot_tx = torch.linspace(-0.15, 0.15, MAX_K)
        slot_ty = torch.zeros(MAX_K)  # leave y unbiased; x-arrangement is enough to disambiguate
        self.register_buffer("slot_tx_bias", slot_tx)
        self.register_buffer("slot_ty_bias", slot_ty)
        # Store fixed ranges:
        self.r_min = 0.1
        self.r_max = 0.9
        self.t_range = 0.7  # tanh range for translations

    def forward(self, labels: torch.Tensor, imgs: torch.Tensor,
                eps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """labels [B,K]; imgs [B,K,1,h,w]; eps [B,D] gaussian.
        Returns canvas [B,1,C,C], patches [B,K,1,C,C], theta [B,K,3] (r, tx, ty)."""
        B, K = labels.shape
        tok = self.enc(imgs, labels)                 # [B,K,d]
        # v3 FiLM: modulate every letter token by eps -> scale,shift
        ss = self.eps_film(eps)                      # [B, 2d]
        scale, shift = ss.chunk(2, dim=-1)           # [B,d] each
        x = tok * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # [B,K,d]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        raw = self.theta_head(x)                     # [B,K,3]
        r = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(raw[..., 0])
        tx = self.t_range * torch.tanh(raw[..., 1])
        ty = self.t_range * torch.tanh(raw[..., 2])
        # v7: add per-slot translation bias (fixed, breaks duplicate-letter symmetry)
        slot_tx = self.slot_tx_bias[:K].view(1, K)
        slot_ty = self.slot_ty_bias[:K].view(1, K)
        tx = tx + slot_tx
        ty = ty + slot_ty
        # clamp within [-t_range, t_range]
        tx = torch.clamp(tx, -self.t_range, self.t_range)
        ty = torch.clamp(ty, -self.t_range, self.t_range)
        theta = torch.stack([r, tx, ty], dim=-1)     # [B,K,3]

        flat_imgs = imgs.view(B * K, 1, imgs.shape[-2], imgs.shape[-1])
        inv_scale = (1.0 / r).view(B * K)
        patches_flat = stn_place(flat_imgs, inv_scale, tx.view(-1), ty.view(-1),
                                 self.cfg.canvas_size)  # [B*K,1,C,C]
        patches = patches_flat.view(B, K, 1, self.cfg.canvas_size, self.cfg.canvas_size)
        canvas = patches.max(dim=1).values           # [B,1,C,C]
        return canvas, patches, theta
