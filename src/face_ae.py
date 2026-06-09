"""v3 — Small face autoencoder, trained on synthetic face proxies.

Provides latent encoder f(canvas) -> z ∈ R^D for D2 drift loss.
Not used in v1/v2. Trained once and cached at checkpoints/face_ae.pt.
"""
from __future__ import annotations
import os
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import synth_face_batch


class FaceAE(nn.Module):
    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.GELU(),     # 64
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),    # 32
            nn.Conv2d(64, 128, 4, 2, 1), nn.GELU(),   # 16
            nn.Conv2d(128, 128, 4, 2, 1), nn.GELU(),  # 8
            nn.Flatten(1),
            nn.Linear(128 * 8 * 8, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8), nn.GELU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 128, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    def forward(self, x: torch.Tensor):
        z = self.enc(x)
        r = self.dec(z)
        return r, z


def train_face_ae(device: torch.device, canvas: int, steps: int = 1500,
                  batch: int = 64, lr: float = 1e-3, latent_dim: int = 64) -> FaceAE:
    model = FaceAE(latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        faces = synth_face_batch(batch, canvas, device)  # [B,1,C,C]
        recon, _ = model(faces)
        loss = F.mse_loss(recon, faces)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 200 == 0:
            print(f"[face_ae] step {s} mse {loss.item():.4f}", flush=True)
    model.eval()
    return model


def save(m: FaceAE, path: str) -> None:
    d = os.path.dirname(path)
    if d: os.makedirs(d, exist_ok=True)
    torch.save(m.state_dict(), path)


def load(path: str, device: torch.device, latent_dim: int = 64) -> FaceAE:
    m = FaceAE(latent_dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m
