"""v1 — Letter classifier for evaluation. Small CNN, trained on the same LetterBank
(the EMNIST or synthetic fallback data) so the bank's distribution matches.
"""
from __future__ import annotations
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LetterCNN(nn.Module):
    def __init__(self, n_classes: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),  # 8
            nn.Flatten(1),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):  # x: [B,1,32,32]
        return self.net(x)


def train_classifier(bank_data: torch.Tensor, device: torch.device,
                     steps: int = 1500, batch: int = 128, lr: float = 1e-3) -> LetterCNN:
    """bank_data: [26, N, 32, 32] in [0,1]."""
    model = LetterCNN(26).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bank = bank_data.to(device)
    C, N = bank.shape[0], bank.shape[1]
    model.train()
    for step in range(steps):
        # balanced sample
        labels = torch.randint(0, C, (batch,), device=device)
        idx = torch.randint(0, N, (batch,), device=device)
        x = bank[labels, idx].unsqueeze(1)  # [B,1,32,32]
        # light augmentation: random shift up to 2 pix
        shift = torch.randint(-2, 3, (2,)).tolist()
        x = F.pad(x, (2, 2, 2, 2))
        x = x[..., 2 + shift[0]: 2 + shift[0] + 32, 2 + shift[1]: 2 + shift[1] + 32]
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            acc = (logits.argmax(-1) == labels).float().mean().item()
            print(f"[cls] step {step} loss {loss.item():.3f} acc {acc:.3f}", flush=True)
    # final eval on bank
    model.eval()
    with torch.no_grad():
        labels = torch.arange(C, device=device).repeat_interleave(min(N, 50))
        idx = torch.randint(0, N, (labels.shape[0],), device=device)
        x = bank[labels, idx].unsqueeze(1)
        acc = (model(x).argmax(-1) == labels).float().mean().item()
    print(f"[cls] final bank acc = {acc:.3f}", flush=True)
    return model


def save(model: LetterCNN, path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    torch.save(model.state_dict(), path)


def load(path: str, device: torch.device) -> LetterCNN:
    m = LetterCNN(26).to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m
