"""v1 — Evaluate: generate grid for test strings, compute letter_acc via STN crop
and the pretrained letter classifier. Also writes metrics.json.
"""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_cfg
from data import LetterBank, string_to_labels
from model import Generator
import classifier as cls_module


def crop_from_theta(canvas: torch.Tensor, theta: torch.Tensor,
                    letter_size: int) -> torch.Tensor:
    """canvas [B,1,C,C], theta [B,K,3] (r, tx, ty). Crops each letter region back to letter_size.

    Build inverse STN: sample canvas at positions of the letter window of size r*canvas at (tx,ty).
    Using affine_grid with matrix M = [[r, 0, tx], [0, r, ty]] on output size letter_size
    samples a r-wide window around (tx,ty).
    """
    B, _, C, C2 = canvas.shape
    K = theta.shape[1]
    r = theta[..., 0].reshape(-1)
    tx = theta[..., 1].reshape(-1)
    ty = theta[..., 2].reshape(-1)
    N = r.shape[0]
    M = torch.zeros(N, 2, 3, device=canvas.device, dtype=canvas.dtype)
    M[:, 0, 0] = r
    M[:, 1, 1] = r
    M[:, 0, 2] = tx
    M[:, 1, 2] = ty
    canvas_rep = canvas.unsqueeze(1).expand(B, K, -1, C, C).reshape(N, 1, C, C2)
    grid = F.affine_grid(M, size=(N, 1, letter_size, letter_size), align_corners=False)
    crop = F.grid_sample(canvas_rep, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)
    return crop  # [N,1,letter_size,letter_size]


@torch.no_grad()
def evaluate(cfg, version: int) -> dict:
    vtag = f"v{version}"
    ckpt_path = Path(f"checkpoints/{vtag}/final.pt")
    if not ckpt_path.exists():
        ckpt_path = Path(f"checkpoints/{vtag}/latest.pt")
    out_samples = Path(f"samples/{vtag}")
    out_samples.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bank = LetterBank.build(root="data", size=cfg.letter_size, device=device)

    gen = Generator(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)
    gen.load_state_dict(state["gen"])
    gen.eval()

    cls_path = Path("checkpoints/letter_cnn.pt")
    classifier = cls_module.load(str(cls_path), device)

    # Grid of 8 strings × eval_samples_per_string
    strings = list(cfg.test_strings)
    NS = cfg.eval_samples_per_string
    rows = len(strings)
    fig, axes = plt.subplots(rows, NS, figsize=(2.4 * NS, 2.4 * rows))
    if rows == 1:
        axes = axes[None, :]
    if NS == 1:
        axes = axes[:, None]

    # For letter acc: gather many strings
    all_correct = []
    detail_per_string = {}

    for ri, s in enumerate(strings):
        lbls = string_to_labels(s).to(device).unsqueeze(0)  # [1,K]
        K = lbls.shape[1]
        imgs = bank.sample(lbls.view(-1)).view(1, K, 1, bank.size, bank.size)
        correct_for_s = 0; total_for_s = 0
        for ci in range(NS):
            eps = torch.randn(1, cfg.d_model, device=device)
            canvas, patches, theta = gen(lbls, imgs, eps)
            crops = crop_from_theta(canvas, theta, cfg.letter_size)  # [K,1,32,32]
            # resize crops: classifier trained on bank (32x32); crops already are 32x32
            logits = classifier(crops)
            pred = logits.argmax(-1)
            gt = lbls.view(-1)
            correct = (pred == gt).float()
            correct_for_s += correct.sum().item(); total_for_s += K
            all_correct.append(correct.cpu().numpy())
            ax = axes[ri, ci]
            ax.imshow(canvas[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{s} [acc={int(correct.sum().item())}/{K}]", fontsize=8)
            ax.axis("off")
        detail_per_string[s] = correct_for_s / max(total_for_s, 1)

    plt.tight_layout()
    plt.savefig(out_samples / "grid.png", dpi=100)
    plt.close()

    letter_acc = float(np.concatenate(all_correct).mean()) if all_correct else 0.0

    # noise baseline: letter_acc on random canvases (random pixels fed through same crop route)
    baseline_correct = []
    for _ in range(20):
        canvas_rand = torch.rand(1, 1, cfg.canvas_size, cfg.canvas_size, device=device)
        # pick random theta similar to what gen produced
        r = torch.full((1, 4), 0.5, device=device)
        tx = torch.zeros(1, 4, device=device)
        ty = torch.zeros(1, 4, device=device)
        theta_rand = torch.stack([r, tx, ty], dim=-1)
        crops = crop_from_theta(canvas_rand, theta_rand, cfg.letter_size)
        pred = classifier(crops).argmax(-1)
        gt = torch.randint(0, 26, (4,), device=device)
        baseline_correct.append((pred == gt).float().cpu().numpy())
    noise_acc = float(np.concatenate(baseline_correct).mean()) if baseline_correct else 0.0

    metrics = {
        "version": version,
        "letter_acc": letter_acc,
        "letter_acc_per_string": detail_per_string,
        "noise_baseline_acc": noise_acc,
        "checkpoint": str(ckpt_path),
    }
    with open(out_samples / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True)
    args = ap.parse_args()
    cfg = get_cfg()
    evaluate(cfg, args.version)


if __name__ == "__main__":
    main()
