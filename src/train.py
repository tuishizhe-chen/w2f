"""Main training loop.

v1: D1 drift in pixel space 128x128. D2 disabled.
v2: D1 drift in 16x16 avg-pooled feature space. Pairwise letter repulsion loss added.

Usage:
  python src/train.py --version N
  python src/train.py --smoke   # 50 steps, fast iteration
"""
from __future__ import annotations
import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torch.nn.functional as F

# make src/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import get_cfg
from data import LetterBank, make_string_batch, sample_positive_placements, synth_face_batch
from drift_loss import drift_loss
from model import Generator
import classifier as cls_module
import face_ae as face_ae_module


def inverse_stn_crop(canvas: torch.Tensor, theta: torch.Tensor,
                     letter_size: int) -> torch.Tensor:
    """Crop canvas regions specified by theta (r, tx, ty) back to letter_size.
    canvas [B,1,C,C]; theta [B,K,3]. Returns [B*K, 1, letter_size, letter_size]."""
    B, _, C, _ = canvas.shape
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
    canvas_rep = canvas.unsqueeze(1).expand(B, K, -1, C, C).reshape(N, 1, C, C)
    grid = F.affine_grid(M, size=(N, 1, letter_size, letter_size), align_corners=False)
    crop = F.grid_sample(canvas_rep, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=False)
    return crop


def _build_negatives(patches_flat: torch.Tensor, labels_flat: torch.Tensor, S: int
                     ) -> torch.Tensor:
    """Build per-gen-sample negatives: same-class OTHER generated patches in batch.
    patches_flat: [BK, S], labels_flat: [BK]. Returns fixed_neg [BK, Cn, S] padded."""
    BK = patches_flat.shape[0]
    device = patches_flat.device
    # For each i, gather indices j != i where labels[j] == labels[i].
    # We'll limit Cn per sample to CN_MAX; pad with zero-weight entries.
    CN_MAX = 4
    fixed_neg = torch.zeros(BK, CN_MAX, S, device=device, dtype=patches_flat.dtype)
    weight_neg = torch.zeros(BK, CN_MAX, device=device, dtype=patches_flat.dtype)
    # simple CPU loop — BK small (~128)
    labels_cpu = labels_flat.detach().cpu().numpy()
    import numpy as np
    for i in range(BK):
        peers = np.where((labels_cpu == labels_cpu[i]) & (np.arange(BK) != i))[0]
        if len(peers) == 0:
            continue
        take = peers[:CN_MAX] if len(peers) >= CN_MAX else peers
        fixed_neg[i, :len(take)] = patches_flat[take]
        weight_neg[i, :len(take)] = 1.0
    return fixed_neg, weight_neg


def train(cfg, version: int, smoke: bool = False) -> None:
    vtag = f"v{version}"
    out_ckpt = Path(f"checkpoints/{vtag}")
    out_samples = Path(f"samples/{vtag}")
    out_logs = Path("logs")
    out_ckpt.mkdir(parents=True, exist_ok=True)
    out_samples.mkdir(parents=True, exist_ok=True)
    out_logs.mkdir(parents=True, exist_ok=True)
    log_path = out_logs / f"{vtag}.log"
    logf = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"[train] {vtag} device={device} cuda_devs={torch.cuda.device_count()}")
    torch.manual_seed(cfg.seed)

    # --- data ---
    bank = LetterBank.build(root="data", size=cfg.letter_size, device=device)

    # --- classifier — train once and cache; also USED in-loop for v3+ ---
    cls_path = Path("checkpoints/letter_cnn.pt")  # shared across versions
    if not cls_path.exists():
        log("[train] training letter classifier (one-time)")
        cls_model = cls_module.train_classifier(bank.data, device,
                                               steps=800 if smoke else 1500)
        cls_module.save(cls_model, str(cls_path))
    else:
        log(f"[train] loading cached classifier {cls_path}")
    classifier_net = cls_module.load(str(cls_path), device)
    for p in classifier_net.parameters():
        p.requires_grad_(False)
    classifier_net.eval()

    # --- face AE — train once and cache if d2_weight > 0 ---
    d2_w = getattr(cfg, "d2_weight", 0.0)
    face_ae = None
    if d2_w > 0:
        fae_path = Path("checkpoints/face_ae.pt")
        if not fae_path.exists():
            log("[train] training face AE (one-time)")
            face_ae = face_ae_module.train_face_ae(device, cfg.canvas_size,
                                                   steps=400 if smoke else 1500)
            face_ae_module.save(face_ae, str(fae_path))
        else:
            face_ae = face_ae_module.load(str(fae_path), device)
        for p in face_ae.parameters():
            p.requires_grad_(False)
        face_ae.eval()

    # --- model ---
    gen = Generator(cfg).to(device)
    n_params = sum(p.numel() for p in gen.parameters())
    log(f"[train] generator params: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(gen.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
                           betas=(0.9, 0.95))

    steps = 50 if smoke else cfg.steps
    start = time.time()
    budget_s = cfg.wall_clock_min * 60

    last_ckpt_t = time.time()
    ckpt_every_s = 300  # 5 minutes

    gen.train()
    for step in range(1, steps + 1):
        B, K = cfg.batch_strings, cfg.K
        if smoke:
            B = 8  # smaller for smoke
        labels, letter_imgs = make_string_batch(bank, B, K)
        eps = torch.randn(B, cfg.d_model, device=device)

        # v4: second eps forward for diversity loss (same labels/imgs, different noise)
        div_w = getattr(cfg, "diversity_weight", 0.0)
        if div_w > 0.0:
            eps2 = torch.randn(B, cfg.d_model, device=device)
            canvas2, _, theta2 = gen(labels, letter_imgs, eps2)
        canvas, patches, theta = gen(labels, letter_imgs, eps)
        # v2: downsample patches+positives to feature space before drift loss
        pool = getattr(cfg, "drift_pool", 1)
        CS = cfg.canvas_size
        FS = CS // pool  # feature spatial size
        S = FS * FS
        patches_flat_spatial = patches.view(B * K, 1, CS, CS)
        if pool > 1:
            patches_feat = F.avg_pool2d(patches_flat_spatial, pool)  # [BK,1,FS,FS]
        else:
            patches_feat = patches_flat_spatial
        patches_flat = patches_feat.view(B * K, S)  # [BK, S]
        labels_flat = labels.view(B * K)

        # D1 positives: per-letter random affine placements (on canvas) of the SAME letter image
        flat_letters = letter_imgs.view(B * K, 1, cfg.letter_size, cfg.letter_size)
        pos_placed = sample_positive_placements(
            flat_letters, cfg.canvas_size,
            cfg.pos_scale_range, cfg.pos_trans_range,
            cfg.pos_per_letter,
        )  # [BK, P, 1, C, C]
        P = cfg.pos_per_letter
        pos_reshape = pos_placed.view(B * K * P, 1, CS, CS)
        if pool > 1:
            pos_feat = F.avg_pool2d(pos_reshape, pool)  # [BK*P,1,FS,FS]
        else:
            pos_feat = pos_reshape
        pos_flat = pos_feat.view(B * K, P, S)

        # D1 negatives: same-class OTHER generated patches within the batch (already feature-space)
        with torch.no_grad():
            fixed_neg, weight_neg = _build_negatives(patches_flat.detach(), labels_flat, S)

        gen_arg = patches_flat.unsqueeze(1)              # [BK, 1, S]
        loss_vec, info = drift_loss(
            gen_arg, pos_flat, fixed_neg,
            weight_neg=weight_neg,
            R_list=cfg.R_list,
        )
        d1 = loss_vec.mean()

        # v6: pairwise repulsion with dual margin (stricter for same-class pairs)
        repulse_w = getattr(cfg, "repulse_weight", 0.0)
        repulse = torch.zeros((), device=device)
        if repulse_w > 0.0 and K > 1:
            margin_other = getattr(cfg, "repulse_margin", 0.35)
            margin_same  = getattr(cfg, "repulse_margin_sameclass", margin_other)
            pos_xy = theta[..., 1:3]  # [B,K,2]
            diff = pos_xy.unsqueeze(2) - pos_xy.unsqueeze(1)  # [B,K,K,2]
            dist = torch.sqrt((diff ** 2).sum(-1) + 1e-6)     # [B,K,K]
            off_diag = 1.0 - torch.eye(K, device=device)
            same = (labels.unsqueeze(2) == labels.unsqueeze(1)).float() * off_diag  # [B,K,K]
            other = off_diag - same                                                 # [B,K,K] (but broadcastable only on K dims? yes off_diag is [K,K])
            # Need proper broadcasting: off_diag [K,K], same [B,K,K]; other = off_diag - same[B,K,K]
            other = off_diag.unsqueeze(0) - same                                    # [B,K,K]
            pen_same = F.relu(margin_same - dist) * same
            pen_other = F.relu(margin_other - dist) * other
            pen = pen_same + pen_other
            repulse = (pen ** 2).sum(dim=(1, 2)) / (K * (K - 1))
            repulse = repulse.mean()

        # v3: classifier-guided D1 (inverse-STN crop → letter CNN → CE with true label)
        cls_w = getattr(cfg, "cls_weight", 0.0)
        cls_loss = torch.zeros((), device=device)
        if cls_w > 0.0:
            crops = inverse_stn_crop(canvas, theta, cfg.letter_size)  # [BK,1,32,32]
            logits = classifier_net(crops)                             # [BK, 26]
            cls_loss = F.cross_entropy(logits, labels_flat)

        # v4: D2 — direct face-mask loss. The canvas should occupy a face-shaped region.
        # Penalty = mean((canvas · (1 - face_mask))^2)  — energy outside the ellipse mask.
        # Bonus = -mean(canvas · face_mask)            — encourage coverage inside the mask.
        # Much stronger supervision than AE-latent drift (which barely moved in v3).
        d2 = torch.zeros((), device=device)
        if d2_w > 0.0:
            # build face mask (cached on device lazily)
            if not hasattr(gen, "_face_mask"):
                ys = torch.linspace(-1, 1, cfg.canvas_size, device=device)
                xs = torch.linspace(-1, 1, cfg.canvas_size, device=device)
                yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                # v8: v4-style ellipse (a=0.65, b=0.80)
                mask = torch.exp(-((xx / 0.65) ** 2 + (yy / 0.80) ** 2) * 3.0)
                mask = torch.clamp(mask, 0, 1)
                gen._face_mask = mask.view(1, 1, cfg.canvas_size, cfg.canvas_size)
            fmask = gen._face_mask
            inv = 1.0 - fmask
            out_energy = (canvas * inv).pow(2).mean()
            cov = (canvas * fmask).mean()
            cov_target = 0.08
            cov_pen = (cov - cov_target).pow(2)
            d2 = out_energy + 0.3 * cov_pen

        # v5: diversity loss — REWARD pixel variance between eps and eps2 canvases, capped.
        # loss = -min(pixel_diff, cap); grad pushes model to add variance until saturated.
        diversity_loss = torch.zeros((), device=device)
        if div_w > 0.0:
            pixel_diff = (canvas - canvas2).pow(2).mean(dim=(1, 2, 3))  # [B]
            cap = 0.03
            diversity_loss = -torch.clamp(pixel_diff, max=cap).mean()  # negative -> minimize loss = maximize diff

        total = (cfg.d1_weight * d1 + repulse_w * repulse +
                 cls_w * cls_loss + d2_w * d2 +
                 div_w * diversity_loss)

        opt.zero_grad()
        total.backward()
        g_norm = torch.nn.utils.clip_grad_norm_(gen.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.log_every == 0 or step == 1 or (smoke and step % 10 == 0):
            with torch.no_grad():
                r_mean = theta[..., 0].mean().item()
                tx_mean = theta[..., 1].abs().mean().item()
                scale_info = info.get("scale", torch.tensor(0.0)).item()
            elapsed = time.time() - start
            log(f"step={step} d1={d1.item():.3f} cls={cls_loss.item():.3f} "
                f"d2={d2.item():.3f} rep={repulse.item():.3f} "
                f"div={diversity_loss.item():.4f} tot={total.item():.3f} "
                f"r={r_mean:.3f} |tx|={tx_mean:.3f} grad={g_norm.item():.2f} "
                f"t={elapsed:.0f}s")
            if not torch.isfinite(d1):
                log("[train] NaN/Inf detected, aborting")
                break

        # ckpt every 5 min
        if (time.time() - last_ckpt_t) > ckpt_every_s and not smoke:
            torch.save({"gen": gen.state_dict(), "step": step,
                       "opt": opt.state_dict()},
                      out_ckpt / "latest.pt")
            last_ckpt_t = time.time()
            log(f"[train] ckpt saved @ step {step}")

        # wall-clock timeout
        if not smoke and (time.time() - start) > budget_s:
            log(f"[train] wall clock budget exceeded at step {step}, stopping")
            break

    # final ckpt
    torch.save({"gen": gen.state_dict(), "step": step},
              out_ckpt / "final.pt")
    log(f"[train] final ckpt saved. total time = {time.time()-start:.0f}s")

    # sanity: save one grid of samples for smoke
    if smoke:
        import matplotlib.pyplot as plt
        gen.eval()
        with torch.no_grad():
            _save_quick_grid(gen, bank, cfg, device, out_samples / "smoke_grid.png")
        log(f"[train] smoke grid saved")
    logf.close()


def _save_quick_grid(gen, bank, cfg, device, path):
    import matplotlib.pyplot as plt
    from data import string_to_labels
    strings = cfg.test_strings[:4]
    fig, axes = plt.subplots(1, len(strings), figsize=(3 * len(strings), 3))
    if len(strings) == 1:
        axes = [axes]
    for ax, s in zip(axes, strings):
        lbls = string_to_labels(s).to(device).unsqueeze(0)
        K = lbls.shape[1]
        imgs = bank.sample(lbls.view(-1)).view(1, K, 1, bank.size, bank.size)
        eps = torch.randn(1, cfg.d_model, device=device)
        canvas, _, _ = gen(lbls, imgs, eps)
        ax.imshow(canvas[0, 0].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(s); ax.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(path, dpi=100); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = get_cfg()
    train(cfg, args.version, smoke=args.smoke)


if __name__ == "__main__":
    main()
