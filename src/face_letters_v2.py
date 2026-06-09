"""Letters phase v2 — G outputs PLACEMENT only, letter content from bank.

Architecture (much simpler):
  LetterPlacer(eps, labels[B,K]) → theta[B,K,3]  (r, tx, ty per letter slot)

For each step:
  sample labels  [B, K]
  sample letter aug from letter_bank[labels]  → letter_imgs[B, K, 1, 64, 64]
  theta = G(eps, labels)
  STN-place each letter → canvas[B, 1, 128, 128]
  drift loss(canvas, face_bank) → push toward face-like composite

Why this is simpler: letters are real letters BY CONSTRUCTION. Only the placement
(scale + position) is learned. The drift loss is the only signal driving the
spatial layout.  Removes the per-class identity collapse problem.
"""
from __future__ import annotations
import argparse, copy, sys, time, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss


def stn_place(letter_imgs, inv_scale, tx, ty, canvas):
    """letter_imgs[N,1,h,w] → placed[N,1,canvas,canvas]."""
    N = letter_imgs.shape[0]
    M = torch.zeros(N, 2, 3, device=letter_imgs.device, dtype=letter_imgs.dtype)
    M[:, 0, 0] = inv_scale
    M[:, 0, 2] = tx
    M[:, 1, 1] = inv_scale
    M[:, 1, 2] = ty
    grid = F.affine_grid(M, size=(N, 1, canvas, canvas), align_corners=False)
    return F.grid_sample(letter_imgs, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)


class LetterPlacer(nn.Module):
    """eps[B,d] + labels[B,K] → theta[B,K,3].  Predicts (r, tx, ty) per slot.

    Each slot's theta is conditioned on eps (sample identity) + that slot's
    letter class. Includes a K-adaptive default grid placement so slots
    spread over the canvas by default.
    """
    def __init__(self, d_noise: int = 128, n_classes: int = 26, K: int = 12,
                 hidden: int = 512, slot_dim: int = 32,
                 dxy_max: float = 0.15, r_min: float = 0.20, r_max: float = 0.50):
        super().__init__()
        self.K = K
        self.dxy_max = dxy_max
        self.r_min = r_min
        self.r_max = r_max
        # per-slot identity (independent of letter) for default placement
        self.slot_emb = nn.Embedding(K, slot_dim)
        # letter class embedding
        self.cls_emb = nn.Embedding(n_classes, 64)
        # shared trunk taking eps + letter class + slot id
        d_in = d_noise + 64 + slot_dim
        self.trunk = nn.Sequential(
            nn.Linear(d_in, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        # head outputs (r_raw, tx_raw, ty_raw)
        self.head = nn.Linear(hidden, 3)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, eps: torch.Tensor, labels: torch.Tensor):
        B, K = labels.shape
        eps_rep = eps.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        cls = self.cls_emb(labels.reshape(-1))
        slot_id = torch.arange(K, device=eps.device).unsqueeze(0).expand(B, K).reshape(-1)
        slot = self.slot_emb(slot_id)
        h = torch.cat([eps_rep, cls, slot], dim=1)
        h = self.trunk(h)
        raw = self.head(h).view(B, K, 3)
        # r and tx, ty deviations are bounded so model can only NUDGE letters
        # around their default grid position. Wider ranges led to all letters
        # collapsing to center (model learned to cancel default placement).
        r = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(raw[..., 0])
        dx = self.dxy_max * torch.tanh(raw[..., 1])
        dy = self.dxy_max * torch.tanh(raw[..., 2])
        # K-adaptive 2D grid default placement (these dominate the placement)
        cols = int(math.ceil(math.sqrt(K)))
        rows = int(math.ceil(K / cols))
        device = eps.device
        bx = torch.tensor([-0.45 + 0.9 * ((k % cols) + 0.5) / cols for k in range(K)], device=device)
        by = torch.tensor([-0.45 + 0.9 * ((k // cols) + 0.5) / rows for k in range(K)], device=device)
        tx = bx.view(1, K) + dx
        ty = by.view(1, K) + dy
        tx = torch.clamp(tx, -0.75, 0.75)
        ty = torch.clamp(ty, -0.75, 0.75)
        return torch.stack([r, tx, ty], dim=-1)            # [B, K, 3]


def update_ema(ema, online, decay):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), online.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # face edge bank
    bank = torch.load(args.bank, weights_only=True)
    N_face = bank.shape[0]; S_bank = bank.shape[-1]
    assert S_bank == 128
    bank_gpu = (bank.float() / 255.0).to(device)
    real_ink = bank_gpu.mean().item()
    print(f"[lp-v2] face bank {tuple(bank.shape)}  real_ink={real_ink:.4f}", flush=True)

    # letter bank
    letter_bank = torch.load(args.letter_bank, weights_only=True)
    n_cls, N_per, ls, _ = letter_bank.shape
    # move to GPU once (cheap, 50 MB)
    letter_bank_gpu = (letter_bank.float() / 255.0).to(device)
    letter_ink = letter_bank_gpu.mean().item()
    print(f"[lp-v2] letter_bank {tuple(letter_bank.shape)}  ink={letter_ink:.4f}", flush=True)

    G = LetterPlacer(d_noise=args.d_noise, n_classes=n_cls, K=args.K,
                     dxy_max=args.dxy_max, r_min=args.r_min, r_max=args.r_max).to(device)
    G_ema = copy.deepcopy(G).to(device)
    for p in G_ema.parameters(): p.requires_grad_(False)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[lp-v2] G params: {n_p/1e6:.2f}M  K={args.K}  Bgen={args.bgen}", flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))
    R_list = tuple(float(x) for x in args.R.split(','))
    fixed_eps = torch.randn(8, args.d_noise, device=device)
    fixed_labels = torch.randint(0, n_cls, (8, args.K), device=device)
    t0 = time.time()

    G.train()
    for step in range(1, args.steps + 1):
        prog = step / max(1, args.steps)
        sharp_w = args.sharpness if args.sharpness_end is None else \
                  args.sharpness + (args.sharpness_end - args.sharpness) * prog

        labels = torch.randint(0, n_cls, (args.bgen, args.K), device=device)
        eps = torch.randn(args.bgen, args.d_noise, device=device)
        theta = G(eps, labels)                                            # [B, K, 3]

        # sample letter image per (b,k) directly from bank
        idx = torch.randint(0, N_per, (args.bgen, args.K), device=device)
        # gather letter_bank_gpu[labels, idx]
        flat_lab = labels.reshape(-1)
        flat_idx = idx.reshape(-1)
        flat_imgs = letter_bank_gpu[flat_lab, flat_idx].unsqueeze(1)      # [B*K, 1, 64, 64]

        # STN-place + max-compose
        r = theta[..., 0].reshape(-1)
        tx = theta[..., 1].reshape(-1)
        ty = theta[..., 2].reshape(-1)
        inv_scale = 1.0 / r
        placed = stn_place(flat_imgs, inv_scale, tx, ty, 128).view(args.bgen, args.K, 1, 128, 128)
        canvas = placed.max(dim=1).values                                  # [B, 1, 128, 128]

        if args.direct_iou > 0:
            # direct loss (IoU/Dice/WL1) against sampled target faces.
            tgt_idx = torch.randint(0, N_face, (args.bgen,))
            targets = bank_gpu[tgt_idx]                                       # [B, 128, 128]
            cf = canvas[:, 0]
            if args.iou_variant == 'iou':
                inter = (cf * targets).sum(dim=(-1, -2))
                union = cf.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2)) - inter
                direct_val = (1.0 - inter / (union + 1e-6)).mean()
            elif args.iou_variant == 'dice':
                inter = (cf * targets).sum(dim=(-1, -2))
                direct_val = (1.0 - 2.0 * inter / (cf.sum(dim=(-1, -2)) + targets.sum(dim=(-1, -2)) + 1e-6)).mean()
            elif args.iou_variant == 'wl1':
                w = 1.0 + 4.0 * targets
                direct_val = (w * (cf - targets).abs()).mean()
            else:
                raise ValueError(args.iou_variant)
            loss = args.direct_iou * direct_val
            info = {'scale': torch.tensor(1.0)}
        else:
            # original drift-loss path
            canvas_feat = canvas.flatten(1)
            Cg = max(1, args.gen_per)
            B = args.bgen // Cg
            gen_arg = canvas_feat[: B * Cg].view(B, Cg, -1)
            pos_idx = torch.randint(0, N_face, (B, args.cp))
            pos_feat = bank_gpu[pos_idx].flatten(2)
            neg_idx = torch.randint(0, N_face, (B, args.cn))
            neg_feat = bank_gpu[neg_idx].flatten(2)
            if args.noise_aug > 0:
                t = args.noise_aug
                gen_arg = (1 - t) * gen_arg + t * torch.randn_like(gen_arg)
                pos_feat = (1 - t) * pos_feat + t * torch.randn_like(pos_feat)
                neg_feat = (1 - t) * neg_feat + t * torch.randn_like(neg_feat)
            loss_vec, info = drift_loss(gen_arg, pos_feat, fixed_neg=neg_feat,
                                        R_list=R_list,
                                        topk_pos=args.topk_pos, topk_neg=args.topk_neg)
            loss = loss_vec.mean()

        if args.cov > 0:
            cov_term = (canvas.mean() - real_ink).pow(2)
            loss = loss + args.cov * cov_term

        # per-region cov: match the spatial ink distribution of real faces
        # (more ink in eye/mouth regions, less in corners). Critical for forcing
        # letter placement to follow face structure, not just match overall ink.
        if args.region_cov > 0:
            # canvas: [B, 1, 128, 128] → [B, 1, n_reg, n_reg] of mean ink per region
            ps = 128 // args.n_regions
            canvas_reg = F.avg_pool2d(canvas, ps)                       # [B, 1, n_reg, n_reg]
            # sample bgen random real faces for region targets
            ridx = torch.randint(0, N_face, (canvas.shape[0],))
            real_reg = F.avg_pool2d(bank_gpu[ridx].unsqueeze(1), ps)    # [B, 1, n_reg, n_reg]
            loss = loss + args.region_cov * F.mse_loss(canvas_reg, real_reg)

        # per-region IoU: force the canvas to have ink IN THE SAME REGIONS as
        # the target face. Encourages eyes/mouth/hair regions to be filled.
        if args.region_iou > 0:
            ps = 128 // args.n_regions
            ridx2 = torch.randint(0, N_face, (canvas.shape[0],))
            tgt = bank_gpu[ridx2]                                       # [B, 128, 128]
            cR = F.avg_pool2d(canvas[:, 0].unsqueeze(1), ps).squeeze(1)
            tR = F.avg_pool2d(tgt.unsqueeze(1), ps).squeeze(1)
            inter = (cR * tR).sum(dim=(-1, -2))
            union = cR.sum(dim=(-1, -2)) + tR.sum(dim=(-1, -2)) - inter
            loss = loss + args.region_iou * (1.0 - inter / (union + 1e-6)).mean()
        if sharp_w > 0:
            sharp_term = (1.0 - 4.0 * (canvas - 0.5).pow(2)).mean()
            loss = loss + sharp_w * sharp_term
        if args.tv > 0:
            dx = (canvas[:, :, :, 1:] - canvas[:, :, :, :-1]).abs().mean()
            dy = (canvas[:, :, 1:, :] - canvas[:, :, :-1, :]).abs().mean()
            loss = loss + args.tv * (dx + dy)

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        if args.ema > 0:
            update_ema(G_ema, G, args.ema)

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                bvar = canvas.var(dim=0).mean().item()
                ink = canvas.mean().item()
                r_mean = theta[..., 0].mean().item()
                r_std = theta[..., 0].std().item()
            print(f"step={step} loss={loss.item():.3f} ink={ink:.3f} bvar={bvar:.4f} "
                  f"r=[{r_mean:.2f}±{r_std:.2f}] grad={gnorm.item():.2f} "
                  f"t={time.time()-t0:.0f}s", flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            G_for_vis = G_ema if args.ema > 0 else G
            _save_grid(G_for_vis, fixed_eps, fixed_labels, letter_bank_gpu, bank_gpu,
                       out / f'letters_step{step:05d}.png', device, N_per)

    torch.save({'G': G.state_dict(), 'G_ema': G_ema.state_dict(),
                'args': vars(args)}, out / 'G_final.pt')
    print(f"[lp-v2] done ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def _save_grid(G, fixed_eps, fixed_labels, letter_bank_gpu, face_bank,
               path, device, N_per):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import numpy as np
    G.eval()
    theta = G(fixed_eps, fixed_labels)
    Nf, K = fixed_labels.shape
    idx = torch.randint(0, N_per, (Nf, K), device=device)
    flat_lab = fixed_labels.reshape(-1)
    flat_idx = idx.reshape(-1)
    flat_imgs = letter_bank_gpu[flat_lab, flat_idx].unsqueeze(1)
    r = theta[..., 0].reshape(-1)
    tx = theta[..., 1].reshape(-1)
    ty = theta[..., 2].reshape(-1)
    placed = stn_place(flat_imgs, 1.0 / r, tx, ty, 128).view(Nf, K, 128, 128).cpu().numpy()
    canvas = placed.max(axis=1)
    G.train()
    cmap = plt.get_cmap('hsv', max(K, 2))
    colors = np.stack([cmap(k)[:3] for k in range(K)], axis=0)
    ridx = torch.randint(0, face_bank.shape[0], (Nf,))
    real = face_bank[ridx].cpu().numpy()
    labels_np = fixed_labels.cpu().numpy()
    fig, axes = plt.subplots(3, Nf, figsize=(Nf * 1.8, 6), facecolor='#0d0f14')
    for i in range(Nf):
        rgb = np.zeros((128, 128, 3), dtype=np.float32)
        for k in range(K):
            rgb += placed[i, k][..., None] * colors[k][None, None, :]
        rgb = np.clip(rgb, 0, 1)
        axes[0, i].imshow(rgb); axes[0, i].axis('off')
        chars = ''.join(chr(65 + int(c)) for c in labels_np[i])
        axes[0, i].set_title(chars, color='white', fontsize=8, pad=2)
        axes[1, i].imshow(canvas[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    fig.suptitle(f'Letters-v2 (bank letters + placement) K={K} — {path.stem}',
                 color='white', fontsize=11)
    plt.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.01,
                        hspace=0.08, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[lp-v2] grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True)
    ap.add_argument('--letter-bank', default='./checkpoints/letter_bank.pt', dest='letter_bank')
    ap.add_argument('--out', default='./samples/letters_v2')
    ap.add_argument('--K', type=int, default=12)
    ap.add_argument('--dxy-max', type=float, default=0.15, dest='dxy_max',
                    help='max deviation of letter position from default grid (in [-1,1] coords). '
                         'too large → all letters collapse to center.')
    ap.add_argument('--r-min', type=float, default=0.20, dest='r_min')
    ap.add_argument('--r-max', type=float, default=0.50, dest='r_max')
    ap.add_argument('--direct-iou', type=float, default=0.0, dest='direct_iou',
                    help='use direct IoU loss against sampled target face per batch '
                         'element instead of drift_loss. >0 enables (with this as weight).')
    ap.add_argument('--iou-variant', default='iou', choices=['iou', 'dice', 'wl1'],
                    dest='iou_variant', help='which direct loss to use')
    ap.add_argument('--region-cov', type=float, default=0.0, dest='region_cov',
                    help='per-region (n×n patch) ink distribution match weight. '
                         'Critical for forcing face-like layout: more ink in eye/mouth, less in corners.')
    ap.add_argument('--n-regions', type=int, default=4, dest='n_regions',
                    help='number of patches per side for per-region cov (default 4 → 4×4 grid).')
    ap.add_argument('--region-iou', type=float, default=0.0, dest='region_iou',
                    help='per-region IoU between canvas and target face (forces ink in '
                         'the same regions as target — eyes, mouth, etc.).')
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--bgen', type=int, default=256)
    ap.add_argument('--cp', type=int, default=64)
    ap.add_argument('--cn', type=int, default=16)
    ap.add_argument('--gen-per', type=int, default=32, dest='gen_per')
    ap.add_argument('--steps', type=int, default=24000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--R', default='0.005,0.02,0.1')
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--cov', type=float, default=5.0)
    ap.add_argument('--tv', type=float, default=0.03)
    ap.add_argument('--sharpness', type=float, default=0.0,
                    help='sharpness penalty (letters are already binary, so usually 0)')
    ap.add_argument('--sharpness-end', type=float, default=None, dest='sharpness_end')
    ap.add_argument('--noise-aug', type=float, default=0.05, dest='noise_aug')
    ap.add_argument('--topk-pos', type=int, default=0, dest='topk_pos')
    ap.add_argument('--topk-neg', type=int, default=0, dest='topk_neg')
    ap.add_argument('--log-every', type=int, default=100, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=2000, dest='sample_every')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
