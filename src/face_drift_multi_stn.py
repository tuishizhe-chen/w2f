"""K-layer multi pixel WITH STN per layer.

Each layer is a small patch (e.g., 48×48) + an STN affine (r, tx, ty) that
places it onto the 128 canvas. K layers → STN-placed → sum-composed (clamped).
Pairwise overlap penalty for disjoint regions, layer-min for anti-collapse.

Key vs face_drift_multi_pixel:
  - Each layer is a SMALL PATCH, not a full 128 canvas. Patches are naturally
    spatially compact → layers are concentrated, not scattered strokes.
  - STN affine learnable → layers position themselves on the canvas to form
    different face features.
"""
from __future__ import annotations
import argparse, copy, math, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss
from face_ae128 import _up


def stn_place(patches, inv_scale, tx, ty, canvas_size):
    """patches[N,1,h,w] → placed[N,1,canvas,canvas]."""
    N = patches.shape[0]
    M = torch.zeros(N, 2, 3, device=patches.device, dtype=patches.dtype)
    M[:, 0, 0] = inv_scale
    M[:, 0, 2] = tx
    M[:, 1, 1] = inv_scale
    M[:, 1, 2] = ty
    grid = F.affine_grid(M, size=(N, 1, canvas_size, canvas_size), align_corners=False)
    return F.grid_sample(patches, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)


class MultiStnGen(nn.Module):
    """eps[B,d] → patches[B,K,1,patch,patch] + thetas[B,K,3].
    K-adaptive default grid for placement, model nudges from there."""
    def __init__(self, d_noise: int = 128, base: int = 192, K: int = 12,
                 patch_size: int = 48, sigmoid_t: float = 1.0,
                 head_refine: int = 2, dxy_max: float = 0.30,
                 r_min: float = 0.20, r_max: float = 0.45):
        super().__init__()
        self.K = K
        self.patch_size = patch_size
        self.sigmoid_t = sigmoid_t
        self.dxy_max = dxy_max
        self.r_min, self.r_max = r_min, r_max
        # patch backbone: outputs K patches via shared trunk + K independent heads.
        # patch is 48x48 so we use a smaller backbone (3 upsamples from 4x4 to 32x32 then
        # bilinear up to 48). Actually simpler: produce 64x64 patch then resize/crop.
        # Let's go: fc → [base, 4, 4] → 3 upsamples → 32 → conv to patch_size via bilinear
        self.base = base
        self.fc = nn.Linear(d_noise, base * 4 * 4)
        self.up = nn.Sequential(_up(base, base), _up(base, base), _up(base, base // 2))
        head_in = base // 2
        # K independent heads producing patches of size 32x32; we'll bilinear-up to patch_size
        self.patch_heads = nn.ModuleList()
        for k in range(K):
            if head_refine > 0:
                layers = []
                for _ in range(head_refine):
                    layers += [nn.Conv2d(head_in, head_in, 3, 1, 1), nn.GELU()]
                layers += [nn.Conv2d(head_in, 1, 1, 1, 0)]
                head = nn.Sequential(*layers)
                with torch.no_grad():
                    head[-1].bias.normal_(mean=-1.0 + 2.0 * k / max(1, K - 1), std=0.2)
            else:
                head = nn.Conv2d(head_in, 1, 3, 1, 1)
                with torch.no_grad():
                    head.bias.normal_(mean=-1.0 + 2.0 * k / max(1, K - 1), std=0.2)
            self.patch_heads.append(head)
        # theta predictor: per-K learnable embedding + shared eps trunk
        self.slot_emb = nn.Embedding(K, 32)
        d_in = d_noise + 32
        self.theta_trunk = nn.Sequential(
            nn.Linear(d_in, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.theta_head = nn.Linear(256, 3)
        nn.init.normal_(self.theta_head.weight, std=0.02)
        nn.init.zeros_(self.theta_head.bias)

    def forward(self, eps: torch.Tensor):
        B = eps.shape[0]
        # patches
        x = self.fc(eps).view(B, self.base, 4, 4)
        x = self.up(x)                                                  # [B, head_in, 32, 32]
        patch_logits = torch.stack([h(x) for h in self.patch_heads], dim=1)  # [B, K, 1, 32, 32]
        patches_small = torch.sigmoid(patch_logits * self.sigmoid_t)
        # bilinear upsample to patch_size
        patches = F.interpolate(
            patches_small.view(B * self.K, 1, 32, 32),
            size=(self.patch_size, self.patch_size),
            mode='bilinear', align_corners=False
        ).view(B, self.K, 1, self.patch_size, self.patch_size)
        # thetas (one per (b, k))
        eps_rep = eps.unsqueeze(1).expand(B, self.K, -1).reshape(B * self.K, -1)
        slot_id = torch.arange(self.K, device=eps.device).unsqueeze(0).expand(B, self.K).reshape(-1)
        slot = self.slot_emb(slot_id)
        h_in = torch.cat([eps_rep, slot], dim=1)
        h = self.theta_trunk(h_in)
        raw = self.theta_head(h).view(B, self.K, 3)
        r = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(raw[..., 0])
        dx = self.dxy_max * torch.tanh(raw[..., 1])
        dy = self.dxy_max * torch.tanh(raw[..., 2])
        # K-adaptive default grid
        cols = int(math.ceil(math.sqrt(self.K)))
        rows = int(math.ceil(self.K / cols))
        device = eps.device
        bx = torch.tensor([-0.45 + 0.9 * ((k % cols) + 0.5) / cols for k in range(self.K)], device=device)
        by = torch.tensor([-0.45 + 0.9 * ((k // cols) + 0.5) / rows for k in range(self.K)], device=device)
        tx = (bx.view(1, self.K) + dx).clamp(-0.75, 0.75)
        ty = (by.view(1, self.K) + dy).clamp(-0.75, 0.75)
        theta = torch.stack([r, tx, ty], dim=-1)                       # [B, K, 3]
        return patches, theta


def update_ema(ema, online, decay):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), online.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bank = torch.load(args.bank, weights_only=True)
    N = bank.shape[0]
    bank_gpu = (bank.float() / 255.0).to(device)
    real_ink = bank_gpu.mean().item()
    print(f"[stn-multi] bank {tuple(bank.shape)}  real_ink={real_ink:.4f}", flush=True)

    G = MultiStnGen(d_noise=args.d_noise, base=args.base, K=args.K,
                    patch_size=args.patch_size, sigmoid_t=args.sigmoid_t,
                    head_refine=args.head_refine, dxy_max=args.dxy_max,
                    r_min=args.r_min, r_max=args.r_max).to(device)
    G_ema = copy.deepcopy(G).to(device)
    for p in G_ema.parameters(): p.requires_grad_(False)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[stn-multi] G params: {n_p/1e6:.2f}M  K={args.K} patch={args.patch_size}", flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))
    R_list = tuple(float(x) for x in args.R.split(','))
    fixed_eps = torch.randn(8, args.d_noise, device=device)
    t0 = time.time()

    G.train()
    for step in range(1, args.steps + 1):
        prog = step / max(1, args.steps)
        if args.sigmoid_t_end is not None:
            cur_sig_t = args.sigmoid_t + (args.sigmoid_t_end - args.sigmoid_t) * prog
            G.sigmoid_t = cur_sig_t; G_ema.sigmoid_t = cur_sig_t
        sharp_w = args.sharpness if args.sharpness_end is None else \
                  args.sharpness + (args.sharpness_end - args.sharpness) * prog

        eps = torch.randn(args.bgen, args.d_noise, device=device)
        patches, theta = G(eps)
        # STN-place each patch onto 128 canvas
        ps = args.patch_size
        flat_p = patches.view(args.bgen * args.K, 1, ps, ps)
        r = theta[..., 0].reshape(-1)
        tx = theta[..., 1].reshape(-1)
        ty = theta[..., 2].reshape(-1)
        placed = stn_place(flat_p, 1.0 / r, tx, ty, 128).view(args.bgen, args.K, 1, 128, 128)
        sum_layers = placed.sum(dim=1)                                 # [B, 1, 128, 128]
        canvas = sum_layers.clamp(0.0, 1.0)
        sum_sq = sum_layers.pow(2)
        sq_sum = placed.pow(2).sum(dim=1)
        overlap_val = ((sum_sq - sq_sum) / 2.0).mean()

        # drift loss on the composite canvas
        canvas_feat = canvas.flatten(1)
        Cg = max(1, args.gen_per)
        B = args.bgen // Cg
        gen_arg = canvas_feat[: B * Cg].view(B, Cg, -1)
        pos_idx = torch.randint(0, N, (B, args.cp))
        pos_feat = bank_gpu[pos_idx].flatten(2)
        neg_idx = torch.randint(0, N, (B, args.cn))
        neg_feat = bank_gpu[neg_idx].flatten(2)
        if args.noise_aug > 0:
            t = args.noise_aug
            gen_arg = (1 - t) * gen_arg + t * torch.randn_like(gen_arg)
            pos_feat = (1 - t) * pos_feat + t * torch.randn_like(pos_feat)
            neg_feat = (1 - t) * neg_feat + t * torch.randn_like(neg_feat)
        loss_vec, info = drift_loss(gen_arg, pos_feat, fixed_neg=neg_feat, R_list=R_list)
        loss = loss_vec.mean()

        if args.cov > 0:
            loss = loss + args.cov * (canvas.mean() - real_ink).pow(2)
        if sharp_w > 0:
            loss = loss + sharp_w * (1.0 - 4.0 * (canvas - 0.5).pow(2)).mean()
        if args.tv > 0:
            dx_ = (canvas[:, :, :, 1:] - canvas[:, :, :, :-1]).abs().mean()
            dy_ = (canvas[:, :, 1:, :] - canvas[:, :, :-1, :]).abs().mean()
            loss = loss + args.tv * (dx_ + dy_)
        if args.overlap > 0:
            loss = loss + args.overlap * overlap_val
        if args.layer_min > 0:
            per_layer_ink = placed.mean(dim=(0, 2, 3, 4))
            dead = torch.relu(args.layer_min - per_layer_ink).sum()
            loss = loss + args.layer_min_w * dead

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        if args.ema > 0:
            update_ema(G_ema, G, args.ema)

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                bvar = canvas.var(dim=0).mean().item()
                ink = canvas.mean().item()
                per_layer_ink = placed.mean(dim=(0, 2, 3, 4)).cpu().numpy()
                r_mean = theta[..., 0].mean().item()
            alive = sum(1 for v in per_layer_ink if v > 0.002)
            print(f"step={step} loss={loss.item():.3f} ink={ink:.3f} bvar={bvar:.4f} "
                  f"r={r_mean:.2f} alive={alive}/{args.K} ovl={overlap_val.item():.4f} "
                  f"grad={gnorm.item():.2f} t={time.time()-t0:.0f}s", flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            G_for_vis = G_ema if args.ema > 0 else G
            _save_grid(G_for_vis, fixed_eps, bank_gpu, out / f'stn_step{step:05d}.png',
                       device, args.K, args.patch_size)

    torch.save({'G': G.state_dict(), 'G_ema': G_ema.state_dict(), 'args': vars(args)},
               out / 'G_final.pt')
    print(f"[stn-multi] done ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def _save_grid(G, fixed_eps, bank, path, device, K, ps):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import numpy as np
    G.eval()
    patches, theta = G(fixed_eps)
    Nf = fixed_eps.shape[0]
    flat_p = patches.view(Nf * K, 1, ps, ps)
    r = theta[..., 0].reshape(-1)
    tx = theta[..., 1].reshape(-1)
    ty = theta[..., 2].reshape(-1)
    placed = stn_place(flat_p, 1.0 / r, tx, ty, 128).view(Nf, K, 128, 128).cpu().numpy()
    sum_layers = placed.sum(axis=1)
    canvas = np.clip(sum_layers, 0, 1)
    G.train()
    cmap = plt.get_cmap('hsv', max(K, 2))
    colors = np.stack([cmap(k)[:3] for k in range(K)], axis=0)
    ridx = torch.randint(0, bank.shape[0], (Nf,))
    real = bank[ridx].cpu().numpy()
    fig, axes = plt.subplots(3, Nf, figsize=(Nf * 1.8, 6), facecolor='#0d0f14')
    for i in range(Nf):
        rgb = np.zeros((128, 128, 3), dtype=np.float32)
        for k in range(K):
            rgb += placed[i, k][..., None] * colors[k][None, None, :]
        rgb = np.clip(rgb, 0, 1)
        axes[0, i].imshow(rgb); axes[0, i].axis('off')
        axes[1, i].imshow(canvas[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    fig.suptitle(f'Multi-layer STN K={K} patch={ps} — {path.stem}', color='white', fontsize=11)
    plt.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[stn-multi] grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True)
    ap.add_argument('--out', default='./samples/multi_stn')
    ap.add_argument('--K', type=int, default=12)
    ap.add_argument('--patch-size', type=int, default=48, dest='patch_size')
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--bgen', type=int, default=256)
    ap.add_argument('--base', type=int, default=192)
    ap.add_argument('--cp', type=int, default=64)
    ap.add_argument('--cn', type=int, default=16)
    ap.add_argument('--gen-per', type=int, default=32, dest='gen_per')
    ap.add_argument('--steps', type=int, default=24000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--R', default='0.005,0.02,0.1')
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--cov', type=float, default=5.0)
    ap.add_argument('--tv', type=float, default=0.03)
    ap.add_argument('--sharpness', type=float, default=0.5)
    ap.add_argument('--sharpness-end', type=float, default=4.0, dest='sharpness_end')
    ap.add_argument('--sigmoid-t', type=float, default=1.0, dest='sigmoid_t')
    ap.add_argument('--sigmoid-t-end', type=float, default=3.0, dest='sigmoid_t_end')
    ap.add_argument('--noise-aug', type=float, default=0.10, dest='noise_aug')
    ap.add_argument('--head-refine', type=int, default=2, dest='head_refine')
    ap.add_argument('--overlap', type=float, default=10.0)
    ap.add_argument('--layer-min', type=float, default=0.005, dest='layer_min')
    ap.add_argument('--layer-min-w', type=float, default=100.0, dest='layer_min_w')
    ap.add_argument('--dxy-max', type=float, default=0.30, dest='dxy_max')
    ap.add_argument('--r-min', type=float, default=0.20, dest='r_min')
    ap.add_argument('--r-max', type=float, default=0.45, dest='r_max')
    ap.add_argument('--log-every', type=int, default=100, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=2000, dest='sample_every')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
