"""K-layer pixel-space drift: G outputs K free layers, max-composed → canvas,
drift loss vs face bank. No letter constraint — each layer is whatever the
model wants. Tests if a K-layer architecture can compose face-like canvases
using only the face drift signal.

Uses all the sweep14 winning tricks (curriculum, refine head, smallR, etc.).
Per-layer bias init differently → forces K layers to start distinct (anti-collapse).
"""
from __future__ import annotations
import argparse, copy, sys, time
from pathlib import Path
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss
from face_ae128 import _up


class MultiLayerPixelGen(nn.Module):
    """eps[B,d] → K layers each [B,1,128,128] in [0,1].
    Shared backbone + K independent heads with different bias init.
    Optional: prior_alpha > 0 adds anatomical log-prior bias to each head's logits
    (breaks K-permutation symmetry — head k naturally inks near anchor[k])."""
    def __init__(self, d_noise: int = 128, base: int = 192, K: int = 12,
                 sigmoid_t: float = 1.0, head_refine: int = 2,
                 prior_alpha: float = 0.0, prior_sigma_scale: float = 1.0,
                 hard_fovea: bool = False):
        super().__init__()
        assert K >= 1
        self.K = K
        self.base = base
        self.sigmoid_t = sigmoid_t
        self.prior_alpha = prior_alpha
        self.hard_fovea = hard_fovea
        # anatomical priors (lazy: only build if needed)
        if prior_alpha > 0 or hard_fovea:
            from face_regions import build_log_priors, build_hard_mask
            assert K in (6, 8, 12), 'spatial-prior mode supports K∈{6,8,12} anchors'
            log_p = build_log_priors(128, 128, sigma_scale=prior_sigma_scale, K=K)
            self.register_buffer('log_priors', log_p)
            if hard_fovea:
                mask = build_hard_mask(128, 128, sigma_scale=prior_sigma_scale,
                                       threshold=0.05, K=K)
                self.register_buffer('hard_mask', mask)
        self.fc = nn.Linear(d_noise, base * 4 * 4)
        # 4 → 8 → 16 → 32 → 64 → 128 (5 upsamples for 128 res)
        self.up = nn.Sequential(
            _up(base, base), _up(base, base),
            _up(base, base // 2), _up(base // 2, base // 4),
            _up(base // 4, base // 8),
        )
        head_in = base // 8
        # K independent heads with refine — different bias per head forces
        # layers to start distinct (avoids the K-1 layers all output 0 collapse).
        self.heads = nn.ModuleList()
        for k in range(K):
            if head_refine > 0:
                layers = []
                for _ in range(head_refine):
                    layers += [nn.Conv2d(head_in, head_in, 3, 1, 1), nn.GELU()]
                layers += [nn.Conv2d(head_in, 1, 1, 1, 0)]
                head = nn.Sequential(*layers)
                # bias of final 1×1 conv differs per layer
                with torch.no_grad():
                    head[-1].bias.normal_(mean=-2.0 + 4.0 * k / max(1, K - 1), std=0.3)
            else:
                head = nn.Conv2d(head_in, 1, 3, 1, 1)
                with torch.no_grad():
                    head.bias.normal_(mean=-2.0 + 4.0 * k / max(1, K - 1), std=0.3)
            self.heads.append(head)

    def forward(self, eps):
        B = eps.shape[0]
        x = self.fc(eps).view(B, self.base, 4, 4)
        x = self.up(x)                                                # [B, base/8, 128, 128]
        layer_logits = torch.stack([h(x) for h in self.heads], dim=1)  # [B, K, 1, 128, 128]
        # anatomical-prior bias (additive, decayable from outside via self.prior_alpha)
        if self.prior_alpha > 0:
            layer_logits = layer_logits + self.prior_alpha * self.log_priors.unsqueeze(0)
        if self.hard_fovea:
            layer_logits = layer_logits.masked_fill(~self.hard_mask.unsqueeze(0), -8.0)
        layers = torch.sigmoid(layer_logits * self.sigmoid_t)
        return layers                                                  # [B, K, 1, 128, 128]


def update_ema(ema, online, decay):
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), online.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bank = torch.load(args.bank, weights_only=True)
    N = bank.shape[0]; S_bank = bank.shape[-1]
    assert S_bank == 128
    bank_gpu = (bank.float() / 255.0).to(device)
    real_ink = bank_gpu.mean().item()
    print(f"[multi-px] bank {tuple(bank.shape)}  real_ink={real_ink:.4f}", flush=True)

    G = MultiLayerPixelGen(d_noise=args.d_noise, base=args.base, K=args.K,
                            sigmoid_t=args.sigmoid_t, head_refine=args.head_refine,
                            prior_alpha=args.prior_alpha,
                            prior_sigma_scale=args.prior_sigma_scale,
                            hard_fovea=bool(args.hard_fovea)).to(device)
    G_ema = copy.deepcopy(G).to(device)
    for p in G_ema.parameters(): p.requires_grad_(False)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[multi-px] G params: {n_p/1e6:.2f}M  K={args.K}  Bgen={args.bgen}", flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))
    R_list = tuple(float(x) for x in args.R.split(','))
    fixed_eps = torch.randn(8, args.d_noise, device=device)
    t0 = time.time()

    # precompute coord grids for locality 2nd-moment penalty
    if args.locality > 0:
        ys_g = torch.linspace(0., 1., 128, device=device).view(1, 1, 128, 1)
        xs_g = torch.linspace(0., 1., 128, device=device).view(1, 1, 1, 128)
    G.train()
    for step in range(1, args.steps + 1):
        prog = step / max(1, args.steps)
        if args.sigmoid_t_end is not None:
            cur_sig_t = args.sigmoid_t + (args.sigmoid_t_end - args.sigmoid_t) * prog
            G.sigmoid_t = cur_sig_t; G_ema.sigmoid_t = cur_sig_t
        sharp_w = args.sharpness if args.sharpness_end is None else \
                  args.sharpness + (args.sharpness_end - args.sharpness) * prog
        # prior_alpha curriculum: start strong, decay to keep anatomical bias
        if args.prior_alpha_end is not None and args.prior_alpha > 0:
            cur_pa = args.prior_alpha + (args.prior_alpha_end - args.prior_alpha) * prog
            G.prior_alpha = cur_pa
            G_ema.prior_alpha = cur_pa

        eps = torch.randn(args.bgen, args.d_noise, device=device)
        layers = G(eps)                                            # [B, K, 1, 128, 128]
        sum_layers = layers.sum(dim=1)                             # [B, 1, 128, 128]
        canvas = sum_layers.clamp(0.0, 1.0)
        # PAIRWISE overlap penalty: sum of (layer_i * layer_j) for all i<j.
        # Equivalent to (sum² - Σ layer²) / 2 — vectorized and cheap.
        # Pushes layers to be DISJOINT (each pixel claimed by at most one layer).
        sum_sq = sum_layers.pow(2)
        sq_sum = layers.pow(2).sum(dim=1)
        overlap_val = ((sum_sq - sq_sum) / 2.0).mean()

        # drift on the COMPOSITE canvas — pixel-space, same recipe as face_drift_pixel
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
            dx = (canvas[:, :, :, 1:] - canvas[:, :, :, :-1]).abs().mean()
            dy = (canvas[:, :, 1:, :] - canvas[:, :, :-1, :]).abs().mean()
            loss = loss + args.tv * (dx + dy)
        # pairwise overlap penalty — user wants this LIGHT (face is hardest constraint)
        if args.overlap > 0:
            loss = loss + args.overlap * overlap_val
        # locality: per-layer ink-weighted 2nd-moment around its centroid.
        # Encourages each layer's ink to cluster spatially (not scattered strokes).
        if args.locality > 0:
            w = layers.squeeze(2)                                   # [B, K, 128, 128]
            mass = w.sum(dim=(2, 3)).clamp_min(1e-3)                # [B, K]
            cy = (w * ys_g).sum(dim=(2, 3)) / mass                  # [B, K]
            cx = (w * xs_g).sum(dim=(2, 3)) / mass
            var = (w * ((ys_g - cy[..., None, None]).pow(2)
                       + (xs_g - cx[..., None, None]).pow(2))).sum(dim=(2, 3)) / mass
            loss = loss + args.locality * var.mean()
        # anti-collapse: force EVERY layer to contribute. Without this + with
        # overlap penalty, model finds trivial minimum of "1 layer carries
        # everything, rest are 0" → no real multi-layer behavior.
        if args.layer_min > 0:
            per_layer_ink = layers.mean(dim=(0, 2, 3, 4))           # [K]
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
                per_layer_ink = layers.mean(dim=(0, 2, 3, 4)).cpu().numpy()
            alive = sum(1 for v in per_layer_ink if v > 0.005)
            print(f"step={step} loss={loss.item():.3f} ink={ink:.3f} bvar={bvar:.4f} "
                  f"alive={alive}/{args.K} ovl={overlap_val.item():.4f} "
                  f"grad={gnorm.item():.2f} t={time.time()-t0:.0f}s", flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            G_for_vis = G_ema if args.ema > 0 else G
            _save_grid(G_for_vis, fixed_eps, bank_gpu, out / f'multi_step{step:05d}.png', device, args.K)

    torch.save({'G': G.state_dict(), 'G_ema': G_ema.state_dict(), 'args': vars(args)},
               out / 'G_final.pt')
    print(f"[multi-px] done ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def _save_grid(G, fixed_eps, bank, path, device, K):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import numpy as np
    G.eval()
    layers = G(fixed_eps).cpu().numpy()                              # [N, K, 1, 128, 128]
    canvas = layers.max(axis=1)[:, 0]                                # [N, 128, 128]
    G.train()
    cmap = plt.get_cmap('hsv', max(K, 2))
    colors = np.stack([cmap(k)[:3] for k in range(K)], axis=0)
    ridx = torch.randint(0, bank.shape[0], (fixed_eps.shape[0],))
    real = bank[ridx].cpu().numpy()
    fig, axes = plt.subplots(3, fixed_eps.shape[0],
                             figsize=(fixed_eps.shape[0] * 1.8, 6), facecolor='#0d0f14')
    for i in range(fixed_eps.shape[0]):
        rgb = np.zeros((128, 128, 3), dtype=np.float32)
        for k in range(K):
            rgb += layers[i, k, 0][..., None] * colors[k][None, None, :]
        rgb = np.clip(rgb, 0, 1)
        axes[0, i].imshow(rgb); axes[0, i].axis('off')
        axes[1, i].imshow(canvas[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    fig.suptitle(f'Multi-layer pixel drift K={K} — {path.stem}', color='white', fontsize=11)
    plt.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[multi-px] grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True)
    ap.add_argument('--out', default='./samples/multi_pixel')
    ap.add_argument('--K', type=int, default=12)
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
    ap.add_argument('--overlap', type=float, default=5.0,
                    help='pairwise overlap penalty weight Σ_{i<j} (layer_i * layer_j). '
                         'User priority: face > letter > no-overlap. Keep LOW.')
    ap.add_argument('--prior-alpha', type=float, default=0.0, dest='prior_alpha',
                    help='spatial-prior (anatomical anchor) additive log-prior bias on '
                         'head logits. >0 breaks K-permutation symmetry. K=12 only.')
    ap.add_argument('--prior-alpha-end', type=float, default=None, dest='prior_alpha_end',
                    help='if set, linear decay of prior_alpha from initial to this value.')
    ap.add_argument('--prior-sigma-scale', type=float, default=1.0, dest='prior_sigma_scale',
                    help='scale on the Gaussian sigma of each anchor prior.')
    ap.add_argument('--hard-fovea', type=int, default=0, dest='hard_fovea',
                    help='if 1: mask logits to -8 outside prior > 0.05 (structurally bound).')
    ap.add_argument('--locality', type=float, default=0.0,
                    help='per-layer 2nd-moment locality penalty weight; in normalized '
                         '[0,1]^2 coords so typical scale 5e-4 to 5e-3.')
    ap.add_argument('--layer-min', type=float, default=0.0, dest='layer_min',
                    help='min mean ink per layer; layers below this get penalized (anti-collapse). '
                         '0 disables — let layers be free.')
    ap.add_argument('--layer-min-w', type=float, default=100.0, dest='layer_min_w',
                    help='weight on the layer-min anti-collapse penalty.')
    ap.add_argument('--log-every', type=int, default=100, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=2000, dest='sample_every')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
