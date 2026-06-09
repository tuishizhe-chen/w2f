"""Direct pixel-space drift at 64x64 (no AE).

The user's hypothesis: in pixel space distances are too uniform → mean-face collapse.
Going to 64x64 (~4k dim, same as our AE latent) might let drift discriminate better.
Plus adaptive-R: scale R proportionally to the distance std of the batch.

The output is just sigmoid(conv), no decoder. Display the gen tensor directly.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss
from face_ae128 import _up


# ─── alternative kernels for drift's cdist ────────────────────────────────────

def cdist_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """L1 distance, Hamming-style for binary [0,1] images. [B,N,D] x [B,M,D] → [B,N,M]."""
    # chunked to avoid the [B,N,M,D] blowup
    B, N, D = x.shape; M = y.shape[1]
    out = torch.zeros(B, N, M, device=x.device, dtype=x.dtype)
    chunk = 32
    for i in range(0, N, chunk):
        diff = (x[:, i:i + chunk].unsqueeze(2) - y.unsqueeze(1)).abs()   # [B, c, M, D]
        out[:, i:i + chunk] = diff.sum(-1)
    return out


def cdist_iou(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Soft IoU distance = 1 - inter/union. Treats values as occupancy in [0,1]."""
    inter = torch.einsum("bnd,bmd->bnm", x, y)
    xs = x.sum(-1, keepdim=True)                                          # [B, N, 1]
    ys = y.sum(-1).unsqueeze(1)                                           # [B, 1, M]
    union = xs + ys - inter
    iou = inter / (union + 1e-6)
    return 1.0 - iou


def cdist_dice(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Dice distance = 1 - 2·inter / (|x|+|y|). Heavier penalty on partial match than IoU."""
    inter = torch.einsum("bnd,bmd->bnm", x, y)
    xs = x.sum(-1, keepdim=True)
    ys = y.sum(-1).unsqueeze(1)
    return 1.0 - 2.0 * inter / (xs + ys + 1e-6)


def cdist_iou_multiscale(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """IoU at 3 scales: native + 2x avg-pool + 4x avg-pool. Averages distances.
    Reshapes to 64x64 for the pooling; expects S = 64*64.
    Encourages matching at multiple structural scales (fine + coarse)."""
    import torch.nn.functional as F
    Bx, N, S = x.shape; M = y.shape[1]
    side = int(S ** 0.5)
    x_im = x.view(Bx * N, 1, side, side)
    y_im = y.view(Bx * M, 1, side, side)
    out = torch.zeros(Bx, N, M, device=x.device, dtype=x.dtype)
    for ks in (1, 2, 4):
        if ks > 1:
            xp = F.avg_pool2d(x_im, ks).flatten(1).view(Bx, N, -1)
            yp = F.avg_pool2d(y_im, ks).flatten(1).view(Bx, M, -1)
        else:
            xp = x; yp = y
        out = out + cdist_iou(xp, yp)
    return out / 3.0


def cdist_chamfer_l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Chamfer-style L1 in pixel space (= Hamming for binary). Simpler than true Chamfer
    but penalizes pixel-by-pixel mismatch directly, which IoU under-weights for sparse data."""
    B, N, D = x.shape; M = y.shape[1]
    out = torch.zeros(B, N, M, device=x.device, dtype=x.dtype)
    chunk = 16
    for i in range(0, N, chunk):
        diff = (x[:, i:i + chunk].unsqueeze(2) - y.unsqueeze(1)).abs()
        out[:, i:i + chunk] = diff.sum(-1)
    return out


# ─── feature-transform kernels (port of hvae JAX distance_metrics.py) ─────────
# These transform features into a structural space, then take IoU.
# Idea: in transformed space, crisp lines and blurry blobs have very different
# representations → drift force will push toward crisp solutions.

_SOBEL_H = None
_SOBEL_V = None


def _sobel_features(im: torch.Tensor) -> torch.Tensor:
    """im [B*N, 1, 64, 64] → [B*N, S']. Concat |dx|, |dy| as features."""
    global _SOBEL_H, _SOBEL_V
    if _SOBEL_H is None or _SOBEL_H.device != im.device:
        kh = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                          dtype=torch.float32, device=im.device) / 4.0
        _SOBEL_H = kh.view(1, 1, 3, 3)
        _SOBEL_V = kh.t().contiguous().view(1, 1, 3, 3)
    import torch.nn.functional as F
    dx = F.conv2d(im, _SOBEL_H, padding=1).abs()
    dy = F.conv2d(im, _SOBEL_V, padding=1).abs()
    return torch.cat([dx, dy], dim=1).flatten(1)


def cdist_gradient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """IoU in Sobel-magnitude space. Strongly rewards edge alignment AND sharpness:
    a soft blob has tiny gradients, a crisp line has large gradients in the right place."""
    B, N, S = x.shape; M = y.shape[1]
    side = int(S ** 0.5)
    xg = _sobel_features(x.view(B * N, 1, side, side)).view(B, N, -1)
    yg = _sobel_features(y.view(B * M, 1, side, side)).view(B, M, -1)
    return cdist_iou(xg, yg)


def cdist_lowfreq(x: torch.Tensor, y: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    """L2 on low-pass FFT features. Weights = 1/(1+alpha*|freq|). Strong structural
    emphasis, suppresses high-freq noise — gets the overall face layout right."""
    B, N, S = x.shape; M = y.shape[1]
    side = int(S ** 0.5)
    xi = x.view(B, N, side, side)
    yi = y.view(B, M, side, side)
    fy = torch.fft.fftfreq(side, device=x.device)
    fx = torch.fft.fftfreq(side, device=x.device)
    w = 1.0 / (1.0 + alpha * torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2))   # [H, W]
    Fx = torch.fft.fft2(xi) * w
    Fy = torch.fft.fft2(yi) * w
    Fx_feat = torch.cat([Fx.real, Fx.imag], dim=-1).flatten(2)                 # [B, N, S']
    Fy_feat = torch.cat([Fy.real, Fy.imag], dim=-1).flatten(2)
    # plain L2 on feature space, [B,N,M]
    xn = (Fx_feat ** 2).sum(-1, keepdim=True)
    yn = (Fy_feat ** 2).sum(-1).unsqueeze(1)
    xy = torch.einsum('bnd,bmd->bnm', Fx_feat, Fy_feat)
    sq = xn + yn - 2 * xy
    return torch.sqrt(torch.clamp(sq, min=1e-8))


def cdist_patch(x: torch.Tensor, y: torch.Tensor, ps: int = 4) -> torch.Tensor:
    """IoU after avg-pool downsample (coarse spatial structure)."""
    import torch.nn.functional as F
    B, N, S = x.shape; M = y.shape[1]
    side = int(S ** 0.5)
    xp = F.avg_pool2d(x.view(B * N, 1, side, side), ps).flatten(1).view(B, N, -1)
    yp = F.avg_pool2d(y.view(B * M, 1, side, side), ps).flatten(1).view(B, M, -1)
    return cdist_iou(xp, yp)


def cdist_iou_grad(x: torch.Tensor, y: torch.Tensor, w: float = 0.5) -> torch.Tensor:
    """Mix: w * gradient_iou + (1-w) * iou. Pixel-overlap for layout + gradient-iou
    for sharpness. The shared scale (both in [0,1]) makes the weighted sum stable."""
    return w * cdist_gradient(x, y) + (1.0 - w) * cdist_iou(x, y)


def patch_drift_loss_kernel(kernel: str):
    """Monkey-patch drift_loss._cdist to use a custom kernel."""
    import drift_loss as dl_mod
    if kernel == 'l1':
        dl_mod._cdist = cdist_l1
    elif kernel == 'iou':
        dl_mod._cdist = cdist_iou
    elif kernel == 'dice':
        dl_mod._cdist = cdist_dice
    elif kernel == 'iou_ms':
        dl_mod._cdist = cdist_iou_multiscale
    elif kernel == 'chamfer_l1':
        dl_mod._cdist = cdist_chamfer_l1
    elif kernel == 'gradient':
        dl_mod._cdist = cdist_gradient
    elif kernel == 'lowfreq':
        dl_mod._cdist = cdist_lowfreq
    elif kernel == 'patch':
        dl_mod._cdist = cdist_patch
    elif kernel == 'iou_grad':
        dl_mod._cdist = cdist_iou_grad
    elif kernel == 'l2':
        pass     # default
    else:
        raise ValueError(f'unknown kernel: {kernel}')


def update_ema(ema_module, online_module, decay: float):
    with torch.no_grad():
        for ep, p in zip(ema_module.parameters(), online_module.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class PixelGen(nn.Module):
    """eps[B,d] → [B, 1, S, S] in [0,1]. S ∈ {64, 128}. sigmoid_t > 1 sharpens.
    head_refine > 0 adds that many extra 3×3 convs (no GroupNorm) at full res to
    give the head logit headroom for crisp 1-px edges."""
    def __init__(self, d_noise: int = 128, base: int = 128, size: int = 64,
                 sigmoid_t: float = 1.0, head_refine: int = 0):
        super().__init__()
        assert size in (64, 128), f'unsupported size {size}'
        self.base = base
        self.size = size
        self.sigmoid_t = sigmoid_t
        self.fc = nn.Linear(d_noise, base * 4 * 4)
        if size == 64:
            # 4 → 8 → 16 → 32 → 64
            self.up = nn.Sequential(
                _up(base, base), _up(base, base),
                _up(base, base // 2), _up(base // 2, base // 4),
            )
            head_in = base // 4
        else:  # 128
            # 4 → 8 → 16 → 32 → 64 → 128
            self.up = nn.Sequential(
                _up(base, base), _up(base, base),
                _up(base, base // 2), _up(base // 2, base // 4),
                _up(base // 4, base // 8),
            )
            head_in = base // 8
        if head_refine > 0:
            layers = []
            for _ in range(head_refine):
                layers += [nn.Conv2d(head_in, head_in, 3, 1, 1), nn.GELU()]
            layers += [nn.Conv2d(head_in, 1, 1, 1, 0)]
            self.head = nn.Sequential(*layers)
        else:
            self.head = nn.Conv2d(head_in, 1, 3, 1, 1)

    def forward(self, eps):
        B = eps.shape[0]
        x = self.fc(eps).view(B, self.base, 4, 4)
        x = self.up(x)
        return torch.sigmoid(self.head(x) * self.sigmoid_t)           # [B, 1, S, S]

    def forward_with_logits(self, eps):
        """Diagnostic: returns (logits, sigmoid(logits))."""
        B = eps.shape[0]
        x = self.fc(eps).view(B, self.base, 4, 4)
        x = self.up(x)
        logits = self.head(x) * self.sigmoid_t
        return logits, torch.sigmoid(logits)


# back-compat alias
Pixel64Gen = PixelGen


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.kernel != 'l2':
        patch_drift_loss_kernel(args.kernel)
        print(f"[px] patched drift_loss kernel → {args.kernel}", flush=True)

    bank = torch.load(args.bank, weights_only=True)                   # uint8 [N, S, S]
    N = bank.shape[0]
    S_bank = bank.shape[-1]
    assert S_bank in (64, 128), f'unexpected bank resolution {S_bank}'
    bank_gpu = (bank.float() / 255.0).to(device)                      # [N, S, S] in [0,1]
    real_ink = bank_gpu.mean().item()
    print(f"[px] real ink ratio = {real_ink:.3f}", flush=True)
    print(f"[px] bank {tuple(bank.shape)}", flush=True)

    # bank histogram + per-image max — quick test of "bank is gray" hypothesis
    with torch.no_grad():
        _bins = [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001]
        _bf = bank_gpu.reshape(-1)
        _hist = [((_bf >= lo) & (_bf < hi)).float().mean().item()
                 for lo, hi in zip(_bins[:-1], _bins[1:])]
        _per_img_max = bank_gpu.amax(dim=(1, 2))
    print(f"[diag] bank hist bins={_bins[:-1]} pcts=[" +
          ','.join(f'{h:.3f}' for h in _hist) + ']', flush=True)
    print(f"[diag] bank per-img max: mean={_per_img_max.mean().item():.3f} "
          f"min={_per_img_max.min().item():.3f}", flush=True)

    # binarize positives: drift targets the bank as {0,1} at threshold args.bin_pos.
    # Recompute real_ink for the cov term to match the binarized bank.
    if args.bin_pos > 0:
        bank_gpu = (bank_gpu > args.bin_pos).float()
        real_ink = bank_gpu.mean().item()
        print(f"[diag] binarized bank (>{args.bin_pos}): "
              f"real_ink = {real_ink:.4f}", flush=True)

    G = PixelGen(d_noise=args.d_noise, base=args.base, size=S_bank,
                 sigmoid_t=args.sigmoid_t, head_refine=args.head_refine).to(device)
    import copy as _copy
    G_ema = _copy.deepcopy(G).to(device)
    for p in G_ema.parameters(): p.requires_grad_(False)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[px] G params: {n_p/1e6:.2f}M  Bgen={args.bgen}  Cp={args.cp}  Cn={args.cn}  "
          f"R={args.R} adaptive={args.adaptive_R} ema={args.ema} kernel={args.kernel}",
          flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))

    R_list = tuple(float(x) for x in args.R.split(','))
    fixed_eps = torch.randn(16, args.d_noise, device=device)
    t0 = time.time()

    G.train()
    for step in range(1, args.steps + 1):
        eps = torch.randn(args.bgen, args.d_noise, device=device)
        gen = G(eps)                                                  # [Bgen, 1, 64, 64]
        gen_feat = gen.flatten(1)                                     # [Bgen, 4096]

        # per-problem multi-gen: B_problems × Cg gen samples per problem.
        # The intra-problem old_gen-as-neg gives the dynamic-diversity force.
        Cg = max(1, args.gen_per)
        B = args.bgen // Cg
        gen_arg = gen_feat[: B * Cg].view(B, Cg, -1)                  # [B, Cg, S]
        pos_idx = torch.randint(0, N, (B, args.cp))
        pos_feat = bank_gpu[pos_idx].flatten(2)                       # [B, Cp, S]
        neg_idx = torch.randint(0, N, (B, args.cn))
        neg_feat = bank_gpu[neg_idx].flatten(2)                       # [B, Cn, S]

        # noise augmentation before drift force computation (port of hvae trick:
        # x_noised = (1-t) * x + t * eps). Anti-collapse + smooths loss landscape.
        if args.noise_aug > 0:
            t = args.noise_aug
            gen_arg = (1 - t) * gen_arg + t * torch.randn_like(gen_arg)
            pos_feat = (1 - t) * pos_feat + t * torch.randn_like(pos_feat)
            neg_feat = (1 - t) * neg_feat + t * torch.randn_like(neg_feat)

        # adaptive R: scale R_list by (current dist std / reference)
        if args.adaptive_R:
            with torch.no_grad():
                # estimate dist scale from a small sub-sample
                samp_g = gen_feat[:32].detach()                       # [32, S]
                samp_r = bank_gpu[torch.randint(0, N, (32,))].flatten(1)  # [32, S]
                dgg = torch.cdist(samp_g, samp_g)
                drr = torch.cdist(samp_r, samp_r)
                ref_std = drr[drr > 0].std().item()
                cur_std = (dgg[dgg > 0].std().item() + drr[drr > 0].std().item()) / 2
                R_scale = max(0.1, min(10.0, cur_std / max(ref_std, 1e-3)))
            R_used = tuple(r * R_scale for r in R_list)
        else:
            R_used = R_list

        # curriculum on sigmoid_t and sharpness: ramp linearly from initial value
        # to *_end across training. Lets drift form layout under soft sigmoid,
        # then crisps as the model approaches the end of training.
        prog = step / max(1, args.steps)
        if args.sigmoid_t_end is not None:
            cur_sig_t = args.sigmoid_t + (args.sigmoid_t_end - args.sigmoid_t) * prog
            G.sigmoid_t = cur_sig_t
            G_ema.sigmoid_t = cur_sig_t
        if args.sharpness_end is not None:
            sharp_w = args.sharpness + (args.sharpness_end - args.sharpness) * prog
        else:
            sharp_w = args.sharpness

        loss_vec, info = drift_loss(gen_arg, pos_feat, fixed_neg=neg_feat, R_list=R_used,
                                    topk_pos=args.topk_pos, topk_neg=args.topk_neg)
        loss = loss_vec.mean()

        # coverage penalty: gen ink ratio should be close to real ink ratio.
        # Pushes G away from all-zero (and all-one) sink.
        if args.cov > 0:
            gen_ink = gen.mean()
            cov_term = (gen_ink - real_ink).pow(2)
            loss = loss + args.cov * cov_term

        # sharpness penalty: punish intermediate pixel values. Hardens edges by
        # pushing gen toward 0/1. sharpness>0 enables (or sharpness_end for curriculum).
        if sharp_w > 0:
            # mean of -4*(x-0.5)^2 + 1, which is 1 at 0.5 and 0 at {0,1}.
            sharp_term = (1.0 - 4.0 * (gen - 0.5).pow(2)).mean()
            loss = loss + sharp_w * sharp_term

        # TV (total variation) regularization: suppresses spurious thin strokes
        # by penalizing |∂x|+|∂y|. Cleans up noisy off-manifold lines while
        # keeping coherent strokes (which have low TV per unit length).
        if args.tv > 0:
            dx = (gen[:, :, :, 1:] - gen[:, :, :, :-1]).abs().mean()
            dy = (gen[:, :, 1:, :] - gen[:, :, :-1, :]).abs().mean()
            loss = loss + args.tv * (dx + dy)

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        if args.ema > 0:
            update_ema(G_ema, G, args.ema)

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                bvar = gen.var(dim=0).mean().item()
                ink = gen.mean().item()
                # logit + pixel histogram probe on first 32 samples
                probe_eps = eps[:32]
                logits_probe, gen_probe = G.forward_with_logits(probe_eps)
                L_abs = logits_probe.abs()
                L_abs_mean = L_abs.mean().item()
                L_p99 = L_abs.flatten().quantile(0.99).item()
                _bins = [0.0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001]
                _gp = gen_probe.flatten()
                px_hist = [((_gp >= lo) & (_gp < hi)).float().mean().item()
                           for lo, hi in zip(_bins[:-1], _bins[1:])]
            t_max = info.get('target_max', torch.tensor(float('nan'))).item()
            t_p99 = info.get('target_p99', torch.tensor(float('nan'))).item()
            t_frac = info.get('target_frac_gt_0p5', torch.tensor(float('nan'))).item()
            print(f"step={step} loss={loss.item():.4f} scale={info['scale'].item():.3f} "
                  f"bvar={bvar:.4f} ink={ink:.3f} grad={gnorm.item():.2f} "
                  f"t={time.time()-t0:.0f}s | L_abs={L_abs_mean:.2f} L_p99={L_p99:.2f} "
                  f"px=[" + ','.join(f'{h:.2f}' for h in px_hist) + f"] "
                  f"tgt_max={t_max:.3f} tgt_p99={t_p99:.3f} tgt>.5={t_frac:.3f}",
                  flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            G_for_vis = G_ema if args.ema > 0 else G
            _save_grid(G_for_vis, fixed_eps, bank_gpu, out / f'px_step{step:05d}.png', device)

    torch.save({'G': G.state_dict(), 'G_ema': G_ema.state_dict(), 'args': vars(args)},
               out / 'G_final.pt')
    print(f"[px] done ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def _save_grid(G, fixed_eps, bank, path, device):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    G.eval()
    gen = G(fixed_eps)                                                # [16, 1, 64, 64]
    img = gen[:, 0].cpu().numpy()
    G.train()
    ridx = torch.randint(0, bank.shape[0], (8,))
    real = bank[ridx].cpu().numpy()
    fig, axes = plt.subplots(3, 8, figsize=(16, 6), facecolor='#0d0f14')
    for i in range(8):
        axes[0, i].imshow(img[i], cmap='gray', vmin=0, vmax=1); axes[0, i].axis('off')
        axes[1, i].imshow(img[i + 8], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    fig.suptitle(f'Pixel 64 drift — {path.stem}  (rows 1-2: gen, row 3: real)',
                 color='white', fontsize=11)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[px] grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='./checkpoints/edge_bank_64.pt')
    ap.add_argument('--out', default='./samples/drift_px64')
    ap.add_argument('--base', type=int, default=128)
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--bgen', type=int, default=256)
    ap.add_argument('--cp', type=int, default=32)
    ap.add_argument('--cn', type=int, default=16)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--R', default='0.02,0.05,0.2')
    ap.add_argument('--adaptive-R', action='store_true', dest='adaptive_R',
                    help='scale R per step based on dist std ratio gen/real')
    ap.add_argument('--kernel', default='l2',
                    choices=['l2', 'l1', 'iou', 'dice', 'iou_ms', 'chamfer_l1',
                             'gradient', 'lowfreq', 'patch', 'iou_grad'],
                    help='distance kernel for drift')
    ap.add_argument('--sharpness', type=float, default=0.0,
                    help='penalty weight pushing pixel values away from 0.5 toward {0,1}')
    ap.add_argument('--sigmoid-t', type=float, default=1.0, dest='sigmoid_t',
                    help='temperature scaling on sigmoid pre-activation; >1 sharpens')
    ap.add_argument('--noise-aug', type=float, default=0.0, dest='noise_aug',
                    help='hvae trick: x_noised = (1-t)*x + t*eps before drift. typical 0.05-0.2.')
    ap.add_argument('--tv', type=float, default=0.0,
                    help='total-variation regularization weight (cleans spurious thin strokes).')
    ap.add_argument('--bin-pos', type=float, default=0.0, dest='bin_pos',
                    help='binarize bank/positives at this threshold (e.g. 0.5). 0 disables.')
    ap.add_argument('--topk-pos', type=int, default=0, dest='topk_pos',
                    help='restrict drift softmax attention to top-K nearest positives per gen row.')
    ap.add_argument('--topk-neg', type=int, default=0, dest='topk_neg',
                    help='restrict drift softmax attention to top-K nearest negatives per gen row.')
    ap.add_argument('--sigmoid-t-end', type=float, default=None, dest='sigmoid_t_end',
                    help='final sigmoid_t value (linear ramp from --sigmoid-t over training).')
    ap.add_argument('--sharpness-end', type=float, default=None, dest='sharpness_end',
                    help='final sharpness weight (linear ramp from --sharpness over training).')
    ap.add_argument('--head-refine', type=int, default=0, dest='head_refine',
                    help='add this many extra 3x3 convs (no GroupNorm) at full res in head.')
    ap.add_argument('--ema', type=float, default=0.0,
                    help='EMA decay for generator (>0 enables EMA, e.g. 0.999)')
    ap.add_argument('--cov', type=float, default=10.0,
                    help='coverage penalty weight (push gen ink toward real ink ratio)')
    ap.add_argument('--gen-per', type=int, default=1, dest='gen_per',
                    help='Cg gen samples per drift problem (>1 enables intra-problem repulsion)')
    ap.add_argument('--log-every', type=int, default=100, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=500, dest='sample_every')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
