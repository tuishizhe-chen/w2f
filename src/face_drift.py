"""D2 face drifting in AE latent space (single-layer backbone, phase 1).

Pipeline:
  G(eps) → latent ẑ ∈ [B, ch, 16, 16]  (single layer)
  drift_loss pulls ẑ toward real face latents (precomputed by face_ae128.py)
  Frozen AE decoder is used ONLY for visualization (decode ẑ → 1×128×128 image).

Stays entirely in pixel-space-equivalent latents (no DINO, no foreign features).
Latent is 16×16×16 = 4096-d: low-dim enough that distances discriminate
(no high-dim mean-collapse) and structured (each cell ≈ a local image region).

Logs `bvar` = batch latent variance. Climbing bvar = generator using eps = no collapse.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss
from face_ae128 import FaceAE128, _up


# ─── 单图层生成器: eps → 一个 16x16xch 的 latent ──────────────────────────────

class LatentGen(nn.Module):
    """单图层：eps[B,d] → conv decoder → [B, ch, 16, 16] latent。"""
    def __init__(self, d_noise: int = 128, ch: int = 16, base: int = 256):
        super().__init__()
        self.d_noise = d_noise
        self.base = base
        self.fc = nn.Linear(d_noise, base * 4 * 4)
        # 加深 + 加宽：让 eps→latent 有足够容量编码多样性
        self.up = nn.Sequential(
            _up(base, base),                 # 4→8
            _up(base, base),                 # 8→16
        )
        self.mid = nn.Sequential(
            nn.Conv2d(base, base, 3, 1, 1), nn.GroupNorm(8, base), nn.GELU(),
            nn.Conv2d(base, base, 3, 1, 1), nn.GroupNorm(8, base), nn.GELU(),
        )
        self.head = nn.Conv2d(base, ch, 3, 1, 1)

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        B = eps.shape[0]
        x = self.fc(eps).view(B, self.base, 4, 4)
        x = self.up(x)
        x = self.mid(x)
        return self.head(x)                                          # [B, ch, 16, 16]


# ─── 训练 ─────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[drift] device={device}", flush=True)

    # 真实 latents（face_ae128 编码好的银行）
    bank_lat = torch.load(args.lat, weights_only=True)               # [N, ch, 16, 16]
    N, ch, ls, _ = bank_lat.shape
    S = ch * ls * ls                                                  # latent flat dim
    print(f"[drift] bank latents {tuple(bank_lat.shape)}  S={S}", flush=True)

    # 冻结的 AE 解码器（只用于 viz）
    ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=True)
    ae_base = ae_ck.get('base', args.ae_base)
    ae_vae = ae_ck.get('vae', False)
    ae = FaceAE128(ch=ae_ck['lat_ch'], base=ae_base, vae=ae_vae).to(device)
    ae.load_state_dict(ae_ck['ae'])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    G = LatentGen(d_noise=args.d_noise, ch=ch).to(device)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[drift] G params: {n_p/1e6:.2f}M  Bgen={args.bgen} Cp={args.cp}", flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))

    R_list = tuple(float(x) for x in args.R.split(','))
    fixed_eps = torch.randn(16, args.d_noise, device=device)         # for vis
    t0 = time.time()

    G.train()
    for step in range(1, args.steps + 1):
        eps = torch.randn(args.bgen, args.d_noise, device=device)
        gen_lat = G(eps)                                              # [Bgen, ch, ls, ls]
        gen_feat = gen_lat.flatten(1)                                 # [Bgen, S]

        # 每个粒子自己一个 drift 问题（B=Bgen, Cg=1）→ 各 eps 被拉向自己
        # 的最近真实邻居（mode-cover），不被同 batch 其他粒子的 swarm 主导。
        # 用 Cn 个**随机其他生成粒子**当 fixed_neg（提供必要的斥力分母）。
        B = args.bgen
        gen_arg = gen_feat.unsqueeze(1)                               # [B, 1, S]
        pos_idx = torch.randint(0, N, (B, args.cp))
        pos_feat = bank_lat[pos_idx].flatten(2).to(device)            # [B, Cp, S]
        neg_idx = torch.randint(0, B, (B, args.cn))
        neg_feat = gen_feat.detach()[neg_idx]                          # [B, Cn, S]
        loss_vec, info = drift_loss(gen_arg, pos_feat, fixed_neg=neg_feat, R_list=R_list)
        loss = loss_vec.mean()

        # 显式多样性（防止 generator 忽略 eps → 所有 latent 塌成一个点）
        div_term = torch.zeros((), device=device)
        if args.div > 0:
            bvar_t = gen_lat.var(dim=0).mean()
            div_term = -torch.clamp(bvar_t, max=args.div_cap)
            loss = loss + args.div * div_term

        # 锐化：把 gen 拉到“它最近的真实邻居”上（防止 drift 收敛到流形附近但不在流形上）
        sharpen_term = torch.zeros((), device=device)
        if args.sharpen > 0:
            with torch.no_grad():
                # gen_feat [B,S], pos_feat [B,Cp,S]：每个粒子在自己的 Cp 里找最近
                d = torch.cdist(gen_feat.unsqueeze(1), pos_feat).squeeze(1)   # [B, Cp]
                nearest = pos_feat[torch.arange(B, device=device), d.argmin(dim=1)]  # [B, S]
            sharpen_term = (gen_feat - nearest.detach()).pow(2).mean()
            loss = loss + args.sharpen * sharpen_term

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                bvar = gen_lat.var(dim=0).mean().item()
            print(f"step={step} loss={loss.item():.4f} scale={info['scale'].item():.3f} "
                  f"bvar={bvar:.4f} grad={gnorm.item():.2f} t={time.time()-t0:.0f}s",
                  flush=True)
            if not torch.isfinite(loss):
                print("[drift] NaN/Inf, abort"); break

        if step % args.sample_every == 0 or step == args.steps:
            _save_grid(G, ae, fixed_eps, bank_lat, out / f'drift_step{step:05d}.png', device)

    torch.save({'G': G.state_dict(), 'args': vars(args)}, out / 'G_final.pt')
    print(f"[drift] done ({time.time()-t0:.0f}s) → {out/'G_final.pt'}", flush=True)


@torch.no_grad()
def _save_grid(G, ae, fixed_eps, bank_lat, path, device):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    G.eval()
    z = G(fixed_eps)                                                   # [16, ch, ls, ls]
    img = torch.sigmoid(ae.dec(z))[:, 0].cpu().numpy()                 # [16, 128, 128]
    G.train()
    # take 8 real bank samples as reference (decoded too, so the comparison is apples-to-apples)
    ridx = torch.randint(0, bank_lat.shape[0], (8,))
    real_z = bank_lat[ridx].to(device)
    real_img = torch.sigmoid(ae.dec(real_z))[:, 0].cpu().numpy()

    fig, axes = plt.subplots(3, 8, figsize=(16, 6), facecolor='#0d0f14')
    for i in range(8):
        axes[0, i].imshow(img[i], cmap='gray', vmin=0, vmax=1); axes[0, i].axis('off')
        axes[1, i].imshow(img[i + 8], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real_img[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    fig.suptitle(f'D2 latent drift — {path.stem}   (rows 1-2: generated decode, row 3: real decode)',
                 color='white', fontsize=11)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[drift] grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat', default='./checkpoints/face_lat.pt')
    ap.add_argument('--ae-ckpt', default='./checkpoints/face_ae128.pt', dest='ae_ckpt')
    ap.add_argument('--ae-base', type=int, default=48, dest='ae_base',
                    help='base channels of the AE checkpoint (must match how it was trained)')
    ap.add_argument('--out', default='./samples/face_drift')
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--bgen', type=int, default=4096)
    ap.add_argument('--cp', type=int, default=64, help='reals per gen particle (per-particle problem)')
    ap.add_argument('--cn', type=int, default=8, help='random other-gen negatives per particle')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--R', default='0.02,0.05,0.2')
    ap.add_argument('--div', type=float, default=0.0,
                    help='diversity weight on latent batch-variance (>0 prevents collapse)')
    ap.add_argument('--div-cap', type=float, default=0.05, dest='div_cap',
                    help='cap for the diversity reward (per-cell latent variance)')
    ap.add_argument('--sharpen', type=float, default=0.0,
                    help='weight on MSE to nearest real (per particle) — sharpens output onto real manifold')
    ap.add_argument('--log-every', type=int, default=50, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=500, dest='sample_every')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
