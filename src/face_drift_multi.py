"""Multi-layer D2 face drift: G outputs K layers' AE-latents, decoded and max-composed.

Pipeline:
  G(eps) → K latents  [B, K, ch, 16, 16]
  for each layer: frozen AE.dec(latent) → layer image [B, 1, 128, 128]
  canvas = max over K layer images → [B, 1, 128, 128]
  frozen AE.enc(canvas) → composite latent [B, ch, 16, 16]
  per-particle drift loss between composite latent and real face latents
  (optional) brightness penalty on a designated "free layer" so it draws minimally

Visualization: each layer rendered in a distinct color + grayscale composite.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss
from face_ae128 import FaceAE128, _up


class MultiLayerGen(nn.Module):
    """eps → [B, K, ch, 16, 16]  (K layers' AE latents)。

    每层一个独立 head（带不同 bias 初始化）→ 强制每层从一开始就不同,
    避免 K-1 个 head 输出 0 的层塌缩问题。
    """
    def __init__(self, d_noise: int = 128, ch: int = 16, K: int = 4, base: int = 256):
        super().__init__()
        self.K, self.ch, self.base = K, ch, base
        self.fc = nn.Linear(d_noise, base * 4 * 4)
        self.up = nn.Sequential(_up(base, base), _up(base, base))           # 4→16
        self.mid = nn.Sequential(
            nn.Conv2d(base, base, 3, 1, 1), nn.GroupNorm(8, base), nn.GELU(),
            nn.Conv2d(base, base, 3, 1, 1), nn.GroupNorm(8, base), nn.GELU(),
        )
        # K 个独立 head，每个 bias 不同 → 各层启动时输出不同
        self.heads = nn.ModuleList([nn.Conv2d(base, ch, 3, 1, 1) for _ in range(K)])
        torch.manual_seed(0)
        for k, h in enumerate(self.heads):
            nn.init.normal_(h.bias, mean=0.0, std=0.6)        # 不同 head 不同 bias
            nn.init.normal_(h.weight, std=0.05)

    def forward(self, eps: torch.Tensor) -> torch.Tensor:
        B = eps.shape[0]
        x = self.fc(eps).view(B, self.base, 4, 4)
        x = self.up(x)
        x = self.mid(x)
        layers = [h(x) for h in self.heads]                                 # K × [B, ch, 16, 16]
        return torch.stack(layers, dim=1)                                   # [B, K, ch, 16, 16]


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bank_lat = torch.load(args.lat, weights_only=True)              # [N, ch, 16, 16]
    N, ch, ls, _ = bank_lat.shape
    S = ch * ls * ls

    ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=True)
    ae = FaceAE128(ch=ae_ck['lat_ch']).to(device)
    ae.load_state_dict(ae_ck['ae']); ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    G = MultiLayerGen(d_noise=args.d_noise, ch=ch, K=args.K).to(device)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[multi] G params: {n_p/1e6:.2f}M  K={args.K}  Bgen={args.bgen}  device={device}",
          flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))
    R_list = tuple(float(x) for x in args.R.split(','))
    fixed_eps = torch.randn(8, args.d_noise, device=device)
    t0 = time.time()

    G.train()
    for step in range(1, args.steps + 1):
        eps = torch.randn(args.bgen, args.d_noise, device=device)
        z = G(eps)                                                   # [B, K, ch, 16, 16]

        # decode each layer → image, max-compose
        zf = z.view(args.bgen * args.K, ch, ls, ls)                  # [B*K, ch, 16, 16]
        layer_logits = ae.dec(zf).view(args.bgen, args.K, 1, 128, 128)
        layer_imgs = torch.sigmoid(layer_logits)                     # [B, K, 1, 128, 128]
        canvas = layer_imgs.max(dim=1).values                        # [B, 1, 128, 128]

        # re-encode composite, drift in latent
        comp_lat = ae.enc(canvas)                                    # [B, ch, 16, 16]
        comp_feat = comp_lat.flatten(1)                              # [B, S]

        # per-particle drift
        gen_arg = comp_feat.unsqueeze(1)                             # [B, 1, S]
        pos_idx = torch.randint(0, N, (args.bgen, args.cp))
        pos_feat = bank_lat[pos_idx].flatten(2).to(device)           # [B, Cp, S]
        neg_idx = torch.randint(0, args.bgen, (args.bgen, args.cn))
        neg_feat = comp_feat.detach()[neg_idx]                       # [B, Cn, S]
        loss_vec, info = drift_loss(gen_arg, pos_feat, fixed_neg=neg_feat, R_list=R_list)
        loss = loss_vec.mean()

        # optional: brightness penalty on the last "free" layer
        if args.free_pen > 0 and args.K >= 2:
            free_ink = layer_imgs[:, -1].mean()
            loss = loss + args.free_pen * free_ink

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                bvar = comp_feat.var(dim=0).mean().item()
                ink_per_layer = layer_imgs.mean(dim=(0, 2, 3, 4)).cpu().numpy()
            print(f"step={step} loss={loss.item():.4f} scale={info['scale'].item():.3f} "
                  f"bvar={bvar:.4f} ink_layers={[f'{v:.3f}' for v in ink_per_layer.tolist()]} "
                  f"grad={gnorm.item():.2f} t={time.time()-t0:.0f}s", flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            _save_colored(G, ae, fixed_eps, bank_lat, out / f'multi_step{step:05d}.png', device)

    torch.save({'G': G.state_dict(), 'args': vars(args)}, out / 'G_final.pt')
    print(f"[multi] done ({time.time()-t0:.0f}s)  → {out/'G_final.pt'}", flush=True)


@torch.no_grad()
def _save_colored(G, ae, fixed_eps, bank_lat, path, device):
    """Top row: per-sample, each of K layers in a distinct color overlaid.
    Bottom row: grayscale max-composite. Plus one row of real edges for reference."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    G.eval()
    z = G(fixed_eps)                                                # [N, K, ch, 16, 16]
    Nf, K = z.shape[:2]
    zf = z.view(Nf * K, *z.shape[2:])
    layer_imgs = torch.sigmoid(ae.dec(zf)).view(Nf, K, 128, 128).cpu().numpy()
    composite = layer_imgs.max(axis=1)                              # [N, 128, 128]
    # color palette for K layers
    cmap = plt.cm.get_cmap('hsv', max(K, 2))
    colors = np.stack([cmap(k)[:3] for k in range(K)], axis=0)      # [K, 3]
    G.train()

    # real samples for reference (decoded)
    ridx = torch.randint(0, bank_lat.shape[0], (Nf,))
    real_dec = torch.sigmoid(ae.dec(bank_lat[ridx].to(device))).cpu().numpy()[:, 0]

    fig, axes = plt.subplots(3, Nf, figsize=(Nf * 1.8, 6), facecolor='#0d0f14')
    for i in range(Nf):
        rgb = np.zeros((128, 128, 3), dtype=np.float32)
        for k in range(K):
            rgb += layer_imgs[i, k][..., None] * colors[k][None, None, :]
        rgb = np.clip(rgb, 0, 1)
        axes[0, i].imshow(rgb); axes[0, i].axis('off')
        axes[1, i].imshow(composite[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real_dec[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    axes[0, 0].set_ylabel('per-layer color', color='w', fontsize=8)
    axes[1, 0].set_ylabel('composite', color='w', fontsize=8)
    axes[2, 0].set_ylabel('real (decoded)', color='w', fontsize=8)
    fig.suptitle(f'Multi-layer D2 drift K={K} — {path.stem}', color='white', fontsize=11)
    plt.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[multi] colored grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat', default='./checkpoints/face_lat.pt')
    ap.add_argument('--ae-ckpt', default='./checkpoints/face_ae128.pt', dest='ae_ckpt')
    ap.add_argument('--out', default='./samples/face_drift_multi')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--bgen', type=int, default=512)
    ap.add_argument('--cp', type=int, default=64)
    ap.add_argument('--cn', type=int, default=8)
    ap.add_argument('--steps', type=int, default=8000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--R', default='0.02,0.05,0.2')
    ap.add_argument('--free-pen', type=float, default=0.0, dest='free_pen',
                    help='brightness penalty on layer K-1 (treat as the free-scribble layer)')
    ap.add_argument('--log-every', type=int, default=50, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=500, dest='sample_every')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
