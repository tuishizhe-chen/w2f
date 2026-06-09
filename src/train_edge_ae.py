"""域匹配边缘 AE（空间 latent 版）：在边缘库上训练 编码器+解码器。

为什么空间 latent：扁平 256-d 瓶颈会把空间布局挤糊（和 DINO 全局向量同病，只是轻些）。
改成 **16×16×ch 的卷积瓶颈** 保留"笔画在哪"的局部信息 → 重建清晰，且仍是个结构化、
比像素(65536)低得多的 drift 空间(默认 16×16×16=4096)。

DINO 对照(train_dino_decoder)：全局特征重建糊成均值脸 → 不可逆 → 不适合当 latent。
本 AE 在边缘图上自训，latent 既能忠实重建又紧凑结构化，才是合适的 latent / drift 空间。

产出：checkpoints/edge_ae.pt、checkpoints/edge_lat_<N>.pt（[N,ch,16,16]）、samples/edge_ae/recon_*.png
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_d2only import _up


class EdgeEncoder(nn.Module):
    """1×256×256 → [B, ch, 16, 16] 空间 latent（4 次下采样到 16×16）。"""
    def __init__(self, ch: int = 16, base: int = 48):
        super().__init__()
        def dn(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 4, 2, 1), nn.GroupNorm(8, co), nn.GELU())
        self.net = nn.Sequential(dn(1, base), dn(base, base * 2),
                                 dn(base * 2, base * 4), dn(base * 4, base * 4))  # 256→16
        self.to_lat = nn.Conv2d(base * 4, ch, 1)

    def forward(self, x):
        return self.to_lat(self.net(x))   # [B, ch, 16, 16]


class SpatialDecoder(nn.Module):
    """[B, ch, 16, 16] → 1×256×256 logits（4 次上采样）。"""
    def __init__(self, ch: int = 16, base: int = 48):
        super().__init__()
        self.from_lat = nn.Conv2d(ch, base * 4, 3, 1, 1)
        self.up = nn.Sequential(_up(base * 4, base * 4), _up(base * 4, base * 2),
                                _up(base * 2, base), _up(base, base))  # 16→256
        self.head = nn.Conv2d(base, 1, 3, 1, 1)

    def forward(self, z):
        return self.head(self.up(self.from_lat(z)))


class EdgeAE(nn.Module):
    def __init__(self, ch: int = 16, base: int = 48):
        super().__init__()
        self.enc = EdgeEncoder(ch, base)
        self.dec = SpatialDecoder(ch, base)

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


@torch.no_grad()
def save_recon(ae, bank, idx, path, device):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ae.eval()
    x = (bank[idx].float() / 255.0).unsqueeze(1).to(device)
    rec = torch.sigmoid(ae(x)[0])[:, 0].cpu().numpy()
    ae.train()
    tgt = (bank[idx].float() / 255.0).numpy()
    n = len(idx)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4), facecolor='#0d0f14')
    for i in range(n):
        axes[0, i].imshow(tgt[i], cmap='gray', vmin=0, vmax=1); axes[0, i].axis('off')
        axes[1, i].imshow(rec[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
    axes[0, 0].set_ylabel('real', color='w'); axes[1, 0].set_ylabel('recon', color='w')
    fig.suptitle(f'Edge-AE (spatial latent) — {path.stem} (top: real, bottom: recon)',
                 color='white', fontsize=11)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[ae] recon → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='checkpoints/edge_bank_256.pt')
    ap.add_argument('--out', default='samples/edge_ae')
    ap.add_argument('--lat-ch', type=int, default=16, dest='lat_ch',
                    help='空间 latent 通道数（latent = ch×16×16）')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--pos-weight', type=float, default=6.0, dest='pos_weight')
    ap.add_argument('--sample-every', type=int, default=500, dest='sample_every')
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bank = torch.load(args.bank, weights_only=True)   # uint8 [N,256,256]
    N = bank.shape[0]
    ae = EdgeAE(ch=args.lat_ch).to(device)
    n_p = sum(p.numel() for p in ae.parameters())
    latdim = args.lat_ch * 16 * 16
    print(f"[ae] bank={tuple(bank.shape)} latent=16x16x{args.lat_ch}={latdim} "
          f"params={n_p/1e6:.2f}M device={device}", flush=True)
    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, betas=(0.9, 0.95))
    pw = torch.tensor([args.pos_weight], device=device)
    fixed = torch.randint(0, N, (8,))
    t0 = time.time()

    ae.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.bs,))
        x = (bank[idx].float() / 255.0).unsqueeze(1).to(device)
        logits, _ = ae(x)
        loss = F.binary_cross_entropy_with_logits(logits, x, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step={step} bce={loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            save_recon(ae, bank, fixed, out / f'recon_step{step:05d}.png', device)

    torch.save({'ae': ae.state_dict(), 'lat_ch': args.lat_ch}, 'checkpoints/edge_ae.pt')
    ae.eval()
    lats = []
    with torch.no_grad():
        for i in range(0, N, 256):
            x = (bank[i:i + 256].float() / 255.0).unsqueeze(1).to(device)
            lats.append(ae.enc(x).cpu())
    lats = torch.cat(lats, 0)   # [N, ch, 16, 16]
    torch.save(lats, f'checkpoints/edge_lat_{N}.pt')
    print(f"[ae] done ({time.time()-t0:.0f}s). ckpt→checkpoints/edge_ae.pt  "
          f"latents→edge_lat_{N}.pt {tuple(lats.shape)}", flush=True)


if __name__ == '__main__':
    main()
