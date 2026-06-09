"""DINO-latent 可逆性测试：训练一个解码器 Dec: DINO特征(384) → 边缘图(256²)。

这是"用 DINO 当 latent 空间"的成败关键(gate):
  如果能从 DINO 的 384 维全局特征**重建出边缘线稿**，说明该 latent 携带了足够信息，
  之后就能"生成器产 latent → 在 DINO 空间 drift → Dec 解码成图"。
  如果重建很糊/千篇一律，说明 DINO 全局特征对我们这种 OOD 线稿太有损，此路不通
  (届时再考虑用 patch tokens 之类更丰富的特征)。

复用缓存：checkpoints/edge_bank_256.pt (uint8 [N,256,256]) + dino_feat_256_8000.pt ([N,384])。
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_d2only import _up   # 2x上采样块


class DinoDecoder(nn.Module):
    """384-dim DINO latent → 1×256×256 logits。"""
    def __init__(self, d: int = 384, base: int = 48, out_size: int = 256):
        super().__init__()
        self.base = base
        self.fc = nn.Linear(d, base * 8 * 4 * 4)
        self.up = nn.Sequential(
            _up(base * 8, base * 8), _up(base * 8, base * 4), _up(base * 4, base * 2),
            _up(base * 2, base * 2), _up(base * 2, base), _up(base, base),
        )
        self.head = nn.Conv2d(base, 1, 3, 1, 1)

    def forward(self, z):
        x = self.fc(z).view(z.shape[0], self.base * 8, 4, 4)
        x = self.up(x)
        return self.head(x)   # logits [B,1,256,256]


@torch.no_grad()
def save_recon(dec, feats, bank, idx, path, device):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    dec.eval()
    z = feats[idx].to(device)
    rec = torch.sigmoid(dec(z))[:, 0].cpu().numpy()
    dec.train()
    tgt = (bank[idx].float() / 255.0).numpy()
    n = len(idx)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4), facecolor='#0d0f14')
    for i in range(n):
        axes[0, i].imshow(tgt[i], cmap='gray', vmin=0, vmax=1); axes[0, i].axis('off')
        axes[1, i].imshow(rec[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
    axes[0, 0].set_ylabel('real', color='w'); axes[1, 0].set_ylabel('recon', color='w')
    fig.suptitle(f'DINO-latent reconstruction — {path.stem} (top: real edge, bottom: decoded from DINO-384)',
                 color='white', fontsize=11)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[dec] recon grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='checkpoints/edge_bank_256.pt')
    ap.add_argument('--feats', default='checkpoints/dino_feat_256_8000.pt')
    ap.add_argument('--out', default='samples/dino_decoder')
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--bs', type=int, default=32)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--pos-weight', type=float, default=6.0, dest='pos_weight',
                    help='BCE 正类权重(边缘像素稀疏,~12%,加权防止全黑解)')
    ap.add_argument('--sample-every', type=int, default=500, dest='sample_every')
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bank = torch.load(args.bank, weights_only=True)        # uint8 [N,256,256]
    feats = torch.load(args.feats, weights_only=True)      # [N,384]
    N = bank.shape[0]
    print(f"[dec] bank={tuple(bank.shape)} feats={tuple(feats.shape)} device={device}", flush=True)

    dec = DinoDecoder(d=feats.shape[1]).to(device)
    print(f"[dec] decoder params: {sum(p.numel() for p in dec.parameters())/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(dec.parameters(), lr=args.lr, betas=(0.9, 0.95))
    pw = torch.tensor([args.pos_weight], device=device)
    fixed = torch.randint(0, N, (8,))
    t0 = time.time()

    dec.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.bs,))
        z = feats[idx].to(device)
        tgt = (bank[idx].float() / 255.0).unsqueeze(1).to(device)
        logits = dec(z)
        loss = F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step={step} bce={loss.item():.4f} t={time.time()-t0:.0f}s", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            save_recon(dec, feats, bank, fixed, out / f'recon_step{step:05d}.png', device)
    torch.save({'dec': dec.state_dict()}, out / 'dec_final.pt')
    print(f"[dec] done ({time.time()-t0:.0f}s) → {out/'dec_final.pt'}", flush=True)


if __name__ == '__main__':
    main()
