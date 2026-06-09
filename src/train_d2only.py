"""D2-only 像素空间 drifting 实验。

目的（按 YX 要求）：完全不考虑每个图层是不是字母，只要求 K 个自由图层
max-compose 出来的画布在**像素空间**上向 CelebA 描边分布做 drifting。
即只有第二个 loss（D2），验证“纯像素空间对描边做 drifting 会不会出问题”。

与成熟版 train.py 的区别：
  - 无字母 / 无 STN / 无 D1 / 无 classifier / 无 repulsion；
  - 生成器是 letter-agnostic 的卷积解码器：eps → [B,K,256,256] → max over K → [B,1,256,256]；
  - D2 = drift_loss，把这一批生成画布当作粒子，拉向采样的真实描边、彼此排斥。

用法：
  python src/train_d2only.py --celeba-root ./data/celeba_full --smoke
  python src/train_d2only.py --celeba-root ./data/celeba_full --steps 4000 --pool 1
"""
from __future__ import annotations
import argparse, sys, time, glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss
from celeba_edges import CelebAEdgeDataset


# ─── 生成器：eps → K 个自由图层 → max 合成 ────────────────────────────────────

def _up(cin, cout):
    """上采样块：2x 最近邻 + 3x3 conv + GroupNorm + GELU。"""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(cin, cout, 3, 1, 1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class PixelGenerator(nn.Module):
    """eps[B,d] → [B,K,256,256]（每层 sigmoid）→ canvas = max over K → [B,1,256,256]。

    K 个图层 + max 合成 = 复用项目的“拼图”结构，但图层是自由的（不约束成字母）。
    """
    def __init__(self, d_noise: int = 128, K: int = 16, base: int = 48, out_size: int = 256):
        super().__init__()
        self.d_noise = d_noise
        self.K = K
        self.out_size = out_size
        self.fc = nn.Linear(d_noise, base * 8 * 4 * 4)
        self.base = base
        # 4 → 8 → 16 → 32 → 64 → 128 → 256 （6 次上采样）
        self.up = nn.Sequential(
            _up(base * 8, base * 8),
            _up(base * 8, base * 4),
            _up(base * 4, base * 2),
            _up(base * 2, base * 2),
            _up(base * 2, base),
            _up(base, base),
        )
        self.head = nn.Conv2d(base, K, 3, 1, 1)

    def forward(self, eps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B = eps.shape[0]
        x = self.fc(eps).view(B, self.base * 8, 4, 4)
        x = self.up(x)                          # [B, base, 256, 256]
        layers = torch.sigmoid(self.head(x))    # [B, K, 256, 256] ∈ (0,1)
        canvas = layers.max(dim=1, keepdim=True).values   # [B,1,256,256]
        return canvas, layers


# ─── 描边正样本库 ─────────────────────────────────────────────────────────────

def build_edge_bank(celeba_root: str, size: int, max_n: int,
                    cache_path: str | None) -> torch.Tensor:
    """从 CelebA 生成描边库，存成 uint8 [N, size, size]（省内存；二值图）。"""
    if cache_path and Path(cache_path).exists():
        print(f"[bank] 从缓存加载 {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=True)

    ds = CelebAEdgeDataset(celeba_root, size=size, max_samples=max_n)  # 默认 REC 档参数
    n = len(ds)
    print(f"[bank] 处理 {n} 张描边 @ {size}px …", flush=True)
    bank = torch.zeros(n, size, size, dtype=torch.uint8)
    for i in range(n):
        edge = ds._extract_edge(ds.paths[i])         # [H,W] float ∈ {0,1}
        bank[i] = torch.from_numpy((edge > 0.5).astype(np.uint8) * 255)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n}", flush=True)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank, cache_path)
        print(f"[bank] 已缓存 {cache_path}  shape={tuple(bank.shape)}", flush=True)
    return bank


def _to_feat(x: torch.Tensor, pool: int) -> torch.Tensor:
    """[*,1,H,W] → [*, S]，可选 avg_pool 降维（pool=1 即真·256² 像素空间）。"""
    if pool > 1:
        x = F.avg_pool2d(x, pool)
    return x.flatten(1)


# ─── DINOv2 特征空间 ──────────────────────────────────────────────────────────

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def build_dino(device):
    """冻结的 DINOv2 ViT-S/14（22M），224 输入。返回 (model, mean, std)。"""
    import timm
    m = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True,
                          num_classes=0, img_size=224, dynamic_img_size=True).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return m, mean, std


def dino_encode(dino, mean, std, canvas):
    """[B,1,H,W]∈[0,1] → DINO 特征 [B,384]（保留梯度）。"""
    x = F.interpolate(canvas, size=224, mode='bilinear', align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    x = (x - mean) / std
    return dino(x)


def precompute_dino_feats(dino, mean, std, bank, device, cache_path, chunk=128):
    """把整个描边库编码成 DINO 特征 [N,384]（一次性，no_grad，缓存）。"""
    if cache_path and Path(cache_path).exists():
        print(f"[dino] 从缓存加载特征 {cache_path}", flush=True)
        return torch.load(cache_path, weights_only=True)
    N = bank.shape[0]
    print(f"[dino] 预编码 {N} 张真实描边 → DINO 特征…", flush=True)
    feats = []
    with torch.no_grad():
        for i in range(0, N, chunk):
            b = (bank[i:i + chunk].float() / 255.0).unsqueeze(1).to(device)
            feats.append(dino_encode(dino, mean, std, b).cpu())
    feats = torch.cat(feats, 0)
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(feats, cache_path)
        print(f"[dino] 已缓存 {cache_path}  shape={tuple(feats.shape)}", flush=True)
    return feats


# ─── 训练 ─────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"[d2only] device={device}", flush=True)
    torch.manual_seed(0)

    size = args.size
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # 描边库
    cache = None if args.smoke else f"checkpoints/edge_bank_{size}.pt"
    max_n = 48 if args.smoke else args.bank_n
    bank = build_edge_bank(args.celeba_root, size, max_n, cache)  # uint8 [N,size,size]
    N = bank.shape[0]
    print(f"[d2only] edge bank: {N} faces", flush=True)

    gen = PixelGenerator(d_noise=args.d_noise, K=args.K, out_size=size).to(device)
    n_p = sum(p.numel() for p in gen.parameters())
    print(f"[d2only] generator params: {n_p/1e6:.2f}M  (K={args.K}, pool={args.pool})", flush=True)
    opt = torch.optim.AdamW(gen.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    # DINO 特征空间：建模型 + 预编码真实库特征（一次性）
    dino = dmean = dstd = real_feats = None
    if args.space == 'dino':
        dino, dmean, dstd = build_dino(device)
        fcache = None if args.smoke else f"checkpoints/dino_feat_{size}_{N}.pt"
        real_feats = precompute_dino_feats(dino, dmean, dstd, bank, device, fcache)  # [N,384] CPU
        print(f"[d2only] DINO 特征空间，dim={real_feats.shape[1]}", flush=True)

    Bgen, Cp, pool = args.bgen, args.cp, args.pool
    steps = 30 if args.smoke else args.steps
    fixed_eps = torch.randn(16, args.d_noise, device=device)  # 固定噪声看进展
    t0 = time.time()

    gen.train()
    for step in range(1, steps + 1):
        eps = torch.randn(Bgen, args.d_noise, device=device)
        canvas, layers = gen(eps)                              # [Bgen,1,H,W]
        idx = torch.randint(0, N, (Cp,))

        if args.space == 'dino':
            gen_feat = dino_encode(dino, dmean, dstd, canvas)  # [Bgen,384] 带梯度
            pos_feat = real_feats[idx].to(device)              # [Cp,384] 预编码
        else:  # pixel
            gen_feat = _to_feat(canvas, pool)                  # [Bgen, S]
            pos = (bank[idx].float() / 255.0).unsqueeze(1).to(device)
            pos_feat = _to_feat(pos, pool)                     # [Cp, S]

        # drift：B=1，把这批生成画布当粒子，拉向真实脸、彼此排斥
        gen_arg = gen_feat.unsqueeze(0)                        # [1, Bgen, S]
        pos_arg = pos_feat.unsqueeze(0)                        # [1, Cp, S]
        loss_vec, info = drift_loss(gen_arg, pos_arg, R_list=args.R_list)
        loss = loss_vec.mean()

        # 显式多样性：奖励 batch 内各样本互不相同（逐像素方差），上限封顶。
        # 直接对治“生成器忽略 eps、所有输出塌成均值脸”。
        div_term = torch.zeros((), device=device)
        if args.div > 0:
            var = canvas.var(dim=0).mean()
            div_term = -torch.clamp(var, max=args.div_cap)   # 越散 → loss 越低
            loss = loss + args.div * div_term

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                ink = canvas.mean().item()
                bvar = canvas.var(dim=0).mean().item()
            print(f"step={step} loss={loss.item():.4f} scale={info['scale'].item():.3f} "
                  f"ink={ink:.3f} bvar={bvar:.4f} div={div_term.item():.4f} "
                  f"grad={gnorm.item():.2f} t={time.time()-t0:.0f}s", flush=True)
            if not torch.isfinite(loss):
                print("[d2only] NaN/Inf，停止"); break

        if step % args.sample_every == 0 or step == steps:
            _save_grid(gen, fixed_eps, bank, out / f"d2_step{step:05d}.png", device)

    torch.save({'gen': gen.state_dict(), 'step': step, 'args': vars(args)},
               out / 'gen_final.pt')
    print(f"[d2only] done. ckpt → {out/'gen_final.pt'}  ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def _save_grid(gen, fixed_eps, bank, path, device):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    gen.eval()
    canvas, _ = gen(fixed_eps)            # [16,1,H,W]
    gen.train()
    imgs = canvas[:, 0].cpu().numpy()
    # 取 8 张真实脸作参照
    ridx = torch.randint(0, bank.shape[0], (8,))
    reals = (bank[ridx].float() / 255.0).numpy()

    fig, axes = plt.subplots(3, 8, figsize=(16, 6), facecolor='#0d0f14')
    for i in range(8):
        axes[0, i].imshow(imgs[i], cmap='gray', vmin=0, vmax=1); axes[0, i].axis('off')
        axes[1, i].imshow(imgs[i + 8], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(reals[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    axes[0, 0].set_ylabel('gen', color='w'); axes[2, 0].set_ylabel('real', color='w')
    fig.suptitle(f'D2-only pixel drift — {path.stem}  (rows1-2: generated, row3: real edges)',
                 color='white', fontsize=12)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[d2only] sample grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--celeba-root', default='./data/celeba_full')
    ap.add_argument('--out', default='samples/d2only')
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--space', default='pixel', choices=['pixel', 'dino'],
                    help='drift 空间：pixel(像素) 或 dino(DINOv2 ViT-S 特征)')
    ap.add_argument('--pool', type=int, default=1, help='drift 前 avg_pool（1=真256²像素空间，仅 pixel 用）')
    ap.add_argument('--K', type=int, default=16, help='自由图层数（max 合成）')
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--bgen', type=int, default=32, help='每步生成粒子数')
    ap.add_argument('--cp', type=int, default=128, help='每步真实正样本数')
    ap.add_argument('--bank-n', type=int, default=8000, dest='bank_n')
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--R', default='0.02,0.05,0.2',
                    help='drift 温度 R_list（逗号分隔）。越小→softmax 越尖→越不易塌成平均脸')
    ap.add_argument('--div', type=float, default=0.0,
                    help='多样性损失权重(奖励 batch 内方差)，>0 打破均值脸塌缩')
    ap.add_argument('--div-cap', type=float, default=0.05, dest='div_cap',
                    help='多样性奖励上限(逐像素方差封顶)')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--log-every', type=int, default=50, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=500, dest='sample_every')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    args.R_list = tuple(float(x) for x in args.R.split(','))
    train(args)


if __name__ == '__main__':
    main()
