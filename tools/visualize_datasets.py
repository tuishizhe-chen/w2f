"""
数据集可视化脚本
用法:
  python scripts/visualize_datasets.py --mode letters   [--data-root ./data]
  python scripts/visualize_datasets.py --mode faces     --celeba-root /path/to/celeba
  python scripts/visualize_datasets.py --mode both      --celeba-root /path/to/celeba
  python scripts/visualize_datasets.py --mode aug-grid  # 对比各增强等级

输出:
  vis_letters.png   — 原始 EMNIST vs 各级增强对比
  vis_faces.png     — CelebA 原图 + Canny 边缘图
  vis_aug_grid.png  — 同一字母在 mild/normal/aggressive/extreme 下的形变矩阵
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─── 辅助：拼图 ──────────────────────────────────────────────────────────────

def make_grid_np(imgs: list, nrow: int, pad: int = 2, bg: float = 0.15) -> np.ndarray:
    """imgs: list of [H,W] numpy float32，拼成网格。"""
    n = len(imgs)
    H, W = imgs[0].shape
    ncol = (n + nrow - 1) // nrow
    canvas = np.full(
        ((H + pad) * nrow + pad, (W + pad) * ncol + pad),
        bg, dtype=np.float32
    )
    for i, img in enumerate(imgs):
        r, c = i // ncol, i % ncol
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        canvas[y0:y0+H, x0:x0+W] = img
    return canvas


# ─── 模式 1：字母增强对比 ─────────────────────────────────────────────────────

def vis_letters(data_root: str, out_path: str = 'vis_letters.png'):
    from data import _try_load_emnist, _fallback_synth_letters
    from aug_letters import augment_letter

    print("[vis] 加载 EMNIST…")
    raw = _try_load_emnist(data_root, size=32)
    if raw is None:
        print("[vis] EMNIST 不可用，使用 PIL fallback")
        raw = _fallback_synth_letters(size=32, per_class=50)

    rng = np.random.default_rng(0)
    levels = ['mild', 'normal', 'aggressive', 'extreme']
    # 选 8 个字母展示
    show_classes = [0, 4, 7, 10, 14, 17, 20, 24]  # A E H K O R U Y
    n_aug = 4   # 每个等级展示 4 个样本

    fig = plt.figure(figsize=(18, 12), facecolor='#0d0f14')
    fig.suptitle('字母数据增强对比', color='white', fontsize=16, fontweight='bold', y=0.98)

    # 列：原图 + 4个等级×4个样本 = 1+16 = 17列
    ncols = 1 + len(levels) * n_aug
    nrows = len(show_classes)
    gs = gridspec.GridSpec(nrows, ncols, figure=fig,
                           hspace=0.08, wspace=0.04,
                           left=0.04, right=0.98, top=0.93, bottom=0.04)

    col_colors = ['#5b8dee', '#3ecf94', '#f0a030', '#e0608e']
    level_names = ['mild', 'normal', 'aggressive', 'extreme']

    for ri, c in enumerate(show_classes):
        letter_name = chr(ord('A') + c)
        src = raw[c].numpy()  # [N, 32, 32]

        # 原图
        ax = fig.add_subplot(gs[ri, 0])
        ax.imshow(src[0], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
        if ri == 0:
            ax.set_title('原图', color='#8899bb', fontsize=9)
        ax.text(-0.25, 0.5, letter_name, transform=ax.transAxes,
                color='white', fontsize=11, fontweight='bold', va='center', ha='center')

        # 各增强等级
        for li, (lv, col) in enumerate(zip(levels, col_colors)):
            for ai in range(n_aug):
                col_idx = 1 + li * n_aug + ai
                ax = fig.add_subplot(gs[ri, col_idx])
                base = src[ai % len(src)]
                aug = augment_letter(base, level=lv, rng=rng)
                ax.imshow(aug, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                ax.axis('off')
                if ri == 0 and ai == 1:
                    ax.set_title(level_names[li], color=col, fontsize=9, fontweight='bold')
                # 顶部彩色横条
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(col if ai == 0 else '#1c2640')
                    spine.set_linewidth(1.5 if ai == 0 else 0.5)

    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#0d0f14')
    plt.close()
    print(f"[vis] 保存 → {out_path}")


# ─── 模式 2：CelebA 人脸边缘图 ───────────────────────────────────────────────

def vis_faces(celeba_root: str, out_path: str = 'vis_faces.png', n: int = 16):
    from celeba_edges import CelebAEdgeDataset
    import cv2
    from PIL import Image as PILImage

    print("[vis] 加载 CelebA 边缘图…")
    ds = CelebAEdgeDataset(
        celeba_root, size=128, split='train',
        canny_low=25, canny_high=90, blur_sigma=1.2,
        max_samples=n * 4,
    )
    actual_n = min(n, len(ds))
    stride = max(1, len(ds) // actual_n)   # 在可用样本里均匀取，且不越界

    fig, axes = plt.subplots(2, actual_n, figsize=(actual_n * 1.8, 4),
                             facecolor='#0d0f14')
    fig.suptitle('CelebA：原图（上） vs Canny 边缘图（下）',
                 color='white', fontsize=13, fontweight='bold')

    img_dir = os.path.join(celeba_root, 'img_align_celeba')
    for i in range(actual_n):
        j = min(i * stride, len(ds) - 1)
        path = ds.paths[j]
        # 原图
        orig = cv2.imread(str(path))
        orig = cv2.resize(orig, (128, 128))
        orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        ax0 = axes[0, i] if actual_n > 1 else axes[0]
        ax0.imshow(orig_rgb)
        ax0.axis('off')
        if i == 0:
            ax0.set_ylabel('原图', color='#8899bb', fontsize=9)

        # 边缘图
        edge = ds[j].squeeze().numpy()
        ax1 = axes[1, i] if actual_n > 1 else axes[1]
        ax1.imshow(edge, cmap='gray', vmin=0, vmax=1)
        ax1.axis('off')
        if i == 0:
            ax1.set_ylabel('边缘图', color='#8899bb', fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0d0f14')
    plt.close()
    print(f"[vis] 保存 → {out_path}")


def vis_faces_synthetic(out_path: str = 'vis_faces.png'):
    """没有 CelebA 时用合成代理脸验证边缘逻辑。"""
    print("[vis] CelebA 不可用，生成合成代理脸可视化")
    import cv2

    fig, axes = plt.subplots(2, 8, figsize=(16, 5), facecolor='#0d0f14')
    fig.suptitle('合成代理脸 + 模拟 Canny 边缘（无 CelebA）',
                 color='white', fontsize=13)

    rng = np.random.default_rng(1)
    for i in range(8):
        # 生成合成人脸
        H = W = 128
        ys, xs = np.linspace(-1, 1, H), np.linspace(-1, 1, W)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        cx, cy = rng.uniform(-0.1, 0.1), rng.uniform(-0.05, 0.05)
        a, b = rng.uniform(0.55, 0.72), rng.uniform(0.72, 0.88)
        face  = np.exp(-((xx - cx)**2 / a**2 + (yy - cy)**2 / b**2) * 3.0)
        ex = 0.22; ey = -0.18
        es = 0.06
        left  = np.exp(-((xx - (cx-ex))**2 + (yy - (cy+ey))**2) / es**2)
        right = np.exp(-((xx - (cx+ex))**2 + (yy - (cy+ey))**2) / es**2)
        mouth = np.exp(-((xx - cx)**2 / 0.25**2 + (yy - (cy+0.3))**2 / 0.05**2))
        img_f = np.clip(face * 0.7 - (left + right) * 0.55 - mouth * 0.45, 0, 1)

        # 伪造边缘
        img_uint8 = (img_f * 255).astype(np.uint8)
        edges = cv2.Canny(cv2.GaussianBlur(img_uint8, (5, 5), 1.2), 20, 80)
        edges_f = edges.astype(np.float32) / 255.0

        axes[0, i].imshow(img_f, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[1, i].imshow(edges_f, cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('合成脸', color='#8899bb', fontsize=9)
    axes[1, 0].set_ylabel('Canny 边缘', color='#8899bb', fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='#0d0f14')
    plt.close()
    print(f"[vis] 保存 → {out_path}")


# ─── 模式 3：增强等级矩阵（对同一字母） ──────────────────────────────────────

def vis_aug_grid(data_root: str, out_path: str = 'vis_aug_grid.png'):
    """
    行 = 字母（选 6 个），列 = 增强等级各 6 个样本
    直观展示弹性形变 + 仿射的效果
    """
    from data import _try_load_emnist, _fallback_synth_letters
    from aug_letters import augment_letter

    raw = _try_load_emnist(data_root, size=32)
    if raw is None:
        raw = _fallback_synth_letters(size=32, per_class=50)

    rng = np.random.default_rng(7)
    show_classes = [7, 0, 24, 4, 14, 17]   # H A Y E O R
    levels = ['原图', 'mild', 'normal', 'aggressive', 'aggressive', 'extreme', 'extreme']
    n_per_level = 5
    lv_real = ['orig', 'mild', 'normal', 'aggressive', 'aggressive', 'extreme', 'extreme']

    col_palette = ['#8899bb', '#5b8dee', '#3ecf94', '#f0a030', '#f0a030', '#e0608e', '#e0608e']

    total_cols = len(levels) * n_per_level
    fig = plt.figure(figsize=(total_cols * 0.9, len(show_classes) * 1.1 + 0.8), facecolor='#090d16')
    fig.suptitle('字母弹性形变 × 仿射增强矩阵', color='white', fontsize=14, fontweight='bold', y=0.99)

    gs = gridspec.GridSpec(len(show_classes), total_cols, figure=fig,
                           hspace=0.05, wspace=0.03,
                           left=0.06, right=0.99, top=0.92, bottom=0.02)

    for ri, c in enumerate(show_classes):
        lname = chr(ord('A') + c)
        src = raw[c].numpy()
        col_idx = 0
        for li, (lv_name, lv_key, col) in enumerate(zip(levels, lv_real, col_palette)):
            for ai in range(n_per_level):
                ax = fig.add_subplot(gs[ri, col_idx])
                if lv_key == 'orig':
                    img = src[ai % len(src)]
                else:
                    img = augment_letter(src[ai % len(src)], level=lv_key, rng=rng)
                ax.imshow(img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                ax.axis('off')
                if ri == 0 and ai == 2:
                    ax.set_title(lv_name, color=col, fontsize=8, fontweight='bold', pad=3)
                for sp in ax.spines.values():
                    sp.set_visible(ai == 0)
                    sp.set_color(col)
                    sp.set_linewidth(1.8)
                col_idx += 1
        # 字母标签
        ax0 = fig.add_subplot(gs[ri, 0])
        ax0.text(-0.7, 0.5, lname, transform=ax0.transAxes,
                 color='white', fontsize=13, fontweight='bold', va='center')

    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='#090d16')
    plt.close()
    print(f"[vis] 保存 → {out_path}")


# ─── 模式 5：26×26 全字母矩阵 ────────────────────────────────────────────────

def vis_26x26(data_root: str, out_path: str = 'vis_26x26.png', level: str = 'aggressive'):
    """
    26 行 × 26 列：
      第 i 行 = 字母 chr('A'+i)，整行是该字母经过 26 个不同 transformation 的结果。
    用于检查“激进增强后笔画是否被裁出画面”这个问题是否已修复。
    """
    from data import _try_load_emnist, _fallback_synth_letters
    from aug_letters import augment_letter

    print(f"[vis] 加载 EMNIST… (level={level})")
    raw = _try_load_emnist(data_root, size=32)
    if raw is None:
        print("[vis] EMNIST 不可用，使用 PIL fallback")
        raw = _fallback_synth_letters(size=32, per_class=50)

    rng = np.random.default_rng(123)
    N = 26  # 26 行 26 列

    fig, axes = plt.subplots(N, N, figsize=(N * 0.5, N * 0.5), facecolor='#090d16')
    fig.suptitle(f'26×26 字母增强矩阵（每行同一字母 × 26 个变换，level={level}）',
                 color='white', fontsize=13, fontweight='bold', y=0.995)

    for ri in range(N):          # 行 = 字母
        src = raw[ri].numpy()    # [n_samples, 32, 32]
        for ci in range(N):      # 列 = 第 ci 个变换
            base = src[rng.integers(0, len(src))]
            aug = augment_letter(base, level=level, rng=rng)
            ax = axes[ri, ci]
            ax.imshow(aug, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color('#1c2640'); sp.set_linewidth(0.4)
            if ci == 0:
                ax.set_ylabel(chr(ord('A') + ri), color='white', fontsize=9,
                              rotation=0, labelpad=8, va='center')

    plt.subplots_adjust(left=0.03, right=0.995, top=0.965, bottom=0.005,
                        hspace=0.05, wspace=0.05)
    plt.savefig(out_path, dpi=150, facecolor='#090d16')
    plt.close()
    print(f"[vis] 保存 → {out_path}")


# ─── 模式 6：每个字母一张图（10 个例子，看得清） ─────────────────────────────

def vis_per_letter(data_root: str, out_dir: str = 'vis_letters_aug',
                   level: str = 'aggressive', n_side: int = 10, size: int = 64):
    """
    为每个字母生成单独一张图，每张里放 n_side×n_side 个增强例子（默认 10×10=100）。
    输出到 out_dir/ 下：A.png, B.png, …, Z.png。
    size: 字母分辨率（默认 64；EMNIST 原生 28×28，会插值上采样到该尺寸）。
    """
    from data import _try_load_emnist, _fallback_synth_letters
    from aug_letters import augment_letter

    print(f"[vis] 加载 EMNIST… (level={level}, size={size})")
    raw = _try_load_emnist(data_root, size=size)
    if raw is None:
        print("[vis] EMNIST 不可用，使用 PIL fallback")
        raw = _fallback_synth_letters(size=size, per_class=50)

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(2024)
    n_examples = n_side * n_side

    for c in range(26):
        lname = chr(ord('A') + c)
        src = raw[c].numpy()
        imgs = [augment_letter(src[rng.integers(0, len(src))], level=level, rng=rng)
                for _ in range(n_examples)]
        mosaic = make_grid_np(imgs, nrow=n_side, pad=2, bg=0.12)  # 单张大拼图

        fig_in = max(8, size * n_side / 110)   # 让 100 个 64px 例子看得清
        fig, ax = plt.subplots(figsize=(fig_in, fig_in + 0.5), facecolor='#090d16')
        fig.suptitle(f'Letter  {lname}   (level={level}, {n_examples} examples, {size}px)',
                     color='white', fontsize=16, fontweight='bold', y=0.985)
        ax.imshow(mosaic, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        ax.axis('off')
        plt.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.005)
        out_path = os.path.join(out_dir, f'{lname}.png')
        plt.savefig(out_path, dpi=150, facecolor='#090d16')
        plt.close()
        if (c + 1) % 5 == 0:
            print(f"  {c+1}/26", flush=True)

    print(f"[vis] 26 张已保存到 {out_dir}/ （A.png … Z.png，每张 {n_examples} 例）")


# ─── 模式 4：快速 smoke test（无数据依赖） ────────────────────────────────────

def vis_smoke(out_path: str = 'vis_smoke.png'):
    """完全不依赖外部数据，只测试增强逻辑本身。"""
    from aug_letters import augment_letter
    from data import _fallback_synth_letters

    print("[vis] smoke test（PIL fallback 字母 + 增强）")
    raw = _fallback_synth_letters(size=32, per_class=10)
    rng = np.random.default_rng(42)

    rows = []
    letters_to_show = [0, 7, 14, 24, 4, 17]  # A H O Y E R
    for c in letters_to_show:
        row = [raw[c, 0].numpy()]  # 原图
        for lv in ['mild', 'normal', 'aggressive', 'extreme']:
            for _ in range(3):
                row.append(augment_letter(raw[c, 0].numpy(), level=lv, rng=rng))
        rows.append(row)

    n_cols = len(rows[0])
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 0.7, n_rows * 0.9),
                             facecolor='#090d16')
    col_tags = ['原图'] + ['mild']*3 + ['normal']*3 + ['aggr.']*3 + ['extreme']*3
    col_cols = ['#8899bb'] + ['#5b8dee']*3 + ['#3ecf94']*3 + ['#f0a030']*3 + ['#e0608e']*3
    lnames   = [chr(ord('A') + c) for c in letters_to_show]

    for ri in range(n_rows):
        for ci in range(n_cols):
            ax = axes[ri, ci] if n_rows > 1 else axes[ci]
            ax.imshow(rows[ri][ci], cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            ax.axis('off')
            if ri == 0:
                ax.set_title(col_tags[ci], color=col_cols[ci], fontsize=7, pad=2)
            if ci == 0:
                ax.set_ylabel(lnames[ri], color='white', fontsize=10, rotation=0,
                              labelpad=10, va='center')
    fig.suptitle('字母增强 smoke test（无外部数据）', color='white', fontsize=12,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight', facecolor='#090d16')
    plt.close()
    print(f"[vis] 保存 → {out_path}")


# ─── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='数据集可视化')
    parser.add_argument('--mode', choices=['letters', 'faces', 'both', 'aug-grid', 'smoke', '26x26', 'per-letter'],
                        default='smoke')
    parser.add_argument('--data-root', default='./data', help='EMNIST 存放目录')
    parser.add_argument('--celeba-root', default=None, help='CelebA 根目录')
    parser.add_argument('--out-dir', default='.', help='输出图片目录')
    parser.add_argument('--level', default='aggressive',
                        choices=['mild', 'normal', 'aggressive', 'extreme'],
                        help='26x26 模式使用的增强等级')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == 'per-letter':
        vis_per_letter(args.data_root, os.path.join(args.out_dir, 'vis_letters_aug'),
                       level=args.level, n_side=10, size=64)

    elif args.mode == '26x26':
        vis_26x26(args.data_root, os.path.join(args.out_dir, 'vis_26x26.png'), level=args.level)

    elif args.mode == 'smoke':
        vis_smoke(os.path.join(args.out_dir, 'vis_smoke.png'))

    elif args.mode == 'letters':
        vis_letters(args.data_root, os.path.join(args.out_dir, 'vis_letters.png'))

    elif args.mode == 'faces':
        if args.celeba_root and os.path.exists(args.celeba_root):
            vis_faces(args.celeba_root, os.path.join(args.out_dir, 'vis_faces.png'))
        else:
            vis_faces_synthetic(os.path.join(args.out_dir, 'vis_faces.png'))

    elif args.mode == 'aug-grid':
        vis_aug_grid(args.data_root, os.path.join(args.out_dir, 'vis_aug_grid.png'))

    elif args.mode == 'both':
        vis_smoke(os.path.join(args.out_dir, 'vis_smoke.png'))
        vis_aug_grid(args.data_root, os.path.join(args.out_dir, 'vis_aug_grid.png'))
        if args.celeba_root and os.path.exists(args.celeba_root):
            vis_faces(args.celeba_root, os.path.join(args.out_dir, 'vis_faces.png'))
        else:
            vis_faces_synthetic(os.path.join(args.out_dir, 'vis_faces.png'))
        vis_letters(args.data_root, os.path.join(args.out_dir, 'vis_letters.png'))
