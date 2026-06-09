"""
激进字母数据增强模块
在 EMNIST 基础上加入：独立 x/y 缩放、旋转、剪切、弹性形变
拓扑约束：gaussian sigma >= 3 保证笔画不断裂

对外接口：
  AugLetterBank  — 替换原来的 LetterBank，采样时实时增强
  build_aug_bank — 预生成并缓存增强后的大数据集 [26, N, 1, H, W]
"""
from __future__ import annotations
import math
import random
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ─── 单张图片的增强（numpy, [H,W] float32, 值域 [0,1]） ───────────────────

def _elastic_deform(img: np.ndarray, sigma: float, alpha: float, rng: np.random.Generator) -> np.ndarray:
    """单次弹性形变：高斯平滑随机位移场。"""
    from scipy.ndimage import gaussian_filter, map_coordinates
    H, W = img.shape
    dx = gaussian_filter(rng.standard_normal((H, W)), sigma=sigma) * alpha
    dy = gaussian_filter(rng.standard_normal((H, W)), sigma=sigma) * alpha
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    coords = [np.clip(y + dy, 0, H - 1), np.clip(x + dx, 0, W - 1)]
    return map_coordinates(img, coords, order=1, mode='nearest').astype(np.float32)


def _compose_elastic(
    img: np.ndarray,
    n: int,
    sigma: float,
    amp_per: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    复合 n 次小弹性形变（pull-back 复合：每步在已扭曲坐标系上再叠加）。
    效果：笔画沿途积累弯曲，产生 S 形、螺旋形等复杂曲线。

    关键参数（与之前版本不同！）：
      sigma   = 位移场的平滑度 / 相关长度（像素）。大 → 全局相干弯曲；小 → 局部抖动。
      amp_per = **每步位移场的 RMS 振幅（像素）**。直接控制“移多远”。

    为什么要显式归一化 amp_per：
      gaussian_filter 平滑白噪声后，振幅会被压得极小（std ≈ 1/(3.5·sigma)）。
      若像旧代码那样只乘一个系数，sigma 一大位移就 < 1px，几乎不形变（这就是之前的 bug）。
      这里把每步位移场重新缩放到 RMS = amp_per 像素，使振幅与 sigma 解耦。

    拓扑守恒：每步梯度 ~ amp_per/sigma < 1 即近似微分同胚；复合多步累积大弯曲仍保拓扑。
    """
    from scipy.ndimage import gaussian_filter, map_coordinates
    H, W = img.shape
    x_base, y_base = np.meshgrid(np.arange(W, dtype=np.float32),
                                  np.arange(H, dtype=np.float32))
    flow_x = np.zeros((H, W), dtype=np.float32)
    flow_y = np.zeros((H, W), dtype=np.float32)

    for _ in range(n):
        ddx = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=sigma)
        ddy = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma=sigma)
        # 归一化到目标像素振幅（关键修复）
        norm = float(np.sqrt(np.mean(ddx * ddx + ddy * ddy))) + 1e-8
        scale = amp_per / norm
        ddx *= scale
        ddy *= scale
        # 在当前累计位置处采样新的位移增量（pull-back 复合）
        cur_x = np.clip(x_base + flow_x, 0, W - 1)
        cur_y = np.clip(y_base + flow_y, 0, H - 1)
        flow_x += map_coordinates(ddx, [cur_y, cur_x], order=1, mode='nearest')
        flow_y += map_coordinates(ddy, [cur_y, cur_x], order=1, mode='nearest')

    coords = [np.clip(y_base + flow_y, 0, H - 1),
              np.clip(x_base + flow_x, 0, W - 1)]
    return map_coordinates(img, coords, order=1, mode='nearest').astype(np.float32)


def _affine_aug_numpy(
    img: np.ndarray,
    sx: float, sy: float,      # 独立 x/y 缩放，值域 (0, +inf)，1.0 = 不变
    angle_deg: float,          # 旋转角度（度）
    shear: float,              # 剪切系数
) -> np.ndarray:
    """
    对 numpy [H,W] 图像做仿射变换（中心对齐）。
    用 cv2 实现，比 scipy 快。
    """
    try:
        import cv2
        H, W = img.shape
        cx, cy = W / 2.0, H / 2.0
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        # 构造仿射矩阵：先缩放+旋转，再加剪切
        M = np.array([
            [sx * cos_a,  -sy * sin_a + shear * sy * cos_a, 0.0],
            [sx * sin_a,   sy * cos_a + shear * sy * sin_a, 0.0],
        ], dtype=np.float32)
        # 修正平移，使变换以图像中心为原点
        M[0, 2] = cx - M[0, 0] * cx - M[0, 1] * cy
        M[1, 2] = cy - M[1, 0] * cx - M[1, 1] * cy
        out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return out.astype(np.float32)
    except ImportError:
        # fallback：用 scipy
        from scipy.ndimage import affine_transform
        H, W = img.shape
        cx, cy = W / 2.0, H / 2.0
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        mat = np.array([[cos_a / sx, sin_a / sy], [-sin_a / sx, cos_a / sy]])
        offset = np.array([cx, cy]) - mat @ np.array([cx, cy])
        return affine_transform(img, mat, offset=offset, order=1, mode='constant', cval=0).astype(np.float32)


def _fit_to_canvas(
    img: np.ndarray,
    out_size: int,
    fill_ratio: float = 0.9,
    jitter: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    把形变后的内容“装回”画面，保证笔画不被裁掉。

    步骤：
      1. 找到非零像素的外接框（bbox）；
      2. 裁出 bbox，保持长宽比缩放，使其长边占据 out_size 的 fill_ratio；
      3. 居中（可选小幅 jitter，但始终保证完整落在画面内）。

    这样无论前面 affine/弹性把字母拉伸成什么样，最终都完整可见。
    fill_ratio 随机化 → 顺便提供“字母大小”多样性（STN 之外的额外变化）。
    """
    ys, xs = np.where(img > 1e-4)
    if len(xs) == 0:
        return np.zeros((out_size, out_size), dtype=np.float32)

    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    crop = img[r0:r1 + 1, c0:c1 + 1]
    ch, cw = crop.shape

    target = out_size * fill_ratio
    scale = target / max(ch, cw)
    new_h = min(out_size, max(1, int(round(ch * scale))))
    new_w = min(out_size, max(1, int(round(cw * scale))))

    try:
        import cv2
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from scipy.ndimage import zoom
        resized = zoom(crop, (new_h / ch, new_w / cw), order=1)[:new_h, :new_w]

    canvas = np.zeros((out_size, out_size), dtype=np.float32)
    free_h, free_w = out_size - new_h, out_size - new_w
    oy, ox = free_h // 2, free_w // 2
    if jitter > 0 and rng is not None:
        oy += int(rng.uniform(-1, 1) * jitter * free_h * 0.5)
        ox += int(rng.uniform(-1, 1) * jitter * free_w * 0.5)
        oy = int(np.clip(oy, 0, free_h))
        ox = int(np.clip(ox, 0, free_w))
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


def augment_letter(
    img: np.ndarray,
    level: str = 'aggressive',
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    对单张字母图片（numpy [H,W] float32）做增强，返回同尺寸图片。

    level 选项：
      'mild'       — 轻微（原 data.py 水平）
      'normal'     — 适中
      'aggressive' — 激进（推荐，默认）
      'extreme'    — 极端（少量混入，约 10%）
    """
    if rng is None:
        rng = np.random.default_rng()

    u = rng.uniform  # 简写

    # 形变哲学：
    #   「压扁 / 改变长宽比」靠 **粗尺度弹性弯曲**（保拓扑地把笔画弯过去，
    #    例：把 e 弯成 p 的形态），而不是各向异性 affine 直接拍扁。
    #   affine 只做近似保形的事：旋转 + 小剪切 + 轻微各向异性。
    #
    # 各级参数（sigma / amp 都用 out_size 的比例表示，自动适配 32/64/…）：
    #   c_* = 粗尺度弹性（大 sigma，全局相干弯曲，主导形状变化/“弯出”长宽比）
    #   f_* = 细尺度弹性（小 sigma，局部笔画抖动）
    #   c_ampf / f_ampf = 每步位移场 RMS 振幅占 out_size 的比例（真正决定“移多远”）
    #   aniso = affine 各向异性幅度（很小，避免“拍扁”观感）
    # 约束：amp/sigma = ampf/sigf < 1 → 每步近似微分同胚，保拓扑。
    # 每步 amp/sigma 比例（=ampf/sigf）控制单步形变强度：
    #   ≤0.27 → 单步稳稳是微分同胚，保拓扑；靠 **加步数 n** 累积出大弯曲
    #   net 弯曲 ≈ sqrt(n) · amp（随机游走累积）
    if level == 'mild':
        c_n, c_sigf, c_ampf = 4,  0.28, 0.040   # net≈5px
        f_n, f_sigf, f_ampf = 4,  0.11, 0.013
        aniso              = 0.12
        angle              = u(-12, 12)
        shear              = u(-0.12, 0.12)
        fill_lo, fill_hi   = 0.80, 0.92
    elif level == 'normal':
        c_n, c_sigf, c_ampf = 7,  0.26, 0.052   # net≈8.8px
        f_n, f_sigf, f_ampf = 6,  0.10, 0.018
        aniso              = 0.16
        angle              = u(-25, 25)
        shear              = u(-0.20, 0.20)
        fill_lo, fill_hi   = 0.68, 0.93
    elif level == 'aggressive':
        c_n, c_sigf, c_ampf = 10, 0.24, 0.065   # sig≈15px, 每步≈4.2px, ratio0.27, net≈13px
        f_n, f_sigf, f_ampf = 8,  0.09, 0.022   # sig≈6px,  每步≈1.4px, 局部抖动
        aniso              = 0.18
        angle              = u(-35, 35)
        shear              = u(-0.25, 0.25)
        fill_lo, fill_hi   = 0.55, 0.95
    elif level == 'aggressive_lowrot':
        # 2026-06-03: aggressive 的强弹性/剪切/fill 多样性，但旋转保持 mild 水平。
        # 动机：decoder 只 condition 在 class 上，看不到 per-sample 旋转，所以 bank 里
        # 的大旋转方差会逼 decoder 输出"旋转平均"的 blob。而 STN 自己就提供 rot=±30°
        # 的贴位旋转，patch 内容里的旋转跟 STN 是重复的。让 STN 管旋转，让 elastic/
        # shear/fill 提供形变多样性（augmentation 真正的卖点），decoder 学接近正立的字形。
        c_n, c_sigf, c_ampf = 10, 0.24, 0.065
        f_n, f_sigf, f_ampf = 8,  0.09, 0.022
        aniso              = 0.18
        angle              = u(-12, 12)         # ← 唯一区别：旋转降到 mild 水平
        shear              = u(-0.25, 0.25)
        fill_lo, fill_hi   = 0.55, 0.95
    else:  # extreme
        c_n, c_sigf, c_ampf = 14, 0.22, 0.078   # ratio0.35, net≈18.7px（偶尔拓扑临界）
        f_n, f_sigf, f_ampf = 10, 0.08, 0.028
        aniso              = 0.26
        angle              = u(-55, 55)
        shear              = u(-0.38, 0.38)
        fill_lo, fill_hi   = 0.50, 0.97

    out_size = img.shape[0]
    sx, sy = u(1 - aniso, 1 + aniso), u(1 - aniso, 1 + aniso)

    # 弹性 sigma / amp 按 out_size 折算成像素
    c_sig, c_amp = out_size * c_sigf, out_size * c_ampf
    f_sig, f_amp = out_size * f_sigf, out_size * f_ampf

    # 1. 先垫到大画布：给“弯出去”和 affine 留足空间，绝不在边界裁断笔画
    grow = (1.0 + abs(shear)) * 1.5 + 0.8
    work = int(np.ceil(out_size * grow))
    work += (work - out_size) % 2          # 两边对称 padding
    pad = (work - out_size) // 2
    big = np.zeros((work, work), dtype=np.float32)
    big[pad:pad + out_size, pad:pad + out_size] = img

    # 2. 粗尺度弹性：全局相干弯曲（“靠弯笔画压扁”的主要来源）
    big = _compose_elastic(big, n=c_n, sigma=c_sig, amp_per=c_amp, rng=rng)
    # 3. 细尺度弹性：局部笔画抖动，叠在粗弯之上
    big = _compose_elastic(big, n=f_n, sigma=f_sig, amp_per=f_amp, rng=rng)

    # 4. 近似保形 affine：归一化长边到 1.0（不放大），只剩旋转 + 小剪切 + 轻微各向异性
    m = max(sx, sy, 1e-6)
    sx, sy = sx / m, sy / m
    big = _affine_aug_numpy(big, sx=sx, sy=sy, angle_deg=angle, shear=shear)

    # 5. 裁回外接框并装进 out_size 画面：笔画完整 + 随机大小多样性
    fill = u(fill_lo, fill_hi)
    out = _fit_to_canvas(big, out_size, fill_ratio=fill, jitter=0.10, rng=rng)

    # 6. 归一化到 [0,1]
    mx = out.max()
    if mx > 1e-6:
        out = out / mx
    return out


# ─── AugLetterBank ────────────────────────────────────────────────────────────

class AugLetterBank:
    """
    替换原 LetterBank，采样时实时做激进增强。
    data: [26, N, H, W] float32 tensor（原始 EMNIST 或 fallback）
    level: 默认 'aggressive'，可按训练进度动态调整
    extreme_ratio: extreme 级别增强的混入比例
    """

    def __init__(
        self,
        data: torch.Tensor,
        device: torch.device,
        level: str = 'aggressive',
        extreme_ratio: float = 0.08,
    ):
        assert data.ndim == 4 and data.shape[0] == 26
        self.data = data  # 留在 CPU，增强在 numpy 做完再转 GPU
        self.device = device
        self.size = data.shape[-1]
        self.level = level
        self.extreme_ratio = extreme_ratio
        self._rng = np.random.default_rng()

    @classmethod
    def build(cls, root: str, size: int, device: torch.device, **kw) -> "AugLetterBank":
        try:
            from .data import _try_load_emnist, _fallback_synth_letters
        except ImportError:
            # Running as a non-package import (e.g. via sys.path append in a
            # script); fall back to absolute import.
            from data import _try_load_emnist, _fallback_synth_letters
        arr = _try_load_emnist(root, size)
        if arr is None:
            print("[aug] 使用 PIL fallback 字母库", flush=True)
            arr = _fallback_synth_letters(size, per_class=200)
        print(f"[aug] AugLetterBank: 原始 shape {tuple(arr.shape)}", flush=True)
        return cls(arr, device, **kw)

    def sample(self, labels: torch.Tensor, augment: bool = True) -> torch.Tensor:
        """
        labels: [N] long in [0,26)
        返回: [N, 1, size, size] 增强后的字母图，已移到 device
        """
        N = labels.shape[0]
        labels_cpu = labels.cpu()
        idx = torch.randint(0, self.data.shape[1], (N,))
        imgs_np = self.data[labels_cpu, idx].numpy()  # [N, H, W]

        if augment:
            out = np.zeros_like(imgs_np)
            for i in range(N):
                lv = 'extreme' if self._rng.random() < self.extreme_ratio else self.level
                out[i] = augment_letter(imgs_np[i], level=lv, rng=self._rng)
            imgs_np = out

        t = torch.from_numpy(imgs_np).unsqueeze(1).to(self.device)  # [N,1,H,W]
        return t.float()


# ─── 预生成并缓存大数据集 ─────────────────────────────────────────────────────

def build_aug_bank(
    raw_data: torch.Tensor,       # [26, N_orig, H, W]
    samples_per_class: int = 2000,
    level: str = 'aggressive',
    extreme_ratio: float = 0.08,
    cache_path: Optional[str] = None,
    seed: int = 42,
) -> torch.Tensor:
    """
    预先生成 [26, samples_per_class, 1, H, W] 增强字母库并缓存到磁盘。
    后续 Drifting Vp 正样本直接从这里采，比实时增强更快。
    """
    if cache_path and Path(cache_path).exists():
        print(f"[aug_bank] 从缓存加载: {cache_path}")
        return torch.load(cache_path, weights_only=True)

    H, W = raw_data.shape[-2], raw_data.shape[-1]
    bank = torch.zeros(26, samples_per_class, 1, H, W, dtype=torch.float32)

    print(f"[aug_bank] 生成 26×{samples_per_class} 增强字母 (parallel)…", flush=True)
    # MP: 26 classes -> 26 workers. Build is single-threaded scipy work per
    # letter (~50ms each); with 26 classes and 208 cores available we run
    # all classes in parallel -> ~1/26th the wall time.
    import multiprocessing as mp
    n_workers = min(26, mp.cpu_count())
    args_list = [
        (c, raw_data[c].numpy(), samples_per_class, level,
         extreme_ratio, seed) for c in range(26)
    ]
    with mp.Pool(n_workers) as pool:
        done = 0
        for c, result in pool.imap_unordered(_build_class_letters, args_list):
            bank[c, :, 0] = torch.from_numpy(result)
            done += 1
            if done % 5 == 0:
                print(f"  {done}/26", flush=True)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank, cache_path)
        print(f"[aug_bank] 已缓存: {cache_path}")

    return bank  # [26, samples_per_class, 1, H, W]


def _build_class_letters(args):
    """Worker for parallel build_aug_bank. Returns (class_idx, [N, H, W])."""
    c, src, samples_per_class, level, extreme_ratio, seed = args
    rng = np.random.default_rng(seed + c)
    N_orig = src.shape[0]
    H, W = src.shape[-2], src.shape[-1]
    result = np.zeros((samples_per_class, H, W), dtype=np.float32)
    for j in range(samples_per_class):
        base = src[j % N_orig]
        lv = 'extreme' if rng.random() < extreme_ratio else level
        result[j] = augment_letter(base, level=lv, rng=rng)
    return c, result
