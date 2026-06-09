"""
CelebA → 边缘图数据集
用 Canny 算子提取人脸边缘，输出 [1, H, W] 归一化灰度张量。
依赖: torchvision, opencv-python, torch
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import cv2
from pathlib import Path


class CelebAEdgeDataset(Dataset):
    """
    从 CelebA 数据集提取 Canny 边缘图。
    首次调用会缓存预处理结果到 cache_dir，之后直接加载。

    参数:
        celeba_root: CelebA 数据集根目录（包含 img_align_celeba/ 子目录）
        size:        输出图片分辨率（正方形），默认 128
        split:       'train' | 'valid' | 'test'
        canny_low:   Canny 低阈值
        canny_high:  Canny 高阈值
        blur_sigma:  Canny 前高斯模糊 sigma，减少噪声边缘
        cache_dir:   预处理缓存目录，None 表示不缓存（每次实时处理）
        max_samples: 最多使用多少张，None 表示全用
    """
    def __init__(
        self,
        celeba_root: str,
        size: int = 256,
        split: str = 'train',
        canny_low: int = 30,       # 保眼睛/嘴/鼻 + 去头发碎纹
        canny_high: int = 90,
        blur_sigma: float = 2.4,
        maxw: int = 4,             # 最大笔宽(px)：笔宽 ∝ 梯度强度 × 五官权重
        feat_lo: float = 0.25,     # 非五官区(头发/外缘)的笔宽下限比例；越小压得越细
        min_area: int = 28,        # 稀疏化：去掉面积 < min_area 的小连通域（碎纹理）
        cache_dir: str = None,
        max_samples: int = None,
    ):
        self.size = size
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.blur_sigma = blur_sigma
        self.maxw = maxw
        self.feat_lo = feat_lo
        self.min_area = min_area
        self._W = self._feature_mask(size)   # 五官重要性权重图（固定，CelebA 已对齐）

        # 收集图片路径
        img_dir = Path(celeba_root) / 'img_align_celeba'
        if not img_dir.exists():
            raise FileNotFoundError(f"找不到 CelebA 图片目录: {img_dir}")

        # 读取官方 split 划分文件
        split_file = Path(celeba_root) / 'list_eval_partition.txt'
        split_id = {'train': 0, 'valid': 1, 'test': 2}[split]

        if split_file.exists():
            self.paths = []
            with open(split_file) as f:
                for line in f:
                    fname, sid = line.strip().split()
                    if int(sid) == split_id:
                        self.paths.append(img_dir / fname)
        else:
            # fallback: 直接列出所有图片
            self.paths = sorted(img_dir.glob('*.jpg'))

        if max_samples is not None:
            self.paths = self.paths[:max_samples]

        # 缓存
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.paths)

    @staticmethod
    def _feature_mask(size: int) -> np.ndarray:
        """五官重要性权重图 W∈[0,1]（CelebA 已对齐，五官位置基本固定）。
        眼鼻嘴区≈1、脸盘≈中、头发/外缘≈低。用来给笔宽加权 → 五官粗、头发细。"""
        ys = np.linspace(-1, 1, size); xs = np.linspace(-1, 1, size)
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        wf = np.exp(-((xx / 0.5) ** 2 + ((yy - 0.05) / 0.45) ** 2) * 2.0)  # 眼鼻嘴
        oval = np.exp(-((xx / 0.7) ** 2 + (yy / 0.95) ** 2) * 2.0)         # 脸盘
        return np.clip(0.15 * oval + 0.85 * wf, 0, 1).astype(np.float32)

    def _extract_edge(self, path: Path) -> np.ndarray:
        """给定图片路径，返回 [H, W] float32 边缘图，值域 [0,1]。

        edge20 版流程：重模糊 → Canny → 去小连通域(稀疏化) →
        每条边的笔宽 = (梯度强度) × (五官权重)，多级膨胀实现「实心、变宽、五官优先」线稿。
        """
        img = cv2.imread(str(path))  # BGR
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if self.blur_sigma > 0:
            blur = cv2.GaussianBlur(gray, (0, 0), self.blur_sigma)
        else:
            blur = gray
        edges = cv2.Canny(blur.astype(np.uint8), self.canny_low, self.canny_high)

        # 稀疏化：按连通域面积过滤，去掉细碎纹理/头发丝
        if self.min_area and self.min_area > 0:
            n, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
            keep = np.zeros_like(edges)
            for c in range(1, n):
                if stats[c, cv2.CC_STAT_AREA] >= self.min_area:
                    keep[labels == c] = 255
            edges = keep

        # 笔宽场：梯度强度(强边→粗) × 五官权重(头发→细)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
        gm = np.sqrt(gx * gx + gy * gy)
        gm = np.clip(gm / (np.percentile(gm, 99) + 1e-6), 0, 1)
        Wf = self.feat_lo + (1.0 - self.feat_lo) * self._W
        width = np.round((1.0 + (self.maxw - 1) * gm) * Wf).astype(int)
        width[edges == 0] = 0

        # 多级膨胀：笔宽≥r 的像素用半径 r 的核膨胀，并集 → 实心、变宽、满亮度
        out = np.zeros((self.size, self.size), dtype=np.float32)
        for r in range(1, self.maxw + 1):
            sel = (((width >= r) & (edges > 0)).astype(np.uint8)) * 255
            if sel.sum() == 0:
                continue
            ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            out = np.maximum(out, cv2.dilate(sel, ker).astype(np.float32) / 255.0)
        return out

    def __getitem__(self, idx) -> torch.Tensor:
        path = self.paths[idx]

        # 尝试读缓存
        if self.cache_dir:
            cache_path = self.cache_dir / (path.stem + '.pt')
            if cache_path.exists():
                return torch.load(cache_path, weights_only=True)

        edge = self._extract_edge(path)
        tensor = torch.from_numpy(edge).unsqueeze(0)  # [1, H, W]

        if self.cache_dir:
            torch.save(tensor, cache_path)

        return tensor


def build_face_bank(
    celeba_root: str,
    size: int = 256,
    max_samples: int = 8000,
    cache_path: str = None,
    **edge_kwargs,
) -> torch.Tensor:
    """
    把前 max_samples 张 CelebA 边缘图预加载成一个大张量 [N, 1, H, W]。
    适合数量不太大时直接放内存，用于 Drifting 正样本采样。

    cache_path: .pt 文件路径，加速重复运行。
    """
    if cache_path and Path(cache_path).exists():
        print(f"[face_bank] 从缓存加载: {cache_path}")
        return torch.load(cache_path, weights_only=True)

    print(f"[face_bank] 正在处理 {max_samples} 张 CelebA 边缘图…")
    ds = CelebAEdgeDataset(
        celeba_root, size=size, split='train',
        max_samples=max_samples, **edge_kwargs
    )
    loader = DataLoader(ds, batch_size=256, num_workers=4, shuffle=False)
    chunks = []
    for batch in loader:
        chunks.append(batch)
    bank = torch.cat(chunks, dim=0)  # [N, 1, H, W]

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(bank, cache_path)
        print(f"[face_bank] 已缓存到: {cache_path}")

    return bank


def visualize_samples(
    celeba_root: str,
    n: int = 16,
    save_path: str = 'edge_samples.png',
    **kwargs,
):
    """保存 n 张边缘图样本，用于肉眼验证质量。"""
    import torchvision.utils as vutils
    ds = CelebAEdgeDataset(celeba_root, max_samples=n, **kwargs)
    imgs = torch.stack([ds[i] for i in range(n)])  # [N, 1, H, W]
    grid = vutils.make_grid(imgs, nrow=4, normalize=True, padding=2)
    from PIL import Image as PILImage
    arr = (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    PILImage.fromarray(arr[:, :, 0], mode='L').save(save_path)
    print(f"[visualize] 保存到 {save_path}")


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else './data/celeba'
    # 快速验证
    visualize_samples(root, n=16, save_path='edge_check.png')
    # 构建 face_bank
    bank = build_face_bank(
        root,
        max_samples=8000,
        cache_path='./w2f/checkpoints/face_bank.pt',
    )
    print(f"face_bank shape: {bank.shape}, dtype: {bank.dtype}")
