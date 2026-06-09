"""Hand-picked 12 face anatomical anchors for K-layer composition priors.

ANCHORS_12[k] = (cx, cy, sigma) in normalized [0, 1] coordinates.
build_log_priors(H, W) returns [K, 1, H, W] tensor of log-Gaussian masks centered
at each anchor — used as additive bias on head logits to break K-permutation
symmetry (head k naturally wants to draw near anchor[k]).
"""
import math
import torch


# normalized (cx, cy, sigma) — y goes top-to-bottom (0 = top of canvas)
# 12-anchor version: granular face anatomy.
ANCHORS_12 = [
    (0.35, 0.42, 0.09),   # 0: left eye
    (0.65, 0.42, 0.09),   # 1: right eye
    (0.35, 0.33, 0.07),   # 2: left brow
    (0.65, 0.33, 0.07),   # 3: right brow
    (0.50, 0.55, 0.08),   # 4: nose
    (0.50, 0.74, 0.10),   # 5: mouth
    (0.22, 0.72, 0.14),   # 6: left jaw
    (0.78, 0.72, 0.14),   # 7: right jaw
    (0.50, 0.88, 0.10),   # 8: chin
    (0.50, 0.13, 0.18),   # 9: hair top center
    (0.18, 0.30, 0.14),   # 10: left hair / temple
    (0.82, 0.30, 0.14),   # 11: right hair / temple
]

# 8-anchor version: merges left/right eyes into pairs and left/right brows;
# fewer slots → each one covers more area → fewer 'forced-fill' noise points.
ANCHORS_8 = [
    (0.50, 0.42, 0.18),   # 0: eyes pair (wide)
    (0.50, 0.33, 0.16),   # 1: brows pair
    (0.50, 0.55, 0.10),   # 2: nose
    (0.50, 0.76, 0.13),   # 3: mouth + chin
    (0.22, 0.65, 0.16),   # 4: left jaw + cheek
    (0.78, 0.65, 0.16),   # 5: right jaw + cheek
    (0.50, 0.15, 0.20),   # 6: hair top
    (0.50, 0.30, 0.30),   # 7: hair sides (wide, covers both temples)
]

# 6-anchor version: coarse face decomposition. eyes + brows = top facial features;
# mouth + chin = bottom facial; left/right jaw; hair (full crown).
ANCHORS_6 = [
    (0.50, 0.38, 0.16),   # 0: eyes + brows (upper facial)
    (0.50, 0.55, 0.10),   # 1: nose
    (0.50, 0.78, 0.13),   # 2: mouth + chin
    (0.22, 0.65, 0.20),   # 3: left side (hair temple + jaw)
    (0.78, 0.65, 0.20),   # 4: right side
    (0.50, 0.18, 0.25),   # 5: hair top (wide)
]


def _anchor_set(K):
    return {6: ANCHORS_6, 8: ANCHORS_8, 12: ANCHORS_12}[K]


def _gaussian_2d(H, W, cx, cy, sigma, device):
    """Returns [H, W] Gaussian centered at (cx, cy) with given sigma, all in
    normalized [0, 1] coords. Peak = 1.0."""
    ys = torch.linspace(0.0, 1.0, H, device=device)
    xs = torch.linspace(0.0, 1.0, W, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    return torch.exp(-((xx - cx).pow(2) + (yy - cy).pow(2)) / (2.0 * sigma * sigma))


def build_priors(H: int = 128, W: int = 128, device='cpu', sigma_scale: float = 1.0, K: int = 12):
    """Build [K, 1, H, W] Gaussian priors, peak=1, one per anchor."""
    anchors = _anchor_set(K)
    priors = torch.zeros(len(anchors), 1, H, W, device=device)
    for k, (cx, cy, sigma) in enumerate(anchors):
        priors[k, 0] = _gaussian_2d(H, W, cx, cy, sigma * sigma_scale, device)
    return priors


def build_log_priors(H: int = 128, W: int = 128, device='cpu', sigma_scale: float = 1.0,
                     eps: float = 1e-6, K: int = 12):
    """log(prior + eps) — use as additive logit bias."""
    return torch.log(build_priors(H, W, device, sigma_scale, K) + eps)


def build_hard_mask(H: int = 128, W: int = 128, device='cpu', sigma_scale: float = 1.0,
                    threshold: float = 0.05, K: int = 12):
    """Boolean [K, 1, H, W] mask: True where prior > threshold (=fovea)."""
    return build_priors(H, W, device, sigma_scale, K) > threshold


if __name__ == '__main__':
    # quick sanity check + viz
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    priors = build_priors(128, 128).numpy()
    K = priors.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9), facecolor='#0d0f14')
    names = ['L eye', 'R eye', 'L brow', 'R brow', 'Nose', 'Mouth',
             'L jaw', 'R jaw', 'Chin', 'Hair top', 'L hair', 'R hair']
    for k in range(K):
        ax = axes[k // 4, k % 4]
        ax.imshow(priors[k, 0], cmap='magma', vmin=0, vmax=1)
        ax.set_title(names[k], color='w', fontsize=10)
        ax.axis('off')
    plt.suptitle('Face anatomical anchor priors (K=12)', color='w')
    plt.savefig('/tmp/face_priors.png', dpi=100, facecolor='#0d0f14')
    print('saved /tmp/face_priors.png')
