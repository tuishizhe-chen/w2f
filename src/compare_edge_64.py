"""Compare 64-res edge styles to find the cleanest for direct pixel drift."""
import cv2, numpy as np, torch
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

S = 64
celeba = Path('./data/celeba/img_align_celeba')
paths = sorted(celeba.glob('*.jpg'))[:8]

def edge_simple(p, blur, lo, hi):
    bgr = cv2.imread(str(p))
    bgr = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if blur > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur)
    edges = cv2.Canny(gray, lo, hi)
    return edges.astype(np.float32) / 255.0

configs = [
    ('A clean: blur1.0 / 30-90 / no-postp', dict(blur=1.0, lo=30, hi=90)),
    ('B sparse: blur1.4 / 50-130 / no-postp', dict(blur=1.4, lo=50, hi=130)),
    ('C very-sparse: blur1.8 / 70-150', dict(blur=1.8, lo=70, hi=150)),
    ('D existing var-width 64 (from bank)', None),
]

bank = torch.load('checkpoints/edge_bank_64.pt', weights_only=True)
fig, axes = plt.subplots(len(configs), 8, figsize=(16, 2.4 * len(configs)), facecolor='#0d0f14')
for r, (name, kw) in enumerate(configs):
    for c in range(8):
        if kw is None:
            img = bank[c].float().numpy() / 255.0
        else:
            img = edge_simple(paths[c], **kw)
        axes[r, c].imshow(img, cmap='gray', vmin=0, vmax=1); axes[r, c].axis('off')
        if c == 0:
            axes[r, c].set_title(name, color='white', fontsize=9, loc='left', pad=2)

plt.subplots_adjust(left=0.005, right=0.995, top=0.96, bottom=0.005, hspace=0.05, wspace=0.04)
plt.savefig('compare_64.png', dpi=120, facecolor='#0d0f14'); plt.close()
print('saved compare_64.png')
