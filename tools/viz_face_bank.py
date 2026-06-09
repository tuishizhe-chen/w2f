"""Render a 2x8 grid of the current FACE edge-map dataset (the D2 drift target)."""
import sys, torch, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

bank_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/edge_bank_128_dil_lo75.pt'
out = sys.argv[2] if len(sys.argv) > 2 else '/root/autodl-tmp/edge_bank_viz.png'

b = torch.load(bank_path, weights_only=True)
print('bank', bank_path, 'shape', tuple(b.shape), 'dtype', b.dtype)
x = b.float()
if x.max() > 1.5:
    x = x / 255.0
g = torch.Generator().manual_seed(7)
idx = torch.randperm(x.shape[0], generator=g)[:16]
sel = x[idx].cpu().numpy()

fig, axes = plt.subplots(2, 8, figsize=(16, 4.3), facecolor='white')
for i, ax in enumerate(axes.flat):
    ax.imshow(sel[i], cmap='gray', vmin=0, vmax=1)
    ax.axis('off')
plt.subplots_adjust(left=0.004, right=0.996, top=0.93, bottom=0.01, wspace=0.06, hspace=0.06)
fig.suptitle('Face edge-map dataset (CelebA -> Canny -> sparsify -> dilate) — 16 random samples',
             fontsize=12)
plt.savefig(out, dpi=120, facecolor='white')
print('saved', out)
