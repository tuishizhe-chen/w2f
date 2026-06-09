"""Sample some letters from letter_bank.pt to verify they look like letters."""
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

b = torch.load('/root/w2f/checkpoints/letter_bank.pt', weights_only=True)
print('bank shape:', tuple(b.shape), 'dtype:', b.dtype)
n_cls, n_per, h, w = b.shape

# show 5 augmentations × 26 classes
fig, axes = plt.subplots(5, 26, figsize=(26 * 1.2, 5 * 1.2), facecolor='#0d0f14')
torch.manual_seed(7)
sample_idx = torch.randperm(n_per)[:5]
for c in range(26):
    for r in range(5):
        ax = axes[r, c]
        ax.imshow(b[c, sample_idx[r]].numpy(), cmap='gray', vmin=0, vmax=255)
        ax.axis('off')
    axes[0, c].set_title(chr(65 + c), color='white', fontsize=10, pad=2)
plt.subplots_adjust(left=0.005, right=0.995, top=0.92, bottom=0.005,
                    hspace=0.05, wspace=0.04)
plt.savefig('/tmp/letter_bank_preview.png', dpi=100, facecolor='#0d0f14')
plt.close()

# also report mean ink ratio per class
inks = []
for c in range(n_cls):
    inks.append(b[c].float().mean().item() / 255)
print('per-class ink ratios:', [round(i, 3) for i in inks])
print('mean ink:', round(sum(inks) / len(inks), 3))
print('saved /tmp/letter_bank_preview.png')
