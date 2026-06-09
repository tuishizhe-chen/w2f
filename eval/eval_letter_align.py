"""Visualize per-slot letter alignment.
For each batch sample shows K columns × 3 rows:
  Row 0: decoded patch (what the model generated)
  Row 1: target letter from aug bank (what L1 loss was pushing toward)
  Row 2: placed on canvas (after STN)
Each column titled with the assigned letter class (A-Z).

Usage:
  python eval_letter_align.py <ckpt.pt> <aug_letter_bank.pt> <out_prefix>
"""
import sys, torch, numpy as np
sys.path.insert(0, '/root/w2f/src')
from face_drift_multi_transformer import HierarchicalSlotGen
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

ckpt_path = sys.argv[1]
aug_bank_path = sys.argv[2]
out_prefix = sys.argv[3]

# Load checkpoint
ckpt = torch.load(ckpt_path, weights_only=False, map_location='cuda')
args = ckpt['args']
print(f"loaded ckpt at step {ckpt.get('step', '?')}, args: K={args['K']} "
      f"d_token={args['d_token']} aug_level={args['letter_aug_level']} "
      f"letter_weight={args['letter_weight']}")

G = HierarchicalSlotGen(
    d_noise=args['d_noise'], K=args['K'], d_token=args['d_token'],
    n_layers=args['n_layers'], n_heads=args['n_heads'],
    patch_size=args['patch_size'], canvas=128,
    sigmoid_t=args.get('sigmoid_t_end', args['sigmoid_t']),
    allow_rotation=not args.get('no_rotation', False),
    r_min=args['r_min'], r_max=args['r_max'], dxy_max=args['dxy_max'],
    letter_mode=True, n_letters=args['n_letters'],
).cuda()
G.load_state_dict(ckpt['G_ema'])
G.eval()

# Load aug letter bank
letter_bank = torch.load(aug_bank_path, weights_only=True).cuda()
print(f"letter_bank shape: {tuple(letter_bank.shape)}")
n_per_class = letter_bank.shape[1]

# Sample
B = 4
K = args['K']
torch.manual_seed(0)
eps = torch.randn(B, args['d_noise'], device='cuda')
letter_classes = torch.randint(0, args['n_letters'], (B, K), device='cuda')
rand_idx = torch.randint(0, n_per_class, (B, K), device='cuda')
letter_targets = letter_bank[letter_classes, rand_idx]  # [B, K, 1, 32, 32]

with torch.no_grad():
    layers, patches = G(eps, letter_classes)  # [B,K,1,128,128], [B,K,1,32,32]
# patches uses the current sigmoid_t — should match training
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

for b in range(B):
    fig, axes = plt.subplots(3, K, figsize=(K * 1.3, 4.2), facecolor='#0d0f14')
    for k in range(K):
        cls = letter_classes[b, k].item()
        axes[0, k].imshow(patches[b, k, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[0, k].set_title(ALPHABET[cls], color='cyan', fontsize=14, pad=2)
        axes[0, k].axis('off')
        axes[1, k].imshow(letter_targets[b, k, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[1, k].axis('off')
        axes[2, k].imshow(layers[b, k, 0].cpu(), cmap='gray', vmin=0, vmax=1)
        axes[2, k].axis('off')
    fig.text(0.01, 0.78, 'decoded', color='white', fontsize=11, rotation=90, va='center')
    fig.text(0.01, 0.50, 'target',  color='white', fontsize=11, rotation=90, va='center')
    fig.text(0.01, 0.20, 'placed',  color='white', fontsize=11, rotation=90, va='center')
    fig.suptitle(f'Sample {b} — letter alignment (decoded vs target vs placed)',
                 color='white', fontsize=11)
    plt.subplots_adjust(left=0.03, right=0.99, top=0.88, bottom=0.02, hspace=0.10, wspace=0.05)
    out_path = f"{out_prefix}_sample{b}.png"
    plt.savefig(out_path, dpi=110, facecolor='#0d0f14')
    plt.close()
    print(f"saved {out_path}")

# Also save canvas composite for these 4 samples
fig, axes = plt.subplots(1, B, figsize=(B * 2.5, 3), facecolor='#0d0f14')
canvas = layers.sum(dim=1).clamp(0, 1)  # [B, 1, 128, 128]
for b in range(B):
    axes[b].imshow(canvas[b, 0].cpu(), cmap='gray', vmin=0, vmax=1)
    axes[b].axis('off')
    letters_str = ''.join(ALPHABET[letter_classes[b, k].item()] for k in range(K))
    axes[b].set_title(letters_str, color='cyan', fontsize=10)
fig.suptitle('Composite canvas for each sample', color='white', fontsize=11)
plt.subplots_adjust(left=0.01, right=0.99, top=0.85, bottom=0.02, wspace=0.05)
out_path = f"{out_prefix}_composite.png"
plt.savefig(out_path, dpi=110, facecolor='#0d0f14')
plt.close()
print(f"saved {out_path}")
print('DONE')
