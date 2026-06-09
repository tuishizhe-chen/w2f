"""Diagnostic: can K=12 letter placements be optimized to match a single face?
Pick a target face, init theta randomly, run Adam on theta, render, save grid.
If even this direct optimization can't form face-like outputs, letter composition
is fundamentally limited and we need a different approach."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '/root/w2f/src')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import numpy as np
import math
from face_letters_v2 import stn_place

device = 'cuda'
torch.manual_seed(0)

# load
bank = torch.load('/root/w2f/checkpoints/edge_bank_128_varied.pt', weights_only=True).float() / 255.
bank = bank.to(device)  # [N, 128, 128]
letter_bank = torch.load('/root/w2f/checkpoints/letter_bank.pt', weights_only=True).float() / 255.
letter_bank = letter_bank.to(device)  # [26, 500, 64, 64]

# pick 8 target faces (diverse)
target_idx = torch.tensor([100, 500, 1000, 2000, 3000, 5000, 8000, 15000])
targets = bank[target_idx]  # [8, 128, 128]

K = 12
N_face = 8
# random letter labels per target
labels = torch.randint(0, 26, (N_face, K), device=device)  # [8, 12]
letter_idx = torch.randint(0, 500, (N_face, K), device=device)
letter_imgs = letter_bank[labels.reshape(-1), letter_idx.reshape(-1)].unsqueeze(1)  # [N*K, 1, 64, 64]

# theta init: K-adaptive grid + small jitter
cols = int(math.ceil(math.sqrt(K)))
rows = int(math.ceil(K / cols))
bx = torch.tensor([-0.45 + 0.9 * ((k % cols) + 0.5) / cols for k in range(K)], device=device)
by = torch.tensor([-0.45 + 0.9 * ((k // cols) + 0.5) / rows for k in range(K)], device=device)

r_init = 0.35
r_log = torch.full((N_face, K), float(math.log(r_init)), device=device, requires_grad=True)
tx_raw = torch.zeros((N_face, K), device=device, requires_grad=True)
ty_raw = torch.zeros((N_face, K), device=device, requires_grad=True)

opt = torch.optim.Adam([r_log, tx_raw, ty_raw], lr=0.02)


def iou_loss(canvas, target):
    """1 - IoU; both in [0,1]. Encourages canvas overlap with target."""
    inter = (canvas * target).sum(dim=(-1, -2))
    union = canvas.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) - inter
    return (1.0 - inter / (union + 1e-6)).mean()


def dice_loss(canvas, target):
    inter = (canvas * target).sum(dim=(-1, -2))
    return (1.0 - 2.0 * inter / (canvas.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2)) + 1e-6)).mean()


def weighted_l1(canvas, target):
    """5x weight on positive pixels of target → covers all target strokes."""
    w = 1.0 + 4.0 * target
    return (w * (canvas - target).abs()).mean()


loss_kind = sys.argv[1] if len(sys.argv) > 1 else 'iou'
loss_fn = {'mse': F.mse_loss, 'iou': iou_loss, 'dice': dice_loss, 'wl1': weighted_l1}[loss_kind]
print(f'using loss={loss_kind}')

# also save snapshots at steps 0, 200, 1000, 5000
snapshots = {}
for step in range(5001):
    r = torch.exp(r_log).clamp(0.10, 0.70)
    tx = bx.view(1, K) + 0.30 * torch.tanh(tx_raw)
    ty = by.view(1, K) + 0.30 * torch.tanh(ty_raw)
    flat_r = r.reshape(-1)
    flat_tx = tx.reshape(-1)
    flat_ty = ty.reshape(-1)
    placed = stn_place(letter_imgs, 1.0 / flat_r, flat_tx, flat_ty, 128).view(N_face, K, 128, 128)
    canvas = placed.max(dim=1).values  # [N, 128, 128]
    loss = loss_fn(canvas, targets)
    opt.zero_grad(); loss.backward(); opt.step()
    if step in (0, 200, 1000, 5000):
        snapshots[step] = (canvas.detach().cpu().numpy().copy(), placed.detach().cpu().numpy().copy())
        print(f'step={step} loss={loss.item():.4f} '
              f'r_mean={r.mean().item():.3f} r_std={r.std().item():.3f} '
              f'ink={canvas.mean().item():.3f}')

# render
fig, axes = plt.subplots(len(snapshots) + 1, N_face,
                         figsize=(N_face * 1.6, (len(snapshots) + 1) * 1.6),
                         facecolor='#0d0f14')
for i in range(N_face):
    axes[0, i].imshow(targets[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1); axes[0, i].axis('off')
axes[0, 0].set_ylabel('target', color='w', fontsize=9)
for row, (step, (canvas, _)) in enumerate(snapshots.items(), start=1):
    for i in range(N_face):
        axes[row, i].imshow(canvas[i], cmap='gray', vmin=0, vmax=1); axes[row, i].axis('off')
    axes[row, 0].set_ylabel(f'step={step}', color='w', fontsize=9)
plt.subplots_adjust(left=0.04, right=0.99, top=0.97, bottom=0.01, hspace=0.05, wspace=0.05)
plt.savefig(f'/tmp/letter_optim_{loss_kind}.png', dpi=100, facecolor='#0d0f14'); plt.close()
print(f'saved /tmp/letter_optim_{loss_kind}.png')
