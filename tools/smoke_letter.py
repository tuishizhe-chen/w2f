"""Smoke test for letter-mode HierarchicalSlotGen."""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, '/root/w2f/src')
from face_drift_multi_transformer import HierarchicalSlotGen

torch.manual_seed(0)

# Build model in letter mode
G = HierarchicalSlotGen(K=12, d_token=320, n_layers=6, n_heads=8,
                        letter_mode=True, n_letters=26).cuda()
n = sum(p.numel() for p in G.parameters())
print(f'params: {n/1e6:.2f}M (letter_mode)')
print(f'letter_embed shape: {G.letter_embed.weight.shape}')

# Forward with random eps + letter_classes
B, K = 4, 12
eps = torch.randn(B, 128, device='cuda')
letter_classes = torch.randint(0, 26, (B, K), device='cuda')
print(f'eps {tuple(eps.shape)}  letter_classes {tuple(letter_classes.shape)}')

layers, patches = G(eps, letter_classes)
print(f'layers {tuple(layers.shape)}  patches {tuple(patches.shape)}')
print(f'layers ink={layers.mean().item():.4f}  patches ink={patches.mean().item():.4f}')

# bvar across batch
canvas = layers.sum(dim=1).clamp(0, 1)
print(f'canvas bvar: {canvas.var(dim=0).mean().item():.6f}')

# Same letter classes across batch members → should NOT make outputs identical
# (eps differs), but should produce similar patches for same (b,k) class
print('\n-- letter consistency test --')
# Pick batch 0 slot 0, find another (b,k) with same class:
target_class = letter_classes[0, 0].item()
print(f'slot[0,0] class: {target_class}')
# Compute patch similarity for slots with same class
matches = []
for b in range(B):
    for k in range(K):
        if letter_classes[b, k].item() == target_class and (b, k) != (0, 0):
            sim = F.cosine_similarity(patches[0, 0].flatten().unsqueeze(0),
                                       patches[b, k].flatten().unsqueeze(0))
            matches.append(((b, k), sim.item()))
if matches:
    print(f'other slots with class {target_class}: {matches[:3]}')

# Backward sanity
loss = layers.mean() + patches.mean()
loss.backward()
print(f'\nbackward OK')
# Check letter_embed got gradient
print(f'letter_embed grad norm: {G.letter_embed.weight.grad.norm().item():.4e}')

print('\n=== SMOKE OK ===')
