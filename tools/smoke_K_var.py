import sys, torch
sys.path.insert(0, '/root/w2f/src')
from face_drift_multi_pixel import MultiLayerPixelGen
for K in [6, 8, 12]:
    G = MultiLayerPixelGen(K=K, base=384, prior_alpha=2.0).cuda()
    eps = torch.randn(2, 128, device='cuda')
    layers = G(eps)
    n_p = sum(p.numel() for p in G.parameters()) / 1e6
    ink = layers.mean(dim=(0, 2, 3, 4))
    print(f'K={K}: layers shape {tuple(layers.shape)}  params {n_p:.2f}M  '
          f'per-layer ink range {ink.min().item():.3f}..{ink.max().item():.3f}')
print('all OK')
