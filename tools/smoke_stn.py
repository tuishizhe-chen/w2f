import sys
sys.path.insert(0, '/root/w2f/src')
import torch
from face_drift_multi_stn import MultiStnGen, stn_place
G = MultiStnGen(d_noise=128, base=192, K=12, patch_size=48, head_refine=2).cuda()
print('G params:', sum(p.numel() for p in G.parameters()) / 1e6, 'M')
eps = torch.randn(4, 128, device='cuda')
patches, theta = G(eps)
print('patches:', tuple(patches.shape))
print('theta:', tuple(theta.shape))
print('r range:', theta[..., 0].min().item(), theta[..., 0].max().item())
print('tx range:', theta[..., 1].min().item(), theta[..., 1].max().item())
flat = patches.view(4 * 12, 1, 48, 48)
r = theta[..., 0].reshape(-1); tx = theta[..., 1].reshape(-1); ty = theta[..., 2].reshape(-1)
placed = stn_place(flat, 1.0 / r, tx, ty, 128).view(4, 12, 1, 128, 128)
canvas = placed.sum(dim=1).clamp(0, 1)
print('canvas:', tuple(canvas.shape), 'mean:', canvas.mean().item())
print('per-layer ink:', placed.mean(dim=(0, 2, 3, 4)).cpu().tolist())
print('all OK')
