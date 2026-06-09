import sys, torch
sys.path.insert(0, '/root/w2f/src')
from face_regions import build_priors, build_log_priors, ANCHORS_12
print('ANCHORS_12 len:', len(ANCHORS_12))
priors = build_priors(128, 128, device='cuda')
print('priors:', tuple(priors.shape), 'peak per-K:', priors.amax(dim=(1, 2, 3)).cpu().tolist()[:4], '...')
log_p = build_log_priors(128, 128, device='cuda')
print('log_priors range:', log_p.min().item(), log_p.max().item())

from face_drift_multi_pixel import MultiLayerPixelGen
G = MultiLayerPixelGen(d_noise=128, base=384, K=12, sigmoid_t=1.0, head_refine=2,
                       prior_alpha=2.0, prior_sigma_scale=1.0).cuda()
print('G params:', sum(p.numel() for p in G.parameters()) / 1e6, 'M')
eps = torch.randn(4, 128, device='cuda')
layers = G(eps)
print('layers:', tuple(layers.shape))
print('per-layer mean ink:', layers.mean(dim=(0, 2, 3, 4)).cpu().tolist())
canvas_sum = layers.sum(dim=1).clamp(0, 1)
print('canvas mean:', canvas_sum.mean().item())
print('all OK')
