"""Smoke test for HierarchicalSlotGen on remote GPU."""
import sys, torch
sys.path.insert(0, '/root/w2f/src')
from face_drift_multi_transformer import HierarchicalSlotGen

print('=== HierarchicalSlotGen smoke ===')
G = HierarchicalSlotGen().cuda()
n = sum(p.numel() for p in G.parameters())
print(f'params: {n/1e6:.2f}M')
print(f'sigmoid_t={G.sigmoid_t}  allow_rotation={G.allow_rotation}  r=[{G.r_min},{G.r_max}]  dxy_max={G.dxy_max}')

eps = torch.randn(2, 128, device='cuda')
y = G(eps)
print(f'forward out: shape={tuple(y.shape)}  min={y.min().item():.4f}  max={y.max().item():.4f}  mean={y.mean().item():.4f}')

ink_per_layer = y.mean(dim=(0, 2, 3, 4))
print('per-layer ink (should differ to show slot diversity):')
for k, v in enumerate(ink_per_layer):
    print(f'  layer {k:2d}: ink={v.item():.5f}')

# Verify zero-init theta_head + to_logit -> initial r at midpoint, tx=ty=rot=0
# i.e. with eps fed through the model, the inverse-affine should produce a
# centered patch of half-width = (r_min+r_max)/2.
print('-- verifying initial theta sits at midpoint --')
with torch.no_grad():
    # Run forward; inspect raw_theta at the slot tokens (after stage1)
    eps_zero = torch.zeros(1, 128, device='cuda')
    # we want to peek inside; replicate forward up to raw_theta
    eps_tok = G.eps_proj(eps_zero).unsqueeze(1) + G.eps_role
    slot_tok = G.slot_embed.unsqueeze(0).expand(1, G.K, G.d_token)
    s1_in = torch.cat([slot_tok, eps_tok], dim=1) + G.stage1_marker
    s1_out = G.stage1(s1_in)
    s1_slots = s1_out[:, :G.K]
    raw = G.theta_head(s1_slots)
    print(f'raw_theta abs mean: {raw.abs().mean().item():.6f} (zero-init head -> SHOULD be small)')
    print(f'raw_theta per-K [0,:,0] (first 6): {raw[0, :6, 0].cpu().tolist()}')

# backward pass check
loss = y.mean()
loss.backward()
print('backward: OK')

# gradient flow: stage1 should receive gradients from STN path (via theta) AND render path
g_stage1 = next(G.stage1.parameters()).grad
g_stage2 = next(G.stage2.parameters()).grad
print(f'stage1 first param grad norm: {g_stage1.norm().item():.6e}')
print(f'stage2 first param grad norm: {g_stage2.norm().item():.6e}')
print('=== SMOKE OK ===')
