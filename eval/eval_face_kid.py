"""Objective FACE-quality metric for the edge-map domain.

Per research: edge-domain KID (unbiased kernel-MMD) in a FROZEN face_ae128
encoder feature space — NOT Inception-FID (domain mismatch), NOT Fréchet
(covariance unstable at small N). Plus Kynkaanniemi Precision/Recall to split
realism vs diversity (mode-collapse detector).

  KID  (x1e3, LOWER = closer to real edge-face distribution)
  prec (realism: fraction of generated inside the real manifold)
  rec  (diversity: fraction of real covered by generated; low = mode collapse)

Usage:
  python .local/eval_face_kid.py <G_ckpt.pt> <ae_ckpt.pt> <edge_bank.pt> [n_gen]
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/w2f/src')
from face_drift_multi_transformer import HierarchicalSlotGen
from face_ae128 import FaceAE128

g_ckpt = sys.argv[1]
ae_ckpt = sys.argv[2]
bank_path = sys.argv[3]
n_gen = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
# optional argv[5]: override the eval-time sigmoid temperature (diagnostic for
# whether late over-sharpening from the sigmoid_t curriculum is what hurts KID).
sig_override = float(sys.argv[5]) if len(sys.argv) > 5 else None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---- frozen AE encoder feature extractor ----
ck = torch.load(ae_ckpt, map_location=device, weights_only=False)
ae = FaceAE128(ch=ck['lat_ch'], base=ck['base'], vae=ck['vae']).to(device)
ae.load_state_dict(ck['ae'])
ae.eval()
for p in ae.parameters():
    p.requires_grad_(False)

@torch.no_grad()
def feats(x):                      # x: [B,1,128,128] in [0,1]
    z = ae.encode(x)               # [B,ch,16,16]
    z = F.adaptive_avg_pool2d(z, 4)  # [B,ch,4,4]  -> coarse spatial structure
    return z.flatten(1)            # [B, ch*16]

# ---- real bank features (cached) ----
bank = torch.load(bank_path, weights_only=True)        # uint8 [N,128,128]
import hashlib
key = hashlib.md5((ae_ckpt + bank_path).encode()).hexdigest()[:8]
cache = f'/root/w2f/checkpoints/_edgefeats_{key}.pt'
import os
if os.path.exists(cache):
    real_f = torch.load(cache, map_location=device, weights_only=True)
else:
    rf = []
    with torch.no_grad():
        for i in range(0, bank.shape[0], 512):
            x = (bank[i:i + 512].float() / 255.0).unsqueeze(1).to(device)
            rf.append(feats(x))
    real_f = torch.cat(rf, 0)
    torch.save(real_f, cache)

# ---- generated canvas features ----
gk = torch.load(g_ckpt, map_location=device, weights_only=False)
a = gk['args']
G = HierarchicalSlotGen(
    d_noise=a['d_noise'], K=a['K'], d_token=a['d_token'], n_layers=a['n_layers'],
    n_heads=a['n_heads'], patch_size=a['patch_size'], canvas=128,
    sigmoid_t=a.get('sigmoid_t_end', a['sigmoid_t']),
    allow_rotation=not a.get('no_rotation', False),
    r_min=a['r_min'], r_max=a['r_max'], dxy_max=a['dxy_max'],
    letter_mode=a.get('letter_mode', False), n_letters=a.get('n_letters', 26),
    alpha_composite=a.get('alpha_composite', False),
    slot_intensity=a.get('slot_intensity', False),
    letter_film=a.get('letter_film', False),
).to(device)
G.load_state_dict(gk['G_ema'])
if sig_override is not None:
    G.set_sigmoid_t(sig_override)
    print(f"[eval] sigmoid_t override -> {sig_override}")
G.eval()
gf = []
torch.manual_seed(0)
with torch.no_grad():
    for i in range(0, n_gen, 256):
        b = min(256, n_gen - i)
        eps = torch.randn(b, a['d_noise'], device=device)
        if a.get('letter_mode', False):
            cls = torch.randint(0, a['n_letters'], (b, a['K']), device=device)
            layers, _ = G(eps, cls)
        else:
            layers = G(eps)
        # match training composition: per-slot uniform gain (slot_intensity),
        # else softmax-alpha (alpha_composite, deprecated), else plain sum.
        if a.get('slot_intensity', False) and getattr(G, 'last_slot_logit', None) is not None:
            gain = torch.sigmoid(G.last_slot_logit).view(layers.shape[0], a['K'], 1, 1, 1)
            canvas = (gain * layers).sum(1).clamp(0, 1)
        elif a.get('alpha_composite', False) and getattr(G, 'last_placed_alpha', None) is not None:
            pa = G.last_placed_alpha                                       # [b,K,1,128,128]
            bg = G.bg_alpha.view(1, 1, 1, 1, 1).expand(pa.shape[0], 1, 1, 128, 128)
            w = torch.softmax(torch.cat([pa, bg], dim=1), dim=1)[:, :a['K']]
            canvas = (w * layers).sum(1).clamp(0, 1)
        else:
            canvas = layers.sum(1).clamp(0, 1)         # [b,1,128,128]
        gf.append(feats(canvas))
gen_f = torch.cat(gf, 0)

# ---- standardize with real stats ----
mu, sd = real_f.mean(0, keepdim=True), real_f.std(0, keepdim=True).clamp_min(1e-6)
real_s = (real_f - mu) / sd
gen_s = (gen_f - mu) / sd

# ---- KID: unbiased polynomial-kernel MMD^2, block estimator ----
def poly_k(X, Y):
    d = X.shape[1]
    return (X @ Y.t() / d + 1.0).pow(3)

def mmd2(X, Y):
    m, n = X.shape[0], Y.shape[0]
    Kxx, Kyy, Kxy = poly_k(X, X), poly_k(Y, Y), poly_k(X, Y)
    sxx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
    syy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
    return (sxx + syy - 2 * Kxy.mean()).item()

vals = []
sub = min(1000, real_s.shape[0], gen_s.shape[0])
for _ in range(10):
    ri = torch.randperm(real_s.shape[0], device=device)[:sub]
    gi = torch.randperm(gen_s.shape[0], device=device)[:sub]
    vals.append(mmd2(real_s[ri], gen_s[gi]))
kid_m = sum(vals) / len(vals)
kid_s = (sum((v - kid_m) ** 2 for v in vals) / len(vals)) ** 0.5

# ---- Precision/Recall (Kynkaanniemi, k=3) on a real subset for speed ----
def radii(X, k=3):
    d = torch.cdist(X, X)
    d.fill_diagonal_(float('inf'))
    return d.topk(k, largest=False, dim=1).values[:, -1]

rsub = real_s[torch.randperm(real_s.shape[0], device=device)[:2048]]
gsub = gen_s[torch.randperm(gen_s.shape[0], device=device)[:2048]]
rr, rg = radii(rsub), radii(gsub)
d_rg = torch.cdist(rsub, gsub)
prec = (d_rg <= rr[:, None]).any(0).float().mean().item()
rec = (d_rg <= rg[None, :]).any(1).float().mean().item()

name = g_ckpt.split('/')[-2] if '/' in g_ckpt else g_ckpt
print(f"=== {name} ===")
print(f"KID(x1e3) = {kid_m*1e3:.3f} +/- {kid_s*1e3:.3f}   prec = {prec:.3f}   rec = {rec:.3f}   (n_gen={n_gen})")
