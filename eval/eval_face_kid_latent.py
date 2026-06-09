"""KID for the DIRECT-LATENT face generator (face_drift.py LatentGen).
Generation: eps -> LatentGen -> latent -> GEN-AE decoder -> face image.
Metric:     face image -> METRIC-AE encoder features -> KID vs real bank.
GEN-AE != METRIC-AE so the KID is NOT circular (fair vs the slot models, which
use no AE in generation).

Usage:
  python .local/eval_face_kid_latent.py <Gfinal.pt> <gen_ae.pt> <metric_ae.pt> <edge_bank.pt> [n_gen]
"""
import sys, os, hashlib
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/w2f/src')
from face_drift import LatentGen
from face_ae128 import FaceAE128

g_ckpt, gen_ae_ckpt, metric_ae_ckpt, bank_path = sys.argv[1:5]
n_gen = int(sys.argv[5]) if len(sys.argv) > 5 else 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_ae(p):
    ck = torch.load(p, map_location=device, weights_only=False)
    ae = FaceAE128(ch=ck['lat_ch'], base=ck['base'], vae=ck['vae']).to(device)
    ae.load_state_dict(ck['ae']); ae.eval()
    for q in ae.parameters(): q.requires_grad_(False)
    return ae, ck['lat_ch']

gen_ae, gen_ch = load_ae(gen_ae_ckpt)       # decodes generated latents
met_ae, _ = load_ae(metric_ae_ckpt)         # independent metric features

@torch.no_grad()
def feats(x):                                # x:[B,1,128,128] in [0,1]
    z = met_ae.encode(x)
    return F.adaptive_avg_pool2d(z, 4).flatten(1)

# real bank features (cache keyed by metric AE + bank)
bank = torch.load(bank_path, weights_only=True)
key = hashlib.md5((metric_ae_ckpt + bank_path).encode()).hexdigest()[:8]
cache = f'/root/w2f/checkpoints/_edgefeats_{key}.pt'
if os.path.exists(cache):
    real_f = torch.load(cache, map_location=device, weights_only=True)
else:
    rf = []
    with torch.no_grad():
        for i in range(0, bank.shape[0], 512):
            x = (bank[i:i+512].float()/255.).unsqueeze(1).to(device)
            rf.append(feats(x))
    real_f = torch.cat(rf, 0); torch.save(real_f, cache)

# generated faces -> features
gk = torch.load(g_ckpt, map_location=device, weights_only=False)
a = gk['args']
G = LatentGen(d_noise=a['d_noise'], ch=gen_ch).to(device)
G.load_state_dict(gk['G']); G.eval()
gf = []
torch.manual_seed(0)
with torch.no_grad():
    for i in range(0, n_gen, 256):
        b = min(256, n_gen - i)
        eps = torch.randn(b, a['d_noise'], device=device)
        face = torch.sigmoid(gen_ae.dec(G(eps)))    # [b,1,128,128]
        gf.append(feats(face))
gen_f = torch.cat(gf, 0)

mu, sd = real_f.mean(0, keepdim=True), real_f.std(0, keepdim=True).clamp_min(1e-6)
real_s, gen_s = (real_f-mu)/sd, (gen_f-mu)/sd

def poly_k(X, Y):
    d = X.shape[1]; return (X @ Y.t()/d + 1.0).pow(3)
def mmd2(X, Y):
    m, n = X.shape[0], Y.shape[0]
    Kxx, Kyy, Kxy = poly_k(X,X), poly_k(Y,Y), poly_k(X,Y)
    sxx = (Kxx.sum()-Kxx.diag().sum())/(m*(m-1)); syy = (Kyy.sum()-Kyy.diag().sum())/(n*(n-1))
    return (sxx+syy-2*Kxy.mean()).item()
sub = min(1000, real_s.shape[0], gen_s.shape[0]); vals=[]
for _ in range(10):
    ri = torch.randperm(real_s.shape[0], device=device)[:sub]
    gi = torch.randperm(gen_s.shape[0], device=device)[:sub]
    vals.append(mmd2(real_s[ri], gen_s[gi]))
kid_m = sum(vals)/len(vals); kid_s = (sum((v-kid_m)**2 for v in vals)/len(vals))**0.5

def radii(X, k=3):
    d = torch.cdist(X, X); d.fill_diagonal_(float('inf'))
    return d.topk(k, largest=False, dim=1).values[:, -1]
rsub = real_s[torch.randperm(real_s.shape[0], device=device)[:2048]]
gsub = gen_s[torch.randperm(gen_s.shape[0], device=device)[:2048]]
rr, rg = radii(rsub), radii(gsub); d_rg = torch.cdist(rsub, gsub)
prec = (d_rg <= rr[:, None]).any(0).float().mean().item()
rec = (d_rg <= rg[None, :]).any(1).float().mean().item()

name = g_ckpt.split('/')[-2] if '/' in g_ckpt else g_ckpt
print(f"=== {name} (DIRECT-LATENT) ===")
print(f"KID(x1e3) = {kid_m*1e3:.3f} +/- {kid_s*1e3:.3f}   prec = {prec:.3f}   rec = {rec:.3f}   (n_gen={n_gen})")
