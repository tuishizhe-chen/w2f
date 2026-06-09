"""KID for the PIXEL-SPACE face generator (face_drift_pixel.py PixelGen) — the
user's real fallbacks sweep12 (plain) + sweep13 (topk). These generate a face
edge map DIRECTLY in pixel space (eps -> conv -> [B,1,128,128]); no letters, no
slots, no AE. KID features come from the INDEPENDENT metric AE (fair vs slot runs).

PixelGen is embedded here (only needs _up from face_ae128) so the class matches
the archived checkpoint regardless of src drift.

Usage:
  python .local/eval_face_kid_pixel.py <G_final.pt> <metric_ae.pt> <ref_bank.pt> [n_gen]
"""
import sys, os, hashlib
import torch, torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/root/w2f/src')
from face_ae128 import _up, FaceAE128


class PixelGen(nn.Module):
    def __init__(self, d_noise=128, base=128, size=128, sigmoid_t=1.0, head_refine=0):
        super().__init__()
        self.base, self.size, self.sigmoid_t = base, size, sigmoid_t
        self.fc = nn.Linear(d_noise, base * 4 * 4)
        if size == 64:
            self.up = nn.Sequential(_up(base, base), _up(base, base),
                                    _up(base, base // 2), _up(base // 2, base // 4))
            head_in = base // 4
        else:
            self.up = nn.Sequential(_up(base, base), _up(base, base),
                                    _up(base, base // 2), _up(base // 2, base // 4),
                                    _up(base // 4, base // 8))
            head_in = base // 8
        if head_refine > 0:
            layers = []
            for _ in range(head_refine):
                layers += [nn.Conv2d(head_in, head_in, 3, 1, 1), nn.GELU()]
            layers += [nn.Conv2d(head_in, 1, 1, 1, 0)]
            self.head = nn.Sequential(*layers)
        else:
            self.head = nn.Conv2d(head_in, 1, 3, 1, 1)

    def forward(self, eps):
        B = eps.shape[0]
        x = self.fc(eps).view(B, self.base, 4, 4)
        x = self.up(x)
        return torch.sigmoid(self.head(x) * self.sigmoid_t)


g_ckpt, metric_ae_ckpt, bank_path = sys.argv[1:4]
n_gen = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ck = torch.load(metric_ae_ckpt, map_location=device, weights_only=False)
met = FaceAE128(ch=ck['lat_ch'], base=ck['base'], vae=ck['vae']).to(device)
met.load_state_dict(ck['ae']); met.eval()
for p in met.parameters(): p.requires_grad_(False)

@torch.no_grad()
def feats(x):
    z = met.encode(x)
    return F.adaptive_avg_pool2d(z, 4).flatten(1)

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
            if x.shape[-1] != 128:
                x = F.interpolate(x, size=128, mode='bilinear', align_corners=False)
            rf.append(feats(x))
    real_f = torch.cat(rf, 0); torch.save(real_f, cache)

gk = torch.load(g_ckpt, map_location=device, weights_only=False)
a = gk['args']
sig = a.get('sigmoid_t_end') or a.get('sigmoid_t', 1.0)
G = PixelGen(d_noise=a.get('d_noise', 128), base=a.get('base', 128), size=128,
             sigmoid_t=sig, head_refine=a.get('head_refine', 0)).to(device)
G.load_state_dict(gk.get('G_ema', gk['G'])); G.eval()
gf = []
torch.manual_seed(0)
with torch.no_grad():
    for i in range(0, n_gen, 256):
        b = min(256, n_gen - i)
        eps = torch.randn(b, a.get('d_noise', 128), device=device)
        face = G(eps)
        if face.shape[-1] != 128:
            face = F.interpolate(face, size=128, mode='bilinear', align_corners=False)
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
print(f"=== {name} (PIXEL-DRIFT fallback) ===")
print(f"KID(x1e3) = {kid_m*1e3:.3f} +/- {kid_s*1e3:.3f}   prec = {prec:.3f}   rec = {rec:.3f}   (n_gen={n_gen}, ref={bank_path.split('/')[-1]})")
