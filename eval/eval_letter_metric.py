"""Objective LETTER-legibility metric: run the frozen LetterCNN over decoded
patches from a G checkpoint and report how often they classify as the intended
letter. Replaces eyeballing the align grid.

Usage:
  python .local/eval_letter_metric.py <G_ckpt.pt> <letter_cnn.pt> [n_samples]

Prints:
  letter_acc  = fraction of B*K patches whose argmax == intended class
  mean_conf   = mean softmax-max over patches (calibration of legibility)
  worst5/best5 per-class accuracy
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/w2f/src')
from face_drift_multi_transformer import HierarchicalSlotGen
from classifier import load as load_cls

ckpt_path = sys.argv[1]
cls_path = sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
a = ckpt['args']
G = HierarchicalSlotGen(
    d_noise=a['d_noise'], K=a['K'], d_token=a['d_token'],
    n_layers=a['n_layers'], n_heads=a['n_heads'], patch_size=a['patch_size'],
    canvas=128, sigmoid_t=a.get('sigmoid_t_end', a['sigmoid_t']),
    allow_rotation=not a.get('no_rotation', False),
    r_min=a['r_min'], r_max=a['r_max'], dxy_max=a['dxy_max'],
    letter_mode=True, n_letters=a['n_letters'],
    alpha_composite=a.get('alpha_composite', False),
    slot_intensity=a.get('slot_intensity', False),
    letter_film=a.get('letter_film', False),
).to(device)
G.load_state_dict(ckpt['G_ema'])
G.eval()
cnn = load_cls(cls_path, device)

K = a['K']
P = a['patch_size']
torch.manual_seed(0)
with torch.no_grad():
    eps = torch.randn(B, a['d_noise'], device=device)
    cls = torch.randint(0, a['n_letters'], (B, K), device=device)
    _layers, patches = G(eps, cls)                 # patches [B,K,1,P,P]
    x = patches.reshape(B * K, 1, P, P)
    if P != 32:
        x = F.interpolate(x, size=32, mode='bilinear', align_corners=False)
    logits = cnn(x)
    prob = logits.softmax(-1)
    pred = logits.argmax(-1)
    tgt = cls.reshape(-1)
    correct = (pred == tgt)
    acc = correct.float().mean().item()
    conf = prob.max(-1).values.mean().item()
    # per-class
    per = []
    for c in range(a['n_letters']):
        m = (tgt == c)
        if m.sum() > 0:
            per.append((c, correct[m].float().mean().item(), int(m.sum())))
per.sort(key=lambda t: t[1])
abc = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
name = ckpt_path.split('/')[-2] if '/' in ckpt_path else ckpt_path
print(f"=== {name} ===")
print(f"letter_acc = {acc:.3f}   mean_conf = {conf:.3f}   (B*K={B*K} patches)")
worst = ', '.join(f"{abc[c]}:{v:.2f}" for c, v, _ in per[:5])
best = ', '.join(f"{abc[c]}:{v:.2f}" for c, v, _ in per[-5:])
print(f"worst5: {worst}")
print(f"best5 : {best}")
