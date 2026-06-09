"""Drift training, closer to the original lambertae/drifting setup.

Differences from our pp version (face_drift.py):
  - **per-problem multi-gen**: B problems × Cg gen samples per problem (vs Cg=1).
    Within a problem, 16 gen samples share random pos+neg → drift_loss applies
    intra-problem repulsion via old_gen-as-neg (this is the key dynamic the
    original paper relies on for diversity).
  - **EMA on generator params** (decay 0.999), evaluated at sampling time.
  - Loads our existing AE checkpoint; drift happens in its latent.

Args you'll want to play with:
  --b-problems (default 32)
  --gen-per (default 16)        Cg per problem
  --cp (default 32)             pos samples per problem
  --cn (default 16)             neg samples per problem (random reals)
"""
from __future__ import annotations
import argparse, sys, time, copy
from pathlib import Path
import torch, torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss
from face_ae128 import FaceAE128, _up
from face_drift import LatentGen   # reuse


def update_ema(ema_module, online_module, decay: float):
    with torch.no_grad():
        for ep, p in zip(ema_module.parameters(), online_module.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bank = torch.load(args.lat, weights_only=True)
    N, ch, ls, _ = bank.shape
    bank_gpu = bank.to(device)
    S = ch * ls * ls
    print(f"[orig] bank {tuple(bank.shape)}  S={S}", flush=True)

    ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=True)
    ae = FaceAE128(ch=ae_ck['lat_ch'], base=ae_ck.get('base', 48),
                   vae=ae_ck.get('vae', False)).to(device)
    ae.load_state_dict(ae_ck['ae']); ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)

    G = LatentGen(d_noise=args.d_noise, ch=ch).to(device)
    G_ema = copy.deepcopy(G).to(device)
    for p in G_ema.parameters(): p.requires_grad_(False)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[orig] G params: {n_p/1e6:.2f}M  B={args.b_problems}  Cg={args.gen_per}  "
          f"Cp={args.cp}  Cn={args.cn}  R={args.R}", flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))
    R_list = tuple(float(x) for x in args.R.split(','))

    fixed_eps = torch.randn(16, args.d_noise, device=device)   # for sample grid
    t0 = time.time()

    G.train()
    for step in range(1, args.steps + 1):
        B = args.b_problems
        Cg = args.gen_per
        total_gen = B * Cg
        eps = torch.randn(total_gen, args.d_noise, device=device)
        gen_lat = G(eps)                                          # [B*Cg, ch, 16, 16]
        gen_feat = gen_lat.flatten(1).view(B, Cg, S)              # [B, Cg, S]

        pos_idx = torch.randint(0, N, (B, args.cp))
        pos_feat = bank_gpu[pos_idx].flatten(2)                   # [B, Cp, S]
        neg_idx = torch.randint(0, N, (B, args.cn))
        neg_feat = bank_gpu[neg_idx].flatten(2)                   # [B, Cn, S]  (random reals, not gens)

        loss_vec, info = drift_loss(gen_feat, pos_feat, fixed_neg=neg_feat, R_list=R_list)
        loss = loss_vec.mean()

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 2.0)
        opt.step()
        update_ema(G_ema, G, args.ema)

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                bvar = gen_lat.var(dim=0).mean().item()
            print(f"step={step} loss={loss.item():.4f} scale={info['scale'].item():.3f} "
                  f"bvar={bvar:.4f} grad={gnorm.item():.2f} t={time.time()-t0:.0f}s",
                  flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            _save_grid(G_ema, ae, fixed_eps, bank_gpu, out / f'orig_step{step:05d}.png', device)

    torch.save({'G': G.state_dict(), 'G_ema': G_ema.state_dict(), 'args': vars(args)},
               out / 'G_final.pt')
    print(f"[orig] done ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def _save_grid(G_ema, ae, fixed_eps, bank, path, device):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    G_ema.eval()
    z = G_ema(fixed_eps)
    img = torch.sigmoid(ae.dec(z))[:, 0].cpu().numpy()
    G_ema.train()
    ridx = torch.randint(0, bank.shape[0], (8,))
    real_img = torch.sigmoid(ae.dec(bank[ridx])).cpu().numpy()[:, 0]
    fig, axes = plt.subplots(3, 8, figsize=(16, 6), facecolor='#0d0f14')
    for i in range(8):
        axes[0, i].imshow(img[i], cmap='gray', vmin=0, vmax=1); axes[0, i].axis('off')
        axes[1, i].imshow(img[i + 8], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real_img[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    fig.suptitle(f'Drift original-style — {path.stem}  (rows 1-2: gen EMA, row 3: real)',
                 color='white', fontsize=11)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[orig] grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat', default='./checkpoints/face_lat_v4.pt')
    ap.add_argument('--ae-ckpt', default='./checkpoints/face_ae_v4.pt', dest='ae_ckpt')
    ap.add_argument('--out', default='./samples/drift_orig')
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--b-problems', type=int, default=32, dest='b_problems')
    ap.add_argument('--gen-per', type=int, default=16, dest='gen_per')
    ap.add_argument('--cp', type=int, default=32)
    ap.add_argument('--cn', type=int, default=16)
    ap.add_argument('--steps', type=int, default=15000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--ema', type=float, default=0.999, help='EMA decay for generator')
    ap.add_argument('--R', default='0.02,0.05,0.2')
    ap.add_argument('--log-every', type=int, default=100, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=500, dest='sample_every')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
