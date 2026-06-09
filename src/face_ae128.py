"""128-tailored spatial edge-AE for CelebA-edge faces.

Encoder: 1×128×128 → [B, ch, 16, 16]   (3 downsamples)
Decoder: [B, ch, 16, 16] → 1×128×128   (3 upsamples)

Default latent = 16×16×16 = 4096-d structured space → reconstructs sharply and is
a good drift space (low-dim enough that distances discriminate, retains spatial layout).

Produces:
  checkpoints/face_ae128.pt   (AE weights)
  checkpoints/face_lat.pt     (encoded bank latents, [N,ch,16,16] for D2 drift)
  samples/face_ae/recon_*.png (real vs recon)
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F


def _up(cin, cout):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(cin, cout, 3, 1, 1),
        nn.GroupNorm(8, cout),
        nn.GELU(),
    )


class Encoder128(nn.Module):
    """128² → [B, ch, 16, 16].
    If vae=True, returns (mu, logvar) for reparameterization."""
    def __init__(self, ch: int = 16, base: int = 48, vae: bool = False):
        super().__init__()
        def dn(ci, co):
            return nn.Sequential(nn.Conv2d(ci, co, 4, 2, 1),
                                 nn.GroupNorm(8, co), nn.GELU())
        self.net = nn.Sequential(dn(1, base), dn(base, base * 2),
                                 dn(base * 2, base * 4))   # 128→16
        self.vae = vae
        if vae:
            self.to_mu = nn.Conv2d(base * 4, ch, 1)
            self.to_logvar = nn.Conv2d(base * 4, ch, 1)
        else:
            self.to_lat = nn.Conv2d(base * 4, ch, 1)

    def forward(self, x):
        h = self.net(x)
        if self.vae:
            return self.to_mu(h), self.to_logvar(h)
        return self.to_lat(h)


class Decoder128(nn.Module):
    def __init__(self, ch: int = 16, base: int = 48):
        super().__init__()
        self.from_lat = nn.Conv2d(ch, base * 4, 3, 1, 1)
        self.up = nn.Sequential(_up(base * 4, base * 4),
                                _up(base * 4, base * 2),
                                _up(base * 2, base))      # 16→128
        self.head = nn.Conv2d(base, 1, 3, 1, 1)

    def forward(self, z):
        return self.head(self.up(self.from_lat(z)))


class FaceAE128(nn.Module):
    def __init__(self, ch: int = 16, base: int = 48, vae: bool = False):
        super().__init__()
        self.vae = vae
        self.enc = Encoder128(ch, base, vae=vae)
        self.dec = Decoder128(ch, base)

    def forward(self, x):
        if self.vae:
            mu, logvar = self.enc(x)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return self.dec(z), z, mu, logvar
        else:
            z = self.enc(x)
            return self.dec(z), z

    def encode(self, x):
        """Always returns deterministic latent (mu if VAE) for downstream use."""
        if self.vae:
            mu, _ = self.enc(x)
            return mu
        return self.enc(x)


@torch.no_grad()
def save_recon(ae, bank, idx, path, device):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ae.eval()
    x = (bank[idx].float() / 255.0).unsqueeze(1).to(device)
    rec = torch.sigmoid(ae(x)[0])[:, 0].cpu().numpy()
    ae.train()
    tgt = (bank[idx].float() / 255.0).numpy()
    n = len(idx)
    fig, axes = plt.subplots(2, n, figsize=(n * 1.7, 4), facecolor='#0d0f14')
    for i in range(n):
        axes[0, i].imshow(tgt[i], cmap='gray', vmin=0, vmax=1); axes[0, i].axis('off')
        axes[1, i].imshow(rec[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
    fig.suptitle(f'Face-AE 128 — {path.stem} (top: real edge, bottom: AE recon)',
                 color='white', fontsize=11)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[ae] recon → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', default='./checkpoints/edge_bank_128.pt')
    ap.add_argument('--out', default='./samples/face_ae')
    ap.add_argument('--ckpt', default='./checkpoints/face_ae128.pt')
    ap.add_argument('--lat-out', default='./checkpoints/face_lat.pt', dest='lat_out')
    ap.add_argument('--lat-ch', type=int, default=16, dest='lat_ch')
    ap.add_argument('--base', type=int, default=48, help='channel base (bigger = more capacity)')
    ap.add_argument('--beta', type=float, default=0.0, help='KL weight for VAE (>0 → VAE mode)')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--pos-weight', type=float, default=6.0, dest='pos_weight')
    ap.add_argument('--sample-every', type=int, default=500, dest='sample_every')
    args = ap.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    bank = torch.load(args.bank, weights_only=True)   # uint8 [N,128,128]
    N = bank.shape[0]
    ae = FaceAE128(ch=args.lat_ch, base=args.base, vae=(args.beta > 0)).to(device)
    n_p = sum(p.numel() for p in ae.parameters())
    print(f"[ae] bank={tuple(bank.shape)} latent=16x16x{args.lat_ch}={args.lat_ch*256} "
          f"params={n_p/1e6:.2f}M device={device}", flush=True)
    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr, betas=(0.9, 0.95))
    pw = torch.tensor([args.pos_weight], device=device)
    fixed = torch.randint(0, N, (8,))
    t0 = time.time()

    ae.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.bs,))
        x = (bank[idx].float() / 255.0).unsqueeze(1).to(device)
        if args.beta > 0:
            logits, _, mu, logvar = ae(x)
            bce = F.binary_cross_entropy_with_logits(logits, x, pos_weight=pw)
            # KL(q(z|x) || N(0,I)) per-element, then mean
            kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = bce + args.beta * kl
        else:
            logits, _ = ae(x)
            bce = F.binary_cross_entropy_with_logits(logits, x, pos_weight=pw)
            kl = torch.zeros((), device=device)
            loss = bce
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            print(f"step={step} bce={bce.item():.4f} kl={kl.item():.4f} t={time.time()-t0:.0f}s", flush=True)
        if step % args.sample_every == 0 or step == args.steps:
            save_recon(ae, bank, fixed, out / f'recon_step{step:05d}.png', device)

    Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'ae': ae.state_dict(), 'lat_ch': args.lat_ch,
                'base': args.base, 'vae': args.beta > 0}, args.ckpt)
    ae.eval(); lats = []
    with torch.no_grad():
        for i in range(0, N, 512):
            x = (bank[i:i + 512].float() / 255.0).unsqueeze(1).to(device)
            lats.append(ae.encode(x).cpu())
    lats = torch.cat(lats, 0)   # [N, ch, 16, 16]
    torch.save(lats, args.lat_out)
    print(f"[ae] done ({time.time()-t0:.0f}s)  ckpt→{args.ckpt}  lats→{args.lat_out} {tuple(lats.shape)}", flush=True)


if __name__ == '__main__':
    main()
