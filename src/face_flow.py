"""Flow matching in AE latent space (alternative to drift).

Standard rectified-flow / OT-flow setup:
  for each step:
    real_lat ~ bank, noise ~ N(0,I)
    t ~ U(0,1), xt = (1-t)*noise + t*real_lat
    v_target = real_lat - noise
    v_pred = G(xt, t)
    loss = MSE(v_pred, v_target)

  Sampling: x0 = randn; integrate dx/dt = G(xt, t) from 0→1 via Euler steps;
            x1 → AE.dec → image.

Known to give sharp samples — drift over-averages, flow snaps to real points.
"""
from __future__ import annotations
import argparse, sys, time, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_ae128 import FaceAE128, _up


def sinusoidal_t_emb(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t [B] → [B, dim] sinusoidal time embedding."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FlowG(nn.Module):
    """G(xt[B,ch,16,16], t[B]) → v_pred[B,ch,16,16]. UNet-ish 16→8→16 with time emb."""
    def __init__(self, ch: int = 16, base: int = 192, t_dim: int = 128):
        super().__init__()
        self.t_proj = nn.Sequential(
            nn.Linear(t_dim, base), nn.SiLU(), nn.Linear(base, base)
        )
        self.t_dim = t_dim
        # in: ch→base at 16x16
        self.in_conv = nn.Conv2d(ch, base, 3, 1, 1)
        # down 16→8
        self.dn = nn.Sequential(
            nn.Conv2d(base, base, 3, 2, 1), nn.GroupNorm(8, base), nn.SiLU(),
            nn.Conv2d(base, base * 2, 3, 1, 1), nn.GroupNorm(8, base * 2), nn.SiLU(),
        )
        # mid at 8x8
        self.mid = nn.Sequential(
            nn.Conv2d(base * 2, base * 2, 3, 1, 1), nn.GroupNorm(8, base * 2), nn.SiLU(),
            nn.Conv2d(base * 2, base * 2, 3, 1, 1), nn.GroupNorm(8, base * 2), nn.SiLU(),
        )
        # up 8→16, with skip
        self.up = nn.Sequential(
            _up(base * 2, base),
            nn.Conv2d(base, base, 3, 1, 1), nn.GroupNorm(8, base), nn.SiLU(),
        )
        # out: base+base(skip)→ch
        self.out_conv = nn.Conv2d(base * 2, ch, 3, 1, 1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        # time addition: per stage we add t_emb to features
        self.t_to_base = nn.Linear(base, base)
        self.t_to_mid = nn.Linear(base, base * 2)

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B = xt.shape[0]
        t_emb = sinusoidal_t_emb(t, self.t_dim)
        t_emb = self.t_proj(t_emb)                                # [B, base]
        h0 = self.in_conv(xt)                                     # [B, base, 16, 16]
        h0 = h0 + self.t_to_base(t_emb)[:, :, None, None]
        h = self.dn(h0)                                           # [B, 2*base, 8, 8]
        h = h + self.t_to_mid(t_emb)[:, :, None, None]
        h = self.mid(h)
        h = self.up(h)                                            # [B, base, 16, 16]
        h = torch.cat([h, h0], dim=1)
        return self.out_conv(h)


def sample(G: nn.Module, n: int, ch: int, ls: int, device, n_steps: int = 50,
           fixed_eps: torch.Tensor = None) -> torch.Tensor:
    """Euler ODE integration from t=0 to t=1."""
    G.eval()
    if fixed_eps is None:
        x = torch.randn(n, ch, ls, ls, device=device)
    else:
        x = fixed_eps.to(device).clone()
    dt = 1.0 / n_steps
    with torch.no_grad():
        for k in range(n_steps):
            t = torch.full((n,), k * dt, device=device)
            v = G(x, t)
            x = x + v * dt
    G.train()
    return x


@torch.no_grad()
def save_grid(G, ae, bank_lat, fixed_eps, path, device, n_steps=50):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    n = fixed_eps.shape[0]
    z = sample(G, n, fixed_eps.shape[1], fixed_eps.shape[2], device, n_steps, fixed_eps)
    img = torch.sigmoid(ae.dec(z))[:, 0].cpu().numpy()
    ridx = torch.randint(0, bank_lat.shape[0], (8,))
    real_img = torch.sigmoid(ae.dec(bank_lat[ridx].to(device)))[:, 0].cpu().numpy()
    fig, axes = plt.subplots(3, 8, figsize=(16, 6), facecolor='#0d0f14')
    for i in range(8):
        axes[0, i].imshow(img[i], cmap='gray', vmin=0, vmax=1); axes[0, i].axis('off')
        axes[1, i].imshow(img[i + 8], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real_img[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    fig.suptitle(f'Flow matching — {path.stem}  (rows 1-2: gen ODE-integrated, row 3: real)',
                 color='white', fontsize=11)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.05, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    bank = torch.load(args.lat, weights_only=True)                # [N, ch, 16, 16]
    N, ch, ls, _ = bank.shape
    bank_gpu = bank.to(device)                                    # all to GPU (320MB, fits)

    ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=True)
    ae_base = ae_ck.get('base', args.ae_base)
    ae_vae = ae_ck.get('vae', False)
    ae = FaceAE128(ch=ae_ck['lat_ch'], base=ae_base, vae=ae_vae).to(device)
    ae.load_state_dict(ae_ck['ae']); ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)

    G = FlowG(ch=ch, base=args.base).to(device)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[flow] G params: {n_p/1e6:.2f}M  bank {tuple(bank.shape)}  device={device}",
          flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))
    fixed_eps = torch.randn(16, ch, ls, ls, device=device)
    t0 = time.time()

    G.train()
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, N, (args.bs,), device=device)
        x1 = bank_gpu[idx]                                        # [B, ch, 16, 16]
        x0 = torch.randn_like(x1)
        t = torch.rand(args.bs, device=device)
        # broadcast t for interpolation
        tb = t.view(-1, 1, 1, 1)
        xt = (1.0 - tb) * x0 + tb * x1
        v_target = x1 - x0
        v_pred = G(xt, t)
        loss = (v_pred - v_target).pow(2).mean()

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(f"step={step} loss={loss.item():.4f} grad={gnorm.item():.2f} "
                  f"t={time.time()-t0:.0f}s", flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            save_grid(G, ae, bank_gpu, fixed_eps,
                      out / f'flow_step{step:05d}.png', device,
                      n_steps=args.sample_n_steps)
            print(f"[flow] grid → flow_step{step:05d}.png", flush=True)

    torch.save({'G': G.state_dict(), 'args': vars(args)}, out / 'G_final.pt')
    print(f"[flow] done ({time.time()-t0:.0f}s)  → {out/'G_final.pt'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lat', default='./checkpoints/face_lat_big.pt')
    ap.add_argument('--ae-ckpt', default='./checkpoints/face_ae128_big.pt', dest='ae_ckpt')
    ap.add_argument('--ae-base', type=int, default=128, dest='ae_base')
    ap.add_argument('--out', default='./samples/face_flow')
    ap.add_argument('--base', type=int, default=192, help='G channel base')
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--steps', type=int, default=10000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--log-every', type=int, default=100, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=500, dest='sample_every')
    ap.add_argument('--sample-n-steps', type=int, default=50, dest='sample_n_steps')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
