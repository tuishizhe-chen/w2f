"""Letters phase: K letter slots → max-composed canvas → D2 face drift + per-letter L1+L2.

Architecture:
  for each sample b in batch, for each slot k in K:
    G(eps_b, letter_label_{b,k}) → letter_image[1,64,64] + theta(r, tx, ty)
  L_letter = L1 + L2 between letter_image and a random aug target for that label (from letter_bank).
  STN-place each letter onto 128 canvas, max-compose K placed images → canvas.
  L_face = per-particle drift on AE-encoded canvas (re-uses face_ae128 decoder/encoder, frozen).

Total = w_letter * L_letter + w_face * L_face.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss
from face_ae128 import FaceAE128, _up


# ─── STN place (copied from data.py to avoid pulling in unrelated deps) ───────

def stn_place(letter_imgs: torch.Tensor, inv_scale: torch.Tensor,
              tx: torch.Tensor, ty: torch.Tensor, canvas: int) -> torch.Tensor:
    """letter_imgs[N,1,h,w] → placed[N,1,canvas,canvas]. inv_scale = 1/r where r∈(0,1] is
    the visual ratio of letter on canvas (smaller r → smaller letter)."""
    N = letter_imgs.shape[0]
    cos = torch.ones_like(inv_scale)
    sin = torch.zeros_like(inv_scale)
    M = torch.zeros(N, 2, 3, device=letter_imgs.device, dtype=letter_imgs.dtype)
    M[:, 0, 0] = inv_scale
    M[:, 0, 1] = 0.0
    M[:, 0, 2] = tx
    M[:, 1, 0] = 0.0
    M[:, 1, 1] = inv_scale
    M[:, 1, 2] = ty
    grid = F.affine_grid(M, size=(N, 1, canvas, canvas), align_corners=False)
    return F.grid_sample(letter_imgs, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)


# ─── Generator ────────────────────────────────────────────────────────────────

class LetterImgGen(nn.Module):
    """eps[B,d_noise] + labels[B,K] → letters[B,K,1,64,64] + theta[B,K,3]."""
    def __init__(self, d_noise: int = 128, n_classes: int = 26,
                 letter_size: int = 64, base: int = 128):
        super().__init__()
        self.letter_size = letter_size
        self.base = base
        self.cls_emb = nn.Embedding(n_classes, 64)
        d_in = d_noise + 64
        self.fc = nn.Linear(d_in, base * 4 * 4)
        # 4 → 8 → 16 → 32 → 64
        self.up = nn.Sequential(
            _up(base, base),
            _up(base, base),
            _up(base, base // 2),
            _up(base // 2, base // 4),
        )
        self.head_img = nn.Conv2d(base // 4, 1, 3, 1, 1)
        self.head_theta = nn.Linear(d_in, 3)
        nn.init.normal_(self.head_theta.weight, std=0.01)
        nn.init.zeros_(self.head_theta.bias)
        # Per-slot placement is computed K-adaptively in forward (2D grid).

    def forward(self, eps: torch.Tensor, labels: torch.Tensor):
        B, K = labels.shape
        eps_rep = eps.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        cls = self.cls_emb(labels.reshape(-1))
        h = torch.cat([eps_rep, cls], dim=1)                  # [B*K, d_in]
        # letter image
        x = self.fc(h).view(B * K, self.base, 4, 4)
        x = self.up(x)
        img = torch.sigmoid(self.head_img(x))                 # [B*K, 1, 64, 64]
        img = img.view(B, K, 1, self.letter_size, self.letter_size)
        # theta
        raw = self.head_theta(h).view(B, K, 3)
        r = 0.20 + 0.30 * torch.sigmoid(raw[..., 0])          # letter visual size in [0.20, 0.50]
        tx = 0.45 * torch.tanh(raw[..., 1])
        ty = 0.45 * torch.tanh(raw[..., 2])
        # K-adaptive 2D grid placement so slots spread over the canvas
        import math
        cols = int(math.ceil(math.sqrt(K)))
        rows = int(math.ceil(K / cols))
        device = eps.device
        bx = torch.tensor([-0.4 + 0.8 * ((k % cols) + 0.5) / cols for k in range(K)], device=device)
        by = torch.tensor([-0.4 + 0.8 * ((k // cols) + 0.5) / rows for k in range(K)], device=device)
        tx = tx + bx.view(1, K)
        ty = ty + by.view(1, K)
        tx = torch.clamp(tx, -0.7, 0.7)
        ty = torch.clamp(ty, -0.7, 0.7)
        theta = torch.stack([r, tx, ty], dim=-1)              # [B, K, 3]
        return img, theta


# ─── Train ────────────────────────────────────────────────────────────────────

def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # frozen AE for face drift / canvas encoding
    ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=True)
    ae = FaceAE128(ch=ae_ck['lat_ch'], base=args.ae_base).to(device)
    ae.load_state_dict(ae_ck['ae']); ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)

    # letter bank: uint8 [26, N_per, 64, 64]
    letter_bank = torch.load(args.letter_bank, weights_only=True)
    n_classes, N_per, ls, _ = letter_bank.shape
    print(f"[letters] letter_bank {tuple(letter_bank.shape)}", flush=True)

    # real face latents
    face_lat = torch.load(args.face_lat, weights_only=True)            # [N, ch, 16, 16]
    N_face = face_lat.shape[0]
    S = face_lat.shape[1] * face_lat.shape[2] * face_lat.shape[3]
    print(f"[letters] face_lat {tuple(face_lat.shape)}", flush=True)

    G = LetterImgGen(d_noise=args.d_noise, n_classes=n_classes, letter_size=ls).to(device)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[letters] G params: {n_p/1e6:.2f}M  K={args.K}  Bgen={args.bgen}  device={device}",
          flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))
    R_list = tuple(float(x) for x in args.R.split(','))
    fixed_eps = torch.randn(8, args.d_noise, device=device)
    fixed_labels = torch.randint(0, n_classes, (8, args.K), device=device)
    t0 = time.time()

    G.train()
    for step in range(1, args.steps + 1):
        # random K-letter strings
        labels = torch.randint(0, n_classes, (args.bgen, args.K), device=device)
        eps = torch.randn(args.bgen, args.d_noise, device=device)
        letter_imgs, theta = G(eps, labels)                            # [B,K,1,64,64], [B,K,3]

        # letter target: random aug sample per (b,k) from the labeled class
        tgt_idx = torch.randint(0, N_per, (args.bgen, args.K))
        target_imgs = (letter_bank[labels.cpu(), tgt_idx].float() / 255.0).to(device)
        target_imgs = target_imgs.unsqueeze(2)                          # [B,K,1,64,64]
        L_l1 = F.l1_loss(letter_imgs, target_imgs)
        L_l2 = F.mse_loss(letter_imgs, target_imgs)
        L_letter = L_l1 + L_l2

        # STN-place each letter onto 128 canvas, max-compose
        flat_imgs = letter_imgs.view(args.bgen * args.K, 1, ls, ls)
        r = theta[..., 0].reshape(-1)
        tx = theta[..., 1].reshape(-1)
        ty = theta[..., 2].reshape(-1)
        inv_scale = 1.0 / r
        placed = stn_place(flat_imgs, inv_scale, tx, ty, 128)           # [B*K, 1, 128, 128]
        placed = placed.view(args.bgen, args.K, 1, 128, 128)
        canvas = placed.max(dim=1).values                                # [B, 1, 128, 128]

        # face drift on AE-encoded canvas (per-particle)
        comp_lat = ae.enc(canvas)
        comp_feat = comp_lat.flatten(1)                                  # [B, S]
        gen_arg = comp_feat.unsqueeze(1)                                 # [B, 1, S]
        pos_idx = torch.randint(0, N_face, (args.bgen, args.cp))
        pos_feat = face_lat[pos_idx].flatten(2).to(device)               # [B, Cp, S]
        neg_idx = torch.randint(0, args.bgen, (args.bgen, args.cn))
        neg_feat = comp_feat.detach()[neg_idx]                            # [B, Cn, S]
        loss_face_vec, info = drift_loss(gen_arg, pos_feat, fixed_neg=neg_feat, R_list=R_list)
        L_face = loss_face_vec.mean()

        loss = args.w_letter * L_letter + args.w_face * L_face

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                ink_canvas = canvas.mean().item()
                r_mean = theta[..., 0].mean().item()
            print(f"step={step} tot={loss.item():.4f} Llet={L_letter.item():.4f} "
                  f"Lface={L_face.item():.3f} ink={ink_canvas:.3f} r={r_mean:.2f} "
                  f"grad={gnorm.item():.2f} t={time.time()-t0:.0f}s", flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            _save_grid(G, ae, fixed_eps, fixed_labels, letter_bank, face_lat,
                       out / f'letters_step{step:05d}.png', device)

    torch.save({'G': G.state_dict(), 'args': vars(args)}, out / 'G_final.pt')
    print(f"[letters] done ({time.time()-t0:.0f}s)  → {out/'G_final.pt'}", flush=True)


@torch.no_grad()
def _save_grid(G, ae, fixed_eps, fixed_labels, letter_bank, face_lat, path, device):
    """Top: per-letter colored composite (K hues).  Middle: canvas grayscale.
       Bottom: real face decoded for reference."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import numpy as np
    G.eval()
    letter_imgs, theta = G(fixed_eps, fixed_labels)                   # [N,K,1,64,64]
    Nf, K = fixed_labels.shape
    ls = letter_imgs.shape[-1]
    flat_imgs = letter_imgs.view(Nf * K, 1, ls, ls)
    inv_scale = (1.0 / theta[..., 0]).view(-1)
    tx = theta[..., 1].view(-1); ty = theta[..., 2].view(-1)
    placed = stn_place(flat_imgs, inv_scale, tx, ty, 128).view(Nf, K, 128, 128).cpu().numpy()
    canvas = placed.max(axis=1)
    G.train()
    # color palette
    cmap = plt.get_cmap('hsv', max(K, 2))
    colors = np.stack([cmap(k)[:3] for k in range(K)], axis=0)
    # real reference
    ridx = torch.randint(0, face_lat.shape[0], (Nf,))
    real_dec = torch.sigmoid(ae.dec(face_lat[ridx].to(device))).cpu().numpy()[:, 0]

    fig, axes = plt.subplots(3, Nf, figsize=(Nf * 1.8, 6), facecolor='#0d0f14')
    letter_labels_np = fixed_labels.cpu().numpy()
    for i in range(Nf):
        rgb = np.zeros((128, 128, 3), dtype=np.float32)
        for k in range(K):
            rgb += placed[i, k][..., None] * colors[k][None, None, :]
        rgb = np.clip(rgb, 0, 1)
        axes[0, i].imshow(rgb); axes[0, i].axis('off')
        # show letter labels above
        chars = ''.join(chr(65 + int(c)) for c in letter_labels_np[i])
        axes[0, i].set_title(chars, color='white', fontsize=7, pad=2)
        axes[1, i].imshow(canvas[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real_dec[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    axes[0, 0].set_ylabel('per-letter color', color='w', fontsize=8)
    axes[1, 0].set_ylabel('canvas', color='w', fontsize=8)
    axes[2, 0].set_ylabel('real face', color='w', fontsize=8)
    fig.suptitle(f'Letters phase — {path.stem}', color='white', fontsize=11)
    plt.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.01, hspace=0.08, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[letters] grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--letter-bank', default='./checkpoints/letter_bank.pt', dest='letter_bank')
    ap.add_argument('--face-lat', default='./checkpoints/face_lat.pt', dest='face_lat')
    ap.add_argument('--ae-ckpt', default='./checkpoints/face_ae128.pt', dest='ae_ckpt')
    ap.add_argument('--ae-base', type=int, default=48, dest='ae_base')
    ap.add_argument('--out', default='./samples/face_letters')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--bgen', type=int, default=128)
    ap.add_argument('--cp', type=int, default=64)
    ap.add_argument('--cn', type=int, default=8)
    ap.add_argument('--w-letter', type=float, default=1.0, dest='w_letter')
    ap.add_argument('--w-face', type=float, default=0.3, dest='w_face')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--R', default='0.02,0.05,0.2')
    ap.add_argument('--log-every', type=int, default=50, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=400, dest='sample_every')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
