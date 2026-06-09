"""Conditional letter generator: K letters composed to match a target face latent.

Training:
  face_target = bank[random]   (or flow_G(eps) for purely-generative target)
  eps = randn; labels = random K letters
  letters, theta = G(eps, labels, face_target)   # letter G conditioned on face
  canvas = max(stn_place(letters, theta))
  composite_lat = AE.encode(canvas)
  L_face = MSE(composite_lat, face_target.detach())
  L_letter = L1+L2(letters, aug_target_letters)
  loss = w_letter * L_letter + w_face * L_face

Inference:
  pick face_target (from real bank or flow-sample) → one-shot letter G → letters.
  No drift involved. Letter G learns to compose into ANY given face.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from face_ae128 import FaceAE128, _up


def stn_place(letter_imgs, inv_scale, tx, ty, canvas):
    N = letter_imgs.shape[0]
    M = torch.zeros(N, 2, 3, device=letter_imgs.device, dtype=letter_imgs.dtype)
    M[:, 0, 0] = inv_scale; M[:, 1, 1] = inv_scale
    M[:, 0, 2] = tx; M[:, 1, 2] = ty
    grid = F.affine_grid(M, size=(N, 1, canvas, canvas), align_corners=False)
    return F.grid_sample(letter_imgs, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


class LetterCondGen(nn.Module):
    """eps + labels + face_target_latent → K letters + STN params."""
    def __init__(self, d_noise=128, n_classes=26, letter_size=64, lat_ch=16, base=192):
        super().__init__()
        self.letter_size = letter_size; self.base = base
        self.cls_emb = nn.Embedding(n_classes, 64)
        # face target latent: flatten + project
        self.face_in = nn.Linear(lat_ch * 16 * 16, 128)
        d_in = d_noise + 64 + 128
        self.fc = nn.Linear(d_in, base * 4 * 4)
        self.up = nn.Sequential(_up(base, base), _up(base, base),
                                _up(base, base // 2), _up(base // 2, base // 4))
        self.head_img = nn.Conv2d(base // 4, 1, 3, 1, 1)
        self.head_theta = nn.Linear(d_in, 3)
        nn.init.normal_(self.head_theta.weight, std=0.01)
        nn.init.zeros_(self.head_theta.bias)
        # placement biases are computed K-adaptive in forward (was a bug to fix K=10 to the
        # left half of a MAX_K=24 linspace).

    def forward(self, eps, labels, face_target):
        B, K = labels.shape
        eps_rep = eps.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        cls = self.cls_emb(labels.reshape(-1))
        face_feat = self.face_in(face_target.flatten(1))                 # [B, 128]
        face_rep = face_feat.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        h = torch.cat([eps_rep, cls, face_rep], dim=1)
        x = self.fc(h).view(B * K, self.base, 4, 4)
        x = self.up(x)
        img = torch.sigmoid(self.head_img(x)).view(B, K, 1, self.letter_size, self.letter_size)
        raw = self.head_theta(h).view(B, K, 3)
        r = 0.30 + 0.40 * torch.sigmoid(raw[..., 0])      # letter size in [0.30, 0.70]
        # FIXED 2D grid placement (model can only perturb by ±0.1, not undo)
        import math
        cols = int(math.ceil(math.sqrt(K)))
        rows = int(math.ceil(K / cols))
        device = eps.device
        bx = torch.tensor([-0.35 + 0.7 * ((k % cols) + 0.5) / cols for k in range(K)], device=device)
        by = torch.tensor([-0.35 + 0.7 * ((k // cols) + 0.5) / rows for k in range(K)], device=device)
        tx = bx.view(1, K) + 0.10 * torch.tanh(raw[..., 1])
        ty = by.view(1, K) + 0.10 * torch.tanh(raw[..., 2])
        tx = torch.clamp(tx, -0.6, 0.6); ty = torch.clamp(ty, -0.6, 0.6)
        return img, torch.stack([r, tx, ty], dim=-1)


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ae_ck = torch.load(args.ae_ckpt, map_location=device, weights_only=True)
    ae = FaceAE128(ch=ae_ck['lat_ch'], base=ae_ck.get('base', 48),
                   vae=ae_ck.get('vae', False)).to(device)
    ae.load_state_dict(ae_ck['ae']); ae.eval()
    for p in ae.parameters(): p.requires_grad_(False)

    bank_face = torch.load(args.face_lat, weights_only=True).to(device)    # [N,ch,16,16]
    bank_letter = torch.load(args.letter_bank, weights_only=True)          # [26,N_per,64,64] uint8
    N_face = bank_face.shape[0]
    n_cls, N_per = bank_letter.shape[:2]

    G = LetterCondGen(d_noise=args.d_noise, n_classes=n_cls,
                      letter_size=bank_letter.shape[-1], lat_ch=ae_ck['lat_ch']).to(device)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[cond] G params: {n_p/1e6:.2f}M  K={args.K}  Bgen={args.bgen}", flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))

    fixed_eps = torch.randn(8, args.d_noise, device=device)
    fixed_labels = torch.randint(0, n_cls, (8, args.K), device=device)
    fixed_target_idx = torch.randint(0, N_face, (8,))
    fixed_target = bank_face[fixed_target_idx].to(device)
    t0 = time.time()

    G.train()
    for step in range(1, args.steps + 1):
        idx_face = torch.randint(0, N_face, (args.bgen,))
        face_target = bank_face[idx_face].to(device)
        labels = torch.randint(0, n_cls, (args.bgen, args.K), device=device)
        eps = torch.randn(args.bgen, args.d_noise, device=device)
        letter_imgs, theta = G(eps, labels, face_target)

        # letter loss
        tgt_idx = torch.randint(0, N_per, (args.bgen, args.K))
        target_letters = (bank_letter[labels.cpu(), tgt_idx].float() / 255.0).unsqueeze(2).to(device)
        L_letter = F.l1_loss(letter_imgs, target_letters) + F.mse_loss(letter_imgs, target_letters)

        # compose → canvas → encode → MSE vs face_target
        ls = letter_imgs.shape[-1]
        flat = letter_imgs.view(args.bgen * args.K, 1, ls, ls)
        r = theta[..., 0].reshape(-1); tx = theta[..., 1].reshape(-1); ty = theta[..., 2].reshape(-1)
        placed = stn_place(flat, 1.0 / r, tx, ty, 128).view(args.bgen, args.K, 1, 128, 128)
        canvas = placed.max(dim=1).values
        comp_lat = ae.encode(canvas)
        L_face = F.mse_loss(comp_lat, face_target.detach())

        loss = args.w_letter * L_letter + args.w_face * L_face
        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            print(f"step={step} tot={loss.item():.4f} Llet={L_letter.item():.4f} "
                  f"Lface={L_face.item():.4f} grad={gnorm.item():.2f} t={time.time()-t0:.0f}s",
                  flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            _save_grid(G, ae, fixed_eps, fixed_labels, fixed_target,
                       out / f'cond_step{step:05d}.png', device)

    torch.save({'G': G.state_dict(), 'args': vars(args)}, out / 'G_final.pt')
    print(f"[cond] done ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def _save_grid(G, ae, fixed_eps, fixed_labels, fixed_target, path, device):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import numpy as np
    G.eval()
    letter_imgs, theta = G(fixed_eps, fixed_labels, fixed_target)
    N, K = fixed_labels.shape
    ls = letter_imgs.shape[-1]
    flat = letter_imgs.view(N * K, 1, ls, ls)
    r = theta[..., 0].reshape(-1); tx = theta[..., 1].reshape(-1); ty = theta[..., 2].reshape(-1)
    placed = stn_place(flat, 1.0 / r, tx, ty, 128).view(N, K, 128, 128).cpu().numpy()
    canvas = placed.max(axis=1)
    G.train()
    target_decoded = torch.sigmoid(ae.dec(fixed_target.to(device))).cpu().numpy()[:, 0]
    cmap = plt.get_cmap('hsv', max(K, 2))
    colors = np.stack([cmap(k)[:3] for k in range(K)], axis=0)

    fig, axes = plt.subplots(3, N, figsize=(N * 1.8, 6), facecolor='#0d0f14')
    labels_np = fixed_labels.cpu().numpy()
    for i in range(N):
        rgb = np.zeros((128, 128, 3), dtype=np.float32)
        for k in range(K):
            rgb += placed[i, k][..., None] * colors[k][None, None, :]
        rgb = np.clip(rgb, 0, 1)
        axes[0, i].imshow(rgb); axes[0, i].axis('off')
        axes[0, i].set_title(''.join(chr(65 + int(c)) for c in labels_np[i]), color='w', fontsize=7, pad=2)
        axes[1, i].imshow(canvas[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(target_decoded[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    axes[1, 0].set_ylabel('composite', color='w'); axes[2, 0].set_ylabel('target', color='w')
    fig.suptitle(f'Cond letters — {path.stem}', color='white', fontsize=11)
    plt.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.01, hspace=0.08, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[cond] grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ae-ckpt', default='./checkpoints/face_ae_v3.pt', dest='ae_ckpt')
    ap.add_argument('--face-lat', default='./checkpoints/face_lat_v3.pt', dest='face_lat')
    ap.add_argument('--letter-bank', default='./checkpoints/letter_bank.pt', dest='letter_bank')
    ap.add_argument('--out', default='./samples/cond_letters')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--bgen', type=int, default=128)
    ap.add_argument('--w-letter', type=float, default=10.0, dest='w_letter')
    ap.add_argument('--w-face', type=float, default=1.0, dest='w_face')
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--log-every', type=int, default=50, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=400, dest='sample_every')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
