"""Letters phase, pixel-space — K letters → STN-place → max-compose → face drift in pixels.

Combines:
  - K-letter LetterImgGen (eps + labels → K letter images + STN theta per letter)
  - the full winning pixel-drift recipe from face_drift_pixel.py
    (smallR + curriculum + refine head + sharpness + TV + noise-aug + cov)
  - per-letter L1+L2 against random aug samples from letter_bank.pt

Loss = w_face * drift(canvas, face_bank) + w_letter * L1L2(letter, letter_bank)
       + cov * (canvas_ink - real_ink)^2 + sharpness * (1 - 4(canvas - 0.5)^2).mean()
       + tv * |∇canvas|
"""
from __future__ import annotations
import argparse, copy, sys, time, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drift_loss import drift_loss
from face_ae128 import _up


def stn_place(letter_imgs: torch.Tensor, inv_scale: torch.Tensor,
              tx: torch.Tensor, ty: torch.Tensor, canvas: int) -> torch.Tensor:
    """letter_imgs[N,1,h,w] → placed[N,1,canvas,canvas]."""
    N = letter_imgs.shape[0]
    M = torch.zeros(N, 2, 3, device=letter_imgs.device, dtype=letter_imgs.dtype)
    M[:, 0, 0] = inv_scale
    M[:, 0, 2] = tx
    M[:, 1, 1] = inv_scale
    M[:, 1, 2] = ty
    grid = F.affine_grid(M, size=(N, 1, canvas, canvas), align_corners=False)
    return F.grid_sample(letter_imgs, grid, mode='bilinear',
                         padding_mode='zeros', align_corners=False)


class LetterImgGen(nn.Module):
    """eps[B,d] + labels[B,K] → letters[B,K,1,64,64] + theta[B,K,3] (r, tx, ty).
    head_refine adds extra full-res convs to the letter head (helps crispness)."""
    def __init__(self, d_noise: int = 128, n_classes: int = 26,
                 letter_size: int = 64, base: int = 128, head_refine: int = 0,
                 sigmoid_t: float = 1.0):
        super().__init__()
        self.letter_size = letter_size
        self.base = base
        self.sigmoid_t = sigmoid_t
        self.cls_emb = nn.Embedding(n_classes, 64)
        d_in = d_noise + 64
        self.fc = nn.Linear(d_in, base * 4 * 4)
        # 4 → 8 → 16 → 32 → 64
        self.up = nn.Sequential(
            _up(base, base), _up(base, base),
            _up(base, base // 2), _up(base // 2, base // 4),
        )
        head_in = base // 4
        if head_refine > 0:
            layers = []
            for _ in range(head_refine):
                layers += [nn.Conv2d(head_in, head_in, 3, 1, 1), nn.GELU()]
            layers += [nn.Conv2d(head_in, 1, 1, 1, 0)]
            self.head_img = nn.Sequential(*layers)
        else:
            self.head_img = nn.Conv2d(head_in, 1, 3, 1, 1)
        self.head_theta = nn.Linear(d_in, 3)
        nn.init.normal_(self.head_theta.weight, std=0.01)
        nn.init.zeros_(self.head_theta.bias)

    def forward(self, eps: torch.Tensor, labels: torch.Tensor):
        B, K = labels.shape
        eps_rep = eps.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)
        cls = self.cls_emb(labels.reshape(-1))
        h = torch.cat([eps_rep, cls], dim=1)                  # [B*K, d_in]
        # letter image
        x = self.fc(h).view(B * K, self.base, 4, 4)
        x = self.up(x)
        img = torch.sigmoid(self.head_img(x) * self.sigmoid_t)
        img = img.view(B, K, 1, self.letter_size, self.letter_size)
        # theta
        raw = self.head_theta(h).view(B, K, 3)
        r = 0.15 + 0.25 * torch.sigmoid(raw[..., 0])          # letter ratio in [0.15, 0.40]
        tx = 0.45 * torch.tanh(raw[..., 1])
        ty = 0.45 * torch.tanh(raw[..., 2])
        # K-adaptive 2D grid placement so slots spread over the canvas
        cols = int(math.ceil(math.sqrt(K)))
        rows = int(math.ceil(K / cols))
        device = eps.device
        bx = torch.tensor([-0.5 + 1.0 * ((k % cols) + 0.5) / cols for k in range(K)], device=device)
        by = torch.tensor([-0.5 + 1.0 * ((k // cols) + 0.5) / rows for k in range(K)], device=device)
        tx = tx + bx.view(1, K)
        ty = ty + by.view(1, K)
        tx = torch.clamp(tx, -0.7, 0.7)
        ty = torch.clamp(ty, -0.7, 0.7)
        theta = torch.stack([r, tx, ty], dim=-1)
        return img, theta


def update_ema(ema_module, online_module, decay: float):
    with torch.no_grad():
        for ep, p in zip(ema_module.parameters(), online_module.parameters()):
            ep.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


def train(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # face edge bank — pixel space target
    bank = torch.load(args.bank, weights_only=True)            # uint8 [N, 128, 128]
    N_face = bank.shape[0]; S_bank = bank.shape[-1]
    assert S_bank == 128, f'letters phase expects 128 bank, got {S_bank}'
    bank_gpu = (bank.float() / 255.0).to(device)
    real_ink = bank_gpu.mean().item()
    print(f"[letters-pix] face bank {tuple(bank.shape)}  real_ink={real_ink:.4f}", flush=True)
    if args.bin_pos > 0:
        bank_gpu = (bank_gpu > args.bin_pos).float()
        real_ink = bank_gpu.mean().item()
        print(f"[letters-pix] binarized → real_ink={real_ink:.4f}", flush=True)

    # letter bank
    letter_bank = torch.load(args.letter_bank, weights_only=True)
    n_classes, N_per, ls, _ = letter_bank.shape
    print(f"[letters-pix] letter_bank {tuple(letter_bank.shape)}", flush=True)

    G = LetterImgGen(d_noise=args.d_noise, n_classes=n_classes, letter_size=ls,
                     base=args.base, head_refine=args.head_refine,
                     sigmoid_t=args.sigmoid_t).to(device)
    G_ema = copy.deepcopy(G).to(device)
    for p in G_ema.parameters(): p.requires_grad_(False)
    n_p = sum(p.numel() for p in G.parameters())
    print(f"[letters-pix] G params: {n_p/1e6:.2f}M  K={args.K}  Bgen={args.bgen}", flush=True)
    opt = torch.optim.AdamW(G.parameters(), lr=args.lr, betas=(0.9, 0.95))
    R_list = tuple(float(x) for x in args.R.split(','))
    fixed_eps = torch.randn(8, args.d_noise, device=device)
    fixed_labels = torch.randint(0, n_classes, (8, args.K), device=device)
    t0 = time.time()

    G.train()
    for step in range(1, args.steps + 1):
        prog = step / max(1, args.steps)
        # curriculum on sigmoid_t and sharpness
        if args.sigmoid_t_end is not None:
            cur_sig_t = args.sigmoid_t + (args.sigmoid_t_end - args.sigmoid_t) * prog
            G.sigmoid_t = cur_sig_t
            G_ema.sigmoid_t = cur_sig_t
        sharp_w = args.sharpness if args.sharpness_end is None else \
                  args.sharpness + (args.sharpness_end - args.sharpness) * prog

        labels = torch.randint(0, n_classes, (args.bgen, args.K), device=device)
        eps = torch.randn(args.bgen, args.d_noise, device=device)
        letter_imgs, theta = G(eps, labels)                            # [B,K,1,64,64], [B,K,3]

        # per-letter target consistency (L1 + L2 against random aug samples)
        if args.w_letter > 0:
            tgt_idx = torch.randint(0, N_per, (args.bgen, args.K))
            target_imgs = (letter_bank[labels.cpu(), tgt_idx].float() / 255.0).to(device)
            target_imgs = target_imgs.unsqueeze(2)                      # [B,K,1,64,64]
            L_l1 = F.l1_loss(letter_imgs, target_imgs)
            L_l2 = F.mse_loss(letter_imgs, target_imgs)
            L_letter = L_l1 + L_l2
        else:
            L_letter = torch.tensor(0.0, device=device)

        # STN place each letter, max-compose to 128 canvas
        flat_imgs = letter_imgs.view(args.bgen * args.K, 1, ls, ls)
        r = theta[..., 0].reshape(-1)
        tx = theta[..., 1].reshape(-1)
        ty = theta[..., 2].reshape(-1)
        inv_scale = 1.0 / r
        placed = stn_place(flat_imgs, inv_scale, tx, ty, 128).view(args.bgen, args.K, 1, 128, 128)
        canvas = placed.max(dim=1).values                               # [B, 1, 128, 128]

        # pixel-space drift on canvas
        canvas_feat = canvas.flatten(1)                                  # [B, 128*128]
        Cg = max(1, args.gen_per)
        B = args.bgen // Cg
        gen_arg = canvas_feat[: B * Cg].view(B, Cg, -1)                  # [B, Cg, S]
        pos_idx = torch.randint(0, N_face, (B, args.cp))
        pos_feat = bank_gpu[pos_idx].flatten(2)                          # [B, Cp, S]
        neg_idx = torch.randint(0, N_face, (B, args.cn))
        neg_feat = bank_gpu[neg_idx].flatten(2)

        if args.noise_aug > 0:
            t = args.noise_aug
            gen_arg = (1 - t) * gen_arg + t * torch.randn_like(gen_arg)
            pos_feat = (1 - t) * pos_feat + t * torch.randn_like(pos_feat)
            neg_feat = (1 - t) * neg_feat + t * torch.randn_like(neg_feat)

        loss_face_vec, info = drift_loss(gen_arg, pos_feat, fixed_neg=neg_feat,
                                         R_list=R_list,
                                         topk_pos=args.topk_pos, topk_neg=args.topk_neg)
        L_face = loss_face_vec.mean()

        loss = args.w_face * L_face + args.w_letter * L_letter

        # cov / sharpness / tv on the composite canvas
        if args.cov > 0:
            cov_term = (canvas.mean() - real_ink).pow(2)
            loss = loss + args.cov * cov_term
        if sharp_w > 0:
            sharp_term = (1.0 - 4.0 * (canvas - 0.5).pow(2)).mean()
            loss = loss + sharp_w * sharp_term
        if args.tv > 0:
            dx = (canvas[:, :, :, 1:] - canvas[:, :, :, :-1]).abs().mean()
            dy = (canvas[:, :, 1:, :] - canvas[:, :, :-1, :]).abs().mean()
            loss = loss + args.tv * (dx + dy)

        opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        if args.ema > 0:
            update_ema(G_ema, G, args.ema)

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                bvar = canvas.var(dim=0).mean().item()
                ink = canvas.mean().item()
                r_mean = theta[..., 0].mean().item()
            print(f"step={step} tot={loss.item():.3f} Lface={L_face.item():.3f} "
                  f"Llet={L_letter.item():.4f} ink={ink:.3f} bvar={bvar:.4f} "
                  f"r={r_mean:.2f} grad={gnorm.item():.2f} t={time.time()-t0:.0f}s",
                  flush=True)

        if step % args.sample_every == 0 or step == args.steps:
            G_for_vis = G_ema if args.ema > 0 else G
            _save_grid(G_for_vis, fixed_eps, fixed_labels, bank_gpu,
                       out / f'letters_step{step:05d}.png', device, args.K)

    torch.save({'G': G.state_dict(), 'G_ema': G_ema.state_dict(),
                'args': vars(args)}, out / 'G_final.pt')
    print(f"[letters-pix] done ({time.time()-t0:.0f}s)", flush=True)


@torch.no_grad()
def _save_grid(G, fixed_eps, fixed_labels, bank, path, device, K):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import numpy as np
    G.eval()
    letter_imgs, theta = G(fixed_eps, fixed_labels)              # [Nf,K,1,64,64], [Nf,K,3]
    Nf, K = fixed_labels.shape
    ls = letter_imgs.shape[-1]
    flat = letter_imgs.view(Nf * K, 1, ls, ls)
    inv_s = (1.0 / theta[..., 0]).view(-1)
    tx = theta[..., 1].view(-1); ty = theta[..., 2].view(-1)
    placed = stn_place(flat, inv_s, tx, ty, 128).view(Nf, K, 128, 128).cpu().numpy()
    canvas = placed.max(axis=1)
    G.train()
    cmap = plt.get_cmap('hsv', max(K, 2))
    colors = np.stack([cmap(k)[:3] for k in range(K)], axis=0)
    ridx = torch.randint(0, bank.shape[0], (Nf,))
    real = bank[ridx].cpu().numpy()
    labels_np = fixed_labels.cpu().numpy()
    fig, axes = plt.subplots(3, Nf, figsize=(Nf * 1.8, 6), facecolor='#0d0f14')
    for i in range(Nf):
        rgb = np.zeros((128, 128, 3), dtype=np.float32)
        for k in range(K):
            rgb += placed[i, k][..., None] * colors[k][None, None, :]
        rgb = np.clip(rgb, 0, 1)
        axes[0, i].imshow(rgb); axes[0, i].axis('off')
        chars = ''.join(chr(65 + int(c)) for c in labels_np[i])
        axes[0, i].set_title(chars, color='white', fontsize=8, pad=2)
        axes[1, i].imshow(canvas[i], cmap='gray', vmin=0, vmax=1); axes[1, i].axis('off')
        axes[2, i].imshow(real[i], cmap='gray', vmin=0, vmax=1); axes[2, i].axis('off')
    fig.suptitle(f'Letters-pixel K={K} — {path.stem}', color='white', fontsize=11)
    plt.subplots_adjust(left=0.02, right=0.99, top=0.92, bottom=0.01, hspace=0.08, wspace=0.05)
    plt.savefig(path, dpi=100, facecolor='#0d0f14'); plt.close()
    print(f"[letters-pix] grid → {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank', required=True, help='face edge bank, 128x128')
    ap.add_argument('--letter-bank', default='./checkpoints/letter_bank.pt', dest='letter_bank')
    ap.add_argument('--out', default='./samples/letters_pixel')
    ap.add_argument('--K', type=int, default=12)
    ap.add_argument('--d-noise', type=int, default=128, dest='d_noise')
    ap.add_argument('--bgen', type=int, default=256)
    ap.add_argument('--base', type=int, default=128)
    ap.add_argument('--cp', type=int, default=64)
    ap.add_argument('--cn', type=int, default=16)
    ap.add_argument('--gen-per', type=int, default=32, dest='gen_per')
    ap.add_argument('--w-letter', type=float, default=1.0, dest='w_letter')
    ap.add_argument('--w-face', type=float, default=1.0, dest='w_face')
    ap.add_argument('--steps', type=int, default=24000)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--R', default='0.005,0.02,0.1')
    ap.add_argument('--ema', type=float, default=0.999)
    ap.add_argument('--cov', type=float, default=5.0)
    ap.add_argument('--tv', type=float, default=0.03)
    ap.add_argument('--sharpness', type=float, default=0.5)
    ap.add_argument('--sharpness-end', type=float, default=4.0, dest='sharpness_end')
    ap.add_argument('--sigmoid-t', type=float, default=1.0, dest='sigmoid_t')
    ap.add_argument('--sigmoid-t-end', type=float, default=3.0, dest='sigmoid_t_end')
    ap.add_argument('--noise-aug', type=float, default=0.10, dest='noise_aug')
    ap.add_argument('--bin-pos', type=float, default=0.0, dest='bin_pos')
    ap.add_argument('--topk-pos', type=int, default=0, dest='topk_pos')
    ap.add_argument('--topk-neg', type=int, default=0, dest='topk_neg')
    ap.add_argument('--head-refine', type=int, default=2, dest='head_refine')
    ap.add_argument('--log-every', type=int, default=100, dest='log_every')
    ap.add_argument('--sample-every', type=int, default=2000, dest='sample_every')
    args = ap.parse_args()
    train(args)


if __name__ == '__main__':
    main()
