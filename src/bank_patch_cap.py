"""Per-patch edge density cap.

Divides each bank image into n_p × n_p patches of size patch_size × patch_size.
For each patch with edge_pixel_count > cap, randomly drops edges down to cap.
Goal: smooth dense regions (hair) without touching sparse regions (face features).
"""
import sys, argparse, time
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--out', dest='out_path', required=True)
    ap.add_argument('--patch-size', type=int, default=16, dest='patch_size')
    ap.add_argument('--cap', type=int, required=True,
                    help='max edge pixels per patch (patches with more get random-pruned)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    bank = torch.load(args.in_path, weights_only=True).numpy()
    N, S, _ = bank.shape
    p = args.patch_size
    assert S % p == 0, f'image size {S} not divisible by patch_size {p}'
    n_p = S // p
    print(f'[cap] bank shape: {bank.shape}  patch_size={p}  n_patches/img={n_p**2}  cap={args.cap}',
          flush=True)
    print(f'[cap] orig ink: {bank.mean()/255:.4f}', flush=True)

    out = bank.copy()
    t0 = time.time()
    capped_patches = 0
    for i in range(N):
        img = out[i]
        for py in range(n_p):
            for px in range(n_p):
                ys, xs = py * p, px * p
                patch = img[ys:ys+p, xs:xs+p]
                on = patch > 0
                cnt = int(on.sum())
                if cnt > args.cap:
                    capped_patches += 1
                    on_idx = np.argwhere(on)  # [cnt, 2]
                    drop_n = cnt - args.cap
                    drop_sel = np.random.choice(cnt, drop_n, replace=False)
                    drop_yx = on_idx[drop_sel]
                    patch[drop_yx[:, 0], drop_yx[:, 1]] = 0
        if (i + 1) % 4000 == 0:
            print(f'  {i+1}/{N}  {time.time()-t0:.0f}s  capped {capped_patches} patches', flush=True)

    out_t = torch.from_numpy(out)
    torch.save(out_t, args.out_path)
    print(f'[cap] saved {args.out_path}  ink={out.mean()/255:.4f}  '
          f'capped {capped_patches}/{N*n_p*n_p} patches  ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
