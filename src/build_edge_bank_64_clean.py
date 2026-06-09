"""Build a clean 64x64 edge bank with plain Canny (no var-width, no dilate)."""
import argparse, time
from pathlib import Path
import cv2, numpy as np, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--celeba-dir', default='./data/celeba/img_align_celeba')
    ap.add_argument('--out', default='./checkpoints/edge_bank_64_clean.pt')
    ap.add_argument('--size', type=int, default=64,
                    help='output resolution (64 or 128)')
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--blur', type=float, default=1.4)
    ap.add_argument('--lo', type=int, default=50)
    ap.add_argument('--hi', type=int, default=130)
    ap.add_argument('--dilate', type=int, default=0,
                    help='post-canny morphological dilation (px radius). 0 disables.')
    ap.add_argument('--min-area', type=int, default=0,
                    help='drop connected components smaller than this many pixels '
                         '(despeckle). 0 disables.')
    args = ap.parse_args()

    S = args.size
    paths = sorted(Path(args.celeba_dir).glob('*.jpg'))[:args.n]
    print(f"[clean] processing {len(paths)} faces", flush=True)
    bank = torch.zeros(len(paths), S, S, dtype=torch.uint8)
    t0 = time.time()
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        bgr = cv2.resize(bgr, (S, S), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (0, 0), args.blur)
        edges = cv2.Canny(gray, args.lo, args.hi)
        # despeckle: drop tiny connected components (噪声短线/孤立点)
        if args.min_area > 0:
            n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(edges, connectivity=8)
            keep_mask = np.zeros_like(edges)
            for cc in range(1, n_cc):  # 0 is background
                if stats[cc, cv2.CC_STAT_AREA] >= args.min_area:
                    keep_mask[labels == cc] = 255
            edges = keep_mask
        if args.dilate > 0:
            ker = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * args.dilate + 1, 2 * args.dilate + 1))
            edges = cv2.dilate(edges, ker)
        bank[i] = torch.from_numpy(edges)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{len(paths)}  {time.time()-t0:.0f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, args.out)
    print(f"[clean] saved {args.out}  shape={tuple(bank.shape)}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
