"""Remote-side: build the augmented letter bank used by the letters-phase trainer.

Loads EMNIST 'letters' (mixed upper+lower, 26 classes) at 64px.
Falls back to PIL-rendered uppercase letters if EMNIST can't be downloaded.
Runs the approved aggressive aug pipeline (see aug_letters.augment_letter).

Saves uint8 [26, samples_per_class, 64, 64] to ./checkpoints/letter_bank.pt.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import _try_load_emnist, _fallback_synth_letters
from aug_letters import augment_letter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--size', type=int, default=64)
    ap.add_argument('--per-class', type=int, default=500, dest='per_class')
    ap.add_argument('--level', default='aggressive')
    ap.add_argument('--out', default='./checkpoints/letter_bank.pt')
    ap.add_argument('--data-root', default='./data', dest='data_root')
    ap.add_argument('--no-emnist', action='store_true', dest='no_emnist',
                    help='skip EMNIST attempt, go straight to PIL fallback')
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[letters] size={args.size}  per-class={args.per_class}  level={args.level}", flush=True)

    if args.no_emnist:
        print("[letters] --no-emnist: using PIL fallback (uppercase only)", flush=True)
        raw = _fallback_synth_letters(size=args.size, per_class=200)
    else:
        raw = _try_load_emnist(args.data_root, size=args.size)
        if raw is None:
            print(f"[letters] EMNIST unavailable → PIL fallback (uppercase only)", flush=True)
            raw = _fallback_synth_letters(size=args.size, per_class=200)
    print(f"[letters] raw shape {tuple(raw.shape)}", flush=True)

    bank = torch.zeros(26, args.per_class, args.size, args.size, dtype=torch.uint8)
    rng = np.random.default_rng(0)
    t0 = time.time()
    for c in range(26):
        src = raw[c].numpy()
        for j in range(args.per_class):
            base = src[j % len(src)]
            aug = augment_letter(base, level=args.level, rng=rng)        # [H,W] float [0,1]
            bank[c, j] = torch.from_numpy((aug * 255).clip(0, 255).astype(np.uint8))
        print(f"  class {c+1}/26 (chr {chr(65+c)})  t={time.time()-t0:.0f}s", flush=True)
    torch.save(bank, str(out))
    print(f"[letters] saved {out}  shape={tuple(bank.shape)}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == '__main__':
    main()
