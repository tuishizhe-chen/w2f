"""Remote-side data prep:
  1) stream N CelebA jpgs from HF mirror nielsr/CelebA-faces into ./data/celeba/
  2) run the edge20 var-width pipeline at 128 → ./checkpoints/edge_bank_128.pt (uint8 [N,128,128])

Designed to run on the AutoDL box (China-side internet → set HF_ENDPOINT=hf-mirror).
Idempotent: re-running skips already-saved jpgs and re-saves the bank.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch


def download_celeba(out_dir: Path, n: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob('*.jpg'))
    if len(existing) >= n:
        print(f"[prep] already have {len(existing)} jpgs, skipping download")
        return
    os.environ.setdefault('HF_HOME', '/root/.cache/hf')
    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    from datasets import load_dataset
    from PIL import Image
    print(f"[prep] streaming {n} CelebA from HF…", flush=True)
    ds = load_dataset('nielsr/CelebA-faces', split='train', streaming=True)
    saved = len(existing)
    for ex in ds:
        if saved >= n: break
        im = next((v for v in ex.values() if isinstance(v, Image.Image)), None)
        if im is None: continue
        im.convert('RGB').save(out_dir / f'{saved:06d}.jpg', quality=92)
        saved += 1
        if saved % 1000 == 0:
            print(f"  {saved}/{n}", flush=True)
    print(f"[prep] downloaded {saved} total", flush=True)


def build_bank(celeba_root: Path, out_path: Path, size: int, n: int):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from celeba_edges import CelebAEdgeDataset
    # CelebAEdgeDataset expects celeba_root/img_align_celeba/*.jpg
    # we saved to celeba_root/img_align_celeba/ directly
    ds = CelebAEdgeDataset(str(celeba_root), size=size, max_samples=n)
    print(f"[prep] building edge bank @ {size} from {len(ds.paths)} jpgs…", flush=True)
    bank = torch.zeros(len(ds.paths), size, size, dtype=torch.uint8)
    import time; t0 = time.time()
    for i, p in enumerate(ds.paths):
        e = ds._extract_edge(p)
        bank[i] = torch.from_numpy((e > 0.5).astype(np.uint8) * 255)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(ds.paths)}  {time.time()-t0:.0f}s", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, str(out_path))
    print(f"[prep] saved {out_path}  shape={tuple(bank.shape)}  ({time.time()-t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--celeba-dir', default='./data/celeba/img_align_celeba')
    ap.add_argument('--bank', default='./checkpoints/edge_bank_128.pt')
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--n', type=int, default=20000)
    ap.add_argument('--skip-download', action='store_true')
    args = ap.parse_args()
    cdir = Path(args.celeba_dir)
    if not args.skip_download:
        download_celeba(cdir, args.n)
    # CelebAEdgeDataset wants celeba_root/img_align_celeba; pass the parent
    build_bank(cdir.parent, Path(args.bank), args.size, args.n)


if __name__ == '__main__':
    main()
