# W2F Backup & Version Control

Set up 2026-06-03 after losing the old `face_ae128.pt` / `face_ae128_big.pt`
AEs to an over-aggressive remote cleanup (made the old `face_drift_*` fallback
faces unreproducible). **Never delete a checkpoint without backing it up first.**

## What lives where

| Asset | Tracked by | Location |
|---|---|---|
| **Code** (`src/`, `.local/`, `scripts/`, docs) | **git** | local repo at project root (`.git`) |
| Critical model weights (`*.pt`) | manual backup | local `w2f/checkpoints/` + `w2f/checkpoints/backup_runs/` |
| Old fallback runs (sweep12/13 + code) | archived | `w2f/fallback_archive/` |
| Datasets, sample images, logs | not backed up | regenerable (see below) |

`*.pt` is **gitignored** — git is for code, not GB-scale binaries.

## Backed-up checkpoints (local `w2f/checkpoints/`)

- `face_ae128_metric.pt` + `face_lat_metric.pt` — frozen edge-AE used as the
  **face-KID metric feature space**. Must never be lost or all KID numbers
  become incomparable across runs.
- `letter_cnn_32_mild.pt` — frozen LetterCNN for the letter-legibility metric.
- `edge_bank_128_dil_lo75.pt` — active CelebA-edge training bank (D2 target).
- `backup_runs/sweep38_R32_*_G_ckpt.pt` — best run so far (KID 532, letter 0.364).

## Regenerable (NOT backed up — how to rebuild)

- **Aug letter banks** `aug_letter_bank_{P}_{level}_{N}.pt`: rebuilt automatically
  by the trainer from EMNIST on first use (`build_aug_bank`). ~5 GB each.
- **Edge banks** from CelebA: `src/celeba_edges.py` (see `[[celeba-download]]` —
  use HF mirror, gdrive is quota-blocked).
- **Metric/AE models**: retrain via `src/face_ae128.py` and
  `.local/train_letter_cnn.py` (minutes each). NOTE: a retrained AE has a NEW
  latent space, so it can't decode latents from an OLD direct-latent generator —
  that's why the deleted AEs made the old fallbacks unreproducible.

## Discipline (the rules I broke, now enforced)

1. **Commit code to git after every meaningful change.** `git add -A && git commit`.
2. **Before `rm` of any `*.pt`**, copy it to `checkpoints/backup_runs/` or download
   it local. The remote AutoDL box is ephemeral (already swapped once).
3. When training an AE that a generator/metric depends on, **keep that AE forever**.
