# Recipes — the five report configurations

Each script is a thin, self-documenting wrapper that reproduces one row of the
paper. Run from anywhere (they `cd` to the repo root themselves):

```bash
GPU=0 bash recipes/recommended_iou_film.sh
```

| Recipe | Script | Model | Face dist. | FiLM | edge-KID ↓ |
|---|---|---|---|---|---|
| **Recommended** | `recommended_iou_film.sh` | transformer + STN | IoU + top-k 4/8 | on | **124 ± 5** |
| Ablation — Euclid + FiLM | `ablation_euclid_film.sh` | transformer + STN | L2 | on | 150 ± 18 |
| Ablation — Euclid, no FiLM | `ablation_euclid_nofilm.sh` | transformer + STN | L2 | off | 169 |
| Baseline — pure face, top-k | `baseline_pure_face_topk.sh` | conv generator | IoU + top-k 4/8 | — | **32.6** (ceiling) |
| Baseline — pure face, no prune | `baseline_pure_face_nopruning.sh` | conv generator | IoU | — | blurry mean face |

The three letter-constrained recipes call **`train_w2f.sh`**
(`src/face_drift_multi_transformer.py`); the two baselines call
**`train_pureface.sh`** (`src/face_drift_pixel.py`). Both launchers expose every
distinguishing knob as an env var — read their headers.

Each run writes to `runs/<name>/` (checkpoints, samples, log) and prints the exact
evaluation command at the end.

## Prerequisites (built once, not committed — see top-level README)

The recipes expect these under `checkpoints/`:

| File | Built by | Used for |
|---|---|---|
| `edge_bank_128_dil_lo75.pt` | `src/celeba_edges.py` (dilated, thresh 75/170) | D2 face drift target (letter models) |
| `edge_bank_128_thin.pt` | `src/celeba_edges.py` (thin, blur 1.0, thresh 40/110) | D2 face drift target (pure-face baselines) |
| `aug_letter_bank_32_aggressive_2000.pt` | `src/build_letter_bank.py` / `src/aug_letters.py` | D1 glyph drift target |
| `letter_cnn_32_aggressive.pt` | `eval/train_letter_cnn.py` (aggressive aug) | CE legibility aux in the loss |
| `letter_cnn_32_mild.pt` | `eval/train_letter_cnn.py` (mild aug) | held-out legibility probe (eval) |
| `face_ae128_metric.pt` | `src/face_ae128.py` | frozen feature space for edge-KID |

See [`../README.md`](../README.md) → *Reproducing the paper* for the build order.
