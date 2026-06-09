# Words to Faces (w2f) — Final Report

## TL;DR

Task: given a string of letters, produce a 128×128 image where the individual
letters remain recognizable and the overall image loosely resembles a face.

8 versions were trained autonomously. **v4 is the best checkpoint** by
letter_acc (0.438 on the 8-string benchmark), reached without GAN-style
training and without a hand-labeled face dataset (a synthetic ellipse-with-mask
face prior was used). Target letter_acc ≥ 0.7 was not met; face-likeness
remains weak. Detailed failure modes and next-step ideas are below.

## Final deliverable paths

| Artifact | Path |
| --- | --- |
| Best checkpoint | `checkpoints/v4/final.pt` |
| Best sample grid | `samples/v4/grid.png` |
| Best metrics | `samples/v4/metrics.json` |
| Training log (best) | `logs/v4_stdout.log` |
| Iteration postmortem | `ITERATION_LOG.md` |
| Pretrained helpers | `checkpoints/letter_cnn.pt`, `checkpoints/face_ae.pt` |

## Results summary

| Version | letter_acc | Noise baseline | Training wall | Notes |
|---------|-----------:|---------------:|--------------:|-------|
| v1 | 0.031 | 0.088 | 9 min | D1 drift in pixel space — fails |
| v2 | 0.039 | 0.025 | 10 min | D1 in 16×16 pooled space — still fails |
| v3 | 0.422 | 0.050 | 15 min | + classifier-CE D1 + FiLM eps + face-AE D2 |
| **v4** | **0.438** | 0.075 | 14 min | + strong repulsion, non-zero FiLM, face-mask D2 |
| v5 | 0.188 | 0.025 | 14 min | Per-slot theta_head with face-layout bias — regression |
| v6 | 0.398 | 0.063 | 16 min | Dual-margin repulsion (same-class=0.8) — did not fix duplicates |
| v7 | 0.375 | 0.050 | 16 min | Small slot-bias + heavy face-mask — modest regression |
| v8 | 0.320 | 0.063 | 16 min | v4 + v7 slot-bias combined — noisy, worse |

Per-string for v4 (`samples/v4/metrics.json`):
```
HELO 1.000, XYZW 0.750, ABCD 0.500, AIAI 0.500, OKOK 0.500,
FACE 0.125, YUXU 0.125, HIHI 0.000
```

## Architecture (v4)

Pipeline: `string → K letter images (32×32) + ε noise → Transformer → θ ∈ ℝ³ per letter → STN → max-compose → canvas (128×128)`.

- **Letter encoder**: 3-stage conv (1→32→64→128 + adaptive-avg-pool) + letter class embedding + positional embedding, projected to d_model=256.
- **FiLM injection**: ε ∈ ℝ²⁵⁶ → MLP → (scale, shift) ∈ ℝ⁵¹² that modulates every letter token: `x_k = x_k * (1 + scale) + shift`. Initialized to near-zero perturbation but not exactly zero (v4 uses N(0, 0.02)).
- **Backbone**: 6× Transformer blocks (d=256, heads=8, mlp=4x), pre-LN, shared across tokens.
- **θ head**: single `Linear(256, 3)` producing (r_raw, tx_raw, ty_raw).
  - `r = 0.1 + 0.8 · σ(r_raw)`  — letter visual size ratio in canvas, ∈ (0.1, 0.9)
  - `tx = 0.7 · tanh(tx_raw)`, `ty = 0.7 · tanh(ty_raw)` — translation in canvas coords
- **STN**: `affine_grid` → `grid_sample`, bilinear, zero-padding. Per-letter canvas patch.
- **Compose**: `canvas = max over K of (patches)`. Simple max — no alpha blending.

Generator params: ~5.0 M.

## Losses (v4 weights)

1. **D1 distribution loss** (w=0.3): drift_loss (PyTorch port of
   `lambertae/drifting`) in 16×16 average-pooled feature space. Positive samples
   are 8 random affine placements of the same letter in canvas; negative samples
   are same-class other-generator patches in the batch. See `src/drift_loss.py`.
   This ends up as a weak regularizer rather than a strong supervision — the
   heavy lifting comes from loss #2.

2. **Classifier-guided letter fidelity** (w=1.0): invert the STN transform to
   crop each letter-region back to 32×32, feed through the frozen letter-CNN
   (pretrained on the same EMNIST-letters bank, >90% test acc), cross-entropy
   against the ground-truth letter label. This is what drives letter_acc up.

3. **Face-mask D2** (w=0.2): soft elliptical mask `exp(−(x²/0.65² + y²/0.80²)·3)`
   on the canvas:
   - `out_of_mask_energy = mean((canvas · (1 − mask))²)` — discourage ink outside
   - `coverage_penalty = (mean(canvas · mask) − 0.08)²`  — encourage a minimum of
     face-region ink

4. **Pairwise repulsion** (w=2.0, margin=0.45): hinge penalty
   `relu(margin − ||tx_i−tx_j||)²` between letter centers, pushing them apart
   so they don't all stack in the canvas centre.

5. **Diversity hinge** (w=0.5, target=0.02): a second forward pass with a fresh
   ε is compared to the first; `relu(target − pixel_diff²)` encourages different
   noise to produce different canvases.

## Data

- **Letters**: `torchvision.datasets.EMNIST(split='letters')`, resized to 32×32,
  limited to 400 samples/class (EMNIST letter orientation is transpose-corrected;
  see `data.py`). Downloads on first run (~560 MB). Fallback: PIL-drawn letters
  (DejaVu Sans Bold) if the download fails.
- **Faces**: a synthetic proxy (ellipse + two eye spots + mouth) via
  `data.synth_face_batch`. Used only for (a) pretraining the face-AE and (b) the
  face-mask prior in the D2 loss. No real face dataset (CelebA / LFW) is needed.
- **Letter classifier**: small CNN (~1.1 M params) trained once on the EMNIST
  bank (`checkpoints/letter_cnn.pt`), 1.5 k steps, achieves 93 % on held-out.

## How to run

All commands assume cwd `/home/wangyanmohan/Projectdl/w2f`.

```bash
# 50-step smoke test (about 5 seconds on H20):
CUDA_VISIBLE_DEVICES=2,3 python src/train.py --smoke --version 4

# Full training (~14 min on 1×H20 for v4):
CUDA_VISIBLE_DEVICES=2,3 python src/train.py --version 4

# Evaluation (grid + per-string letter accuracy + noise baseline):
CUDA_VISIBLE_DEVICES=2,3 python src/evaluate.py --version 4
```

Reproducibility: seed is fixed (`cfg.seed=42`) but CUDA non-determinism + the
per-class data sampling make individual runs vary by ~5 pp on letter_acc.
Versions 3/4/6/7 all cluster in the 37–44% range when using the classifier-CE
signal; selecting the best of a few short runs is the easy improvement.

## What worked

1. **Classifier-CE on inverse-STN crops is the critical signal** (v3→v4). The
   letter_acc jumped 10× the moment this was added. Drift-only D1 plateaus at
   ~0.04.
2. **Non-zero FiLM initialization** (v3→v4) lets ε actually propagate; zero-init
   gets stuck at eps-independence.
3. **Pairwise letter repulsion** (weight 2.0, margin 0.45) prevents the letters
   from stacking at the canvas centre. Without it, the generator collapses to a
   central blob.
4. **Soft face mask as D2** is much more informative than AE-latent drift on
   such a small face proxy distribution.

## What did not work

1. **Drift loss alone in pixel or pooled-pixel space** (v1, v2). The 16 K-dim
   pixel manifold drowns the drift signal; 256-dim pooled space helps marginally
   but still no meaningful letter shape emerges.
2. **Per-slot θ head with face-layout biases** (v5). Strong biases push letters
   away from where the classifier wants them → acc drops to 0.19.
3. **Small slot biases** (v7, v8). They fix duplicate-letter collapse (HIHI goes
   from 0 to ~0.25) but hurt overall acc because they perturb easy strings like
   XYZW.
4. **Dual-margin repulsion for same-class pairs** (v6). Even with margin=0.8 and
   weight=2.0, duplicate letters still collapsed: the generator + shared θ head
   is nearly permutation-equivariant over same-label tokens, so the repulsion
   gradient is symmetric and sums to ≈0 at the collapsed point.
5. **Diversity hinge/reward for ε**. At all reasonable weights, the classifier
   loss prefers a single deterministic canvas per string and the FiLM weights
   are driven toward zero regardless.

## Open issues (ranked)

1. **Duplicate-letter symmetry (HIHI = 0)**. The shared θ head produces nearly
   identical (r, tx, ty) for two equal-class tokens. Fix suggestions:
   (a) Add a small per-slot learnable offset *additively inside the Transformer*
       (e.g. a slot-token concat, not just a pos-embedding add).
   (b) Use a KV-cache style recurrent θ head that sees previously emitted θ
       values (like ObjDet DETR) to force distinct outputs.
   (c) Curriculum: forbid repeated letters in the first 2 k steps of training,
       then mix them in.

2. **ε is ignored (mode collapse across noise)**. The classifier loss prefers
   determinism. Fix suggestions:
   (a) Train with a real face dataset (CelebA) and push ε to control things like
       ink thickness, global pose, colour palette that the classifier is
       invariant to.
   (b) Add an auxiliary "ε reconstruction" loss: a small probe that must recover
       ε from the canvas features (force ε into the bitstream).
   (c) VAE-style KL on ε posterior inferred from canvas.

3. **Face likeness is only a "blob in an ellipse"**. The face-mask signal pushes
   letters inside the ellipse but does not produce the two-eyes / one-mouth
   layout. Fix suggestions:
   (a) Multi-region mask (eye-region + mouth-region + cheek-region) and encourage
       K letters to distribute across regions.
   (b) Swap the synthetic face proxy for CelebA; D2 drift in a face-MAE latent
       space should then carry actual face structure (as the original slides
       propose).

## Time budget actual vs. planned

- Single version ≤ 90 min: never exceeded (longest v8 = 16 min).
- 8 versions total: used.
- Wall-clock for full 8-version sweep: ~110 min on a single H20 GPU (only one of
  the two requested GPUs was used — `CUDA_VISIBLE_DEVICES=2,3` but
  single-GPU training sufficient at batch=32×K=5).

## File map

```
w2f/
├── FINAL.md                  this document
├── ITERATION_LOG.md          per-version postmortem (written as agent ran)
├── PROJECT.md                spec
├── EVAL_CRITERIA.md          evaluation rubric
├── src/
│   ├── config.py             hyperparameters (v4 ships here)
│   ├── data.py               EMNIST loader + synth face + STN helpers
│   ├── drift_loss.py         PyTorch port of lambertae/drifting
│   ├── classifier.py         letter CNN (~1.1 M) + trainer
│   ├── face_ae.py            tiny face AE (unused in v4; kept from v3)
│   ├── model.py              Generator (MLP + Transformer + STN)
│   ├── train.py              full training loop, wall-clock guarded
│   └── evaluate.py           grid + letter_acc + noise baseline
├── checkpoints/
│   ├── letter_cnn.pt         shared across versions
│   ├── face_ae.pt            shared across versions
│   └── v{1..8}/final.pt      per-version generator weights
├── samples/v{1..8}/          grid.png + metrics.json per version
└── logs/v{1..8}_stdout.log   per-version raw training logs
```

## Bottom line

The 0.7 letter-acc goal was not reached; 0.44 is well above the 0.03 random
baseline but still mid-tier. The pipeline **does** demonstrate the core
mechanism from the slides — MLP+Transformer+STN one-shot generation with drift
loss as a training signal — and the Transformer successfully learns
string-specific compositions when augmented with a classifier-CE fidelity loss.
The two biggest remaining obstacles (duplicate-letter collapse and ε-induced
mode collapse) both stem from the shared θ-head's permutation equivariance
plus the classifier loss's preference for determinism; they are the natural
target of a follow-up iteration.
