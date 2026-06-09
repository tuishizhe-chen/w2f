# W2F — One-Step Word-to-Face Generative Typography via Drifting

W2F composes a recognizable human face out of a fixed set of **K = 12 alphabet
glyphs**. A single feed-forward generator predicts, for every glyph, *what* to
draw (a 32×32 letter image) and *where* to put it (an affine pose); a Spatial
Transformer Network (STN) places the glyphs onto a 128×128 canvas. There is **no
GAN discriminator and no iterative diffusion sampler** — the model is trained
*only* by the particle-based **Drifting** loss, applied at two levels: a per-glyph
signal that keeps each placed image on its letter manifold, and a composite
signal that pulls the whole canvas toward the distribution of real face
line-drawings.

Authors: **Yuxuan Chen, Junhao Huang, Yanmohan Wang.**
Full write-up: [`report/W2F_Final_Report.pdf`](report/W2F_Final_Report.pdf).

> This repository is the **code ecosystem** for the project — the full model,
> losses, data pipelines, evaluation, and every experiment harness we built along
> the way (including approaches we later dropped). It deliberately omits the
> figure-making / image-curation tooling used only for the report and slides.

---

## The idea in one picture

```
noise ε + 12 slot tokens + 12 letter classes
        │
   Stage-1 transformer ── affine poses θ ─┐  (orange path: θ feeds BOTH
        │                                 │   the renderer and the STN)
   Stage-2 transformer (layout-aware)     │
        │                                 │
   per-slot CNN decoder ─ 32×32 glyph ─ STN(θ) ─ place on 128×128 ─┐
        ▲  (FiLM letter conditioning)                              │
        └───────────────────────── sum + clamp ────────────────────┴─→ face
```

Two Drifting signals supervise it:

- **D1 (glyph drift)** — each decoded patch is pulled toward real same-class
  letters and repelled from sibling generated glyphs of the same class.
- **D2 (face drift)** — the composited canvas is pulled toward real face-edge
  drawings and repelled from other generated faces.

A small frozen-classifier **cross-entropy** term keeps the deformed glyphs legible
(auxiliary only — Drifting stays the main letter signal).

---

## Repository layout

```
w2f/
├── src/               core code — the model, losses, data, and every variant
│   ├── face_drift_multi_transformer.py   ★ the full W2F model (transformer+STN)
│   ├── face_drift_pixel.py               ★ pure-face conv baseline generator
│   ├── drift_loss.py                     Drifting force (multi-bandwidth, top-k)
│   ├── data.py / aug_letters.py          EMNIST loading + aggressive augmentation
│   ├── celeba_edges.py                   CelebA → Canny face-edge bank builder
│   ├── classifier.py / face_ae128.py     frozen aux models (legibility / KID)
│   ├── …                                 earlier variants (pixel/STN/latent/…),
│   │                                     kept even where superseded
│   └── README.md          ★ guided tour of src/ — every file + research route
├── recipes/           ★ the 5 report configurations (2 baselines, recommended,
│                        2 ablations) — see recipes/README.md
├── eval/              edge-KID, precision/recall, and letter-legibility metrics
├── tools/             dataset/bank inspection, smoke tests, visualizations
├── docs/              design spec, eval criteria, iteration log, handoff notes
├── report/            the write-up — W2F_Final_Report.pdf + LaTeX source
├── presentation/      self-contained HTML slide deck (+ its image assets)
└── third_party/       link to the official Drifting repo (not vendored)
```

The `src/` tree keeps **all** model variants we built — pixel-space drift, the
STN-only and latent-AE attempts, the per-letter composers — even the ones the
final model superseded, so the full research trajectory is legible. See
[`src/README.md`](src/README.md) for a guided tour of every file and the idea
behind it.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate     # or conda
pip install -r requirements.txt
# install a CUDA build of torch/torchvision for your GPU from pytorch.org
```

A single 96 GB GPU trains an 8000-step W2F run in ~2 h; the pure-face baselines
(24000 steps, batch 256) are lighter.

---

## Reproducing the paper

Large binaries (datasets, banks, frozen aux models, checkpoints) are **not
committed** — build them once, then run any recipe.

**1. Data + banks** (written to `data/` and `checkpoints/`):

```bash
python src/prep_data.py                 # fetch EMNIST + CelebA caches
python src/build_letter_bank.py         # aggressive 32² glyph bank (D1 target)
python src/celeba_edges.py --dilate     # dilated face-edge bank (D2, letter models)
python src/celeba_edges.py --thin       # thin face-edge bank (D2, pure-face baselines)
```

**2. Frozen auxiliary models:**

```bash
python src/face_ae128.py                          # → checkpoints/face_ae128_metric.pt  (KID features)
python eval/train_letter_cnn.py --aug aggressive  # → checkpoints/letter_cnn_32_aggressive.pt (CE aux)
python eval/train_letter_cnn.py --aug mild        # → checkpoints/letter_cnn_32_mild.pt (held-out probe)
```

**3. Train a recipe** (each prints its own eval command when done):

```bash
GPU=0 bash recipes/recommended_iou_film.sh        # the headline model
GPU=0 bash recipes/baseline_pure_face_topk.sh     # the quality ceiling
```

**4. Evaluate:**

```bash
# edge-domain face KID (×10³, lower better) + improved precision/recall
python eval/eval_face_kid.py runs/<name>/G_final.pt \
       checkpoints/face_ae128_metric.pt checkpoints/edge_bank_128_dil_lo75.pt

# held-out letter legibility (use the OTHER-strength CNN, not the one in the loss)
python eval/eval_letter_metric.py runs/<name>/G_final.pt checkpoints/letter_cnn_32_mild.pt
```

> Flag names in the build commands above mirror what the scripts expect; check
> each script's `--help` for the exact options on your checkout.

---

## The five recipes & headline numbers

| Recipe | Face dist. | FiLM | edge-KID ↓ | prec | rec | letter acc |
|---|---|---|---|---|---|---|
| `baseline_pure_face_topk` (ceiling) | IoU + top-k | — | **32.6** | 0.55 | 0.59 | — |
| `baseline_pure_face_nopruning` | IoU | — | blurry mean face | — | — | — |
| `recommended_iou_film` | IoU + top-k | on | **124 ± 5** | 0.88 | 0.15 | 0.38 |
| `ablation_euclid_film` | L2 | on | 150 ± 18 | 0.84 | 0.20 | 0.52 |
| `ablation_euclid_nofilm` | L2 | off | 169 | 0.92 | 0.11 | 0.53 |

The central finding is a sharp **legibility ↔ realism trade-off**: every lever
that improves the face costs letter legibility, and vice-versa. The gap from the
pure-face ceiling (32.6) to the recommended model (124) is the price of spelling
the face out of *legible* glyphs. See `recipes/README.md` and `docs/` for detail.

---

## Design constraints (the "hard corner")

1. **One-step** generation — a single forward pass, no iterative refinement.
2. **No adversarial loss** — no discriminator, no critic, no minimax game.
3. **No face priors** — no landmarks, anchors, templates, or face-specific
   similarity (generic structural measures like IoU are allowed).
4. **No paired data** — one unpaired face-edge bank, one unpaired letter bank;
   never a (text → face) target pair.

Composition is **sum-then-clamp only** — no free per-pixel transparency mask, so
the face must be *spelled* out of real letter strokes, not painted by a hidden
channel.

---

## Credits

- Built on the **Drifting** objective ([lambertae/drifting](https://github.com/lambertae/drifting),
  arXiv 2602.04770) — `src/drift_loss.py` is our re-implementation; see
  `third_party/README.md`. Its own license applies.
- Data: CelebA (faces) and EMNIST (letters); metrics use a frozen edge-domain
  autoencoder plus improved precision/recall.
- Implementation produced with Claude Code under the authors' direction; the
  authors designed the experiments, made all research decisions, and verified the
  reported results.
