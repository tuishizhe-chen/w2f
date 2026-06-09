# `src/` — the W2F code ecosystem

This directory holds the **entire model/loss/data stack** the project ever grew —
the final model *and* every earlier route we tried on the way to it. We keep the
superseded variants on purpose: they are the record of *why* the final design
looks the way it does.

**Status legend**

| | meaning |
|---|---|
| ★ | **live** — used by the published recipes |
| ✓ | **supporting** — data / banks / frozen aux models / config |
| ◦ | **superseded** — an earlier route, kept for the record |
| ✗ | **abandoned** — tried, then deliberately dropped (violated a design rule) |

> Nothing imports across routes in a way that breaks if you ignore the ◦/✗ files;
> they are self-contained training scripts. Start from the ★ files.

---

## ★ The final pipeline (what the recipes run)

| File | Role |
|---|---|
| **`face_drift_multi_transformer.py`** | The full **W2F** model. K=12 slot tokens + noise + per-slot letter embedding → **Stage-1** transformer emits an affine pose θ and a content seed per slot → θ is re-encoded and fed back → **Stage-2** transformer renders each glyph → per-slot CNN decoder → 32² patch → **STN** places it on the 128² canvas → **sum + clamp**. Trained by two-level Drifting (D1 glyph + D2 face) + a small CE legibility term; supports FiLM letter conditioning, structural face kernels, region-IoU, per-slot intensity. The three letter recipes drive this file. |
| **`face_drift_pixel.py`** | The **pure-face baseline** generator: a plain conv net (noise → fc → 5 nearest-upsample blocks, widths 192·192·192·96·48·24 → 128² sigmoid). No letters, no slots, no STN — the single map *is* the face. Face drift only. Output size is read from the bank (64 or 128). The two baseline recipes drive this file. |
| **`drift_loss.py`** | The **Drifting force** itself: data-adaptive scale, self-masking, nearest-neighbour (top-k) pruning, doubly-normalized geometric-mean affinity, multi-bandwidth aggregation, attraction − repulsion under a stop-gradient. The distance is modular (L2 / IoU / Dice / multi-scale IoU / gradient / low-freq). Re-implemented from the official [Drifting](https://github.com/lambertae/drifting) repo (see `third_party/README.md`) with our modifications. |

---

## ✓ Data, banks, frozen models, config

| File | Role |
|---|---|
| `prep_data.py` | Fetch / cache EMNIST (letters) and CelebA (faces). |
| `data.py` | EMNIST loader (+ an early synthetic face proxy, now unused). |
| `aug_letters.py` | The aggressive two-stage **elastic + affine** glyph augmentation — the deformation-tolerant letter prior. |
| `build_letter_bank.py` | Bake the augmented 32² glyph bank — the **D1 positives**. |
| `celeba_edges.py` | CelebA → blur → **Canny** → sparsify → (dilate) edge maps — the **D2 face bank**. Builds both the dilated bank (letter models) and the thin bank (pure-face baselines). |
| `build_edge_bank_64_clean.py`, `bank_patch_cap.py`, `compare_edge_64.py` | Edge-bank builders / cleaners / style comparison from the 64-res pixel-drift era. |
| `face_ae128.py` | The 128 edge-domain autoencoder — its frozen bottleneck is the **edge-KID** feature space (and decodes latents in the AE route). |
| `train_edge_ae.py`, `face_ae.py` | Train the edge-matched AE / a small AE on the early synthetic proxies. |
| `classifier.py` | The small **LetterCNN** used for the CE legibility term and the held-out legibility probe. |
| `config.py` | Central hyperparameter defaults (early scripts). |

---

## ◦ The research trajectory (superseded routes)

These are the four routes we explored before the slot-transformer converged. Each
is a complete, runnable training script; read them as "what we tried and what it
taught us".

### Route 1 — direct face generation by drift alone
*Idea: before adding letters, can Drifting on its own form a face? In what space?*

| File | What it is |
|---|---|
| `face_drift.py` | D2 face drift in **AE-latent** space, single-layer backbone (phase 1). |
| `face_drift_orig.py` | A variant kept close to the original `lambertae/drifting` setup. |
| `face_flow.py` | **Flow-matching** in AE latent as an alternative to drift. |
| `train_d2only.py` | A D2-only latent-drifting experiment harness. |
| `train_dino_decoder.py` | Decode from frozen **DINO** features instead of an AE. |

**Lesson.** Latent-space drift gives clean but blurry "mean faces"; **pixel-space**
drift with a structural (IoU) kernel + nearest-prototype pruning gives sharp faces
(this became `face_drift_pixel.py`, the KID-32.6 ceiling).

### Route 2 — multi-layer composition (precursor to slots)
*Idea: build the face out of K independently-drifting layers.*

| File | What it is |
|---|---|
| `face_drift_multi.py` | K layers as AE-latents, decoded and **max-composed**. |
| `face_drift_multi_pixel.py` | K free **pixel** layers, max-composed. |
| `face_drift_multi_stn.py` | K pixel layers, each placed by its **own STN**. |

**Lesson.** Per-layer STN placement worked and became the core of the final model;
**max-compose** was replaced by **sum + clamp** (max lets a hidden layer "paint"
the face; sum-clamp forces it to be spelled). The "K layers" became "K letter
slots".

### Route 3 — early letter composers (pre-transformer)
*Idea: actually place real letters so they read as a face.*

| File | What it is |
|---|---|
| `model.py` | The first generator: per-letter MLP + ε → Transformer → affine θ → STN → max-compose. |
| `train.py` | Its training loop. |
| `face_letters.py` | K letter slots → max-composed canvas → D2 face drift **+ per-letter L1/L2**. |
| `face_letters_pixel.py` | Same in pixel space (STN-place → max-compose). |
| `face_letters_v2.py` | G predicts **placement only**; letter content comes frozen from the bank. |
| `face_letters_cond.py` | **Conditional**: compose K letters to match a *given* target face latent. |

**Lesson.** Letters must be supervised by **D1 drift**, not L1/L2 (which forces
canonical glyphs and fights the face); placement-only and target-conditioned
variants were dropped as too rigid / not unpaired. This whole route matured into
the two-stage `face_drift_multi_transformer.py`.

### ✗ Route 4 — anatomical priors (abandoned)

| File | What it is |
|---|---|
| `face_regions.py` | Hand-picked 12 eye/nose/mouth **anchors** to seed slot placement. |

**Abandoned by design.** Hand-placed facial anchors are a **face prior** — one of
the things the project explicitly forbids (see the design constraints in the
top-level README). Kept only to document the rejected shortcut.

---

## Small utilities

`evaluate.py` (early all-in-one eval: sample grid + STN-crop letter accuracy +
latent Fréchet — superseded by [`../eval/`](../eval/)), `inspect_lat.py` (latent
inspection), `_kill_drift.py` (kill stray training processes).

---

### One-line history

> latent drift (blurry) → **pixel drift + IoU + top-k** (sharp faces, the ceiling)
> → K max-composed layers → **K STN-placed slots** → letters via **L1/L2** (rigid)
> → letters via **D1 drift** → two-stage **slot transformer + STN + FiLM** = the
> model in the report.
