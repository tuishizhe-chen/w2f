#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train_w2f.sh — parametrized launcher for the full W2F model
# (two-stage transformer + STN, letters composed into a face).
#
# All knobs that DISTINGUISH the report configurations are env vars; everything
# else is the locked "face-forming sweet spot" (do not change r/region-iou/
# overlap or faces fall apart -- see docs/HANDOFF_NIGHT.md §4).
#
#   Env            Default   Meaning
#   NAME           w2f_run   run name (-> runs/$NAME)
#   KERNEL         iou       face drift distance: iou | l2 | dice | gradient ...
#   FILM           1         FiLM letter conditioning (1=on, 0=off)
#   DW             0.3       glyph (D1) drift weight  [report: 0.3 fixed]
#   CLSW           0.3       CE legibility aux weight [report: 0.15 or 0.3]
#   LETTERR        0.0125,0.05,0.25   glyph-drift bandwidths (temperature)
#   STEPS          8000      training steps (model peaks ~6000)
#   GPU            0         CUDA device
#
# The 5 thin recipe scripts in this folder just set these and call this file.
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

PY=${PY:-python}
NAME=${NAME:-w2f_run}
KERNEL=${KERNEL:-iou}
FILM=${FILM:-1}
DW=${DW:-0.3}
CLSW=${CLSW:-0.3}
LETTERR=${LETTERR:-0.0125,0.05,0.25}
STEPS=${STEPS:-8000}
GPU=${GPU:-0}

# face distance: structural kernels prune to nearest prototypes (top-k 4/8);
# plain L2 uses the full softmax-weighted average (top-k disabled).
if [ "$KERNEL" = "l2" ]; then
  FACE_KARGS="--drift-kernel l2 --face-topk-pos 0 --face-topk-neg 0"
else
  FACE_KARGS="--drift-kernel ${KERNEL} --face-topk-pos 4 --face-topk-neg 8"
fi
FILMARG=""; [ "$FILM" = "1" ] && FILMARG="--letter-film"

OUT=runs/${NAME}
mkdir -p runs

CUDA_VISIBLE_DEVICES=${GPU} $PY -u src/face_drift_multi_transformer.py \
  --bank checkpoints/edge_bank_128_dil_lo75.pt \
  --K 12 --bgen 2048 --gen-per 64 --cp 64 --cn 16 --steps ${STEPS} --lr 2e-4 \
  --sample-every 2000 --ckpt-every 2000 --keep-ckpts \
  --R 0.005,0.02,0.1 --ema 0.999 --noise-aug 0.10 --tv 0.03 --cov 8.0 \
  --sharpness 0.5 --sharpness-end 0.5 --sigmoid-t 1.0 --sigmoid-t-end 2.5 \
  --d-token 320 --n-layers 6 --n-heads 8 --patch-size 32 \
  --r-min 0.10 --r-max 0.40 --dxy-max 0.6 \
  --layer-min 0.005 --layer-min-w 150 --overlap 5 --locality 1e-3 \
  --region-iou 1.5 --n-regions 16 \
  ${FACE_KARGS} --letter-drift-kernel l2 \
  --letter-mode --n-letters 26 \
  --letter-weight 0.5 --letter-weight-end 0.5 \
  --letter-samples-per-class 2000 --letter-data-root ./data/emnist \
  --letter-loss-mode drift --letter-drift-cp 32 --letter-drift-cn 32 \
  --letter-drift-R ${LETTERR} --letter-noise-aug 0.10 \
  --letter-drift-weight ${DW} --letter-aug-level aggressive \
  --letter-cls-weight ${CLSW} --letter-cls-ckpt checkpoints/letter_cnn_32_aggressive.pt \
  ${FILMARG} \
  --out ${OUT}

echo "[train_w2f] ${NAME} done -> ${OUT}"
echo "[train_w2f] checkpoints: G_final.pt + per-step G_stepNNNNN.pt (model peaks ~step 6000;"
echo "             pick the best per-step ckpt by edge-KID rather than the last)."
echo "[train_w2f] evaluate with:"
echo "    $PY eval/eval_face_kid.py ${OUT}/G_final.pt checkpoints/face_ae128_metric.pt checkpoints/edge_bank_128_dil_lo75.pt"
echo "    $PY eval/eval_letter_metric.py ${OUT}/G_final.pt checkpoints/letter_cnn_32_mild.pt"
