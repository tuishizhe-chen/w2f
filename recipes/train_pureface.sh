#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# train_pureface.sh — parametrized launcher for the PURE-FACE baseline.
#
# A plain convolutional generator (noise -> fc -> 5 nearest-upsample conv blocks,
# widths 192,192,192,96,48,24 -> 128x128 sigmoid). NO letters, NO slots, NO STN:
# the single output map *is* the face. Face drift only. This sets the face-quality
# ceiling of the method (report Sec. "Baseline: pure-drifting face generation").
#
#   Env       Default   Meaning
#   NAME      pureface  run name (-> runs/$NAME)
#   TOPK      1         nearest-prototype pruning (1 -> top-k 4/8, 0 -> off)
#   STEPS     24000     training steps
#   GPU       0         CUDA device
#   BANK      checkpoints/edge_bank_128_thin.pt   THIN (non-dilated) Canny bank
#
# TOPK=1 -> sharp, recovered eyes/nose/mouth, KID ~= 32.6  (quality ceiling).
# TOPK=0 -> clean but blurry "mean face" (the averaging artifact).
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

PY=${PY:-python}
NAME=${NAME:-pureface}
TOPK=${TOPK:-1}
STEPS=${STEPS:-24000}
GPU=${GPU:-0}
BANK=${BANK:-checkpoints/edge_bank_128_thin.pt}

TOPKARG=""
[ "$TOPK" = "1" ] && TOPKARG="--topk-pos 4 --topk-neg 8"

OUT=runs/${NAME}
mkdir -p runs

CUDA_VISIBLE_DEVICES=${GPU} $PY -u src/face_drift_pixel.py \
  --bank ${BANK} --base 192 --bgen 256 \
  --kernel iou --R 0.005,0.02,0.1 ${TOPKARG} \
  --gen-per 32 --cp 64 --cn 16 --cov 5.0 \
  --ema 0.999 --noise-aug 0.10 --tv 0.03 --sharpness 1.0 \
  --steps ${STEPS} --sample-every 2000 \
  --out ${OUT}

echo "[train_pureface] ${NAME} done -> ${OUT}"
echo "[train_pureface] evaluate with:"
echo "    $PY eval/eval_face_kid_pixel.py ${OUT}/G_final.pt checkpoints/face_ae128_metric.pt ${BANK}"
