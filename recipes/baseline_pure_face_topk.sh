#!/usr/bin/env bash
# BASELINE 1 — pure-face, top-k nearest-prototype pruning (report Table, "top-k").
#   Conv generator, face drift only, IoU distance, drift target = average of the
#   4 nearest real faces -> sharp eyes/nose/mouth.  edge-KID ~= 32.6 (prec 0.55,
#   rec 0.59).  THE FACE-QUALITY CEILING of the method.
set -euo pipefail
NAME=baseline_pure_face_topk \
TOPK=1 \
  exec "$(dirname "$0")/train_pureface.sh"
