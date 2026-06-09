#!/usr/bin/env bash
# BASELINE 2 — pure-face, NO pruning (report Table, "no pruning").
#   Identical to baseline 1 except the drift target averages over the WHOLE
#   collection -> features smear out -> a clean but BLURRY "mean face". Shows
#   exactly what nearest-prototype pruning buys.
set -euo pipefail
NAME=baseline_pure_face_nopruning \
TOPK=0 \
  exec "$(dirname "$0")/train_pureface.sh"
