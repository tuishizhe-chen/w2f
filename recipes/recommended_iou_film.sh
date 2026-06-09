#!/usr/bin/env bash
# RECOMMENDED W2F model (report Table 2, recommended row).
#   IoU face distance + Euclidean glyphs + FiLM letter conditioning + CE aux.
#   Best & most stable faces.  edge-KID = 124 +/- 5  (prec 0.88, rec 0.15, letter acc 0.38)
set -euo pipefail
NAME=recommended_iou_film \
KERNEL=iou \
FILM=1 \
DW=0.3 \
CLSW=0.3 \
LETTERR=0.0125,0.05,0.25 \
  exec "$(dirname "$0")/train_w2f.sh"
