#!/usr/bin/env bash
# ABLATION 1 — Euclidean face distance + FiLM (report Table 2).
#   Swaps the IoU face distance for plain L2; keeps FiLM. Looser, noisier faces;
#   crisper letters.  edge-KID = 150 +/- 18  (prec 0.84, rec 0.20, letter acc 0.52)
set -euo pipefail
NAME=ablation_euclid_film \
KERNEL=l2 \
FILM=1 \
DW=0.3 \
CLSW=0.15 \
LETTERR=0.0025,0.01,0.05 \
  exec "$(dirname "$0")/train_w2f.sh"
