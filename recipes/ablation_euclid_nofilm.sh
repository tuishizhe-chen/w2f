#!/usr/bin/env bash
# ABLATION 2 — Euclidean face distance, NO FiLM (report Table 2).
#   The FiLM-off counterpart: turns OFF feature-wise letter conditioning. Higher
#   precision but markedly lower diversity / weaker facial structure.
#   edge-KID = 169  (prec 0.92, rec 0.11, letter acc 0.53)
set -euo pipefail
NAME=ablation_euclid_nofilm \
KERNEL=l2 \
FILM=0 \
DW=0.3 \
CLSW=0.3 \
LETTERR=0.005,0.02,0.1 \
  exec "$(dirname "$0")/train_w2f.sh"
