#!/usr/bin/env bash
set -euo pipefail

python run_forgery.py \
  --config configs/tree_ring_stage1.yaml \
  --mode both \
  --limit 10 \
  --iterations 3000 \
  --run-name core_10x3000
python evaluate.py \
  --config configs/tree_ring_stage1.yaml \
  --run-dir outputs/tree_ring_stage1/attacks/core_10x3000

