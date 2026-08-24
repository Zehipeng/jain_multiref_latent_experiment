#!/usr/bin/env bash
set -euo pipefail

python prepare_references.py --config configs/tree_ring_stage1.yaml --verify
python run_forgery.py \
  --config configs/tree_ring_stage1.yaml \
  --mode both \
  --limit 2 \
  --iterations 200 \
  --run-name smoke_2x200
python evaluate.py \
  --config configs/tree_ring_stage1.yaml \
  --run-dir outputs/tree_ring_stage1/attacks/smoke_2x200 \
  --no-lpips

