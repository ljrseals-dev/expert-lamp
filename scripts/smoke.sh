#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m training.train --dry-run
python -m training.train \
  --n-samples 64 --n-val 16 --epochs 2 --batch-size 4 \
  --out-dir checkpoints/smoke --device cpu
python -m pytest tests -q
echo "smoke OK"
