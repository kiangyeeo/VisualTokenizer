#!/usr/bin/env bash
set -euo pipefail

python -m baselines.dinov2_probe.eval_linear_probe \
  --models dinov2_vits14 \
  --datasets cifar10 \
  --smoke \
  --epochs 1 \
  --feature-batch-size 32 \
  --train-batch-size 128 \
  --allow-download \
  --dataset-download \
  "$@"

python -m baselines.dinov2_probe.collect_results "$@"

