#!/usr/bin/env bash
set -euo pipefail

python -m baselines.dinov2_probe.eval_imagenet_knn "$@"

