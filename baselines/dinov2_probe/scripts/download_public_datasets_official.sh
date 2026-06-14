#!/usr/bin/env bash
set -euo pipefail

python -m baselines.dinov2_probe.download_public_datasets "$@"
python -m baselines.dinov2_probe.check_data "$@" || true

