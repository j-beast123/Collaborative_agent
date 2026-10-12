#!/usr/bin/env bash
# No arguments: checks only. --launch: rubric-only online training.
set -euo pipefail
cd "$(dirname "$0")"
export SHERPA_FIVE_GPU_CONFIG=configs/group_online.json
exec bash run.sh "$@"
