#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SHERPA_FIVE_GPU_CONFIG=configs/online_rubric_only_parallel_chunks.json
exec bash run.sh "$@"
