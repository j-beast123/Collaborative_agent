#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SHERPA_FIVE_GPU_CONFIG=configs/online_parallel_bank.json
exec bash run.sh "$@"
