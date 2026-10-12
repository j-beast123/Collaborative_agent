#!/usr/bin/env bash
# Check only by default; --launch explicitly starts training.
set -euo pipefail
cd "$(dirname "$0")"
export SHERPA_FIVE_GPU_CONFIG=configs/online_openai20.json
exec bash run.sh "$@"
