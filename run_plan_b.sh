#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/online_rl/runtime_env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONUNBUFFERED=1
exec "$SHERPA_PYTHON" scripts/online_rl/run_plan_b.py "$@"
