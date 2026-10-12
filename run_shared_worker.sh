#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/online_rl/runtime_env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_HUB_OFFLINE=1
export SHERPA_AGENT_REPLICAS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$SHERPA_PYTHON" scripts/online_rl/shared_worker.py --config configs/group_shared.json "$@"
