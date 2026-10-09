#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"  # logical GPU0-4 -> physical
export SHERPA_AGENT_REPLICAS=1
export TOKENIZERS_PARALLELISM=false
: "${SHERPA_ENV_FILE:=$PWD/.env}"
export SHERPA_ENV_FILE
PYTHON_BIN="${SHERPA_PYTHON:-python}"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
CONFIG="${SHERPA_FIVE_GPU_CONFIG:-configs/online_rl/tree_v21_five_gpu_shared5.json}"
# Validate settings/data before starting any GPU server or loading credentials.
"$PYTHON_BIN" scripts/online_rl/run_five_gpu.py --config "$CONFIG"
"$PYTHON_BIN" -c 'from scripts.online_rl.run_tree import load_key; assert load_key().strip(), "OPENROUTER_API_KEY is required for rubric generation/scoring"'
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 5, "CUDA GPUs 0..4 are required"'
"$PYTHON_BIN" scripts/online_rl/start_five_gpu_doctors.py --config "$CONFIG" --launch
exec "$PYTHON_BIN" scripts/online_rl/run_five_gpu.py --config "$CONFIG" --launch "$@"
