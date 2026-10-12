#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SHERPA_PYTHON="${SHERPA_PYTHON:-$PWD/runtime/python}"
export SHERPA_ENV_FILE="${SHERPA_ENV_FILE:-$PWD/runtime/credentials.env}"
export SHERPA_OLLAMA_BINARY="${SHERPA_OLLAMA_BINARY:-$PWD/runtime/ollama}"
export SHERPA_OLLAMA_MODELS="${SHERPA_OLLAMA_MODELS:-$PWD/runtime/ollama_models}"
export HF_HOME="${HF_HOME:-$PWD/runtime/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export SHERPA_AGENT_REPLICAS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
exec "$SHERPA_PYTHON" scripts/online_rl/shared_worker.py --config configs/group_shared.json "$@"
