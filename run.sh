#!/usr/bin/env bash
# No arguments: local checks only. --launch: start online training.
set -euo pipefail
cd "$(dirname "$0")"
export SHERPA_PYTHON="${SHERPA_PYTHON:-$PWD/runtime/python}"
export SHERPA_ENV_FILE="${SHERPA_ENV_FILE:-$PWD/runtime/credentials.env}"
export SHERPA_OLLAMA_BINARY="${SHERPA_OLLAMA_BINARY:-$PWD/runtime/ollama}"
export SHERPA_OLLAMA_MODELS="${SHERPA_OLLAMA_MODELS:-$PWD/runtime/ollama_models}"
export HF_HOME="${HF_HOME:-$PWD/runtime/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export SHERPA_AGENT_REPLICAS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
CONFIG="${SHERPA_FIVE_GPU_CONFIG:-configs/group_online.json}"
if [[ $# -gt 0 && "$1" != --launch ]]; then echo 'Usage: bash run.sh [--launch [runner arguments]]' >&2; exit 2; fi
"$SHERPA_PYTHON" scripts/online_rl/run_five_gpu.py --config "$CONFIG"
"$SHERPA_PYTHON" scripts/online_rl/preflight_chunk8_2gpu.py --config "$CONFIG"
"$SHERPA_PYTHON" scripts/online_rl/verify_wiring.py --config "$CONFIG"
if [[ $# -eq 0 ]]; then exit 0; fi
shift
"$SHERPA_PYTHON" scripts/online_rl/start_five_gpu_doctors.py --config "$CONFIG" --launch
exec "$SHERPA_PYTHON" scripts/online_rl/run_five_gpu.py --config "$CONFIG" --launch "$@"
