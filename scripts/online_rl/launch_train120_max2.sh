#!/usr/bin/env bash
# Active fixed-P0 training entry point; all model/data checks remain in the runner.
set -euo pipefail
cd "$(dirname "$0")/../.."
export SHERPA_FIVE_GPU_CONFIG="${SHERPA_FIVE_GPU_CONFIG:-configs/experiments/train120_max2_fixed_p0.json}"
export SHERPA_PYTHON="${SHERPA_PYTHON:-python}"
export SHERPA_ENV_FILE="${SHERPA_ENV_FILE:-$PWD/.env}"
export SHERPA_OLLAMA_BINARY="${SHERPA_OLLAMA_BINARY:-ollama}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}"
export SHERPA_AGENT_BATCH_SIZE=4
export SHERPA_DOCTOR_CONCURRENCY=1
export SHERPA_RUBRIC_SCORE_BATCH_SIZE=4
export SHERPA_PARALLEL_STRATA=1
export PYTHONUNBUFFERED=1
exec bash scripts/online_rl/launch_five_gpu.sh "$@"
