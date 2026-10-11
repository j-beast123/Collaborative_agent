#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${SHERPA_PYTHON:-/home/common/miniconda3/bin/python}"
# Resume with the exact configuration identity that created this run.  The
# qwen/gemma/luna role manifest is documentation-only; using its wrapper path
# would change the persisted configuration hash even though model roles agree.
CONFIG="${SHERPA_FIVE_GPU_CONFIG:-configs/experiments/train120_max4_direct_luna_v9_1.json}"
SOURCE_RUN="${SHERPA_TRAIN120_RUN_DIR:-outputs/train120_max4_idle_eval_v1_20261011/r05_o05__p0fixed__gemma3doctor__seed0__train120_max4_idle_eval_v1}"
EVAL_ROOT="${SHERPA_TRAIN120_IDLE_EVAL_ROOT:-outputs/train120_max4_idle_eval_v1_622_20261011}"
PRIORITY_DIR="${SHERPA_IDLE_EVAL_PRIORITY_DIR:-$SOURCE_RUN/idle_eval_priority}"
DOCTOR_BINARY="${SHERPA_OLLAMA_BINARY:-/home/kai/.local/ollama-v0.12.5/bin/ollama}"

mkdir -p logs "$EVAL_ROOT" "$PRIORITY_DIR"
export SHERPA_PYTHON="$PYTHON"
export SHERPA_FIVE_GPU_CONFIG="$CONFIG"
export SHERPA_IDLE_EVAL_PRIORITY_DIR="$PRIORITY_DIR"

evaluate_checkpoints() {
  while [[ ! -f "$SOURCE_RUN/run_manifest.json" ]]; do sleep 1; done
  for checkpoint in 30 60 90 120; do
    "$PYTHON" scripts/online_rl/run_idle_checkpoint_eval.py \
      --config "$CONFIG" \
      --source-run "$SOURCE_RUN" \
      --output "$EVAL_ROOT/checkpoint_$(printf '%06d' "$checkpoint")" \
      --checkpoint-cases "$checkpoint" \
      --priority-dir "$PRIORITY_DIR" \
      --gpus 0,1,2,3,4 \
      --doctor-binary "$DOCTOR_BINARY" \
      --doctor-base-port $((12100 + checkpoint))
  done
}

evaluate_checkpoints > logs/train120_max4_idle_eval_worker.log 2>&1 &
EVAL_PID=$!
cleanup() {
  kill -TERM "$EVAL_PID" 2>/dev/null || true
  wait "$EVAL_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

scripts/online_rl/launch_five_gpu.sh
