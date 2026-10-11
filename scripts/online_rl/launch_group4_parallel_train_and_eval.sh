#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="configs/experiments/train120_max4_group4_parallel_gate_eval622_v1.json"
RUN="$ROOT/outputs/train120_max4_group4_parallel_gate_eval622_v1_20261012/r05_o05__p0fixed__gemma3doctor__seed0__max4_group4_parallel_gate_eval622_v1"
EVAL_ROOT="$ROOT/outputs/train120_max4_group4_parallel_gate_eval622_v1_idle622_20261012"
PRIORITY_DIR="$RUN/idle_eval_priority"
PYTHON_BIN="${SHERPA_PYTHON:-/home/common/miniconda3/bin/python}"
DOCTOR_BIN="${SHERPA_OLLAMA_BIN:-/home/kai/.local/ollama-v0.12.5/bin/ollama}"

mkdir -p logs "$EVAL_ROOT"

export SHERPA_FIVE_GPU_CONFIG="$CONFIG"
export SHERPA_TRAIN_RUN="$RUN"
export SHERPA_IDLE_EVAL_ROOT="$EVAL_ROOT"
export SHERPA_IDLE_EVAL_PRIORITY_DIR="$PRIORITY_DIR"
export SHERPA_GATE_PARALLEL_STAGES=1

nohup env \
  SHERPA_PYTHON="$PYTHON_BIN" \
  SHERPA_ENV_FILE="${SHERPA_ENV_FILE:-/home/kai/Collaborative_agent/.env}" \
  SHERPA_GATE_PARALLEL_STAGES=1 \
  SHERPA_FIVE_GPU_CONFIG="$CONFIG" \
  scripts/online_rl/launch_five_gpu.sh \
  > logs/train120_group4_parallel_gate.log 2>&1 < /dev/null &
TRAIN_PID="$!"
echo "$TRAIN_PID" > logs/train120_group4_parallel_gate.pid

# The evaluator pins the immutable training manifest. Do not race it against
# trainer initialization.
for _ in $(seq 1 300); do
  [[ -f "$RUN/run_manifest.json" ]] && break
  if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    wait "$TRAIN_PID"
    exit "$?"
  fi
  sleep 1
done
if [[ ! -f "$RUN/run_manifest.json" ]]; then
  kill -TERM "$TRAIN_PID" 2>/dev/null || true
  echo "training manifest was not created within 300 seconds" >&2
  exit 1
fi

nohup env \
  SHERPA_PYTHON="$PYTHON_BIN" \
  SHERPA_FIVE_GPU_CONFIG="$CONFIG" \
  SHERPA_TRAIN_RUN="$RUN" \
  SHERPA_IDLE_EVAL_ROOT="$EVAL_ROOT" \
  SHERPA_IDLE_EVAL_PRIORITY_DIR="$PRIORITY_DIR" \
  SHERPA_IDLE_DOCTOR_BASE_PORT=17450 \
  SHERPA_OLLAMA_BIN="$DOCTOR_BIN" \
  scripts/online_rl/launch_natural_idle622_checkpoints.sh \
  > logs/train120_group4_parallel_eval622.log 2>&1 < /dev/null &
EVAL_PID="$!"
echo "$EVAL_PID" > logs/train120_group4_parallel_eval622.pid

printf 'training_pid=%s\nevaluation_pid=%s\nrun=%s\n' \
  "$(cat logs/train120_group4_parallel_gate.pid)" \
  "$(cat logs/train120_group4_parallel_eval622.pid)" \
  "$RUN"

# Keep a supervising parent alive. Some managed shells reap detached descendants
# when the launcher exits even when nohup is used.
trap 'kill -TERM "$TRAIN_PID" "$EVAL_PID" 2>/dev/null || true' TERM INT
set +e
wait "$TRAIN_PID"
TRAIN_RC="$?"
if [[ "$TRAIN_RC" -ne 0 ]]; then
  kill -TERM "$EVAL_PID" 2>/dev/null || true
  wait "$EVAL_PID" 2>/dev/null || true
  exit "$TRAIN_RC"
fi
wait "$EVAL_PID"
exit "$?"
