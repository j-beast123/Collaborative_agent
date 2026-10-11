#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONFIG="${SHERPA_FIVE_GPU_CONFIG:-configs/experiments/train120_max4_natural_sample8_v1.json}"
RUN="${SHERPA_TRAIN_RUN:-$ROOT/outputs/train120_max4_natural_sample8_relationfix_v2_20261012/r05_o05__p0fixed__gemma3doctor__seed0__max4_natural_sample8_relationfix_v2}"
EVAL_ROOT="${SHERPA_IDLE_EVAL_ROOT:-$ROOT/outputs/train120_max4_natural_sample8_relationfix_v2_idle622_20261012}"
PRIORITY_DIR="${SHERPA_IDLE_EVAL_PRIORITY_DIR:-$RUN/idle_eval_priority}"
PYTHON_BIN="${SHERPA_PYTHON:-/home/common/miniconda3/bin/python}"
DOCTOR_BIN="${SHERPA_OLLAMA_BIN:-/home/kai/.local/ollama-v0.12.5/bin/ollama}"
DOCTOR_BASE_PORT="${SHERPA_IDLE_DOCTOR_BASE_PORT:-17350}"

mkdir -p "$EVAL_ROOT" logs

for checkpoint in 0 30 60 90 120; do
  output="$EVAL_ROOT/C$(printf '%03d' "$checkpoint")"
  "$PYTHON_BIN" - "$RUN" "$output" "$PRIORITY_DIR" "$checkpoint" <<'PY'
import json, os, sys, tempfile
from pathlib import Path
run, output, priority, checkpoint = sys.argv[1:]
path = Path(run) / 'idle_eval_runtime.json'
value = {
    'contract': 'idle_checkpoint_evaluation_v1',
    'output_dir': output,
    'priority_dir': priority,
    'checkpoint_cases': int(checkpoint),
}
path.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix='.idle-eval-', dir=path.parent)
try:
    with os.fdopen(fd, 'w') as handle:
        json.dump(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
  "$PYTHON_BIN" -u scripts/online_rl/run_idle_checkpoint_eval.py \
    --config "$CONFIG" \
    --source-run "$RUN" \
    --output "$output" \
    --checkpoint-cases "$checkpoint" \
    --priority-dir "$PRIORITY_DIR" \
    --gpus 0,1,2,3,4 \
    --doctor-binary "$DOCTOR_BIN" \
    --doctor-base-port "$DOCTOR_BASE_PORT"
done
