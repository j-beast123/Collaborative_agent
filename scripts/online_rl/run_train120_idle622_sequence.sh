#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="$ROOT/configs/experiments/train120_max4_consolidation8_random5_idle622_v1.json"
SOURCE="$ROOT/outputs/train120_max4_consolidation8_random5_idle622_v1_20261011/r05_o05__p0fixed__gemma3doctor__seed0__max4_consolidation8_random5_idle622_v1"
EVAL_ROOT="$ROOT/outputs/train120_max4_consolidation8_random5_idle622_evaluations_20261011"
PRIORITY="$SOURCE/idle_eval_priority"
PYTHON_BIN="${SHERPA_PYTHON:-/home/common/miniconda3/bin/python}"
DOCTOR_BIN="${SHERPA_EVAL_OLLAMA:-/home/kai/.local/ollama-v0.12.5/bin/ollama}"

mkdir -p "$EVAL_ROOT"
for CASES in 30 60 90 120; do
  OUTPUT="$EVAL_ROOT/C${CASES}"
  "$PYTHON_BIN" - "$SOURCE" "$OUTPUT" "$PRIORITY" "$CASES" <<'PY'
import json, sys, time
from pathlib import Path
source, output, priority, cases = sys.argv[1:]
path = Path(source) / 'idle_eval_runtime.json'
value = {
    'output_dir': output,
    'priority_dir': priority,
    'checkpoint_cases': int(cases),
    'dataset_cases': 622,
    'updated_at': time.time(),
}
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(value, indent=2) + '\n')
temporary.replace(path)
PY
  "$PYTHON_BIN" "$ROOT/scripts/online_rl/run_idle_checkpoint_eval.py" \
    --config "$CONFIG" \
    --source-run "$SOURCE" \
    --output "$OUTPUT" \
    --checkpoint-cases "$CASES" \
    --priority-dir "$PRIORITY" \
    --gpus 0,1,2,3,4 \
    --doctor-binary "$DOCTOR_BIN" \
    --doctor-base-port 12135
done
