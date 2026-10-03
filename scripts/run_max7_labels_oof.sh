#!/usr/bin/env bash
set -euo pipefail
# Retired: these prerequisites and paths belong to the GPT Doctor diagnostic protocol.
# A Gemma-main workflow must regenerate D1/base/reach/E2E and bind a new identity.
echo "BLOCKED: GPT Doctor diagnostic artifacts cannot be used for Gemma main. See docs/CHANGELOG.md (2026-10-02, Gemma Doctor)." >&2
exit 1
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
export CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -a
. ${ENV_FILE:-../.env}
set +a
OUT=outputs/effectiveness/v8_agentclinic_max7/effectiveness_oof_v1
export API_USAGE_LOG=$OUT/api_usage.jsonl API_USAGE_STAGE=effectiveness_labels
export PYTHONPATH=../.pydeps/grpc${PYTHONPATH:+:$PYTHONPATH}
PY=${PY:-../.venv-qwen3/bin/python}
$PY -c 'import json; from pathlib import Path; p=Path("outputs/serl/e2e_max7_5case_20261002-051742_4009b088/checks.json"); d=json.loads(p.read_text()); assert d["passed"], d'
if [ ! -f "$OUT/labels.freeze.json" ]; then
  echo "[$(date -u '+%FT%TZ')] label generation started"
  $PY -m scripts.run_agentclinic_turns --config configs/effectiveness/v8_labels_oof.json --scope full candidates labels report > "$OUT/labels_stage.json" 2>> "$OUT/labels_stage.err"
  echo "[$(date -u '+%FT%TZ')] label generation complete"
  $PY -m scripts.train_max7 audit --output-dir "$OUT"
fi
for stage in cv final evaluate parity; do
  if [ "$stage" = cv ] && [ -f "$OUT/cv_summary.json" ]; then continue; fi
  if [ "$stage" = parity ] && [ -f "$OUT/integration.json" ]; then continue; fi
  echo "[$(date -u '+%FT%TZ')] stage=$stage"
  $PY -m scripts.train_max7 "$stage" --output-dir "$OUT" >> "$OUT/$stage.log" 2>&1
done
"$PY" -m scripts.api_cost "$API_USAGE_LOG" > "$OUT/api_cost.json"
"$PY" -m scripts.train_max7 report --output-dir "$OUT"
echo "[$(date -u '+%FT%TZ')] Effectiveness complete; no rubric/GRPO/TEST stage follows. Starting frozen inference service."
exec "$PY" -m scripts.serve_effectiveness --config configs/serl_agentclinic_max7.json --device cuda:0 --port 50557 --log-dir "$OUT/grpc_logs"
