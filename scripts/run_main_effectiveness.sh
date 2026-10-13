#!/usr/bin/env bash
# Main environment: Agent Qwen3-4B, Doctor Gemma-3-12B (after scripts/run_gemma_effectiveness.sh). Resume matching artifacts; stop on any failed prerequisite.
# base -> validity -> TRAIN reach + protocol freeze -> wiring E2E 1/2/5 -> STOP until the candidate-generator and compliance-judge roles are resolved.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY=${PY:-../.venv-qwen3/bin/python}
CONFIG=configs/effectiveness/v10_qwen4b_gemma12b_max7.json
MAIN=configs/serl_agentclinic_qwen4b_max7.json
BASE=outputs/effectiveness/v10_qwen4b_gemma12b_max7
OUT=$BASE/effectiveness_oof_v1
export CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=../.pydeps/grpo:../.pydeps/grpc${PYTHONPATH:+:$PYTHONPATH}
set -a
. ${ENV_FILE:-../.env}
set +a
mkdir -p "$OUT"
export API_USAGE_LOG=$OUT/api_usage.jsonl API_USAGE_STAGE=base
if [ ! -f "$BASE/protocol.frozen.json" ]; then
  # The generation process can abort at interpreter exit (status 134) after every collaboration is written; the
  # completeness check below decides, not its exit status. Finished collaborations are kept on a rerun.
  if [ "$(wc -l < "$BASE/base_trajectories.jsonl" 2>/dev/null || echo 0)" -lt 121 ]; then
    "$PY" -m scripts.run_base_collaborations --config "$CONFIG" generate > "$OUT/base_stage.json" 2>> "$OUT/base.err" || true
  fi
  "$PY" -c 'import json; from pathlib import Path; r=[json.loads(x) for x in Path("outputs/effectiveness/v10_qwen4b_gemma12b_max7/base_trajectories.jsonl").read_text().splitlines()]; assert len(r)==121 and all(x["trajectory_status"]=="valid" for x in r), "incomplete/invalid base trajectories"'
  "$PY" -m scripts.run_base_collaborations --config "$CONFIG" report freeze > "$OUT/reach_freeze.json" 2>> "$OUT/base.err"
fi
for n in 1 2 5; do
  ID=$OUT/e2e_${n}.run_id
  if [ -s "$ID" ]; then
    "$PY" -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]).read_text().strip(); assert json.loads((Path("outputs/serl")/r/"checks.json").read_text())["passed"]' "$ID"
  else
    PREREQ=()
    if [ "$n" != 1 ]; then PREREQ=(--prerequisite-run "$(cat "$OUT/e2e_$PREV.run_id")"); fi
    "$PY" -m scripts.run_e2e_max7 --config "$MAIN" --cases "$n" "${PREREQ[@]}" --write-run-id "$ID" >> "$OUT/e2e_${n}.log" 2>&1
  fi
  PREV=$n
done
"$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); p=out/'workflow.json'; c=json.loads(p.read_text())
c['e2e_runs']={str(n):(out/f'e2e_{n}.run_id').read_text().strip() for n in (1,2,5)}
p.write_text(json.dumps(c,indent=2)+'\n')
if not c.get('compliance_role_resolved'):
    raise SystemExit('Base/reach/freeze/E2E complete. STOP: compliance evaluator role needs resolution before labels.')
PY
export API_USAGE_STAGE=gemma_effectiveness_labels
if [ ! -f "$OUT/labels.freeze.json" ]; then
  "$PY" -m scripts.run_agentclinic_turns --config "$CONFIG" --scope smoke candidates labels report > "$OUT/smoke_stage.json" 2>> "$OUT/labels.err"
  "$PY" -m scripts.run_agentclinic_turns --config "$CONFIG" --scope full candidates labels report > "$OUT/labels_stage.json" 2>> "$OUT/labels.err"
  "$PY" -m scripts.train_max7 audit --output-dir "$OUT" >> "$OUT/audit.log" 2>&1
fi
for stage in cv final evaluate parity; do
  if [ "$stage" = cv ] && [ -f "$OUT/cv_summary.json" ]; then continue; fi
  if [ "$stage" = parity ] && [ -f "$OUT/integration.json" ]; then continue; fi
  "$PY" -m scripts.train_max7 "$stage" --output-dir "$OUT" >> "$OUT/$stage.log" 2>&1
done
"$PY" -m scripts.api_cost "$API_USAGE_LOG" > "$OUT/api_cost.json"
"$PY" -m scripts.train_max7 report --output-dir "$OUT"
echo 'Gemma Effectiveness complete. No rubric/GRPO/TEST stage follows.'
exec "$PY" -m scripts.serve_effectiveness --config "$MAIN" --device cuda:0 --port 50557 --log-dir "$OUT/grpc_logs"
