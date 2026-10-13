#!/usr/bin/env bash
# Effectiveness of the main environment, from the label smoke to the predictor's DEV evaluation, then STOP.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY=${PY:-../.venv-qwen3/bin/python}
CONFIG=configs/effectiveness/v10_qwen4b_gemma12b_max7.json
OUT=outputs/effectiveness/v10_qwen4b_gemma12b_max7/effectiveness_oof_v1
# GPU: the GPU of the label generation (default 7). PREDICTOR_GPU: the GPU of the predictor training (default: the same).
# WAIT_FOR: tmux sessions that must be gone before the predictor starts (default outcome0; WAIT_FOR= for none, when the
# predictor has a GPU of its own). MODEL_PATH_MAP maps the configured model directories on another machine.
GPU=${GPU:-7}
PREDICTOR_GPU=${PREDICTOR_GPU:-$GPU}
WAIT_FOR=${WAIT_FOR-outcome0}
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=../.pydeps/grpo:../.pydeps/grpc
set -a; . ${ENV_FILE:-../.env}; set +a
export API_USAGE_LOG=$OUT/api_usage.jsonl
stamp() { echo "[$(date '+%F %T')] $*"; }

labels() {  # labels <scope>: candidates and labels; the process may abort at interpreter exit after everything is written
  local scope=$1 attempt
  for attempt in 1 2 3; do
    stamp "labels $scope (attempt $attempt)"
    $PY -m scripts.run_agentclinic_turns --config $CONFIG --scope $scope candidates labels > $OUT/labels_${scope}_stage.json 2>> $OUT/labels_$scope.err
    stamp "labels $scope attempt $attempt ended with status $?"
    $PY -m scripts.run_agentclinic_turns --config $CONFIG --scope $scope report > $OUT/labels_${scope}_report.json 2>> $OUT/labels_$scope.err && return 0
    sleep 30
  done
  return 1
}

if [ ! -f $OUT/labels.freeze.json ]; then
  if [ ! -f $OUT/labels/smoke_report.json ] || ! $PY -c "import json,sys; sys.exit(0 if json.load(open('$OUT/labels/smoke_report.json'))['correctness_passed'] else 1)"; then
    export API_USAGE_STAGE=effectiveness_labels_smoke
    labels smoke || { stamp "STOP: the smoke labels did not complete"; exit 1; }
    $PY -m scripts.label_smoke_report --config $CONFIG --usage $API_USAGE_LOG --stderr $OUT/labels_smoke.err > $OUT/smoke_report.out 2>> $OUT/labels_smoke.err \
      || { stamp "STOP: the smoke report did not pass (see $OUT/labels/smoke_report.json)"; exit 1; }
    stamp "smoke passed: $(grep -m1 estimated_full_hours $OUT/labels/smoke_report.json)"
  fi
  export API_USAGE_STAGE=effectiveness_labels
  labels full || { stamp "STOP: the full labels did not complete"; exit 1; }
  $PY -m scripts.train_max7 audit --output-dir $OUT >> $OUT/audit.log 2>&1 || { stamp "STOP: the label audit failed (see $OUT/audit.log)"; exit 1; }
  stamp "labels audited and frozen"
fi

stamp "waiting for GPU $PREDICTOR_GPU (sessions: ${WAIT_FOR:-none})"
while :; do
  busy=0
  for session in $WAIT_FOR; do tmux has-session -t "$session" 2>/dev/null && busy=1; done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $PREDICTOR_GPU)
  [ "$busy" = 0 ] && [ "$used" -lt 3000 ] && break
  sleep 300
done
export CUDA_VISIBLE_DEVICES=$PREDICTOR_GPU
for stage in cv final evaluate; do
  if [ "$stage" = cv ] && [ -f $OUT/cv_summary.json ]; then continue; fi
  if [ "$stage" = evaluate ] && [ -f $OUT/dev_report.json ]; then continue; fi
  stamp "predictor: $stage"
  $PY -m scripts.train_max7 $stage --output-dir $OUT >> $OUT/$stage.log 2>&1 || { stamp "STOP: predictor stage $stage failed (see $OUT/$stage.log)"; exit 1; }
done
$PY -m scripts.api_cost $API_USAGE_LOG > $OUT/api_cost.json 2>/dev/null
stamp "PREDICTOR DEV EVALUATION DONE: $OUT/dev_report.json. STOP. Next, by hand: python -m scripts.train_max7 parity --output-dir $OUT"
