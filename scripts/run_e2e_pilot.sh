#!/usr/bin/env bash
# End-to-end pipeline check on 1 or 2 TRAIN cases, without any weight update (scripts/run_e2e_pilot.py).
#   GPU=7 CASES=1 bash scripts/run_e2e_pilot.sh
set -uo pipefail

CONFIG=${CONFIG:-configs/serl_audit_pilot.json}
CASES=${CASES:-1}
GPU=${GPU:?set GPU to the physical GPU of the policy}
SERVICE_GPU=${SERVICE_GPU:-$GPU}
PY=${PY:-python}
ENV_FILE=${ENV_FILE:-.env}
DEPS=${DEPS:-../.pydeps}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
for d in "$DEPS/grpo" "$DEPS/grpc"; do [ -d "$d" ] && export PYTHONPATH="$(cd "$d" && pwd)${PYTHONPATH:+:$PYTHONPATH}"; done
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set"; exit 1; }

stamp() { echo "[$(date '+%F %T')] $*"; }
PLACEMENT=$($PY -c "from src.common.config import read_config; print(read_config('$CONFIG')['devices']['placement'])") || exit 1
PREREQ=()
[ "$CASES" = 2 ] && PREREQ=(--prerequisite-run "${PREREQUISITE_RUN:?CASES=2 needs PREREQUISITE_RUN (a passed 1-case run)}")
SERVICE_PID=""
stop_service() {
  if [ -n "$SERVICE_PID" ] && kill -0 "$SERVICE_PID" 2>/dev/null; then
    kill -TERM "$SERVICE_PID"; wait "$SERVICE_PID" 2>/dev/null; stamp "service stopped"
  fi
}
trap stop_service EXIT

start_service() {  # start_service <log dir>: returns when the service answers, fails when it exits
  mkdir -p "$1"
  CUDA_VISIBLE_DEVICES=$SERVICE_GPU $PY -m scripts.serve_effectiveness --config "$CONFIG" --log-dir "$1" > "$1/service.log" 2>&1 &
  SERVICE_PID=$!
  for _ in $(seq 1 120); do
    grep -q '"status": "SERVING"' "$1/service.log" 2>/dev/null && { stamp "service is serving (pid $SERVICE_PID, GPU $SERVICE_GPU)"; return 0; }
    kill -0 "$SERVICE_PID" 2>/dev/null || { stamp "FAILED: the service exited"; tail -5 "$1/service.log"; return 1; }
    sleep 5
  done
  stamp "FAILED: the service did not start within 10 minutes"; return 1
}

if [ "$PLACEMENT" = sequential ]; then
  stamp "rollout ($CASES case(s), policy on GPU $GPU)"
  OUT=$(mktemp); ID_FILE=$(mktemp)
  CUDA_VISIBLE_DEVICES=$GPU $PY -m scripts.run_e2e_pilot rollout --config "$CONFIG" --cases "$CASES" ${PREREQ[@]+"${PREREQ[@]}"} \
    --write-run-id "$ID_FILE" > "$OUT" 2>&1 || { stamp "FAILED: rollout"; tail -30 "$OUT"; exit 1; }
  RUN=$(cat "$ID_FILE"); [ -n "$RUN" ] || { stamp "FAILED: no run id"; exit 1; }
  cp "$OUT" "outputs/serl/$RUN/rollout_stage.log"
  stamp "run id $RUN"
  start_service "outputs/serl/$RUN/grpc_logs" || exit 1
  stamp "parity"
  $PY -m scripts.run_e2e_pilot parity --config "$CONFIG" --run-id "$RUN" > "outputs/serl/$RUN/parity_stage.log" 2>&1 || { stamp "FAILED: parity"; tail -20 "outputs/serl/$RUN/parity_stage.log"; exit 1; }
  stamp "reward"
  $PY -m scripts.run_e2e_pilot reward --config "$CONFIG" --run-id "$RUN" > "outputs/serl/$RUN/reward_stage.log" 2>&1; STATUS=$?
  tail -12 "outputs/serl/$RUN/reward_stage.log"
else
  LOGS=$(mktemp -d)
  start_service "$LOGS" || exit 1
  stamp "rollout + reward ($CASES case(s), policy on GPU $GPU, service on GPU $SERVICE_GPU)"
  OUT=$(mktemp); ID_FILE=$(mktemp)
  CUDA_VISIBLE_DEVICES=$GPU $PY -m scripts.run_e2e_pilot run --config "$CONFIG" --cases "$CASES" ${PREREQ[@]+"${PREREQ[@]}"} \
    --write-run-id "$ID_FILE" --set "pilot.service_request_log=\"$LOGS/effectiveness_requests.jsonl\"" > "$OUT" 2>&1; STATUS=$?
  tail -12 "$OUT"
  RUN=$(cat "$ID_FILE"); [ -n "$RUN" ] && cp "$LOGS/service.log" "outputs/serl/$RUN/grpc_logs/service.log" && cp "$OUT" "outputs/serl/$RUN/run_stage.log"
fi
stop_service
[ -n "${RUN_ID_OUT:-}" ] && [ -n "${RUN:-}" ] && echo "$RUN" > "$RUN_ID_OUT"
[ "$STATUS" = 0 ] && stamp "done: checks passed (outputs/serl/$RUN)" || stamp "FAILED: checks did not pass (outputs/serl/${RUN:-?})"
exit "$STATUS"
