#!/usr/bin/env bash
# End-to-end check of the main protocol (max 7 turns, early stopping) on TRAIN cases, without any weight update
# (scripts/run_e2e_max7.py).
#   GPU=7 CASES=1 bash scripts/run_e2e_max7.sh
set -uo pipefail

CONFIG=${CONFIG:-configs/serl_agentclinic_max7.json}
CASES=${CASES:-1}
GPU=${GPU:?set GPU to the physical GPU of the policy}
PY=${PY:-python}
ENV_FILE=${ENV_FILE:-.env}
DEPS=${DEPS:-../.pydeps}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
for d in "$DEPS/grpo" "$DEPS/grpc"; do [ -d "$d" ] && export PYTHONPATH="$(cd "$d" && pwd)${PYTHONPATH:+:$PYTHONPATH}"; done
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set"; exit 1; }
stamp() { echo "[$(date '+%F %T')] $*"; }
PREREQ=()
[ "$CASES" != 1 ] && PREREQ=(--prerequisite-run "${PREREQUISITE_RUN:?CASES=$CASES needs PREREQUISITE_RUN}")
OUT=$(mktemp); ID_FILE=$(mktemp)
stamp "end-to-end check, $CASES case(s), policy on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU $PY -m scripts.run_e2e_max7 --config "$CONFIG" --cases "$CASES" ${PREREQ[@]+"${PREREQ[@]}"} \
  --write-run-id "$ID_FILE" > "$OUT" 2>&1; STATUS=$?
grep -v "Loading checkpoint\|generation flags" "$OUT" | tail -40
RUN=$(cat "$ID_FILE"); [ -n "$RUN" ] && cp "$OUT" "outputs/serl/$RUN/run_stage.log"
[ -n "${RUN_ID_OUT:-}" ] && [ -n "${RUN:-}" ] && echo "$RUN" > "$RUN_ID_OUT"
[ "$STATUS" = 0 ] && stamp "done: checks passed (outputs/serl/$RUN)" || stamp "FAILED: checks did not pass (outputs/serl/${RUN:-?})"
exit "$STATUS"
