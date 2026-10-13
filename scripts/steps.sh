# Shared by the pipeline scripts (source it; not run on its own).
# The caller defines run_step <name> and ALL_STEPS (and DEFAULT_STEPS if not every step runs by default),
# then calls: run_steps "$@"
STEP_ATTEMPTS=${STEP_ATTEMPTS:-3}
PY=${PY:-python}
ENV_FILE=${ENV_FILE:-.env}

stamp() { echo "[$(date '+%F %T')] $*"; }

# OPENROUTER_API_KEY from the environment or ENV_FILE; never printed.
load_api_key() {
  if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
  [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY is not set"; exit 1; }
}

# Runs the given steps (default: DEFAULT_STEPS, else ALL_STEPS) in order.
run_steps() {
  local steps=("$@") step attempt
  export API_USAGE_LOG=${API_USAGE_LOG:-$OUT/api_usage.jsonl}
  [ ${#steps[@]} -gt 0 ] || { if [ -n "${DEFAULT_STEPS+x}" ]; then steps=("${DEFAULT_STEPS[@]}"); else steps=("${ALL_STEPS[@]}"); fi; }
  for step in "${steps[@]}"; do
    [[ " ${ALL_STEPS[*]} " == *" $step "* ]] || { echo "unknown step $step (steps: ${ALL_STEPS[*]})"; exit 2; }
  done
  if declare -F check_step > /dev/null; then
    for step in "${steps[@]}"; do
      [ -f "$OUT/.done_$step" ] || check_step "$step" || exit 2
    done
  fi
  for step in "${steps[@]}"; do
    if [ "${DONE_MARKERS:-0}" = 1 ] && [ -f "$OUT/.done_$step" ]; then stamp "step $step: already done, skipped"; continue; fi
    export API_USAGE_STAGE=$step
    for attempt in $(seq 1 "$STEP_ATTEMPTS"); do
      stamp "step $step (attempt $attempt/$STEP_ATTEMPTS)"
      if run_step "$step"; then break; fi
      [ "$attempt" -lt "$STEP_ATTEMPTS" ] || { stamp "FAILED: step $step"; exit 1; }
      sleep "${RETRY_SLEEP:-60}"
    done
    [ "${DONE_MARKERS:-0}" != 1 ] || [[ " ${ALWAYS_RUN:-} " == *" $step "* ]] || touch "$OUT/.done_$step"
  done
  stamp "done: ${steps[*]}"
  [ ! -f "$API_USAGE_LOG" ] || $PY -m scripts.api_cost "$API_USAGE_LOG" > "$OUT/api_cost.json" || true
}
