#!/usr/bin/env bash
# One SERL iteration on the AgentClinic protocol: theta_i's failures -> rubric set R_i -> GRPO -> theta_(i+1) ->
# theta_(i+1)'s trajectories.
#   ITERATION=0 bash scripts/run_serl.sh [step ...]                 # creates a new run and prints its RUN_ID
#   python -m scripts.run_grpo check --config configs/serl_agentclinic.json --iteration 0
set -euo pipefail

CONFIG=${CONFIG:-configs/serl_agentclinic.json}
ITERATION=${ITERATION:?set ITERATION (0 for the first SERL iteration)}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/serl}       # must be the config's output_root
export API_USAGE_ITERATION=$ITERATION
GRPO_DEPS=${GRPO_DEPS:-../.pydeps/grpo}
GRPC_DEPS=${GRPC_DEPS:-../.pydeps/grpc}
DONE_MARKERS=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
for deps in "$GRPO_DEPS" "$GRPC_DEPS"; do
  if [ -d "$deps" ]; then export PYTHONPATH="$(cd "$deps" && pwd)${PYTHONPATH:+:$PYTHONPATH}"; fi
done

. "$(dirname "$0")/steps.sh"
load_api_key
SET=()
[ -n "${FIXED_RUBRIC_SET:-}" ] && SET+=(--set "rubric_registry.fixed_rubric_set=\"$FIXED_RUBRIC_SET\"")
# INITIAL_RUBRICS=outputs/serl/<run>/iter0/rubrics: iteration 0 takes its rubric step from that run (the same base
# collaborations), so that every condition of an experiment starts from the same R_0 / the same raw candidates.
[ -n "${INITIAL_RUBRICS:-}" ] && SET+=(--set "rubric_registry.initial_rubrics=\"$INITIAL_RUBRICS\"")
[ -n "${SEED:-}" ] && SET+=(--set "seed=$SEED" --set "grpo.seed=$SEED")
# A run is one directory $OUTPUT_ROOT/$RUN_ID. Iteration 0 without RUN_ID creates a new run; every later
# call names the run it continues. A run directory is never overwritten.
if [ -z "${RUN_ID:-}" ]; then
  [ "$ITERATION" = 0 ] || { echo "set RUN_ID (the run that iteration $ITERATION continues)"; exit 2; }
  RUN_ID=$($PY -m scripts.run_grpo init --config "$CONFIG" ${SET[@]+"${SET[@]}"} | $PY -c "import json,sys; print(json.load(sys.stdin)['run_id'])") || exit 1
  echo "new run: $RUN_ID"
fi
RUN_DIR=$OUTPUT_ROOT/$RUN_ID
OUT=$RUN_DIR/iter$ITERATION
mkdir -p "$OUT"
ALL_STEPS=(base rubrics train report generate)
if [ "$ITERATION" = 0 ]; then
  TRAJECTORIES=$RUN_DIR/base/trajectories.jsonl
  PREVIOUS=()
else
  TRAJECTORIES=$RUN_DIR/iter$((ITERATION - 1))/trajectories.jsonl
  PREVIOUS=(--previous-categories "$RUN_DIR/iter$((ITERATION - 1))/rubrics/categories.json"
            --previous-rubric-set "$RUN_DIR/iter$((ITERATION - 1))/rubrics/rubric_set.json")
  DEFAULT_STEPS=(rubrics train report generate)
fi

run_step() {
  case $1 in
    base) $PY -m scripts.run_grpo base --config "$CONFIG" --run-id "$RUN_ID" ${SET[@]+"${SET[@]}"} || return 1 ;;
    rubrics) $PY -m scripts.build_rubrics --config "$CONFIG" --trajectories "$TRAJECTORIES" \
               --run-id "$RUN_ID" --output-dir "$OUT/rubrics" --iteration "$ITERATION" ${SET[@]+"${SET[@]}"} \
               ${PREVIOUS[@]+"${PREVIOUS[@]}"} || return 1 ;;
    train) $PY -m scripts.run_grpo check --config "$CONFIG" --run-id "$RUN_ID" --iteration "$ITERATION" ${SET[@]+"${SET[@]}"} || return 1
           $PY -m scripts.run_grpo train --config "$CONFIG" --run-id "$RUN_ID" --iteration "$ITERATION" --resume ${SET[@]+"${SET[@]}"} || return 1 ;;
    report) $PY -m scripts.reward_report --log "$OUT/reward_log.jsonl" --config "$CONFIG" --output "$OUT/reward_report.json" \
              --rubric-set "${FIXED_RUBRIC_SET:-$OUT/rubrics/rubric_set.json}" --selection "$OUT/rubrics/active_selection.json" > /dev/null || return 1 ;;
    generate) $PY -m scripts.run_grpo generate --config "$CONFIG" --run-id "$RUN_ID" --iteration "$ITERATION" ${SET[@]+"${SET[@]}"} || return 1 ;;
  esac
}

run_steps "$@"
