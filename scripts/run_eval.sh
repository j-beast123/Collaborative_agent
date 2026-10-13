#!/usr/bin/env bash
# Evaluation experiments on the held-out split (TEST for the final evaluation; trajectories come from `python -m
# scripts.run_grpo generate --run-id RUN --split test --allow-test --output FILE`): RQ1, RQ2 (states ...
#   REGISTRY=outputs/serl/<run_id>/iter3/rubrics/categories.json bash scripts/run_eval.sh [step ...]
set -euo pipefail

OUT=${OUT:-outputs/eval}
CONFIG=${CONFIG:-configs/eval/agentclinic.json}
RUNS=${RUNS:-}                    # rq1: JSON list of {dataset, system, checkpoint, iteration, trajectories, standalone}
BASE_TRAJ=${BASE_TRAJ:-}          # Base agent's EVAL trajectories
SERL_TRAJ=${SERL_TRAJ:-}          # SERL agent's EVAL trajectories
REFERENCE=${REFERENCE:-$BASE_TRAJ}  # shared histories of rq2_controlled and rq2_effectiveness
REGISTRY=${REGISTRY:-}            # rq3: category registry of the last iteration (categories.json)
PROBE_RESULTS=${PROBE_RESULTS:-}  # rq3, optional: fixed probe set results per iteration
TOP_KEYWORDS=${TOP_KEYWORDS:-8}
DONE_MARKERS=1
ALWAYS_RUN=plots                  # figures are redrawn from whatever results exist
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

. "$(dirname "$0")/steps.sh"
load_api_key
mkdir -p "$OUT"
ALL_STEPS=(rq1 rq2_states rq2_controlled rq2_effectiveness rq3 plots)

need() {  # need <variable name>...: each must name an existing file
  local name
  for name in "$@"; do
    [ -n "${!name}" ] && [ -f "${!name}" ] || { echo "step $step needs $name (a file); got '${!name}'"; return 1; }
  done
}

check_step() {
  local step=$1
  case $step in
    rq1) need CONFIG RUNS ;;
    rq2_states) need CONFIG BASE_TRAJ SERL_TRAJ ;;
    rq2_controlled) need CONFIG REFERENCE ;;
    rq2_effectiveness) need CONFIG BASE_TRAJ SERL_TRAJ ;;
    rq3) need REGISTRY ;;
  esac
}

run_step() {
  case $1 in
    rq1)
      $PY -m scripts.run_eval rq1 --config "$CONFIG" --runs "$RUNS" --output-dir "$OUT" || return 1 ;;
    rq2_states)
      $PY -m scripts.run_eval rq2-states --config "$CONFIG" --trajectories base="$BASE_TRAJ" \
        --trajectories serl="$SERL_TRAJ" --output-dir "$OUT" || return 1 ;;
    rq2_controlled)
      $PY -m scripts.run_eval rq2-controlled --config "$CONFIG" --reference "$REFERENCE" --output-dir "$OUT" || return 1 ;;
    rq2_effectiveness)
      $PY -m scripts.run_eval rq2-effectiveness --config "$CONFIG" --reference "$REFERENCE" \
        --trajectories base="$BASE_TRAJ" --trajectories serl="$SERL_TRAJ" --output-dir "$OUT" || return 1 ;;
    rq3)
      $PY -m scripts.run_eval rq3 --registry "$REGISTRY" --top-keywords "$TOP_KEYWORDS" --output-dir "$OUT" \
        ${PROBE_RESULTS:+--probe-results "$PROBE_RESULTS"} || return 1 ;;
    plots)
      $PY -m scripts.run_eval plots --output-dir "$OUT" || return 1 ;;
  esac
}

run_steps "$@"
