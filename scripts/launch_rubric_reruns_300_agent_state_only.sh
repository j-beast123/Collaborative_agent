#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/home/kai/venvs/qwen3/bin/python}"
TRAJECTORIES="outputs/collaboration/gemma12b_qwen3_8b_300/trajectories_300.jsonl"
FAILURES="outputs/collaboration/gemma12b_qwen3_8b_300/trajectories_300_failures.jsonl"
CANDIDATES="${CANDIDATE_INPUT:-outputs/rubric_candidates/rubric_candidates_300_gpt56_sol_behavior_only_no_belief.jsonl}"
OUT="${OUT_DIR:-outputs/rubric_validation/gemma12b_qwen3_8b_300_gpt56_sol_agent_state_only}"
RERUN_AGENT_TURNS="${RERUN_AGENT_TURNS:-1}"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set" >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$OUT"

for gpu in 0 1 2 3 4 5 6; do
  shard_dir="$OUT/gpu_${gpu}"
  mkdir -p "$shard_dir"
  log="$shard_dir/validation.log"
  pid_file="$shard_dir/pid"

  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    echo "gpu_${gpu} already running with pid $(<"$pid_file")"
    continue
  fi

  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u \
    scripts/run_rubric_reruns_sharded.py \
    --trajectories "$TRAJECTORIES" \
    --failures "$FAILURES" \
    --candidate-input "$CANDIDATES" \
    --validation-output "$shard_dir/validations.jsonl" \
    --agent-model /home/kai/models/Qwen3-8B \
    --doctor-model google/gemma-3-12b-it \
    --reconstructor-model google/gemini-2.5-flash-lite \
    --doctor-prompt prompts/user_simulator.txt \
    --max-tokens 512 \
    --reconstruction-max-tokens 4096 \
    --num-reruns 3 \
    --min-successes 2 \
    --rerun-agent-turns "$RERUN_AGENT_TURNS" \
    --shard-index "$gpu" \
    --num-shards 7 \
    >"$log" 2>&1 &
  echo $! >"$pid_file"
  echo "started gpu_${gpu} pid $(<"$pid_file")"
done
