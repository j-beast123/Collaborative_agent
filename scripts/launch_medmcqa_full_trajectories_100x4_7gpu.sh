#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON_BIN:-/home/kai/venvs/qwen3/bin/python}"
INPUT="${INPUT:-data/interim/medmcqa/medicine_train_random300_seed20261004.jsonl}"
RUBRIC_SET="${RUBRIC_SET:-outputs/rubric_candidates/rubric_guidance_set_observable_v2_validated.json}"
OUT="${OUT:-outputs/medmcqa_full_trajectories_100x4_shards}"
ROUNDS="${ROUNDS:-7}"
STARTS=(0 15 30 44 58 72 86)
COUNTS=(15 15 14 14 14 14 14)

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set" >&2
  exit 1
fi
if [[ ! -s "$ROOT/$RUBRIC_SET" ]]; then
  echo "validated rubric set is missing: $RUBRIC_SET" >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$OUT/logs"
for gpu in 0 1 2 3 4 5 6; do
  shard_dir="$OUT/gpu_${gpu}"
  log="$OUT/logs/gpu_${gpu}.log"
  pid_file="$OUT/logs/gpu_${gpu}.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    echo "gpu_${gpu} already running with pid $(<"$pid_file")"
    continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u \
    scripts/run_medmcqa_full_trajectory_smoke.py \
    --input "$INPUT" \
    --start-index "${STARTS[$gpu]}" \
    --num-cases "${COUNTS[$gpu]}" \
    --rollouts-per-case 4 \
    --rounds "$ROUNDS" \
    --output "$shard_dir" \
    --rubric-set "$RUBRIC_SET" \
    --model /home/kai/models/Qwen3-8B \
    --doctor-model google/gemma-3-12b-it \
    --generation-temperature 0.7 \
    --top-p 0.9 \
    --rubric-temperature 0.5 \
    --rubric-weight 0.3 \
    --seed "$gpu" \
    >"$log" 2>&1 &
  echo $! >"$pid_file"
  echo "started gpu_${gpu} pid $(<"$pid_file")"
done
