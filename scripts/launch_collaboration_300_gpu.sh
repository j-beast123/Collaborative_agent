#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/opt/anonymous/venvs/qwen3/bin/python"
INPUT="data/interim/medmcqa/medicine_sample_500_label_corrected.jsonl"
OUTPUT_ROOT="outputs/collaboration/gemma12b_qwen3_8b_300"

if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "OPENROUTER_API_KEY is not set" >&2
  exit 1
fi

# 300 input rows split without overlap across GPUs 0 through 6.
starts=(0 43 86 129 172 215 258)
counts=(43 43 43 43 43 43 42)

for gpu in {0..6}; do
  output_dir="$ROOT/$OUTPUT_ROOT/gpu_${gpu}"
  mkdir -p "$output_dir/logs"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" -u \
    "$ROOT/scripts/run_collaboration_pilot.py" \
    --input "$INPUT" \
    --output "$OUTPUT_ROOT/gpu_${gpu}/trajectories.jsonl" \
    --model /opt/anonymous/models/Qwen3-8B \
    --doctor-model google/gemma-3-12b-it \
    --reconstructor-model google/gemini-2.5-flash-lite \
    --max-cases "${counts[$gpu]}" \
    --start-index "${starts[$gpu]}" \
    --rounds 7 \
    --max-tokens 512 \
    --shard-index 0 \
    --num-shards 1 \
    --final-decision-role doctor \
    > "$output_dir/logs/trajectory.log" 2>&1 &
  echo "started gpu=${gpu} start=${starts[$gpu]} count=${counts[$gpu]} pid=$!"
done
