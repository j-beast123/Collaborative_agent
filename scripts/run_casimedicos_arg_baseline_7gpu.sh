#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${OPENROUTER_API_KEY:?OPENROUTER_API_KEY must be set}"

INPUT="${INPUT:-data/interim/casimedicos_arg/en_test.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/collaboration/casimedicos_arg_en_test_baseline}"
MODEL="${MODEL:-/opt/anonymous/models/Qwen3-8B}"
PYTHON="${PYTHON:-/opt/anonymous/miniconda3/envs/qwen-vllm/bin/python}"
DOCTOR_MODEL="${DOCTOR_MODEL:-google/gemma-3-12b-it}"
RECONSTRUCTOR_MODEL="${RECONSTRUCTOR_MODEL:-google/gemini-2.5-flash-lite}"
CASES="${CASES:-125}"
ROUNDS="${ROUNDS:-7}"
RESUME_FLAG=()
if [[ "${RESUME:-0}" == "1" ]]; then
  RESUME_FLAG=(--resume)
fi

for gpu in $(seq 0 6); do
  shard="$OUTPUT_ROOT/gpu_$gpu"
  mkdir -p "$shard/logs"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" scripts/run_collaboration_pilot.py \
    --input "$INPUT" \
    --output "$shard/trajectories.jsonl" \
    --model "$MODEL" \
    --doctor-model "$DOCTOR_MODEL" \
    --reconstructor-model "$RECONSTRUCTOR_MODEL" \
    --max-cases "$CASES" \
    --start-index 0 \
    --rounds "$ROUNDS" \
    --max-tokens 512 \
    --final-decision-role doctor \
    --disable-belief \
    --shard-index "$gpu" \
    --num-shards 7 \
    "${RESUME_FLAG[@]}" \
    > "$shard/logs/trajectory.log" 2>&1 &
  echo $! > "$shard/pid"
  echo "GPU$gpu PID $(cat "$shard/pid")"
done

echo "Started $CASES CasiMedicos-Arg test cases across GPU 0-6."
echo "Progress: watch -n 5 'for g in {0..6}; do printf \"GPU%s \" \"\$g\"; grep -E \"^[0-9]+/[0-9]+\" \"$OUTPUT_ROOT/gpu_\$g/logs/trajectory.log\" 2>/dev/null | tail -1; done'"
