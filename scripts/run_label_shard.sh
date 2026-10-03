#!/usr/bin/env bash
# One shard of the full Effectiveness label run (scripts/run_agentclinic_turns.py --shard), on one GPU, in tmux:
# Candidates and labels of the shard's histories go to <labels_dir>/shards/nNN/shardII; written rows are kept ...
#   SHARD=3 SHARDS=14 GPU=4 MODEL_PATH_MAP=... bash scripts/run_label_shard.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY=${PY:-../.venv-qwen3/bin/python}
CONFIG=${CONFIG:-configs/effectiveness/v10_qwen4b_gemma12b_max7.json}
OUT=${OUT:-outputs/effectiveness/v10_qwen4b_gemma12b_max7/effectiveness_oof_v1}
SHARD=${SHARD:?set SHARD (0-based)}; SHARDS=${SHARDS:?set SHARDS}; GPU=${GPU:?set GPU}
LOGS=$OUT/shard_logs/n$(printf %02d "$SHARDS"); mkdir -p "$LOGS"
NAME=shard$(printf %02d "$SHARD")
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=../.pydeps/grpo:../.pydeps/grpc
set -a; . ${ENV_FILE:-../.env}; set +a
export API_USAGE_LOG=$LOGS/api_usage_$NAME.jsonl API_USAGE_STAGE=effectiveness_labels
stamp() { echo "[$(date '+%F %T')] $NAME (GPU $GPU): $*"; }
for attempt in 1 2 3 4 5; do
  stamp "labels (attempt $attempt)"
  $PY -m scripts.run_agentclinic_turns --config "$CONFIG" --scope full --shard "$SHARD/$SHARDS" candidates labels > "$LOGS/${NAME}_stage.json" 2>> "$LOGS/$NAME.err"
  stamp "attempt $attempt ended with status $?"
  if $PY -m scripts.label_shards status --config "$CONFIG" --shards "$SHARDS" --shard "$SHARD" --require-complete > "$LOGS/${NAME}_status.json" 2>> "$LOGS/$NAME.err"; then
    stamp "COMPLETE"; exit 0
  fi
  sleep 30
done
stamp "STOP: not complete after 5 attempts (see $LOGS/${NAME}_status.json)"; exit 1
