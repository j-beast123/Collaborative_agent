#!/usr/bin/env bash
# After the label shards (scripts/run_label_shard.sh): wait for them, then check every shard complete -> merge
# into the label directory -> report stage (split files, leakage audit, integrity) -> label audit and freeze -> ...
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY=${PY:-../.venv-qwen3/bin/python}
CONFIG=${CONFIG:-configs/effectiveness/v10_qwen4b_gemma12b_max7.json}
OUT=${OUT:-outputs/effectiveness/v10_qwen4b_gemma12b_max7/effectiveness_oof_v1}
SHARDS=${SHARDS:?set SHARDS}
FOLD_GPUS=(${FOLD_GPUS:-1 2 3})          # one GPU per CV fold; the first one also does the final fit and the DEV evaluation
SESSION_PREFIX=${SESSION_PREFIX:-lab}    # tmux sessions of the shards
LOGS=$OUT/shard_logs/n$(printf %02d "$SHARDS")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=../.pydeps/grpo:../.pydeps/grpc
set -a; . ${ENV_FILE:-../.env}; set +a
stamp() { echo "[$(date '+%F %T')] $*"; }

if [ ! -f $OUT/labels.freeze.json ]; then
  if [ ! -f $OUT/labels/shard_merge_manifest.json ]; then
    while tmux ls 2>/dev/null | grep -q "^$SESSION_PREFIX[0-9]"; do sleep 60; done
    stamp "label shards ended"
    CUDA_VISIBLE_DEVICES= $PY -m scripts.label_shards status --config $CONFIG --shards $SHARDS --require-complete > $LOGS/status_before_merge.json 2>> $LOGS/merge.err \
      || { stamp "STOP: a shard is not complete (see $LOGS/status_before_merge.json)"; exit 1; }
    CUDA_VISIBLE_DEVICES= $PY -m scripts.label_shards merge --config $CONFIG --shards $SHARDS > $LOGS/merge.json 2>> $LOGS/merge.err \
      || { stamp "STOP: the merge was refused (see $LOGS/merge.err)"; exit 1; }
    cat $LOGS/api_usage_shard*.jsonl >> $OUT/api_usage.jsonl
    stamp "shards merged into $OUT/labels"
  fi
  CUDA_VISIBLE_DEVICES= $PY -m scripts.run_agentclinic_turns --config $CONFIG --scope full report > $OUT/labels_full_report.json 2>> $OUT/labels_full.err \
    || { stamp "STOP: the label report did not pass (see $OUT/labels_full_report.json)"; exit 1; }
  CUDA_VISIBLE_DEVICES= $PY -m scripts.train_max7 audit --output-dir $OUT >> $OUT/audit.log 2>&1 || { stamp "STOP: the label audit failed (see $OUT/audit.log)"; exit 1; }
  stamp "labels audited and frozen"
fi

if [ ! -f $OUT/cv_summary.json ]; then
  for gpu in "${FOLD_GPUS[@]}"; do
    while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu)" -ge 3000 ]; do sleep 30; done
  done
  stamp "predictor: cv folds on GPUs ${FOLD_GPUS[*]}"
  pids=()
  for fold in 0 1 2; do
    CUDA_VISIBLE_DEVICES=${FOLD_GPUS[$fold]} $PY -m scripts.train_max7 cv --fold $fold --output-dir $OUT >> $OUT/cv_fold$fold.log 2>&1 &
    pids+=($!)
  done
  failed=0
  for pid in "${pids[@]}"; do wait $pid || failed=1; done
  [ $failed = 0 ] || { stamp "STOP: a CV fold failed (see $OUT/cv_fold*.log)"; exit 1; }
  stamp "predictor: cv selection (epoch, calibration) on the TRAIN out-of-fold logits"
  CUDA_VISIBLE_DEVICES=${FOLD_GPUS[0]} $PY -m scripts.train_max7 cv --output-dir $OUT >> $OUT/cv.log 2>&1 || { stamp "STOP: predictor stage cv failed (see $OUT/cv.log)"; exit 1; }
fi
export CUDA_VISIBLE_DEVICES=${FOLD_GPUS[0]}
for stage in final evaluate; do
  if [ "$stage" = evaluate ] && [ -f $OUT/dev_report.json ]; then continue; fi
  stamp "predictor: $stage"
  $PY -m scripts.train_max7 $stage --output-dir $OUT >> $OUT/$stage.log 2>&1 || { stamp "STOP: predictor stage $stage failed (see $OUT/$stage.log)"; exit 1; }
done
$PY -m scripts.api_cost $OUT/api_usage.jsonl > $OUT/api_cost.json 2>/dev/null
stamp "PREDICTOR DEV EVALUATION DONE: $OUT/dev_report.json. STOP. Next, by hand: python -m scripts.train_max7 parity --output-dir $OUT"
