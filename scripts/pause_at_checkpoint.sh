#!/usr/bin/env bash
# Stop a running GRPO iteration at a completed step, for a move to another machine: Waits for the trainer's next
# checkpoint (policy adapter, optimizer, scheduler, global step, RNG state; written every grpo.save_steps ...
#   bash scripts/pause_at_checkpoint.sh <tmux session> <run dir> <iteration>
set -euo pipefail
SESSION=$1; RUN=$2; ITER=$3
T=$RUN/iter$ITER/trainer
last() { ls -d $T/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1; }
START=$(last); START=${START:-0}
echo "[$(date '+%F %T')] last checkpoint: $START; waiting for the next one"
while :; do
  NOW=$(last); NOW=${NOW:-0}
  if [ "$NOW" -gt "$START" ] && [ -f "$T/checkpoint-$NOW/trainer_state.json" ] && [ -f "$T/checkpoint-$NOW/optimizer.pt" ]; then break; fi
  tmux has-session -t "$SESSION" 2>/dev/null || { echo "session $SESSION is gone"; exit 1; }
  sleep 5
done
sleep 5
tmux kill-session -t "$SESSION"
echo "[$(date '+%F %T')] paused: session $SESSION stopped after checkpoint-$NOW ($T/checkpoint-$NOW)"
