#!/usr/bin/env bash
# Run one arm with launch_production.sh and resume it after a crash (e.g. a transient API failure such as Gemini
# finish_reason=error), with the same command and environment, until run_state reaches schedule.cases.
# Runtime only: research config, seeds and caches are untouched; resume uses the runner's own committed state.
# Cleanup before a resume: SIGTERM the arm's own process group (orphaned rollout workers) and the Doctor servers recorded in
# <output_dir>/doctor_servers/*.pid, then wait for the arm's GPUs to be free. Never SIGKILL; never touches other processes.
# usage: supervise_arm.sh <arm> [max_restarts=3]   (same environment variables as launch_production.sh)
set -u
cd "$(dirname "$0")/../.."
ARM="${1:?usage: supervise_arm.sh <arm> [max_restarts]}"; MAX="${2:-3}"; PY="${SHERPA_PYTHON:-python}"
case "$ARM" in dev_*) CFG=configs/experiments/dev_opt/${ARM#dev_}.json;; t2dev_*) CFG=configs/experiments/fixed_t2/dev/${ARM#t2dev_}.json;; t2smoke) CFG=configs/experiments/fixed_t2/smoke_r05.json;; t2rw_*) CFG=configs/experiments/fixed_t2/dev_reward/${ARM#t2rw_}.json;; t2dis_*) CFG=configs/experiments/fixed_t2/dis/${ARM#t2dis_}.json;; t2f120_*) CFG=configs/experiments/fixed_t2/final120/${ARM#t2f120_}.json;; t2bal_*) CFG=configs/experiments/fixed_t2/bal/${ARM#t2bal_}.json;; t2p_*) CFG=configs/experiments/fixed_t2/pilot/${ARM#t2p_}.json;; t2m_*) CFG=configs/experiments/fixed_t2/main120/${ARM#t2m_}.json;;
  t2_*) CFG=configs/experiments/fixed_t2/${ARM#t2_}.json;; *) echo "supervise_arm.sh supports dev_*, t2dev_* and t2_* arms" >&2; exit 2;; esac
read -r OUT CASES < <("$PY" -c "import sys;sys.path.insert(0,'.');from src.online_rl.tree import experiment as X;c=X.load_config('$CFG');print(c['output_dir'],c['schedule']['cases'])")
mkdir -p "$OUT"; LOG="$OUT/SUPERVISOR.log"
log() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
cursor() { [ -f "$OUT/run_state.json" ] && "$PY" -c "import json;print(json.load(open('$OUT/run_state.json'))['case_cursor'])" || echo 0; }
gpus_busy() { local busy=0; for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
  m=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits); [ "${m:-0}" -gt 1000 ] && busy=1; done; return $((1-busy)); }
restarts=0
while :; do
  log "launch $ARM (restart $restarts of at most $MAX), committed cursor $(cursor) of $CASES"
  setsid bash scripts/online_rl/launch_production.sh "$ARM" & CHILD=$!
  wait $CHILD; rc=$?
  now=$(cursor); log "runner exited rc=$rc at committed cursor $now of $CASES"
  for f in "$OUT"/doctor_servers/*.pid; do [ -f "$f" ] && kill -TERM "$(cat "$f")" 2>/dev/null; done
  kill -TERM -- -"$CHILD" 2>/dev/null
  for i in $(seq 1 120); do gpus_busy || break; sleep 5; done
  gpus_busy && { log "GPUs ${CUDA_VISIBLE_DEVICES} still busy after cleanup; stopping supervision"; exit 3; }
  if [ "$now" -ge "$CASES" ]; then log "complete: cursor $now"; exit 0; fi
  if [ "$restarts" -ge "$MAX" ]; then log "not complete after $MAX restarts; stopping supervision"; exit 1; fi
  restarts=$((restarts+1)); sleep 30
done
