#!/bin/bash
# FRESH Rubric-only 100-case run (case 1 -> 100) of the frozen rubric protocol v2.1 (configs/online_rl/tree_v21_witness.json:
# bounded_witness_v1 + feedback behavior_observer_v4 + equal-weight R_A + rubric.source_consistency = source_locus_v1).
# Outcome-only keeps its v2 run (no rubric path). GPUs 3-7: Agent primary GPU3 + M2 process workers on GPUs 4-7, co-located
# Doctors :11453-11457. Checkpoint-0 VAL50 under the v2.1 protocol runs first. Execution only: single-flight cache, parallel strata.
cd "$(dirname "$0")/../.."                                  # repository root
export SHERPA_ENV_FILE=${SHERPA_ENV_FILE:-.env} HF_HUB_OFFLINE=1      # .env holds OPENROUTER_API_KEY (never committed)
export SHERPA_API_WORKERS=8 SHERPA_MEMBER_WORKERS=3 SHERPA_AGENT_PROCESS_WORKERS=1 SHERPA_PARALLEL_STRATA=1
PY=${PYTHON:-python}; O=outputs/sherpa_tree_v21_witness; C=configs/online_rl/tree_v21_witness.json
mkdir -p $O/checkpoint0_val50 $O/rubric_only
( export CUDA_VISIBLE_DEVICES=3,4,5,6,7 SHERPA_AGENT_REPLICAS=5 SHERPA_ARM=rubric_only SHERPA_DOCTOR_ENDPOINTS=http://127.0.0.1:11453,http://127.0.0.1:11454,http://127.0.0.1:11455,http://127.0.0.1:11456,http://127.0.0.1:11457
  echo "=== CHECKPOINT-0 VAL50 START $(date -u +%FT%TZ)"
  SHERPA_EXEC_LOG=$O/checkpoint0_val50/exec_calls.jsonl $PY scripts/online_rl/eval_checkpoint.py --config $C --initial --out $O/checkpoint0_val50 --launch
  echo "=== RUBRIC-ONLY START $(date -u +%FT%TZ)"
  SHERPA_EXEC_LOG=$O/rubric_only/exec_calls.jsonl $PY scripts/online_rl/run_tree.py --config $C --condition rubric_only --launch --max-cases 100
  echo "=== RUBRIC-ONLY END rc=$? $(date -u +%FT%TZ)" ) > $O/rubric_only_100.log 2>&1 &
wait
