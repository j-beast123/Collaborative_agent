#!/bin/bash
# FRESH main 100-case runs (case 1 -> 100) of the FROZEN tree v2 protocol (docs/FROZEN_TREE_V2.md): bounded_witness_v1 +
# feedback behavior_observer_v4 + equal-weight R_A. Outcome-only and Rubric-only from case 0; v1 runs are not reused.
# Execution only: M2 process workers (one model copy per GPU), single-flight cache, parallel strata.
# Outcome-only: GPUs 0-2 (primary GPU0, Doctors :11450-11452); checkpoint-0 VAL50 under the v2 protocol runs first.
# Rubric-only:  GPUs 3-7 (primary GPU3, Doctors :11453-11457).
cd "$(dirname "$0")/../.."                                  # repository root
export SHERPA_ENV_FILE=${SHERPA_ENV_FILE:-.env} HF_HUB_OFFLINE=1      # .env holds OPENROUTER_API_KEY (never committed)
export SHERPA_API_WORKERS=8 SHERPA_MEMBER_WORKERS=3 SHERPA_AGENT_PROCESS_WORKERS=1 SHERPA_PARALLEL_STRATA=1
PY=${PYTHON:-python}; O=outputs/sherpa_tree_v2_witness; C=configs/online_rl/tree_v2_witness.json
mkdir -p $O/checkpoint0_val50 $O/outcome_only $O/rubric_only
( export CUDA_VISIBLE_DEVICES=0,1,2 SHERPA_AGENT_REPLICAS=3 SHERPA_ARM=outcome_only SHERPA_DOCTOR_ENDPOINTS=http://127.0.0.1:11450,http://127.0.0.1:11451,http://127.0.0.1:11452
  echo "=== CHECKPOINT-0 VAL50 START $(date -u +%FT%TZ)"
  SHERPA_EXEC_LOG=$O/checkpoint0_val50/exec_calls.jsonl $PY scripts/online_rl/eval_checkpoint.py --config $C --initial --out $O/checkpoint0_val50 --launch
  echo "=== OUTCOME-ONLY START $(date -u +%FT%TZ)"
  SHERPA_EXEC_LOG=$O/outcome_only/exec_calls.jsonl $PY scripts/online_rl/run_tree.py --config $C --condition outcome_only --launch --max-cases 100
  echo "=== OUTCOME-ONLY END rc=$? $(date -u +%FT%TZ)" ) > $O/outcome_only_100.log 2>&1 &
( export CUDA_VISIBLE_DEVICES=3,4,5,6,7 SHERPA_AGENT_REPLICAS=5 SHERPA_ARM=rubric_only SHERPA_DOCTOR_ENDPOINTS=http://127.0.0.1:11453,http://127.0.0.1:11454,http://127.0.0.1:11455,http://127.0.0.1:11456,http://127.0.0.1:11457
  echo "=== RUBRIC-ONLY START $(date -u +%FT%TZ)"
  SHERPA_EXEC_LOG=$O/rubric_only/exec_calls.jsonl $PY scripts/online_rl/run_tree.py --config $C --condition rubric_only --launch --max-cases 100
  echo "=== RUBRIC-ONLY END rc=$? $(date -u +%FT%TZ)" ) > $O/rubric_only_100.log 2>&1 &
wait
