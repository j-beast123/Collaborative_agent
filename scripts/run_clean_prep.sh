#!/usr/bin/env bash
# Clean TRAIN30 rerun, preparation (no GRPO): from the finished human adjudication to the full-reward E2E.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); cd "$REPO"
PY=${PY:-../.venv-qwen3/bin/python}
A=outputs/audit/equivalence_adjudication_20261002
ENV=configs/effectiveness/v13_qwen4b_gemma12b_max7_train30_clean_majority.json
PARENT=configs/effectiveness/v11_qwen4b_gemma12b_max7_train30_labels_v2.json
MAIN=configs/serl_agentclinic_qwen4b_max7_train30_clean.json
BASE=outputs/effectiveness/v13_qwen4b_gemma12b_max7_train30_clean_majority
OUT=$BASE/effectiveness_oof_v1
M=outputs/main_experiments_train30_4b_clean; L=$M/launch
GPUS=(${PREP_GPUS:?set PREP_GPUS to two or three free GPUs})
G1=${GPUS[0]}; G2=${GPUS[1]}; G3=${GPUS[2]:-}
export MODEL_PATH_MAP=${MODEL_PATH_MAP:-} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_LIBCUDA_PATH=../.triton_libcuda
export PYTHONPATH=../.pydeps/grpo:../.pydeps/grpc
set -a; . ${ENV_FILE:-../.env}; set +a
mkdir -p $M $L
export API_USAGE_LOG=$M/prep_api_usage.jsonl
stamp() { echo "[$(date '+%F %T')] $*"; }
die() { stamp "STOP: $*"; exit 1; }
free() { while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1)" -ge 1500 ]; do sleep 30; done; }

# No human adjudication (user decision of 2026-10-02, option B): the mapping of human decisions is empty and the judge
# (gpt-4o-mini with the revised prompt) is frozen in the environment config; Sol is an independent audit only.
MAPPING=$M/no_human_decisions.jsonl
[ -f $MAPPING ] || : > $MAPPING
grep -q '"diagnosis_judge_frozen"' $ENV || die "the environment config names no frozen judge"
if [ ! -f $BASE/protocol.frozen.json ]; then
  [ -d $BASE ] || { API_USAGE_STAGE=clean_relabel CUDA_VISIBLE_DEVICES= $PY -m scripts.derive_clean_environment --config $ENV --parent $PARENT --human-mapping $MAPPING --decisions outputs/audit/equivalence_adjudication_20261002/majority_v1/majority_mapping_v1.jsonl > $M/derive.out 2>&1 || die "derivation failed (see $M/derive.out)"; }
  CUDA_VISIBLE_DEVICES= $PY -m scripts.run_base_collaborations --config $ENV report freeze > $M/reach_freeze.out 2>&1 || die "reach report / protocol freeze failed"
  stamp "clean environment frozen: $(grep -o '"protocol_identity": "[0-9a-f]*"' $M/reach_freeze.out | tail -1)"
fi
if [ ! -f $OUT/labels.freeze.json ]; then
  mkdir -p $OUT
  CUDA_VISIBLE_DEVICES= $PY -m scripts.run_agentclinic_turns --config $ENV --scope full report > $OUT/labels_full_report.json 2>> $OUT/labels_full.err || die "label report did not pass"
  [ -f $OUT/workflow.json ] || $PY - <<PY
import json
json.dump({"compliance_role_resolved": True, "environment_config": "$ENV", "main_config": "$MAIN",
           "derived_from": "v11 environment, every dialogue reused, outcomes from the clean diagnosis-equivalence decisions", "e2e_runs": {}}, open("$OUT/workflow.json", "w"), indent=2)
PY
  CUDA_VISIBLE_DEVICES= $PY -m scripts.train_max7 audit --output-dir $OUT >> $OUT/audit.log 2>&1 || die "label audit failed (see $OUT/audit.log)"
  stamp "labels audited and frozen"
fi
if [ ! -f $OUT/cv_summary.json ]; then
  stamp "predictor: cv folds on GPUs ${GPUS[*]}"
  fold() { free $2; CUDA_VISIBLE_DEVICES=$2 $PY -m scripts.train_max7 cv --fold $1 --output-dir $OUT >> $OUT/cv_fold$1.log 2>&1; }
  if [ -n "$G3" ]; then fold 0 $G1 & fold 1 $G2 & fold 2 $G3 & wait
  else fold 0 $G1 & fold 1 $G2 & wait; fold 2 $G1; fi
  for f in 0 1 2; do [ -f $OUT/cv/fold${f}_logits.json ] || die "CV fold $f did not finish (see $OUT/cv_fold$f.log)"; done
  CUDA_VISIBLE_DEVICES=$G1 $PY -m scripts.train_max7 cv --output-dir $OUT >> $OUT/cv.log 2>&1 || die "cv selection failed"
fi
for stage in final evaluate; do
  [ "$stage" = final ] && [ -f $OUT/checkpoint.freeze.json ] && continue
  [ "$stage" = evaluate ] && [ -f $OUT/dev_report.json ] && continue
  stamp "predictor: $stage"; free $G1
  CUDA_VISIBLE_DEVICES=$G1 $PY -m scripts.train_max7 $stage --output-dir $OUT >> $OUT/$stage.log 2>&1 || die "predictor stage $stage failed"
done
if [ ! -f $OUT/integration.json ]; then
  cp -p $MAIN $M/main_config_before_parity.json
  CUDA_VISIBLE_DEVICES=$G1 $PY -m scripts.train_max7 parity --output-dir $OUT > $OUT/parity.log 2>&1 || die "gRPC parity failed"
  $PY - <<PY
import json
old = json.load(open("$M/main_config_before_parity.json")); path = "$MAIN"; c = json.load(open(path))
c["notes"] = old["notes"] + " Effectiveness predictor connected after gRPC parity ($OUT)."
c["reward"]["effectiveness"]["endpoint"] = "127.0.0.1:50558"   # the pilot's service keeps port 50557
json.dump(c, open(path, "w"), indent=2); open(path, "a").write("\n")
PY
  stamp "parity passed; predictor connected (service port 50558)"
fi
if ! tmux has-session -t effserve_clean 2>/dev/null; then
  free $G1
  tmux new-session -d -s effserve_clean "cd $REPO && CUDA_VISIBLE_DEVICES=$G1 PYTHONPATH=$PYTHONPATH MODEL_PATH_MAP=$MODEL_PATH_MAP $PY -m scripts.serve_effectiveness --config $MAIN --device cuda:0 --port 50558 --log-dir $OUT/grpc_logs >> $OUT/serve.log 2>&1"
  until grep -q "schema_version" $OUT/serve.log 2>/dev/null; do sleep 5; tmux has-session -t effserve_clean 2>/dev/null || die "the service did not start (see $OUT/serve.log)"; done
  stamp "Effectiveness service running on GPU $G1, port 50558"
fi
if [ ! -f $OUT/service_benchmark.json ]; then
  free $G2
  CUDA_VISIBLE_DEVICES=$G2 $PY -m scripts.diagnostics.benchmark_effectiveness_service --config $MAIN --service-gpu $G1 --requests 60 --clients 1 2 5 --output $OUT/service_benchmark.json > $OUT/service_benchmark.log 2>&1 || die "service benchmark failed"
  grep -q '"correct": true' $OUT/service_benchmark.json || die "service benchmark: results differ from the offline predictor or requests failed"
fi
CUDA_VISIBLE_DEVICES= $PY -m scripts.main_experiments plan --matrix configs/experiments/train30_4b_clean/matrix.json > $M/plan.out 2>&1 || die "condition configs differ in more than the allowed fields (see $M/plan.out)"
if ! grep -q "done: iteration 0 steps base rubrics" $L/ours_seed0.log 2>/dev/null; then
  stamp "R_0: rubric step of Ours on GPU $G2"; free $G2
  CONDITION=ours GPU=$G2 bash $L/condition.sh base rubrics >> $L/ours_seed0.log 2>&1 || die "the rubric step failed (see $L/ours_seed0.log)"
fi
R=outputs/serl_clean/$(cat $L/ours_seed0.run_id)
CUDA_VISIBLE_DEVICES= $PY -m scripts.diagnostics.rubric_funnel --run $R --output $M/rubric_funnel/ours_iter0.json > /dev/null 2>&1
E=$M/full_reward_e2e; mkdir -p $E; PREV=
for n in 1 2 5; do
  if [ -s $E/e2e_$n.run_id ] && grep -q '"checks_passed": true' $E/e2e_$n.log 2>/dev/null; then PREV=$(cat $E/e2e_$n.run_id); continue; fi
  PRE=(); [ -n "$PREV" ] && PRE=(--prerequisite-run "$PREV")
  free $G2
  API_USAGE_STAGE=full_reward_e2e_$n CUDA_VISIBLE_DEVICES=$G2 $PY -m scripts.run_e2e_max7 --config configs/experiments/train30_4b_clean/ours.json --cases $n "${PRE[@]}" \
    --rubric-set $R/iter0/rubrics/rubric_set.json --write-run-id $E/e2e_$n.run_id > $E/e2e_$n.log 2>&1 || die "full-reward E2E with $n case(s) failed (see $E/e2e_$n.log)"
  PREV=$(cat $E/e2e_$n.run_id); stamp "full-reward E2E $n case(s) passed"
done
stamp "CLEAN PREPARATION DONE: predictor $OUT/dev_report.json, R_0 $R/iter0/rubrics, E2E passed. GRPO is not started here."
