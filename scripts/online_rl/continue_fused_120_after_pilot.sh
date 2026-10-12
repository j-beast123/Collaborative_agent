#!/usr/bin/env bash
set -euo pipefail

pilot_unit=sherpa-traj120-doctor-roles-fused-pilot-v3.service
run_dir=/home/kai/Collaborative_agent_max4_v331_lane_test/outputs/train120_max4_trajectory_grpo_g4_rubric_only_revised_bank13_gpu03_doctor_roles_fused_v1_20261012/r1_o0__p0fixed__gemma3doctor__seed0__traj_g4_bank13_doctor_roles_fused_v1

while systemctl --user is-active --quiet "$pilot_unit"; do
  sleep 5
done

# Continue only from a fully committed first batch. A failed or partial pilot
# must never cause an overlapping rollout or duplicate optimizer update.
/home/common/miniconda3/bin/python - "$run_dir/run_state.json" <<'PY'
import json, sys
state = json.load(open(sys.argv[1]))
if state.get('case_cursor') != 4 or state.get('batch_cursor') != 1:
    raise SystemExit(f"pilot did not commit exactly one batch: {state}")
PY

cd /home/kai/Collaborative_agent_max4_v331_lane_test
export SHERPA_ENV_FILE=/home/kai/Collaborative_agent/.env
exec /home/common/miniconda3/bin/python -u scripts/online_rl/run_five_gpu.py \
  --config configs/experiments/train120_max4_trajectory_grpo_g4_rubric_only_revised_bank13_gpu03_doctor_roles_fused_v1.json \
  --launch
