#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec bash scripts/online_rl/test_active_training.sh \
  tests/test_b200_partner.py tests/test_b200_need_accuracy.py tests/test_b200_runner.py tests/test_b200_doctor_setup.py tests/test_parallel_bank.py tests/test_b200_hub.py tests/test_hf_doctor.py "$@"
