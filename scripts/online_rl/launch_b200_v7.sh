#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec bash scripts/online_rl/launch_b200.sh --config configs/experiments/train120_max4_v7_rubric_only_b200.json "$@"
