#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/online_rl/runtime_env.sh
exec "$SHERPA_PYTHON" monitor.py "$@"
