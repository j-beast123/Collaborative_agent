#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

config="configs/online_rl/paper_reproduction.json"
launch=0
while (($#)); do
  case "$1" in
    --full) config="configs/online_rl/paper_full_120.json" ;;
    --launch) launch=1 ;;
    -h|--help)
      echo "Usage: bash run_paper.sh [--full] [--launch]"
      echo "  default: exact reported-checkpoint 30-case configuration"
      echo "  --full:  paper-scale 120-case configuration"
      echo "  without --launch: local preflight only"
      exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

python_bin="${SHERPA_PYTHON:-python}"
if ((launch)); then
  export SHERPA_FIVE_GPU_CONFIG="$config"
  exec bash scripts/online_rl/launch_five_gpu.sh
fi
exec "$python_bin" scripts/online_rl/run_five_gpu.py --config "$config"
