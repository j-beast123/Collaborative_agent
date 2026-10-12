#!/usr/bin/env bash
# No arguments: describe configuration. --launch: execute the selected phase.
set -euo pipefail
cd "$(dirname "$0")"
source scripts/online_rl/runtime_env.sh
export HF_HUB_OFFLINE=1
export SHERPA_AGENT_REPLICAS=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
CONFIG="${SHERPA_FIVE_GPU_CONFIG:-configs/sherpa.json}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  CUDA_VISIBLE_DEVICES=$("$SHERPA_PYTHON" -c 'import sys; from src.online_rl.tree.experiment import load_config; from src.online_rl.tree.devices import gpu_count; cfg=load_config(sys.argv[1]); base="base-rollouts" in sys.argv[2:] or "--phase=base-rollouts" in sys.argv[2:]; n=cfg["prompt_optimization"].get("rollout_replicas",1) if base else gpu_count(cfg); print(",".join(map(str, range(n))))' "$CONFIG" "$@")
  export CUDA_VISIBLE_DEVICES
fi
if [[ $# -gt 0 && "$1" != --launch ]]; then echo 'Usage: bash run.sh [--launch [runner arguments]]' >&2; exit 2; fi
if [[ $# -eq 0 ]]; then exec "$SHERPA_PYTHON" scripts/online_rl/run_sherpa.py --config "$CONFIG"; fi
shift
exec "$SHERPA_PYTHON" scripts/online_rl/run_sherpa.py --config "$CONFIG" --launch "$@"
