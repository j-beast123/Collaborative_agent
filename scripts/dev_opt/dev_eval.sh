#!/usr/bin/env bash
# DEV evaluation of one finished DEV arm: theta0 (checkpoint_000000) and every due checkpoint (30, 60) on the fixed held-out
# MedMCQA DEV split (configs/experiments/dev_opt/dev_eval.json), with the production async evaluator and its own Doctors.
# CasiMedicos622 is never evaluated. Usage: dev_eval.sh dev_base|dev_diversity|dev_optimizer|dev_both|t2dev_lr1e5|t2dev_lr3e5
# Runtime inputs: SHERPA_EVAL_GPUS (default 5,6,7), SHERPA_EVAL_PORTS (default 11545,11546,11547), SHERPA_OLLAMA_BINARY, SHERPA_PYTHON
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM="${1:?usage: dev_eval.sh dev_base|dev_diversity|dev_optimizer|dev_both|t2dev_lr1e5|t2dev_lr3e5}"
case "$ARM" in dev_base|dev_diversity|dev_optimizer|dev_both) CFG=configs/experiments/dev_opt/${ARM#dev_}.json;;
  t2dev_lr1e5|t2dev_lr3e5) CFG=configs/experiments/fixed_t2/dev/${ARM#t2dev_}.json;;
  t2rw_o10_r00|t2rw_o09_r01|t2rw_o07_r03|t2rw_o05_r05) CFG=configs/experiments/fixed_t2/dev_reward/${ARM#t2rw_}.json;;
  t2dis_o10_r00|t2dis_o09_r01) CFG=configs/experiments/fixed_t2/dis/${ARM#t2dis_}.json;; *) echo "unknown DEV arm $ARM" >&2; exit 2;; esac
PY="${SHERPA_PYTHON:-python}"
OUT=$("$PY" -c "import sys;sys.path.insert(0,'.');from src.online_rl.tree import experiment as X;print(X.load_config('$CFG')['output_dir'])")
[ -f "$OUT/run_state.json" ] || { echo "no run at $OUT" >&2; exit 2; }
exec "$PY" scripts/online_rl/launch_async_eval.py --config "$CFG" --gpus "${SHERPA_EVAL_GPUS:-5,6,7}" --ports "${SHERPA_EVAL_PORTS:-11545,11546,11547}" \
  --eval-root "$OUT/dev_evaluation" --eval-override configs/experiments/dev_opt/dev_eval.json --include-initial --once
