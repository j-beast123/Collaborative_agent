#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[1/5] Frozen artifact hashes"
(cd artifacts/reported_checkpoint && sha256sum -c SHA256SUMS)

echo "[2/5] Python syntax"
python -m py_compile \
  scripts/online_rl/eval_checkpoint.py \
  src/online_rl/runtime.py \
  src/online_rl/integrated/continuation.py

echo "[3/5] Read-only reported checkpoint evaluation preflight"
python scripts/online_rl/eval_checkpoint.py

echo "[4/5] reported checkpoint training preflight"
python scripts/online_rl/run_five_gpu.py \
  --config configs/online_rl/paper_reproduction.json

echo "[5/5] Focused artifact tests"
pytest -q tests/test_checkpoint_handoff.py

echo "reported checkpoint package verification complete."
