#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
PYTHON_BIN="${SHERPA_PYTHON:-/home/kai/.venvs/gpu1_probe/bin/python}"
BENCHMARK_OUT="${SHERPA_BENCHMARK_OUT:-outputs/diagnostics/agent_doctor_124_ollama0125_20261008}"
BENCHMARK_PORT="$($PYTHON_BIN -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')"
exec "$PYTHON_BIN" scripts/diagnostics/doctor_parallel_bench.py \
  --out "$BENCHMARK_OUT" --gpu 1 --port "$BENCHMARK_PORT" \
  --arms 1 2 4 --repeats 2 --both --limit 8 --num-batch 64 \
  --ollama-bin /home/kai/.local/opt/ollama-0.12.5/bin/ollama
