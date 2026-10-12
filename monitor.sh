#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${SHERPA_PYTHON:-$PWD/runtime/python}" monitor.py "$@"
