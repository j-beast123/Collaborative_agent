#!/usr/bin/env bash
# Source from launchers after changing to the repository root.
# Explicit environment settings take precedence over local assets and PATH.
if [[ -z "${SHERPA_PYTHON:-}" ]]; then
  for sherpa_python_candidate in "$PWD/runtime/python" "$PWD/.venv/bin/python" "$PWD/venv/bin/python"; do
    if [[ -x "$sherpa_python_candidate" ]]; then
      export SHERPA_PYTHON="$sherpa_python_candidate"
      break
    fi
  done
  if [[ -z "${SHERPA_PYTHON:-}" ]]; then
    export SHERPA_PYTHON="$(command -v python3 || command -v python || true)"
  fi
fi
if [[ -z "$SHERPA_PYTHON" ]] || ! command -v "$SHERPA_PYTHON" >/dev/null 2>&1; then
  echo 'Python executable missing. Set SHERPA_PYTHON to your training environment Python.' >&2
  return 1
fi
if [[ -z "${SHERPA_ENV_FILE:-}" ]]; then
  if [[ -f "$PWD/runtime/credentials.env" ]]; then
    export SHERPA_ENV_FILE="$PWD/runtime/credentials.env"
  else
    export SHERPA_ENV_FILE="$PWD/.env"
  fi
fi
if [[ -z "${HF_HOME:-}" ]]; then
  if [[ -d "$PWD/runtime/huggingface" ]]; then
    export HF_HOME="$PWD/runtime/huggingface"
  else
    export HF_HOME="${XDG_CACHE_HOME:-$HOME/.cache}/huggingface"
  fi
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
unset sherpa_python_candidate
