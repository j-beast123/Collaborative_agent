#!/usr/bin/env bash
# Launch a 300-case production reward-ablation arm with the accepted runtime profile.
# Usage: launch_production.sh r03_o07|r07_o03|r05_o05_evolving|r05_o05_p0_fixed [--dry-run]
# Runtime inputs (never written into research configs):
#   CUDA_VISIBLE_DEVICES   five physical GPU ids (logical 0-4; logical 4 runs the update)
#   SHERPA_OLLAMA_BINARY   Ollama 0.12.5 executable
#   SHERPA_OLLAMA_MODELS   Ollama model store with gemma3:12b-it-q8_0 (default ~/.ollama/models)
#   SHERPA_DOCTOR_PORTS    optional five ports, e.g. 11535,11536,11537,11538,11539
#   SHERPA_RUNTIME_OVERLAY optional JSON file with host-specific five_gpu.doctor_urls / doctor_servers
#   SHERPA_PYTHON, SHERPA_ENV_FILE (destination secrets; default ./.env)
set -euo pipefail
cd "$(dirname "$0")/../.."
ARM="${1:?usage: launch_production.sh r03_o07|r07_o03|r05_o05_evolving|r05_o05_p0_fixed [--dry-run]}"; DRY="${2:-}"
PROD=configs/experiments/reward_prompt_matrix/production
case "$ARM" in r03_o07|r07_o03) NAME=${ARM}_evolving_300;; r05_o05_evolving|r05_o05_p0_fixed) NAME=${ARM}_300;; *) echo "unknown arm $ARM" >&2; exit 2;; esac
CFG=$PROD/$NAME.json
PY="${SHERPA_PYTHON:-python}"
: "${CUDA_VISIBLE_DEVICES:?set five physical GPU ids, e.g. 0,1,2,3,4}"
: "${SHERPA_OLLAMA_BINARY:?set the Ollama 0.12.5 executable}"
export CUDA_VISIBLE_DEVICES SHERPA_OLLAMA_BINARY
# Accepted runtime profile (shared by every arm).
STRATA=$("$PY" -c "import json,sys;p=json.load(open('$PROD/runtime_profile.json'));sys.exit('runtime profile not decided') if p.get('status')!='decided' else print(int(bool(p['parallel_strata'])))")
export SHERPA_PARALLEL_STRATA=$STRATA
RUNCFG=$CFG
if [ -n "${SHERPA_DOCTOR_PORTS:-}" ]; then
  # Runtime-only overlay: Doctor endpoints change the config hash, never the research fingerprint.
  RUNCFG=outputs/reward_prompt_matrix/runtime/$NAME.ports.json
  mkdir -p "$(dirname "$RUNCFG")"
  "$PY" - "$CFG" "$RUNCFG" "$SHERPA_DOCTOR_PORTS" <<'EOF'
import json, sys
cfg, out, ports = sys.argv[1], sys.argv[2], sys.argv[3].split(',')
if len(ports) != 5 or len(set(ports)) != 5: sys.exit('SHERPA_DOCTOR_PORTS needs five distinct ports')
json.dump({'extends': cfg, 'five_gpu': {'doctor_urls': [f'http://127.0.0.1:{p}' for p in ports]}}, open(out, 'w'), indent=2)
EOF
fi
if [ -n "${SHERPA_RUNTIME_OVERLAY:-}" ]; then
  # Host runtime overlay: only Doctor placement keys; the research fingerprint must not change.
  BASE=$RUNCFG; RUNCFG=outputs/reward_prompt_matrix/runtime/${ARM}_evolving_300.runtime.json
  mkdir -p "$(dirname "$RUNCFG")"
  "$PY" - "$BASE" "$SHERPA_RUNTIME_OVERLAY" "$RUNCFG" <<'EOF'
import json, sys
base, overlay, out = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
allowed = {'doctor_urls', 'doctor_servers'}
if set(overlay) != {'five_gpu'} or not set(overlay['five_gpu']) <= allowed: sys.exit(f'runtime overlay may only set five_gpu.{sorted(allowed)}')
json.dump({'extends': base, **overlay}, open(out, 'w'), indent=2)
EOF
fi
"$PY" scripts/online_rl/preflight_destination.py --config "$CFG" ${RUNCFG:+--runtime-config "$RUNCFG"} \
  ${SHERPA_EXPECTED_COMMIT:+--expected-commit "$SHERPA_EXPECTED_COMMIT"} $([ "$DRY" = --dry-run ] || echo --write-provenance)
if [ "$DRY" = --dry-run ]; then
  echo "DRY RUN: SHERPA_PARALLEL_STRATA=$STRATA SHERPA_FIVE_GPU_CONFIG=$RUNCFG bash scripts/online_rl/launch_five_gpu.sh"; exit 0
fi
SHERPA_FIVE_GPU_CONFIG=$RUNCFG exec bash scripts/online_rl/launch_five_gpu.sh
