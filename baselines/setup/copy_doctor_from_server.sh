#!/usr/bin/env bash
# Usage: bash .../copy_doctor_from_server.sh user@server /absolute/source/models
# Reads the remote store; writes only our separate account-owned target store.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: script user@server /absolute/source/models' >&2; exit 2; }
doctor_source=$1
doctor_source_models=${2%/}
[[ $doctor_source != -* && $doctor_source_models == /* ]] || exit 2
asset_root=/convei_nas2/zzuhyun/ollama_baselines
models="$asset_root/models_gemma3_88c2e99b4a44"
manifest_rel=manifests/registry.ollama.ai/library/gemma3/12b-it-q8_0
mkdir -p "$asset_root/incoming" "$models/blobs"
stage=$(mktemp -d "$asset_root/incoming/gemma3.XXXXXX")
# Fetch tiny manifest first. A wrong source model stops before copying any blobs.
rsync -a --protect-args -- "$doctor_source:$doctor_source_models/$manifest_rel" "$stage/manifest"
python -m baselines.common.doctor_assets --models "$models" --manifest "$stage/manifest" \
  --manifest-only --blob-list "$stage/blob-files.txt"
if [[ -e "$models/$manifest_rel" ]]; then
  python -m baselines.common.doctor_assets --models "$models" --manifest-only
fi
# --ignore-existing preserves any existing blobs; complete hash verification below
# rejects corrupted/incomplete ones instead of silently replacing them.
rsync -a --protect-args --partial --ignore-existing --info=progress2 \
  --files-from="$stage/blob-files.txt" -- "$doctor_source:$doctor_source_models/" "$models/"
python -m baselines.common.doctor_assets --models "$models" --manifest "$stage/manifest"
mkdir -p "$models/manifests/registry.ollama.ai/library/gemma3"
if [[ ! -e "$models/$manifest_rel" ]]; then cp -n -- "$stage/manifest" "$models/$manifest_rel"; fi
python -m baselines.common.doctor_assets --models "$models"
printf 'Verified store: %s\n' "$models"
