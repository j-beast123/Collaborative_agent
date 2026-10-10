#!/usr/bin/env bash
# User-invoked direct registry transfer, no Ollama daemon or GPU inference.
# The newly approved digest is pinned; future tag changes stop before blob transfer.
set -euo pipefail
asset_root=/convei_nas2/zzuhyun/ollama_baselines
models="$asset_root/models_gemma3_88c2e99b4a44"
manifest_rel=manifests/registry.ollama.ai/library/gemma3/12b-it-q8_0
registry=https://registry.ollama.ai/v2/library/gemma3
mkdir -p "$asset_root/incoming" "$models/blobs"
stage=$(mktemp -d "$asset_root/incoming/registry.XXXXXX")
curl --fail --location --retry 3 --header 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
  --output "$stage/manifest" "$registry/manifests/12b-it-q8_0"
python -m baselines.common.doctor_assets --models "$models" --manifest "$stage/manifest" \
  --manifest-only --blob-list "$stage/blob-files.txt"
if [[ -e "$models/$manifest_rel" ]]; then
  python -m baselines.common.doctor_assets --models "$models" --manifest-only
fi
while IFS= read -r blob; do
  [[ -f "$models/$blob" ]] && continue
  blob_hash=${blob#blobs/sha256-}
  curl --fail --location --retry 3 --continue-at - --output "$models/$blob.partial" "$registry/blobs/sha256:$blob_hash"
  printf '%s  %s\n' "$blob_hash" "$models/$blob.partial" | sha256sum --check -
  mv -n -- "$models/$blob.partial" "$models/$blob"
done < "$stage/blob-files.txt"
python -m baselines.common.doctor_assets --models "$models" --manifest "$stage/manifest"
mkdir -p "$models/manifests/registry.ollama.ai/library/gemma3"
if [[ ! -e "$models/$manifest_rel" ]]; then cp -n -- "$stage/manifest" "$models/$manifest_rel"; fi
python -m baselines.common.doctor_assets --models "$models"
