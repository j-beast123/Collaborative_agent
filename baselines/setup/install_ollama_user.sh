#!/usr/bin/env bash
# User-invoked only. No sudo, services, model downloads, or running-process changes.
set -euo pipefail
asset_root=/convei_nas2/zzuhyun/ollama_baselines
install_root="$asset_root/opt/ollama-0.12.5"
archive="$asset_root/downloads/ollama-linux-amd64-0.12.5.tgz"
[[ $(uname -m) == x86_64 ]] || { echo 'This archive requires x86_64' >&2; exit 1; }
[[ ! -e "$install_root" ]] || { echo "Already exists; left unchanged: $install_root" >&2; exit 1; }
mkdir -p "$asset_root/downloads"
curl --fail --location --retry 3 --continue-at - \
  --output "$archive" https://github.com/ollama/ollama/releases/download/v0.12.5/ollama-linux-amd64.tgz
printf '%s  %s\n' e0d20dac3358403ef17f880168f32a59b911f3875c509d2d256bd70acce9360e "$archive" | sha256sum --check -
mkdir -p "$install_root"
tar --extract --gzip --no-same-owner --file "$archive" --directory "$install_root"
# Explicit non-default endpoint; this CLI call does not start a server.
OLLAMA_HOST=http://127.0.0.1:11931 "$install_root/bin/ollama" --version
printf 'Installed only in %s\n' "$install_root"
