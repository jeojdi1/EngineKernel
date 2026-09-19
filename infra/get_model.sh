#!/usr/bin/env bash
# Fetch the pinned checkpoint the benchmark uses (hub >=0.34 renamed the CLI to `hf`).
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
hf download Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --local-dir "$HOME/qwen3-4b"
du -sh "$HOME/qwen3-4b"
