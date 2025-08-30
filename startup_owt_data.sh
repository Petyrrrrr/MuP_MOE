#!/usr/bin/env bash

set -euo pipefail

# Paths
PROJ="/home/ubuntu/MuP_MOE"
VENV="$PROJ/venv"
SCRIPT="$PROJ/data/openwebtext/prepare.py"

echo "==> Activating venv at $VENV"
# Create venv if missing
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

echo "==> Upgrading pip tooling"
pip install -U pip setuptools wheel

echo "==> Pinning Hugging Face libs compatible with loader scripts"
pip install -U "datasets<4.0.0" "huggingface_hub<0.25.0"

# Avoid the interactive trust prompt for dataset loader scripts
# (you can also pass trust_remote_code=True in code; this env var forces 'yes')
export HF_DATASETS_TRUST_REMOTE_CODE=1

# Optional: faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "==> Running $SCRIPT"
python "$SCRIPT" "$@"