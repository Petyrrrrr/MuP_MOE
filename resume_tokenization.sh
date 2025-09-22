#!/usr/bin/env bash

set -euo pipefail

# Paths
PROJ="/home/ubuntu/MuP_MOE"
VENV="$PROJ/venv"
SCRIPT="$PROJ/data/cccc/fast_resume.py"

source "$VENV/bin/activate"
python "$SCRIPT"

# Copy to backup location if successful
if [ -f "$PROJ/data/cccc/train.bin" ] && [ -f "$PROJ/data/cccc/val.bin" ]; then
    echo ""
    echo "==> Backing up tokenized data to /mnt/linky-b/..."
    cp -r "$PROJ/data/cccc" /mnt/linky-b/
    echo "✅ Backup complete!"
fi
