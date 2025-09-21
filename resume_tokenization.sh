#!/usr/bin/env bash

set -euo pipefail

# Paths
PROJ="/home/ubuntu/MuP_MOE"
VENV="$PROJ/venv"
SCRIPT="$PROJ/data/cccc/fast_resume.py"

echo "========================================"
echo "  C4 Dataset Resume Tokenization"
echo "========================================"
echo ""
echo "This will re-download the same 25% of C4 that you downloaded before"
echo "(91.25M train + 91.25k validation examples) and tokenize it."
echo ""
echo "The script will:"
echo "1. Check for existing checkpoints"
echo "2. Re-download if needed (with progress saving)"
echo "3. Tokenize the data"
echo "4. Save binary files for training"
echo ""
echo "You can safely interrupt (Ctrl+C) and resume at any time!"
echo ""
echo "----------------------------------------"

# Activate venv
echo "==> Activating virtual environment..."
source "$VENV/bin/activate"

# Run the resume script
echo "==> Starting download/tokenization..."
python "$SCRIPT"

# Copy to backup location if successful
if [ -f "$PROJ/data/cccc/train.bin" ] && [ -f "$PROJ/data/cccc/val.bin" ]; then
    echo ""
    echo "==> Backing up tokenized data to /mnt/linky-b/..."
    cp -r "$PROJ/data/cccc" /mnt/linky-b/
    echo "✅ Backup complete!"
fi

echo ""
echo "========================================"
echo "  ✨ All done!"
echo "========================================"