#!/usr/bin/env bash

set -euo pipefail

BIG_DIR="/mnt/local"
PROJ="/home/ubuntu/MuP_MOE"
DATA_DIR="$PROJ/data/fineweb"
VENV="$PROJ/venv"
DATA_DOWNLOAD="$DATA_DIR/data_download.py"
TOKENIZER="$DATA_DIR/tokenize_pkl_to_bin.py"
PICKLE_OUT="$BIG_DIR/fineweb_checkpoint/train.pkl"

mkdir -p "$DATA_DIR"

echo "==> Activating venv at $VENV"
if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"

python "$DATA_DOWNLOAD" --big-dir "$BIG_DIR" --limit 10000000 --checkpoint-interval 2500000 "$@"

if [[ ! -f "$PICKLE_OUT" ]]; then
  echo "[error] Expected pickle at $PICKLE_OUT" >&2
  exit 1
fi

echo "==> Tokenizing $PICKLE_OUT"
python "$TOKENIZER" --in "$PICKLE_OUT" --out "$DATA_DIR" --encoding gpt2 --add-eos

echo "==> Done. Tokenized files are in $DATA_DIR"
