#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/ubuntu/MuP_MOE"
ENV_FILE="${PROJECT_DIR}/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -o allexport
    source "$ENV_FILE"
    set +o allexport
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "WANDB_API_KEY not set in $ENV_FILE"
    exit 1
fi

source "${PROJECT_DIR}/venv/bin/activate"
wandb login --relogin "$WANDB_API_KEY"