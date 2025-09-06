#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults
METHOD="nodesource"        # nodesource | nvm
NODE_TARGET="lts"          # 18 | 20 | 22 | lts
UPGRADE_NPM=1              # 1=yes, 0=no

usage() {
  cat <<EOF
Usage: sudo bash install-node.sh [--method nodesource|nvm] [--node 18|20|22|lts] [--no-npm-upgrade]
Examples:
  sudo bash install-node.sh                 # system-wide latest LTS (recommended)
  sudo bash install-node.sh --node 18       # system-wide Node 18.x
  bash install-node.sh --method nvm --node lts
EOF
}

# Parse flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --method) METHOD="${2:-}"; shift 2;;
    --node) NODE_TARGET="${2:-}"; shift 2;;
    --no-npm-upgrade) UPGRADE_NPM=0; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

# sudo helpers
if [[ $EUID -ne 0 ]]; then
  SUDO="sudo -H"
  SUDO_BASH="sudo -E bash -"
else
  SUDO=""
  SUDO_BASH="bash -"
fi

# Noninteractive apt to avoid prompts
export DEBIAN_FRONTEND=noninteractive
export NEEDRESTART_MODE=a

ensure_pkg() {
  if ! dpkg -s "$1" >/dev/null 2>&1; then
    $SUDO apt-get update -y
    $SUDO apt-get install -y "$1"
  fi
}

log() { printf '\n==> %s\n' "$*"; }

if [[ "$METHOD" == "nodesource" ]]; then
  log "Installing Node via NodeSource (system-wide)"
  ensure_pkg curl
  ensure_pkg ca-certificates

  # Remove ancient repo versions if present
  $SUDO apt-get remove -y nodejs npm || true

  # Decide major version
  case "$NODE_TARGET" in
    lts|LTS) NODE_MAJOR="22";;   # change when LTS changes
    18|20|22) NODE_MAJOR="$NODE_TARGET";;
    *) echo "Unsupported --node '$NODE_TARGET' (use 18|20|22|lts)"; exit 1;;
  esac

  log "Adding NodeSource repo for Node ${NODE_MAJOR}.x"
  curl -fsSL "https://deb.nodesource.com/setup_${NODE_MAJOR}.x" | $SUDO_BASH

  log "Installing nodejs + build tools"
  $SUDO apt-get install -y nodejs build-essential

  if [[ $UPGRADE_NPM -eq 1 ]]; then
    log "Upgrading npm to latest"
    $SUDO npm i -g npm@latest
  fi

elif [[ "$METHOD" == "nvm" ]]; then
  log "Installing Node via nvm (per-user)"
  ensure_pkg curl

  # Install nvm if missing
  if [[ ! -d "${HOME}/.nvm" ]]; then
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  fi

  # Load nvm
  export NVM_DIR="${HOME}/.nvm"
  # shellcheck disable=SC1090
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

  case "$NODE_TARGET" in
    lts|LTS) NVM_TARGET="--lts";;
    18|20|22) NVM_TARGET="$NODE_TARGET";;
    *) echo "Unsupported --node '$NODE_TARGET' (use 18|20|22|lts)"; exit 1;;
  esac

  log "nvm install $NVM_TARGET"
  nvm install $NVM_TARGET
  nvm use --delete-prefix $NVM_TARGET
  nvm alias default "$(node -v)"

  if [[ $UPGRADE_NPM -eq 1 ]]; then
    log "Upgrading npm to latest (user)"
    npm i -g npm@latest
  fi

  if command -v corepack >/dev/null 2>&1; then
    log "Enabling Corepack (Yarn/pnpm shims)"
    corepack enable || true
  fi
else
  echo "Unknown --method '$METHOD' (use nodesource or nvm)"; exit 1
fi

echo
echo "✅ Done."
echo "Node: $(node -v)"
echo "npm:  $(npm -v)"


sudo npm install -g @anthropic-ai/claude-code

sudo npm install -g @openai/codex