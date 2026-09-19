#!/usr/bin/env bash
# Push the repo to a Lambda box and set it up.  usage: infra/deploy.sh <ip> [--setup]
set -euo pipefail
IP="$1"; shift || true
KEY="${LAMBDA_SSH_KEY:-$HOME/.ssh/lambda_splice}"
SSH=(ssh -i "$KEY" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "ubuntu@$IP")

rsync -az --delete -e "ssh -i $KEY -o StrictHostKeyChecking=accept-new" \
  --exclude '.git' --exclude '__pycache__' --exclude '.env' --exclude '*.safetensors' \
  "$(dirname "$0")/.." "ubuntu@$IP:~/EngineKernel/"

if [ "${1:-}" = "--setup" ]; then
  "${SSH[@]}" 'bash ~/EngineKernel/infra/remote_setup.sh'
fi
echo "deployed to $IP"
