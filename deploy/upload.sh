#!/usr/bin/env bash
# Run this in your Codespaces / local terminal:   bash deploy/upload.sh USER@VM_PUBLIC_IP [path/to/key.pem]
# It copies the project (model + .env included) to the VM over SSH and starts it there.
set -euo pipefail

TARGET="${1:?Usage: bash deploy/upload.sh USER@VM_PUBLIC_IP [path/to/key.pem]}"
KEY="${2:-}"
SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
if [ -n "$KEY" ]; then
  chmod 600 "$KEY"
  SSH_OPTS+=(-i "$KEY")
fi

cd "$(dirname "$0")/.."
for f in model/dr_model.keras model/model_meta.json .env; do
  [ -f "$f" ] || { echo "Missing $f  (see README: copy the model files into model/, and cp .env.example .env)"; exit 1; }
done

echo "==> Uploading the project to $TARGET ..."
tar czf - --exclude='.git' --exclude='__pycache__' --exclude='chat_images' --exclude='.venv' --exclude='tests' . \
  | ssh "${SSH_OPTS[@]}" "$TARGET" 'mkdir -p ~/dr_chatbot && tar xzf - -C ~/dr_chatbot'

echo "==> Building and starting the app on the VM (the first time takes 10-15 minutes) ..."
ssh -t "${SSH_OPTS[@]}" "$TARGET" 'cd ~/dr_chatbot && bash deploy/setup_vm.sh'
