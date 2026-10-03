#!/usr/bin/env bash
# Runs ON the VM (upload.sh calls it). Safe to run again: adds swap, installs Docker, (re)starts the app.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! swapon --show | grep -q swapfile; then
  echo "==> Creating a 2 GB swap file (safety net for 4 GB RAM)"
  sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "==> Installing Docker"
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER" || true
fi

echo "==> Building and starting the containers"
sudo docker compose up -d --build
echo
sudo docker compose ps

DOMAIN_VALUE="$(grep -E '^DOMAIN=' .env | cut -d= -f2- | tr -d '"' || true)"
echo
if [ -z "$DOMAIN_VALUE" ] || [ "${DOMAIN_VALUE:0:1}" = ":" ]; then
  echo "Done. Open  http://<your VM public IP>  (plain HTTP, because DOMAIN is not set)."
else
  echo "Done. Open  https://$DOMAIN_VALUE  (the HTTPS certificate can take a minute on first start)."
fi
echo "Logs:  cd ~/dr_chatbot && sudo docker compose logs -f app"
