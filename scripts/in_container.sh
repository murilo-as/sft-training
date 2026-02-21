#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-/data/config/config.yaml}"

cd /data

if [ ! -f "$CONFIG_PATH" ]; then
  echo "[CONTAINER] ERRO: config nao encontrada: $CONFIG_PATH" >&2
  exit 1
fi

if [ ! -f requirements.txt ]; then
  echo "[CONTAINER] ERRO: requirements.txt nao encontrado" >&2
  exit 1
fi

pip install -r requirements.txt

python train_sft.py
