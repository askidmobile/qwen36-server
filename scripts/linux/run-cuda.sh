#!/usr/bin/env bash
# Запуск qwen36-server на Linux + CUDA.
# Сборка — scripts/build_linux.sh (вызывается автоматически, если бинарника нет).
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVER_DIR"

BIN="$SERVER_DIR/target/release/qwen36-server"
if [ ! -x "$BIN" ]; then
    echo "[run] бинарника нет — собираю: scripts/build_linux.sh"
    bash "$(dirname "$0")/../build_linux.sh"
fi

if [ ! -f .env ]; then
    echo "[run] .env не найден — скопируйте: cp .env.example .env и заполните API_KEYS/MODEL"
    exit 1
fi

echo "[run] $BIN"
exec "$BIN"
