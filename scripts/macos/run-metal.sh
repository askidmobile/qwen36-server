#!/usr/bin/env bash
# Запуск qwen36-server на macOS для разработки (Metal, Apple Silicon).
# Сборка при необходимости: scripts/build_linux.sh не подходит (CUDA) —
# здесь просто cargo run с фичей metal.
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$SERVER_DIR"

if [ ! -f .env ]; then
    echo "[run] .env не найден — скопируйте: cp .env.example .env и заполните QWEN36_API_KEYS/QWEN36_MODEL"
    exit 1
fi

echo "[run] cargo run --release --features metal"
exec cargo run --release --features metal
