#!/usr/bin/env bash
# Сборка qwen36-server под Linux + CUDA (цель: RTX 4090 24 GB, sm_89).
#
# Зависимости (Debian/Ubuntu):
#   sudo apt install -y build-essential pkg-config git curl
#   + Rust: https://rustup.rs
#   + CUDA toolkit (nvcc в PATH): apt nvidia-cuda-toolkit ИЛИ toolkit с nvidia.com
#
# Раскладка директорий (path-зависимости относительные):
#   <workdir>/candle-fork-qwen35-batch   (форк, ветка feat/qwen35-batching)
#   <workdir>/Qwen3.6 27B                (этот репо, сервер)
set -euo pipefail

SERVER_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK_DIR="$(dirname "$SERVER_DIR")"
FORK_DIR="$WORK_DIR/candle-fork-qwen35-batch"

# RTX 4090 = Ada Lovelace, sm_89. Для другой карты: CUDA_COMPUTE_CAP=86/90/...
export CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-89}"

if [ ! -d "$FORK_DIR" ]; then
    echo "[setup] клонирую форк в $FORK_DIR"
    git clone --depth 50 -b feat/qwen35-batching \
        https://github.com/askidmobile/candle.git "$FORK_DIR"
fi

command -v nvcc >/dev/null || { echo "ERROR: nvcc не найден (нужен CUDA toolkit)"; exit 1; }
command -v cargo >/dev/null || { echo "ERROR: cargo не найден (нужен rustup)"; exit 1; }

cd "$SERVER_DIR"
echo "[build] cargo build --release --features cuda (CUDA_COMPUTE_CAP=$CUDA_COMPUTE_CAP)"
cargo build --release --features cuda

echo "[ok] бинарник: $SERVER_DIR/target/release/qwen36-server"
echo "[run] QWEN36_API_KEY=<ключ> QWEN36_MODEL=<путь к gguf> ./target/release/qwen36-server"
