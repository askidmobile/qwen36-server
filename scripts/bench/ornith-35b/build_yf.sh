#!/usr/bin/env bash
export CUDA_PATH=/usr/local/cuda-12.8 CUDA_HOME=/usr/local/cuda-12.8 CUDA_ROOT=/usr/local/cuda-12.8
export PATH=/usr/local/cuda-12.8/bin:$HOME/.cargo/bin:$PATH
export CUDA_COMPUTE_CAP=89
export CARGO_TARGET_DIR=/root/target
export LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
cd /root/yttri-inference/qwen36-server
cargo build --release --features cuda --bin yforge 2>&1
echo "BUILD_RC=$?"
