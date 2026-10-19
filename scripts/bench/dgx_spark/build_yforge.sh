#!/bin/bash
set -euo pipefail
ROOT=${1:-/workspace/build/qwen36}
source /workspace/.gpuwerk-init
cd "$ROOT"
CUDA_COMPUTE_CAP=121 cargo rustc --release --features cuda --bin yforge -- \
  -C link-arg=-L/usr/local/cuda-13.0/targets/sbsa-linux/lib \
  -C link-arg=-lcudart
ls -lh target/release/yforge
ldd target/release/yforge | grep -E 'cudart|not found' || true
