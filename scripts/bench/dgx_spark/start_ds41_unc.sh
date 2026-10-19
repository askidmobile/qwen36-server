#!/bin/bash
set -euo pipefail
MODEL_DIR=${MODEL_DIR:-/workspace/models/ds41-1017-unc}
SERVER=${SERVER:-/workspace/build/ds4-dfm-rs/ds4-server}
MEM_BUDGET_MB=${MEM_BUDGET_MB:-110000}
EXTRA_FLAGS=()
if [ "${NO_ENGRAM:-0}" = "1" ]; then EXTRA_FLAGS+=(--v41-no-engram); fi
MODEL="$MODEL_DIR/DeepSeek-V4.1-Flash-vq8sh14-q4k-mtpnative.gguf"
source /workspace/.gpuwerk-init
exec "$SERVER" --cuda \
  -m "$MODEL" \
  --engram-dir "$MODEL_DIR/deepseek-engram" \
  --vision-gguf "$MODEL_DIR/DeepSeek-V4.1-Flash.vision.gguf" \
  --mem-budget-mb "$MEM_BUDGET_MB" \
  --mem-floor-gb 2 \
  --mtp-mode auto \
  --max-seqs 1 \
  --prefix-reuse off \
  --model-id deepseek-v4.1-flash-1017-uncensored-vq \
  --host 0.0.0.0 \
  --port 8000 \
  --no-update-check \
  "${EXTRA_FLAGS[@]}"
