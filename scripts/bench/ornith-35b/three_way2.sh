#!/usr/bin/env bash
# Трёхсторонний замер на одинаковой методике: vLLM (FP8) -> yforge -> llama.cpp.
# Одно окно на все три, свежий промпт на повтор.
set -u
CTX="${1:-32768}"; PT="${2:-16384}"; MT="${3:-64}"; REPS="${4:-3}"
PORT_V=18900; M=/root/models/ornith-fp8
kill_gpu() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -x yforge 2>/dev/null; pkill -9 -x llama-server 2>/dev/null
  pkill -9 -f "vllm serve" 2>/dev/null
  sleep 5
  for _ in $(seq 1 200); do
    U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
    [ "${U:-9999}" -le 400 ] && return 0; sleep 3
  done
}
kill_gpu
echo "=== vLLM FP8 ctx=$CTX ==="
export VLLM_LOGGING_LEVEL=INFO VLLM_USE_V1=1
setsid nohup /root/vllm-env/bin/vllm serve "$M" \
  --host 127.0.0.1 --port $PORT_V --served-model-name ornith \
  --max-model-len "$CTX" --gpu-memory-utilization 0.90 --max-num-seqs 1 \
  --trust-remote-code > /root/logs/vllm3-$CTX.log 2>&1 < /dev/null &
for _ in $(seq 1 500); do
  curl -s -m 3 "http://127.0.0.1:$PORT_V/health" >/dev/null 2>&1 && break
  sleep 3
done
V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
echo "vLLM VRAM: $V MiB"
PORT=$PORT_V CTX=$CTX PROMPT_TOKENS=$PT MAXTOK=$MT REPS=$REPS VARIANT=vllm3-$CTX \
  python3 /root/vllm_bench.py 2>&1 | tail -4
kill_gpu
echo "=== yforge ctx=$CTX ==="
CTX=$CTX PROMPT_TOKENS=$PT MAXTOK=$MT REPS=$REPS PREFILL_CHUNK=4096 \
  ENGINE=yforge VARIANT=yf3-$CTX EXTRA='{"PREFILL_CHUNK":"4096","SAMPLING_LOCK":"0"}' PORT=18820 \
  python3 /root/ornith_ab.py 2>&1 | grep -E "^\[yf3"
kill_gpu
echo "=== llama.cpp ctx=$CTX ==="
CTX=$CTX PROMPT_TOKENS=$PT MAXTOK=$MT REPS=$REPS \
  ENGINE=llamacpp VARIANT=ll3-$CTX EXTRA='{}' PORT=18820 \
  python3 /root/ornith_ab.py 2>&1 | grep -E "^\[ll3"
kill_gpu
echo THREE_WAY2_DONE
