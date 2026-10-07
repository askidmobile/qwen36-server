#!/usr/bin/env bash
# Декод-профиль для одного режима DELTA_DECODE: nsys окно только на декоде.
set -u
DEC="${1:-tile}"; CTX="${2:-8192}"; DELAY="${3:-18}"; DUR="${4:-40}"; PORT=18910
OUT=/root/logs/pd-$DEC; mkdir -p "$OUT"
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p"; done
sleep 3
for _ in $(seq 1 200); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1); [ "${U:-9999}" -le 400 ] && break; sleep 3; done
NSYS=/usr/local/cuda-12.8/bin/nsys
setsid nohup env MODEL=/root/models/Ornith-1.5-35B-Q8_0.gguf CTX=$CTX CONTEXT_LIMIT=$CTX SLOTS=1 \
  PORT=$PORT API_KEYS="[{\"key\":\"x\",\"name\":\"m\"}]" GPU_ONLY=1 GPU_LAYERS=999 \
  KV_POOL_Q8=1 KV_CACHE_TYPE=q8 CUDA_GRAPHS=1 PGRAPH=on FLASH_ATTN=1 \
  PREFIX_CACHE_MIB=0 SAMPLING_LOCK=1 RUST_LOG=warn MOE_EXPERTS=auto PREFILL_CHUNK=4096 \
  DELTA_DECODE=$DEC \
  $NSYS profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --delay=$DELAY --duration=$DUR -o "$OUT/dec" --force-overwrite=true \
  /root/target/release/yforge > "$OUT/serve.log" 2>&1 < /dev/null &
for _ in $(seq 1 400); do
  curl -s -m 5 -H "Authorization: Bearer x" -H "Content-Type: application/json" \
    -d "{\"model\":\"ornith-1.5-35b\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
    http://127.0.0.1:$PORT/v1/chat/completions 2>/dev/null | grep -q "\"content\"" && break
  sleep 2
done
echo "ready t+$(date +%s)"
python3 /root/load_loop.py $PORT 0 "$((DELAY+DUR+25))" ornith-1.5-35b 500 x 20 fixed
sleep 20
pkill -9 -x yforge 2>/dev/null
sleep 5
ls -la "$OUT"/dec.nsys-rep 2>/dev/null
echo PDEC_TILE_DONE
