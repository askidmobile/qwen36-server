#!/usr/bin/env bash
# stability_smoke.sh — прогон критерия v1 (BD-008): 4 параллельных клиента,
# длинные генерации, процесс жив, VRAM без роста, все стримы Done.
# План: tests/stability_plan.md. Portable bash (macOS + Linux), set -u only.
#
# Использование:
#   KEY=master-key ./scripts/stability_smoke.sh
#   KEY=... HOST=http://192.168.2.89:8080 SSH_HOST=yttri-win ./scripts/stability_smoke.sh
set -u

HOST="${HOST:-http://192.168.2.89:8080}"
KEY="${KEY:-}"
CLIENTS="${CLIENTS:-4}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
SSH_HOST="${SSH_HOST:-}"
OUT_DIR="${OUT_DIR:-out/stability-$(date +%Y%m%d-%H%M%S)}"
POLL_SEC="${POLL_SEC:-30}"
VRAM_TOLERANCE_MIB="${VRAM_TOLERANCE_MIB:-512}"

if [ -z "$KEY" ]; then
    echo "FATAL: KEY (master API key) не задан" >&2
    exit 2
fi
mkdir -p "$OUT_DIR"

PASS=0
FAIL=0
report() { # $1=имя проверки $2=PASS|FAIL|SKIP $3=детали
    printf '%-24s %-4s %s\n' "$1" "$2" "$3"
    case "$2" in
        PASS) PASS=$((PASS + 1)) ;;
        FAIL) FAIL=$((FAIL + 1)) ;;
    esac
}

# ── helpers ──────────────────────────────────────────────────────────────────

vram_used_mib() { # пустая строка если недоступно
    if [ -n "$SSH_HOST" ]; then
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_HOST" \
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null
    elif command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
    fi
}

server_alive() { # 0 = жив (HTTP отвечает); процессная проверка — по HTTP
    curl -sf -o /dev/null -m 5 -H "Authorization: Bearer $KEY" "$HOST/v1/models"
}

# ── 1. preflight ─────────────────────────────────────────────────────────────

echo "== preflight: $HOST (clients=$CLIENTS, max_tokens~$MAX_TOKENS, out=$OUT_DIR)"
models_json="$OUT_DIR/models.json"
if ! server_alive; then
    report "preflight" "FAIL" "сервер не отвечает на /v1/models"
    exit 1
fi
curl -sf -m 10 -H "Authorization: Bearer $KEY" "$HOST/v1/models" > "$models_json"
grep -q '"slots"' "$models_json" \
    && report "preflight" "PASS" "$(tr -d '\n ' < "$models_json" | head -c 160)" \
    || report "preflight" "FAIL" "нет поля slots в /v1/models"

V0="$(vram_used_mib | head -n1 | tr -d ' ')"
if [ -n "$V0" ]; then
    echo "vram baseline: ${V0} MiB"
else
    echo "vram baseline: недоступно (нет SSH_HOST/nvidia-smi) — VRAM-проверки SKIP"
fi
echo "$(date +%s) $V0" >> "$OUT_DIR/vram.log"

# ── 2. 4 параллельных клиента ────────────────────────────────────────────────

prompt_for() { # $1 = client idx → длинный развёрнутый prompt
    case "$1" in
        0) echo "Напиши подробный технический разбор устройства планировщика непрерывного батчинга для LLM-инференса: слоты, prefill vs decode, освобождение слотов. Минимум 40 абзацев с деталями." ;;
        1) echo "Опиши исчерпывающе архитектуру гибридной модели DeltaNet+attention: рекуррентное состояние, KV-cache, RoPE, sliding window. Длинный ответ, по пунктам, с примерами." ;;
        2) echo "Напиши очень подробный учебник по численным методам решения дифференциальных уравнений: Эйлер, Рунге-Кутта, устойчивость, порядок точности. Как можно больше разделов." ;;
        *) echo "Составь максимально подробный план миграции монолитного приложения на микросервисы: этапы, риски, метрики, откат. Длинный структурированный текст." ;;
    esac
}

pids=""
i=0
while [ "$i" -lt "$CLIENTS" ]; do
    mt=$MAX_TOKENS
    [ "$i" -eq 1 ] && mt=$((MAX_TOKENS * 3 / 2))
    [ "$i" -ge 2 ] && mt=$((MAX_TOKENS * 2))
    body=$(printf '{"model":"qwen3.6-27b","stream":true,"max_tokens":%s,"messages":[{"role":"user","content":%s}]}' \
        "$mt" "$(prompt_for "$i" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))' 2>/dev/null || echo "\"prompt $i\"")")
    # -N: без буферизации SSE. Запрос идёт до терминального chunk / ошибки curl.
    curl -sN -m 3600 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "$body" "$HOST/v1/chat/completions" > "$OUT_DIR/client-$i.sse" 2> "$OUT_DIR/client-$i.err" &
    pids="$pids $!"
    i=$((i + 1))
done
echo "started clients:$pids"

# ── 3. контроль во время прогона ─────────────────────────────────────────────

died=""
while :; do
    alive_count=0
    for p in $pids; do
        kill -0 "$p" 2>/dev/null && alive_count=$((alive_count + 1))
    done
    [ "$alive_count" -eq 0 ] && break
    if ! server_alive; then
        died="сервер перестал отвечать во время прогона"
        break
    fi
    v="$(vram_used_mib | head -n1 | tr -d ' ')"
    echo "$(date +%s) ${v:-nan}" >> "$OUT_DIR/vram.log"
    sleep "$POLL_SEC"
done

for p in $pids; do
    wait "$p" 2>/dev/null
done

# ── 4. проверки ──────────────────────────────────────────────────────────────

if [ -z "$died" ] && server_alive; then
    report "process-alive" "PASS" "сервер отвечает после прогона"
else
    report "process-alive" "FAIL" "${died:-сервер не отвечает после прогона}"
fi

i=0
streams_ok=0
while [ "$i" -lt "$CLIENTS" ]; do
    f="$OUT_DIR/client-$i.sse"
    if grep -q 'data: \[DONE\]' "$f" && grep -q '"finish_reason"' "$f"; then
        streams_ok=$((streams_ok + 1))
    else
        echo "client-$i: стрим не завершился корректно (tail):" >&2
        tail -c 400 "$f" >&2
    fi
    i=$((i + 1))
done
if [ "$streams_ok" -eq "$CLIENTS" ]; then
    report "streams-done" "PASS" "$streams_ok/$CLIENTS стримов с finish_reason + [DONE]"
else
    report "streams-done" "FAIL" "$streams_ok/$CLIENTS стримов завершились"
fi

V1="$(vram_used_mib | head -n1 | tr -d ' ')"
echo "$(date +%s) ${V1:-nan}" >> "$OUT_DIR/vram.log"
if [ -n "$V0" ] && [ -n "$V1" ]; then
    delta=$((V1 - V0))
    [ "$delta" -lt 0 ] && delta=$((-delta))
    if [ "$delta" -le "$VRAM_TOLERANCE_MIB" ]; then
        report "vram-stable" "PASS" "baseline=${V0}MiB end=${V1}MiB delta=${delta}MiB<=${VRAM_TOLERANCE_MIB}MiB"
    else
        report "vram-stable" "FAIL" "baseline=${V0}MiB end=${V1}MiB delta=${delta}MiB>${VRAM_TOLERANCE_MIB}MiB"
    fi
else
    report "vram-stable" "SKIP" "nvidia-smi недоступен; лог: $OUT_DIR/vram.log"
fi

# reuse: одиночный запрос после прогона
reuse_body='{"model":"qwen3.6-27b","stream":false,"max_tokens":32,"messages":[{"role":"user","content":"ping"}]}'
if curl -sf -m 300 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -d "$reuse_body" "$HOST/v1/chat/completions" | grep -q '"content"'; then
    report "engine-reuse" "PASS" "одиночный запрос после прогона отвечает"
else
    report "engine-reuse" "FAIL" "одиночный запрос после прогона не отвечает"
fi

# ── 5. сводка ────────────────────────────────────────────────────────────────

echo "──"
echo "итог: PASS=$PASS FAIL=$FAIL  артефакты: $OUT_DIR"
[ "$FAIL" -eq 0 ]
