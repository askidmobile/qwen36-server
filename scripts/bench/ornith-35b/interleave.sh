#!/usr/bin/env bash
# Чередующийся A/B: yforge -> llama.cpp -> yforge -> llama.cpp на одном окне.
# Каждый запуск = отдельный процесс со свежими уникальными промптами.
set -u
CTX="${1:-8192}"; PT="${2:-4096}"; MT="${3:-128}"; REPS="${4:-3}"
YF='{"PREFILL_CHUNK":"4096"}'
run_yf() { CTX=$CTX PROMPT_TOKENS=$PT MAXTOK=$MT REPS=$REPS ENGINE=yforge \
   VARIANT="$1" EXTRA="$YF" PORT=18820 python3 /root/ornith_ab.py 2>&1 | grep -E "^\[yf" ; }
run_ll() { CTX=$CTX PROMPT_TOKENS=$PT MAXTOK=$MT REPS=$REPS ENGINE=llamacpp \
   VARIANT="$1" EXTRA='{}' PORT=18820 python3 /root/ornith_ab.py 2>&1 | grep -E "^\[ll" ; }
for i in 1 2 3; do
  run_yf "yf-r$i"
  run_ll "ll-r$i"
done
echo INTERLEAVE_DONE
