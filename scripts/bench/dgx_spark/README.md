# DGX Spark HTTP benchmark

`http_bench.py` меняет только запросы, поэтому одинаково работает с `yforge` и
`ds4-server`, если сервер уже поднят. Каждый запрос содержит новый nonce;
`usage.completion_tokens` берётся из финального SSE-чанка, а не считается по
числу чанков.

GB10 использует unified memory и `nvidia-smi` отдаёт `memory.used=N/A`.
Поэтому вместо несуществующего VRAM-значения скрипт пишет пик `VmRSS`/`VmHWM`
сервера и минимум `MemAvailable`; при интерпретации учитывать, что страницы
модели могут быть mmap-backed и не входить в RSS целиком.

```bash
python3 scripts/bench/dgx_spark/http_bench.py \
  --engine yforge --base-url http://127.0.0.1:18099 \
  --api-key "$API_KEY" --pid "$SERVER_PID" \
  --contexts 2048,8192,32768,65536,131072 --repeats 2 \
  --outdir /workspace/forge-run/bench/yforge
```

Для DeepSeek используется тот же набор точек плюс повтор одинакового запроса
для Engram. `--repeat-context 307200 --repeat-count 2` запускает его отдельной
фазой; сервер при этом должен быть запущен с `--no-kv-reuse`, иначе повтор
будет попаданием в prefix cache, а не повторным Engram-префиллом.

## GB10 build note

`build_yforge.sh` собирает только `yforge` и добавляет `-lcudart` на финальную
линковку. Это обход того, что `cudart_static` из `candle-flash-attn/build.rs`
не попадает в итоговую команду линкера на aarch64. Для CUDA 13 PTX-модули
`sm_121f` должны сохранять `.version 9.x`; понижение до `.version 8.6`
приводит к `CUDA_ERROR_INVALID_PTX`.
