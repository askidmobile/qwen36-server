# yttri-forge против llama.cpp: Ornith-1.5-9B Q4_K_M на RTX 3060

**Дата:** 14 сентября 2026.
**Хост:** `yttri-win`, Windows, RTX 3060 12 GB.
**Модель:** `D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf`
(один и тот же файл для обоих движков; `general.architecture=qwen35`, 33 блока,
встроенный `blk.32.nextn.*`).
**llama.cpp:** сборка `8e7f22b` (13.08.26), CUDA 12.4, `-ngl 999 -fa on`.
**yttri-forge:** `yforge.exe` SHA-256
`C3BC0BC3D9EB9C4328ED0BDAFBAC0FCD5B7AA4EF4279532336A308B4A55515B8`.

Один стенд для обоих движков, обращение по OpenAI-совместимому API.
Методика повторяет `docs/plans/2026-08-28-vs-llamacpp-ornith.md`.

---

## Итог

Паритет по параллелизму, отставание по prefill и декоду одного слота.

| критерий | llama.cpp | yttri-forge | разница |
|---|---:|---:|---:|
| префил ~7K, ток/с | 1654 | 1352 | −18% |
| префил ~14K, ток/с | 1622 | 1266 | −22% |
| префил ~25K, ток/с | 1545 | 1117 | −28% |
| декод 1 слот, ток/с | 52,4 | 49,3 | −6% |
| 4 клиента разом, ток/с | 114,1 | 110,8 | −3% |
| VRAM, 4 слота × 16K | 7040 МиБ | 6779 МиБ | **−261 МиБ** |

Замеры prefill — однократные на точку, декод и параллель — три и два прогона
соответственно; разброс внутри прогонов меньше процента.

Отставание на префиле растёт с длиной контекста: 18% на 7K, 28% на 25K.
Декод и параллелизм практически в паритете.

---

## Что мерили

**Префил:** длинный промпт, `max_tokens=1` — вся стена времени и есть префил.
К каждому промпту добавлялся уникальный маркер, чтобы префикс-кеш не дал
ложного ускорения (в прошлом сравнении вложенные промпты дали невозможные
7781 ток/с у llama.cpp).

**Декод:** короткий промпт, 300 токенов, три прогона, время делится на выданные
токены из `usage`.

**Параллель:** 4 одновременных клиента по 300 токенов. Первый прогон в
`compare_bench.ps1` шёл через один слот (то есть очередь, а не параллелизм), и
эти числа выброшены; в таблице — отдельный замер с настоящей параллельностью.

**Память:** пик `nvidia-smi --query-gpu=memory.used` за прогон.

### Конфигурации, приведённые к общему знаменателю

| параметр | yttri-forge | llama.cpp |
|---|---|---|
| KV | `KV_POOL_Q8=1`, пул q8 | `-ctk q8_0 -ctv q8_0` |
| Flash Attention | `FLASH_ATTN=1` | `-fa on` |
| слоёв на GPU | `GPU_LAYERS=999` | `-ngl 999` |
| окно | `CTX=131072`, `SLOTS=1` | `-c 131072`, `-np 1` |
| MTP | `MTP=0` | без `--spec-type draft-mtp` |
| графы | `CUDA_GRAPHS=1`, `PGRAPH=on` | встроенные |

Отдельно снят контроль llama.cpp на f16 KV (`-ctk f16 -ctv f16`): префил
1672/1625/1557, декод 52,5 — то есть квантование KV у llama.cpp на скорость
почти не влияет, меняется только память (9788 против 8350 МиБ).

---

## MTP не участвовал ни с одной стороны

В том же файле лежат тензоры `blk.32.nextn.*` — встроенная голова MTP.
llama.cpp их явно игнорирует:

```
W model has unused tensor blk.32.nextn.eh_proj.weight (18874368 bytes) -- ignoring
```

Отдельного MTP-артефакта для Ornith на хосте нет (поиск по `D:\Models`
находит единственный gguf), поэтому `MTP=0` у обоих движков. Сравнение чистое
по «сырому» движку, но не покрывает спекулятивный декод.

---

## Попутные находки

### 1. `PREFILL_CHUNK=64` в старом профиле резал префил вдвое

Профиль `.env.bak-ornith-pgraph-on` нёс `PREFILL_CHUNK=64`. Замер на том же
промпте:

| chunk | префил 7K | префил 14K | префил 25K |
|---|---:|---:|---:|
| 64 | 627 | 587 | 535 |
| 512 | 1352 | 1266 | 1117 |

Причины для 64 ни в `docs/`, ни в `README` не нашлось; дефолт проекта — 512.
В рабочем профиле выставлено **512**.

### 2. `PGRAPH=off` ускоряет префил, но ломает декод на int8-пуле

| режим | префил 7K | префил 14K | префил 25K | декод |
|---|---:|---:|---:|---:|
| `PGRAPH=on`, checkpoints=1 | 1352 | 1266 | 1117 | 49,3 |
| `PGRAPH=off`, checkpoints=0 | 1512 | 1475 | 1381 | 43,9 |

`PGRAPH=off` даёт +9…+24% на префиле, но декод падает на 11%: в логе

```
[graphs] graphed decode failed, eager fallback: migrate KV slot 0:
int8-пул: обратная миграция пока не поддержана — graph-путь требует
batched-кэш (PGRAPH=on, это умолчание), счёт 1/3
```

То есть графовый префил — предусловие графового декода: он пишет KV сразу в
page-пул, а из int8-пула обратной миграции нет. Выигрыш `PGRAPH=off` на
префиле недостижим без починки миграции. **В рабочем профиле оставлен
`PGRAPH=on`.**

### 3. Branch-point checkpoints стоят ~5% префила

| режим | префил 7K | префил 14K | префил 25K |
|---|---:|---:|---:|
| `PREFIX_CACHE_CHECKPOINTS=1` | 1352 | 1266 | 1117 |
| `PREFIX_CACHE_CHECKPOINTS=0` | 1418 | 1307 | 1135 |

Это цена снятия нескольких host-снимков состояния на запрос. Она окупается на
ветвящейся агентской нагрузке (см.
`docs/research/2026-09-13-sglang-validation-report.md`: divergent branch
3,176 → 1,989 с), но на одиночных длинных промптах видна как −4…−5%.

### 4. llama.cpp игнорирует MTP-тензоры

Побочно видно, что 33-й блок в llama.cpp не используется вовсе — это ожидаемо
для text-only загрузки, но означает, что веса `blk.32.*` у llama.cpp просто
висят на диске.

---

## Чего сравнение не покрывает

- Качество выдачи. Мерилась только скорость и память.
- Спекулятивный декод: у обоих `MTP=0`.
- Контекст свыше 25K на префиле и свыше 16K на слот в параллельном режиме.
- Мультимодальность: GGUF text-only, обе стороны text-only.
- Влияние квантования KV на качество (q8 против f16 у llama.cpp по скорости
  совпало, качество не мерялось).

---

## Сырые данные и команды

**yforge:** `D:\Projects\yttri-inference\logs\bench-yforge-ornith.json`,
`bench-yforge-ornith-pgoff.json`, `prefill-on.json`, `prefill-off.json`,
`prefill-pgoff.json`, `server.log`.

**llama.cpp:** `D:\Projects\yttri-inference\logs\bench-llamacpp-ornith.json`,
`bench-llamacpp-ornith-q8.json`, `llama-ornith*.log/.err`.

**Параллельность:** `D:\Projects\yttri-inference\logs\concurrency.ps1`.

llama-server для замера:

```bat
llama-server.exe ^
  -m "D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf" ^
  --host 127.0.0.1 --port 18098 ^
  -ngl 999 -c 131072 -fa on -np 1 -ctk q8_0 -ctv q8_0 ^
  -a ornith-1.5-9b
```

Для 4 слотов: `-c 65536 -np 4`.

**Важно:** `llama-server.exe` из `D:\Projects\yttri-inference\llama.cpp-bin`
без CUDA 12.4 в `PATH` падает с `0xC0000135` (DLL not found) — это то же
требование, что зафиксировано в `scripts/README.md` для `yforge.exe`.

---

## Рабочий профиль после теста

`qwen36-server\.env` переведён на Ornith-1.5-9B:

```env
MODEL=D:\Models\deepreinforce-ai\Ornith-1.5-9B-GGUF\Ornith-1.5-9B-Q4_K_M.gguf
CTX=131072
SLOTS=1
KV_POOL_Q8=1
CUDA_GRAPHS=1
PGRAPH=on
PREFIX_CACHE_CHECKPOINTS=1
PREFILL_CHUNK=512
MTP=0
```

Предыдущий профиль сохранён как `.env.bak-pre-ornith-20260914` и
`.env.bak-ornith-pgraph-on`.

Сервер поднят, проверен живым запросом: `prompt=24 completion=200`,
`finish_reason=length`, выдача связная.

Временные задачи `llama-ornith-bench`, `llama-ornith-q8`, `llama-ornith-4slot`
удалены. Предсуществующие задачи `Llama27bFull`, `Llama27bIQ`, `Llama4bYttri`,
`llama_ornith15`, `llama_srv` не трогались.
