# DGX Spark: DeepSeek V4.1 Flash 1017 Uncensored FP8

Дата аудита: 2026-10-19. Стенд: `dgx-forge.gpuwerk.com`, NVIDIA GB10,
`compute_cap=12.1`, 48 SM, доступно 121 GiB unified memory.

## Запрошенный артефакт

[`dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8)
— weight-level abliteration официального `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
Формат натива: FP8 E4M3 с ue8m0 block scales и FP4 routed experts. Это не GGUF
и не VQ-артефакт.

| Параметр | Значение |
|---|---:|
| HF revision | `c19ca5ca4eaeb79d29fdf08aea8327da95b1116a` |
| Всего параметров по safetensors metadata | 763,205,315,794 |
| Размер репозитория | 510,296,921,671 B (475.3 GiB) |
| Base weights без двух Engram shards | 307,223,844,095 B (286.1 GiB) |
| Два Engram shard | 203,073,077,576 B (189.1 GiB) |
| Контекст | 1,048,576 токенов |
| Reference runtime | `inference/` + `convert.py`, tensor parallel (`MP=4` или `MP=8`) |

## Вывод по памяти

Даже если Engram-таблицы читать с диска и не считать их resident, базовые веса
занимают 286.1 GiB. Это больше всей доступной unified memory GB10 (121 GiB)
ещё до KV, workspace, CUDA context и vision tower. Reference README запускает
модель через `torchrun` на 4×H200 (`MP=4`) и предлагает `MP=8` для большего
числа GPU; single-GPU пути для этого checkpoint нет.

Практический вывод: **нативный UNCENSORED FP8 на одном DGX Spark запустить
нельзя**. Это не ограничение текущей сборки Forge, а размер checkpoint.

## Почему существует VQ-GGUF

[`Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF`](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-VQ-GGUF)
сжимает base weights до 113,556,639,424 B (105.8 GiB), оставляя Engram на диске.
Именно поэтому он помещается в 128 GB Spark. Это отдельный lossy VQ-формат,
который читает `ds4-dfm-rs`; исходный FP8 checkpoint им не читается.

VQ-GGUF собран из **официального 1017 base**, не из UNCENSORED FP8. Подменять
один артефакт другим без отдельного прогона качества нельзя. Model card
UNCENSORED прямо предупреждает, что наивное низкобитное квантование FP8-весов
может ослабить abliteration; правильный путь — повторить abliteration на
целевой точности.

## Другие варианты

[`dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED`](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED)
имеет 43 base shard размером около 212.8 GB (198.2 GiB) плюс те же 203.1 GB
Engram. Base всё ещё больше 121 GiB; кроме того, `ds4-dfm-rs` не заявляет
поддержку EXL3 checkpoint.

Рабочие варианты:

1. Запускать UNCENSORED FP8 на 4×H200 (или большем числе GPU) через его
   reference `inference/` runtime — не на этом стенде.
2. Сделать новый abliterated VQ-артефакт по пайплайну модели и проверить
   качество; это отдельный тяжёлый этап и не гарантируется текущим VQ-кодеком.
3. Для проверки инфраструктуры GB10 отдельно запустить опубликованный
   1017 VQ-GGUF в `ds4-dfm-rs`, явно обозначив, что это base, а не
   UNCENSORED-вариант.

## Обновление: найден UNCENSORED VQ-GGUF

После аудита FP8 найден новый артефакт
[`Baekpica/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF`](https://huggingface.co/Baekpica/DeepSeek-V4.1-Flash-1017-UNCENSORED-VQ-GGUF),
revision `610a2d71400313f63228589bf4cea4182ad360e9`. Он собран из того же
dealignai FP8 checkpoint тем же `vq8sh14-q4k-mtpnative` кодеком.

- Репозиторий: 317,601,046,990 B (295.8 GiB).
- Base GGUF: 113,556,639,424 B (105.8 GiB), помещается в unified memory Spark.
- Vision sidecar: 971,066,016 B.
- Engram: 203,073,077,576 B, читается с NVMe.
- Loader уже проверен автором: `bind: slots=1000 bound=1000 required-missing=0`.
- 54 из 1,000 тензоров отличаются от base VQ — ровно затронутые abliteration.

Это позволяет провести performance- и runtime-тест на DGX Spark. Важное
ограничение: автор не публиковал измерений скорости и качества для этого
артефакта; VQ lossy, и uncensored-поведение после квантования может отличаться
от FP8. Для теста инфраструктуры и производительности артефакт пригоден; для
утверждения о качестве нужен отдельный quality gate.
