# REPORT: вывод для qwen36-server

Основание: делегированное Pi исследование в
`2026-08-30-sampling-ollama-lmstudio-vllm.md` + приёмка текущего кода
`src/config.rs`, `src/api.rs`, `src/api/openai.rs`, `src/api/anthropic.rs`.

При приёмке найден и исправлен дефект поверх исследованного commit `c0edf49`:
`reasoning_effort=high|xhigh` ошибочно выбирал Ornith-пресет
`thinking-coding` (`temperature=0.6`, `presence_penalty=0.0`). Теперь effort
управляет только thinking/template, а sampling остаётся на общем
model-card-пресете и явных env overrides.

## Факты

- В `qwen36-server` commit `c0edf49` вводит `SamplingPolicy.lock`; default — locked, если `SAMPLING_LOCK` не равен `0`.
- OpenAI path (`src/api/openai.rs`) и Anthropic path (`src/api/anthropic.rs`) сначала берут model-card preset по режиму, затем накладывают явные env overrides. Sampling-поля клиента применяются только при `SAMPLING_LOCK=0`.
- Reasoning/thinking в коде — отдельная ось: `thinking`, `chat_template_kwargs.enable_thinking`, `reasoning_effort` управляют template behavior; `reasoning_effort` не выбирает sampling preset и не заменяет `temperature/top_p/...`.
- Проверенные Ollama, LM Studio и vLLM в обычном API-контракте принимают per-request generation params. vLLM явно: request value wins over generation_config defaults when not `None`; Ollama source явно: defaults → model options → request options.
- Официального общего server-side lock для запрета клиентских sampling overrides у Ollama/LM Studio/vLLM в просмотренных docs/source не найдено. Это не доказывает, что закрытых/недокументированных способов нет.

## Оценка `SAMPLING_LOCK=1`

### Соответствие обычной OpenAI-compatible семантике

Полный `SAMPLING_LOCK=1`, который молча игнорирует `temperature`, `top_p`, penalties и похожие поля клиента, **не соответствует обычной OpenAI-compatible семантике**. В совместимых API клиентские поля считаются параметрами конкретного запроса, а server/model/generation_config значения — defaults, когда клиент поле не задал.

Исключения вроде server cap на `max_tokens`/context/resource limits нормальны; полный lock всех sampling-полей — deployment policy, а не стандартная совместимость.

### Разумность как policy для конкретного agent endpoint

Для выделенного agent endpoint это **разумная защитная policy**, если явно задокументирована:

- есть воспроизведённый инцидент: клиент прислал `temperature=0`, модель вошла в tool-loop; model-card preset + `presence_penalty=1.5` разрывал конкретный синтетический цикл;
- endpoint обслуживает не произвольный OpenAI-compatible traffic, а конкретный агент, где сервер владеет model-card profile;
- `/v1/models` уже должен показывать effective sampling и `sampling_locked`, чтобы клиент не думал, что его поля применены.

Но для публичного/general OpenAI-compatible endpoint silent ignore опасен: ломает воспроизводимость, A/B тесты клиента и ожидания SDK.

## Какой контракт лучше

Рекомендованный контракт: **profiles/endpoints + allowlist/clamp**.

- `client-wins`: оставить для generic `/v1/chat/completions` совместимости. Request fields override defaults, но проходят валидацию и resource caps.
- `env-wins`: допустим для sealed internal agent endpoint, но должен быть явным: отдельный model id/profile/endpoint и response metadata с effective params.
- `allowlist/clamp`: лучший middle ground для agent endpoint. Например разрешить клиенту `max_tokens`, `stop`, возможно `seed`; sampling (`temperature/top_p/top_k/min_p/presence/repetition`) либо игнорировать, либо clamp в безопасный диапазон профиля.
- `profiles/endpoints`: лучше, чем один глобальный lock. Например `qwen36-agent`, `qwen36-compat`, `qwen36-coding`; каждый профиль фиксирует reasoning mode, sampling preset, max token policy и список allowed overrides.

Для текущего выделенного Ornith endpoint принят `env-wins`: lock включён по
умолчанию, активные значения публикуются в `/v1/models`, а возврат к
client-wins требует явного `SAMPLING_LOCK=0`. Разделение на agent/compat
profiles остаётся рекомендуемым следующим шагом, если endpoint станет общим.

## Reasoning отдельно от sampling

- Не позиционировать reasoning как замену sampling.
- Для Qwen-like моделей держать отдельный контракт: `thinking`/`reasoning_effort`/`chat_template_kwargs.enable_thinking` → template/reasoning mode; `temperature/top_p/...` → stochastic decoding.
- Нужны отдельные caps: reasoning on/off/effort/budget не должны случайно зависеть от `temperature`.
- Если hidden reasoning нужно убрать из ответа, это output formatting/include policy; генерация reasoning при этом может продолжаться.

## Повторяющиеся tool calls

Breaker должен жить в agent/tool loop layer, а не в sampler:

- повтор одинаковых законных tool calls определяется по истории, tool name, args, tool result и progress между раундами; sampler видит только текущий token stream;
- sampling может снизить вероятность залипания, но после 12–20 одинаковых законных вызовов нужна cycle detection/history compaction;
- если сервер только парсит `tool_calls`, breaker обязан быть у клиента/агента;
- если сервер сам выполняет MCP/tools, breaker обязан быть в server-owned integration loop.

Минимальный контракт breaker:

- max tool rounds / max tool calls per response/session;
- cap на одинаковую сигнатуру `(tool_name, normalized_args)` подряд и в sliding window;
- no-progress detector: одинаковый tool result + одинаковая следующая call signature;
- история failed/invalid calls компактируется, а не копится как few-shot пример ошибки;
- unknown tool call — 400/structured error или явный log/metric; не молча продолжать.
