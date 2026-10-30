# Урок 2026-10-30: req_timeout захардкожен в BatchedEngine

## Симптом
Ornith-1.5-35B, prefill 128k-промпта: ровно на ~600-й секунде (чанк ~104,
pos 53 248 при ~5,8 с/чанк) клиент получал
`{"error":{"type":"internal_error","message":"timeout"}}`, сервер молча
останавливал prefill. Воспроизводилось на двух длинах промпта (131k и 54.5k)
на одной и той же позиции по времени, а не по токенам.

## Причина — две наложившиеся
1. `main.rs` создавал `BatchConfig` с захардкоженным
   `req_timeout: Duration::from_secs(600)`, игнорируя `cfg.req_timeout`
   (REQ_TIMEOUT из env/CLI). Лог печатал 7200, вотчдог в
   `engine_batched::dispatch_loop` жил с 600.
2. Вотчдог меряет время от последнего прогресса, но `last_progress`
   обновляется только на декоде (engine_batched.rs:1401). Долгий prefill
   без декода выглядит как «тишина» — вотчдог убивает живой запрос.

## Исправление
`main.rs`: `req_timeout: Duration::from_secs(cfg.req_timeout)` (коммит
`1a88430`). Для длинных prefill теперь достаточно поднять REQ_TIMEOUT.

## Осторожно на будущее
- Любая долгая фаза без `StreamEvent::Delta` (большие prefill, vision,
  большие снапшоты prefix-cache) требует либо большего REQ_TIMEOUT, либо
  обновления `last_progress` в этой фазе.
- Диагностика затруднялась тем, что ошибка не логировалась на сервере:
  вотчдог молча делает `free_slot`. При отладке таймаутов сначала сверять
  реальный `BatchConfig.req_timeout` в dispatch-потоке, а не в логе старта.
