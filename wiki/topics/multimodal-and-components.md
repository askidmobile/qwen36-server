---
topic: Multimodal and Components
slug: multimodal-and-components
last_compiled: 2026-08-30
sources: 10
status: active
---

# Multimodal and Components

## Purpose [coverage: high — 10 sources]

Multimodal subsystem принимает image/video, хранит bytes ограниченное время, готовит inputs внешним helper и допускает их только в runtime с реальной capability. Profiles связывают text GGUF с vision/MTP artifacts.

## Architecture [coverage: high — 10 sources]

- `MediaStore`: owner-scoped temporary objects, TTL/reaper и одноразовое использование.
- `fetch/prepare`: проверка MIME/лимитов, download и preprocessing.
- `qwen36-media-helper`: отдельный процесс/protocol для media transforms.
- `ComponentManager`: leases, warm TTL и приоритет освобождения VRAM.
- `profile.rs`: versioned manifests, hashes, artifact inventory и declared capabilities.

## API Surface [coverage: high — 7 sources]

- Media upload raw/multipart за тем же Bearer auth.
- OpenAI/Anthropic content blocks преобразуются в `ContentBlock`/`MediaSource`.
- Text-only runtime возвращает `component_unavailable` до inference.

## Key Decisions [coverage: high — 6 sources]

- BD-022 заменил text-only границу валидированным Vision/Video runtime.
- BD-026 разрешает только временные пользовательские media bytes.
- BD-030 заменил lazy MTP startup-load; Vision lifecycle остаётся отдельным.

## Gotchas [coverage: high — 8 sources]

- Наличие `mmproj` рядом с GGUF не означает runtime support.
- Declared profile capability должна пересекаться с `Engine::supports_*`.
- Media bytes и request identifiers нельзя писать в logs.
- Старый `ComponentManager` комментарий о lazy MTP не соответствует BD-030/main runtime и не должен использоваться как источник текущего поведения.

## Sources

- [src/media/mod.rs](../../src/media/mod.rs)
- [src/media/store.rs](../../src/media/store.rs)
- [src/media/fetch.rs](../../src/media/fetch.rs)
- [src/media/prepare.rs](../../src/media/prepare.rs)
- [src/media/helper.rs](../../src/media/helper.rs)
- [src/media/helper_protocol.rs](../../src/media/helper_protocol.rs)
- [src/api/media.rs](../../src/api/media.rs)
- [src/component_manager.rs](../../src/component_manager.rs)
- [src/profile.rs](../../src/profile.rs)
- [src/bin/qwen36-media-helper.rs](../../src/bin/qwen36-media-helper.rs)
