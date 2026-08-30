# Wiki Schema — qwen36-server

Last compiled: 2026-08-31

## Topics

| Slug | Name | Description |
|---|---|---|
| project-overview | Project Overview | Фактический scope Yttri Self-Inference Server, runtime и решения |
| engine-layer | Engine Layer | Engine contract, Batched/Candle/Swappable engines, GPU lifecycle |
| http-api-layer | HTTP API Layer | Inference/admin/HF/media API, auth, SSE и tools |
| configuration-and-profiles | Configuration and Profiles | `.env`, model profiles, capabilities и sampling policy |
| prefix-cache | Prefix Cache | Host-backed state snapshots, prefix matching, LRU и int8 pool |
| multimodal-and-components | Multimodal and Components | Media TTL pipeline, profiles, vision/video/MTP lifecycle |
| web-chat | Web UI and Studio Proxy | Unsloth Studio proxy и встроенный fallback WebUI |
| testing | Testing and Validation | Unit/API/CUDA parity, cancellation, memory и stability gates |

## Concepts

| Slug | Name | Connects |
|---|---|---|
| candle-fork-coupling | Runtime Path-Dependency on yttri-forge | project-overview, engine-layer, prefix-cache, testing |
| not-send-serialization | Single-Owner GPU State | engine-layer, prefix-cache, testing |

## Evolution Log

- 2026-08-30: Зафиксирован BD-032 и внешне проверенный sampling/reasoning/tool-loop contract.
- 2026-08-30: 5 → 8 topics; актуализирован runtime `yttri-forge`, добавлены profiles, components и prefix reuse.
- 2026-08-08: Initial schema generated from 5 topics and 2 concepts.
