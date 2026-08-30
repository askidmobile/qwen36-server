## 2026-08-31

**Topics updated:** prefix-cache, engine-layer, testing
**New topics:** none
**Sources scanned:** 54
**Sources changed:** yttri-forge/engine/qwen35-batch/src/real/adapter.rs, yttri-forge/engine/qwen35-batch/src/real/model_weights.rs; MTP graph lifetime сверена с yttri-forge/engine/qwen35-batch/src/real/mtp.rs
**Mode:** codebase, external runtime contract verification
**Notes:** Зафиксированы recurrent-only final snapshot, немедленный host transfer границы cache, лимит 65536 токенов и живой WDDM A/B без shared spill.

---

## 2026-08-30

**Topics updated:** none (актуальные статьи уже содержат изменения `f1110e0`)
**New topics:** none
**Sources scanned:** 51
**Sources changed:** none
**Mode:** codebase, incremental verification
**Notes:** Повторная компиляция сверила schema-aware tool parsing с live source; drift между исходниками и wiki не найден.

---

## 2026-08-30

**Topics updated:** http-api-layer, prefix-cache, engine-layer, testing
**Sources scanned:** 51
**Sources changed:** src/api.rs, src/api/openai.rs, src/api/anthropic.rs, src/api/tools.rs + связанный runtime contract yttri-forge
**Mode:** codebase
**Notes:** Schema-declared string tool arguments больше не меняют тип; CUDA A/B отделил Q8 KV от cache/MTP проблем; зафиксирован per-slot MTP alignment contract и очистка stale recurrent-state owner.

---

## 2026-08-30

**Topics updated:** project-overview, engine-layer, http-api-layer, web-chat, testing
**Topics created:** configuration-and-profiles, prefix-cache, multimodal-and-components
**Concepts updated:** candle-fork-coupling, not-send-serialization
**Sources scanned:** 51
**Sources changed:** full recompile + incremental sampling-policy update
**Mode:** codebase
**Notes:** Live Cargo/source contract now points to `yttri-forge`; sampling contract синхронизирован с BD-032 и официальным исследованием Ollama/LM Studio/vLLM.

## 2026-08-08

**Topics created:** project-overview, engine-layer, http-api-layer, web-chat, testing
**New topics:** project-overview, engine-layer, http-api-layer, web-chat, testing
**Concepts created:** candle-fork-coupling, not-send-serialization
**Sources scanned:** 25
**Sources changed:** 25 (first compile — all treated as new)
**Mode:** codebase
