//! Prefix cache (FR-T328): snapshot state после prefill → повторный prompt
//! без полного prefill. Уникально для hybrid DeltaNet моделей (llama.cpp их
//! не кэширует: «forcing full prompt re-processing due to recurrent memory»).
//!
//! Поиск по ПРЕФИКСУ, не по точному совпадению: в агентской переписке промпт
//! хода N+1 — это промпт хода N плюс ответ и новый вопрос, точные ключи не
//! совпадают никогда. Как в llama.cpp/vLLM: хеш по блокам фиксированной длины
//! (BLOCK_TOKENS = страница paged-пула), кандидат — по хешу блочной границы,
//! решение — сверка токенов: коллизия хеша даёт промах, чужое состояние не
//! подставляется.
//!
//! Снимок покрывает ровно `position` токенов (длину закешированного промпта),
//! поэтому попадание всегда возвращает prefix_len = длине записи; остаток
//! промпта досчитывается обычным prefill'ом (scheduler.submit_primed). Полное
//! совпадение (prefix_len == tokens.len()) не отдаётся: primed-пути нужен
//! хотя бы один токен на последний прогон через модель.
//!
//! LRU по суммарному размеру snapshot'ов (MiB). Эвикт: самый давно
//! неиспользованный. Снимки хранятся в СИСТЕМНОЙ памяти (host), не в VRAM:
//! иначе кеш вытесняет KV-пул за пределы видеопамяти (WDDM 2026-08-23).
//! `put` переносит снимок на CPU, `find` возвращает его уже на устройстве
//! модели.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use candle_core::Device;
use qwen35_batch::real::model_weights::StateSnapshot;

/// Блок поиска = страница paged KV-пула (PAGE_SIZE = 64): снимок всё равно
/// режется страницами.
pub const BLOCK_TOKENS: usize = 64;

/// Асинхронный перенос снимков в системную память.
///
/// `StateSnapshot::to_host()` — это D2H примерно 0.5 ГиБ на 30k контекста
/// (8 слоёв внимания × K/V). В потоке планировщика он держит первый токен
/// ответа: замер 2026-09-17 — TTFT 19.86…19.95 с с кешем против 19.66 с
/// `PREFIX_CACHE_MIB=0`. Поэтому снимки уезжают воркеру, а планировщик
/// забирает уже готовые host-снимки на ближайшей итерации: клиент получает
/// токен, не дожидаясь копии, кеш наполняется через ~0.2 с после префила.
pub struct HostSnapshotWorker {
    tx: std::sync::mpsc::Sender<(Vec<u32>, Vec<(usize, StateSnapshot)>, Option<Vec<f32>>)>,
    rx: std::sync::mpsc::Receiver<(Vec<u32>, Vec<(usize, StateSnapshot)>, Option<Vec<f32>>)>,
}

impl HostSnapshotWorker {
    pub fn new() -> Self {
        let (job_tx, job_rx) =
            std::sync::mpsc::channel::<(Vec<u32>, Vec<(usize, StateSnapshot)>, Option<Vec<f32>>)>();
        let (res_tx, res_rx) =
            std::sync::mpsc::channel::<(Vec<u32>, Vec<(usize, StateSnapshot)>, Option<Vec<f32>>)>();
        let spawned = std::thread::Builder::new()
            .name("pcache-to-host".to_string())
            .spawn(move || {
                while let Ok((tokens, snaps, logits)) = job_rx.recv() {
                    let mut host = Vec::with_capacity(snaps.len());
                    for (pos, snap) in snaps {
                        match snap.to_host() {
                            Ok(host_snap) => host.push((pos, host_snap)),
                            Err(err) => {
                                eprintln!("[pcache] to_host не удался: {err}");
                            }
                        }
                    }
                    if host.is_empty() {
                        continue;
                    }
                    if res_tx.send((tokens, host, logits)).is_err() {
                        break;
                    }
                }
            });
        if let Err(err) = spawned {
            eprintln!("[pcache] воркер переноса в host не запустился: {err}");
        }
        Self { tx: job_tx, rx: res_rx }
    }

    /// Отдать снимки воркеру (не блокирует). `logits` — логиты последней
    /// позиции промпта, если снимок снят ровно на его длине.
    pub fn submit(
        &self,
        tokens: Vec<u32>,
        snapshots: Vec<(usize, StateSnapshot)>,
        logits: Option<Vec<f32>>,
    ) {
        if snapshots.is_empty() {
            return;
        }
        let _ = self.tx.send((tokens, snapshots, logits));
    }

    /// Забрать готовые host-снимки, не блокируясь.
    pub fn drain(&self) -> Vec<(Vec<u32>, Vec<(usize, StateSnapshot)>, Option<Vec<f32>>)> {
        self.rx.try_iter().collect()
    }
}

impl Default for HostSnapshotWorker {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct PrefixHit {
    pub snap: StateSnapshot,
    /// Длина совпавшего префикса в токенах; кратна BLOCK_TOKENS не обязана быть —
    /// равна длине закешированного промпта.
    pub prefix_len: usize,
    /// Логиты последней позиции префикса — есть только у записи, снятой на
    /// границе ПОЛНОГО промпта. С ними попадание «ровно в длину» не требует
    /// prefill'а вообще: берём логиты из записи и сразу сэмплируем.
    pub logits: Option<Vec<f32>>,
}

struct Entry {
    id: u64,
    /// Хеш блочной границы записи (по которому её находит find).
    key: u64,
    snap: StateSnapshot,
    /// Логиты последней позиции (см. PrefixHit::logits).
    logits: Option<Vec<f32>>,
    size_bytes: usize,
    /// Полный токенный состав закешированного промпта — для защиты от
    /// hash collision: find сверяет токены, а не только хеш.
    tokens: Vec<u32>,
}

pub struct PrefixCache {
    /// key (хеш блочной границы) → id записей. Несколько записей в корзине —
    /// коллизии/общие префиксы; разбираются сверкой токенов.
    buckets: HashMap<u64, Vec<u64>>,
    by_id: HashMap<u64, Entry>,
    /// LRU: front = самый старый.
    lru: VecDeque<u64>,
    next_id: u64,
    total_bytes: usize,
    budget_bytes: usize,
}

/// Хеш цепочки блоков: h_n = hash(h_{n-1}, tokens[n*B..(n+1)*B]).
/// Только ПОЛНЫЕ блоки: граница записи (len/B блоков) обязана совпадать
/// с границей входящего промпта при равном содержимом, иначе хеш последнего
/// неполного блока зависел бы от длины конкретного промпта.
fn boundary_hashes(tokens: &[u32]) -> Vec<u64> {
    use std::hash::{Hash, Hasher};
    let n_blocks = tokens.len() / BLOCK_TOKENS;
    let mut out = Vec::with_capacity(n_blocks);
    let mut prev: u64 = 0;
    for block in tokens[..n_blocks * BLOCK_TOKENS].chunks(BLOCK_TOKENS) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        prev.hash(&mut h);
        block.hash(&mut h);
        prev = h.finish();
        out.push(prev);
    }
    out
}

impl PrefixCache {
    pub fn new(budget_mib: usize) -> Self {
        Self {
            buckets: HashMap::new(),
            by_id: HashMap::new(),
            lru: VecDeque::new(),
            next_id: 0,
            total_bytes: 0,
            budget_bytes: budget_mib.saturating_mul(1024 * 1024),
        }
    }

    /// Положить снимок, покрывающий `tokens` целиком (snap.position == len).
    /// Промпты короче блока не кэшируются — попадание с них невозможно.
    /// Возвращает true, если запись сохранена.
    pub fn put(&mut self, tokens: Vec<u32>, snap: StateSnapshot) -> bool {
        self.put_with_logits(tokens, snap, None)
    }

    /// Как `put`, но вместе со снимком кладёт логиты последней позиции:
    /// запись становится пригодной для попадания «ровно в длину».
    pub fn put_with_logits(
        &mut self,
        tokens: Vec<u32>,
        snap: StateSnapshot,
        logits: Option<Vec<f32>>,
    ) -> bool {
        debug_assert_eq!(
            snap.position,
            tokens.len(),
            "снимок не совпадает с токенами"
        );
        if tokens.len() < BLOCK_TOKENS {
            return false;
        }
        let size_bytes = snap
            .size_bytes()
            .saturating_add(tokens.len().saturating_mul(std::mem::size_of::<u32>()));
        if size_bytes > self.budget_bytes {
            return false; // snapshot больше всего бюджета — не кэшируем
        }
        let key = boundary_hashes(&tokens)[tokens.len() / BLOCK_TOKENS - 1];
        // Дедуп: запись с теми же токенами уже лежит (прирост промпта за ход
        // меньше чанка — граница повторяется, снимки эквивалентны). Дубликат
        // не создаём: снимок выбрасываем, позицию записи в LRU обновляем.
        // Ключ — хеш блочной границы, равенство ключа обычно означает
        // равенство токенов, но корректность держим на сверке токенов.
        if let Some(id) = self.buckets.get(&key).and_then(|ids| {
            ids.iter().copied().find(|&id| {
                self.by_id
                    .get(&id)
                    .is_some_and(|e| e.tokens.as_slice() == tokens.as_slice())
            })
        }) {
            self.touch(id);
            return true;
        }
        // Хранение — в системной памяти: VRAM оставляем живому KV-пулу модели.
        let snap = match snap.to_host() {
            Ok(s) => s,
            Err(err) => {
                eprintln!("[pcache] put: перенос снимка в host-память не удался: {err}");
                return false;
            }
        };
        while self.total_bytes.saturating_add(size_bytes) > self.budget_bytes
            && !self.lru.is_empty()
        {
            self.evict_one();
        }
        let id = self.next_id;
        self.next_id += 1;
        self.total_bytes += size_bytes;
        self.lru.push_back(id);
        self.buckets.entry(key).or_default().push(id);
        self.by_id.insert(
            id,
            Entry {
                id,
                key,
                snap,
                logits,
                size_bytes,
                tokens,
            },
        );
        true
    }

    /// Положить несколько checkpoint'ов одного prompt'а.
    ///
    /// Это branch-point вариант поверх обычного `put`: один и тот же prompt
    /// сохраняется в нескольких точках, поэтому divergent ветка может попасть
    /// в более ранний общий префикс, а не только в последнюю границу чанка.
    /// Позиции за пределами prompt'а или нулевые отбрасываются.
    pub fn put_many(
        &mut self,
        tokens: &[u32],
        snapshots: Vec<(usize, StateSnapshot)>,
        logits: Option<Vec<f32>>,
    ) -> usize {
        let mut saved = 0usize;
        for (pos, snap) in snapshots {
            if pos == 0 || pos > tokens.len() {
                continue;
            }
            // Логиты относятся только к записи во всю длину промпта.
            let entry_logits = if pos == tokens.len() { logits.clone() } else { None };
            if self.put_with_logits(tokens[..pos].to_vec(), snap, entry_logits) {
                saved += 1;
            }
        }
        saved
    }

    /// Самый длинный закешированный префикс `tokens`. Кандидат — по хешам
    /// блочных границ от старших к младшим, решение — сверка токенов.
    /// Полное совпадение не возвращается: на primed-пути должен остаться
    /// хотя бы один токен prefill'а.
    /// Снимок возвращается уже перенесённым на `device` модели.
    pub fn find(&mut self, tokens: &[u32], device: &Device) -> Option<PrefixHit> {
        let hashes = boundary_hashes(tokens);
        for &h in hashes.iter().rev() {
            let Some(ids) = self.buckets.get(&h).map(Vec::as_slice) else {
                continue;
            };
            // Длиннейшая запись корзины, чьи токены — реальный префикс.
            let mut best: Option<u64> = None;
            let mut best_len = 0usize;
            for &id in ids {
                let Some(e) = self.by_id.get(&id) else {
                    continue;
                };
                // Полное совпадение отдаём только тогда, когда в записи есть
                // логиты последней позиции: иначе primed-пути неоткуда взять
                // логиты для первого сэмпла (см. submit_fully_primed).
                let fits = e.tokens.len() < tokens.len()
                    || (e.tokens.len() == tokens.len() && e.logits.is_some());
                if e.tokens.len() > best_len
                    && fits
                    && e.tokens.as_slice() == &tokens[..e.tokens.len()]
                {
                    best = Some(id);
                    best_len = e.tokens.len();
                }
            }
            if let Some(id) = best {
                self.touch(id);
                let e = &self.by_id[&id];
                // Снимок хранится в host-памяти — возвращаем на устройство.
                let snap = match e.snap.to_device(device) {
                    Ok(s) => s,
                    Err(err) => {
                        eprintln!("[pcache] find: перенос снимка на устройство не удался: {err}");
                        return None;
                    }
                };
                return Some(PrefixHit {
                    snap,
                    prefix_len: e.tokens.len(),
                    logits: e.logits.clone(),
                });
            }
            // Коллизия хеша / содержимое разошлось — ищем по младшим границам.
        }
        // Диагностика промаха: длиннейшее совпадение по всем записям. Нужна
        // потому, что промах молчалив и неотличим от выключенного кеша, а
        // разошедшийся хвост промпта виден только этим числом.
        if let Some(best) = self
            .by_id
            .values()
            .map(|e| {
                e.tokens
                    .iter()
                    .zip(tokens)
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .max()
        {
            eprintln!(
                "[pcache] miss: prompt {} tok, longest common prefix {best}",
                tokens.len()
            );
        }
        None
    }

    fn evict_one(&mut self) {
        while let Some(id) = self.lru.pop_front() {
            if let Some(e) = self.by_id.remove(&id) {
                self.total_bytes -= e.size_bytes;
                if let Some(v) = self.buckets.get_mut(&e.key) {
                    v.retain(|&x| x != id);
                    if v.is_empty() {
                        self.buckets.remove(&e.key);
                    }
                }
                return;
            }
        }
    }

    fn touch(&mut self, id: u64) {
        self.lru.retain(|&x| x != id);
        self.lru.push_back(id);
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

pub type SharedPrefixCache = Mutex<PrefixCache>;

#[cfg(test)]
impl PrefixCache {
    /// Тестам: положить запись под принудительным ключом (моделирование
    /// коллизии хеша у разных токенов).
    fn put_forced(&mut self, key: u64, tokens: Vec<u32>, snap: StateSnapshot) {
        let size_bytes = snap.size_bytes();
        let id = self.next_id;
        self.next_id += 1;
        self.total_bytes += size_bytes;
        self.lru.push_back(id);
        self.buckets.entry(key).or_default().push(id);
        self.by_id.insert(
            id,
            Entry {
                id,
                key,
                snap,
                logits: None,
                size_bytes,
                tokens,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use qwen35_batch::real::model_weights::{BlockStateSnap, DeltaNetStateSnap};

    fn fake_snap_bytes(bytes: usize) -> StateSnapshot {
        let floats = bytes / std::mem::size_of::<f32>();
        StateSnapshot {
            model_nonce: 1,
            position: 0, // выставит вызывающий (put проверяет совпадение)
            blocks: vec![BlockStateSnap::DeltaNet(DeltaNetStateSnap {
                conv_buf: vec![0.0; floats / 2],
                ssm_state: vec![0.0; floats / 2],
            })],
        }
    }

    fn snap_for(tokens: &[u32], bytes: usize) -> StateSnapshot {
        let mut s = fake_snap_bytes(bytes);
        s.position = tokens.len();
        s
    }

    fn tokens_from(start: u32, len: usize) -> Vec<u32> {
        (start..start + len as u32).collect()
    }

    #[test]
    fn extending_prompt_hits_full_cached_prefix() {
        let mut c = PrefixCache::new(64);
        let cached = tokens_from(100, 2 * BLOCK_TOKENS);
        assert!(c.put(cached.clone(), snap_for(&cached, 4096)));
        // Ход N+1 = ход N + ответ + вопрос → префикс совпадает целиком.
        let mut next = cached.clone();
        next.extend(tokens_from(9000, 37));
        let hit = c.find(&next, &Device::Cpu).expect("hit");
        assert_eq!(hit.prefix_len, 2 * BLOCK_TOKENS);
        assert_eq!(hit.snap.position, 2 * BLOCK_TOKENS);
    }

    #[test]
    fn exact_repeat_is_miss() {
        let mut c = PrefixCache::new(64);
        let cached = tokens_from(1, 2 * BLOCK_TOKENS);
        c.put(cached.clone(), snap_for(&cached, 4096));
        // Совпадение до последнего токена: primed-пути нужен ≥1 токен prefill'а.
        assert!(c.find(&cached, &Device::Cpu).is_none());
    }

    #[test]
    fn divergent_prompt_is_miss() {
        let mut c = PrefixCache::new(64);
        let cached = tokens_from(1, 2 * BLOCK_TOKENS);
        c.put(cached, snap_for(&tokens_from(1, 2 * BLOCK_TOKENS), 4096));
        // Общий первый блок, расхождение во втором.
        let mut other = tokens_from(1, BLOCK_TOKENS);
        other.extend(tokens_from(7777, BLOCK_TOKENS));
        assert!(c.find(&other, &Device::Cpu).is_none());
        // Расхождение в первом блоке.
        assert!(c
            .find(&tokens_from(500, 4 * BLOCK_TOKENS), &Device::Cpu)
            .is_none());
        assert!(c
            .find(&tokens_from(1, BLOCK_TOKENS), &Device::Cpu)
            .is_none());
    }

    #[test]
    fn longest_cached_prefix_wins() {
        let mut c = PrefixCache::new(64);
        let short = tokens_from(1, 2 * BLOCK_TOKENS);
        c.put(short.clone(), snap_for(&short, 4096));
        let mut long = short.clone();
        long.extend(tokens_from(500, BLOCK_TOKENS));
        c.put(long.clone(), snap_for(&long, 4096));
        // Ход продолжается третьим блоком → попадает длинная запись.
        let mut next = long.clone();
        next.extend(tokens_from(800, BLOCK_TOKENS));
        assert_eq!(
            c.find(&next, &Device::Cpu).unwrap().prefix_len,
            3 * BLOCK_TOKENS
        );
        // Третий блок другой → отпадает длинная, остаётся короткая.
        let mut next2 = short.clone();
        next2.extend(tokens_from(999, BLOCK_TOKENS));
        assert_eq!(
            c.find(&next2, &Device::Cpu).unwrap().prefix_len,
            2 * BLOCK_TOKENS
        );
    }

    #[test]
    fn hash_collision_falls_back_to_token_verification() {
        let mut c = PrefixCache::new(64);
        let cached = tokens_from(1, 2 * BLOCK_TOKENS);
        assert!(c.put(cached.clone(), snap_for(&cached, 4096)));
        // Чужая запись под тем же ключом (симуляция коллизии хеша):
        // подстановка её состояния запрещена, сверка токенов обязана отсеять.
        let stranger = tokens_from(42, 2 * BLOCK_TOKENS);
        let key = boundary_hashes(&cached)[cached.len() / BLOCK_TOKENS - 1];
        c.put_forced(key, stranger.clone(), snap_for(&stranger, 4096));
        let mut next = cached.clone();
        next.extend(tokens_from(9000, 10));
        let hit = c
            .find(&next, &Device::Cpu)
            .expect("настоящая запись находится");
        assert_eq!(hit.snap.position, 2 * BLOCK_TOKENS);
        assert_eq!(hit.prefix_len, 2 * BLOCK_TOKENS);
        // Промпт «чужой» записи не получает её состояние (общий хеш, разное
        // содержимое) — conservative промах.
        let mut stranger_next = stranger.clone();
        stranger_next.extend(tokens_from(7000, 10));
        assert!(c.find(&stranger_next, &Device::Cpu).is_none());
    }

    #[test]
    fn short_prompts_are_not_cached() {
        let mut c = PrefixCache::new(64);
        let small = tokens_from(1, 10);
        assert!(!c.put(small.clone(), snap_for(&small, 4096)));
        assert!(c.find(&small, &Device::Cpu).is_none());
    }

    #[test]
    fn lru_eviction_by_budget() {
        let entry_bytes = 1024 * 1024 - BLOCK_TOKENS * std::mem::size_of::<u32>();
        let mut c = PrefixCache::new(3);
        let mk = |start: u32| {
            let t = tokens_from(start, BLOCK_TOKENS);
            (t.clone(), snap_for(&t, entry_bytes))
        };
        for start in [1, 2, 3] {
            let (t, s) = mk(start);
            assert!(c.put(t, s));
        }
        assert_eq!(c.len(), 3);
        // 4-й не влезает → вытесняет самый старый (start=1).
        let (t4, s4) = mk(4);
        assert!(c.put(t4, s4));
        let mut q1 = tokens_from(1, BLOCK_TOKENS);
        q1.extend(tokens_from(9000, BLOCK_TOKENS));
        assert!(c.find(&q1, &Device::Cpu).is_none());
        let mut q2 = tokens_from(2, BLOCK_TOKENS);
        q2.extend(tokens_from(9000, BLOCK_TOKENS));
        assert!(c.find(&q2, &Device::Cpu).is_some());
        let mut q4 = tokens_from(4, BLOCK_TOKENS);
        q4.extend(tokens_from(9000, BLOCK_TOKENS));
        assert!(c.find(&q4, &Device::Cpu).is_some());
        // get(2) поднял его в LRU → следующим вытесняется start=3.
        let (t5, s5) = mk(5);
        assert!(c.put(t5, s5));
        let mut q3 = tokens_from(3, BLOCK_TOKENS);
        q3.extend(tokens_from(9000, BLOCK_TOKENS));
        assert!(c.find(&q3, &Device::Cpu).is_none());
        assert!(c.find(&q2, &Device::Cpu).is_some());
    }

    /// Снимок, положенный в кеш (put переносит в host), после find возвращается
    /// на устройстве, запрошенном в find. Здесь CPU: тензоры внимания в тест
    /// не положить (KvCacheSnap::from_pool pub(crate) — конструирование только
    /// внутри крейта); перенос настоящих K/V-тензоров проверяет тест
    /// to_host_then_to_device_roundtrip в движке (model_weights.rs).
    #[test]
    fn find_returns_snapshot_for_requested_device() {
        let mut c = PrefixCache::new(64);
        let cached = tokens_from(1, 2 * BLOCK_TOKENS);
        assert!(c.put(cached.clone(), snap_for(&cached, 4096)));
        let mut next = cached.clone();
        next.extend(tokens_from(9000, 37));
        let hit = c.find(&next, &Device::Cpu).expect("hit");
        assert_eq!(hit.prefix_len, 2 * BLOCK_TOKENS);
        assert_eq!(hit.snap.position, 2 * BLOCK_TOKENS);
        for b in &hit.snap.blocks {
            match b {
                BlockStateSnap::DeltaNet(dn) => {
                    assert_eq!(dn.conv_buf, vec![0.0; dn.conv_buf.len()]);
                }
                BlockStateSnap::Attention(kv) => assert!(kv.is_none()),
            }
        }
    }

    /// Повторный put того же префикса не создаёт дубликат: len == 1,
    /// total_bytes не вырос (снимок дубликата выбрасывается до to_host).
    #[test]
    fn duplicate_prefix_is_deduplicated() {
        let mut c = PrefixCache::new(64);
        let cached = tokens_from(1, 2 * BLOCK_TOKENS);
        assert!(c.put(cached.clone(), snap_for(&cached, 4096)));
        let bytes = c.total_bytes();
        let snap = snap_for(&cached, 4096);
        assert!(c.put(cached, snap));
        assert_eq!(c.len(), 1);
        assert_eq!(c.total_bytes(), bytes);
    }

    /// Дедуп-put поднимает запись в LRU: дальше вытесняется сосед, не дубль.
    #[test]
    fn duplicate_put_touches_lru() {
        let entry_bytes = 1024 * 1024 - BLOCK_TOKENS * std::mem::size_of::<u32>();
        let mut c = PrefixCache::new(3);
        let mk = |start: u32| {
            let t = tokens_from(start, BLOCK_TOKENS);
            (t.clone(), snap_for(&t, entry_bytes))
        };
        for start in [1, 2, 3] {
            let (t, s) = mk(start);
            assert!(c.put(t, s));
        }
        let (t1, s1) = mk(1);
        assert!(c.put(t1, s1));
        assert_eq!(c.len(), 3);
        // 4-я запись вытесняет самую старую — start=2, а не поднятый дубль start=1.
        let (t4, s4) = mk(4);
        assert!(c.put(t4, s4));
        let mut q1 = tokens_from(1, BLOCK_TOKENS);
        q1.extend(tokens_from(9000, BLOCK_TOKENS));
        assert!(c.find(&q1, &Device::Cpu).is_some());
        let mut q2 = tokens_from(2, BLOCK_TOKENS);
        q2.extend(tokens_from(9000, BLOCK_TOKENS));
        assert!(c.find(&q2, &Device::Cpu).is_none());
    }

    #[test]
    fn multi_boundary_put_hits_earlier_branch_point() {
        let mut c = PrefixCache::new(64);
        let prefix = tokens_from(1, 3 * BLOCK_TOKENS);
        let snapshots = vec![
            (BLOCK_TOKENS, snap_for(&prefix[..BLOCK_TOKENS], 1024)),
            (
                2 * BLOCK_TOKENS,
                snap_for(&prefix[..2 * BLOCK_TOKENS], 2048),
            ),
        ];
        assert_eq!(c.put_many(&prefix, snapshots, None), 2);

        // Ветка расходится после второго блока: длинный checkpoint не подходит,
        // а ранний checkpoint на первом блоке остаётся валидным.
        let mut divergent = tokens_from(1, BLOCK_TOKENS);
        divergent.extend(tokens_from(7777, 2 * BLOCK_TOKENS));
        let hit = c.find(&divergent, &Device::Cpu).expect("early hit");
        assert_eq!(hit.prefix_len, BLOCK_TOKENS);
        assert_eq!(hit.snap.position, BLOCK_TOKENS);
    }
}
