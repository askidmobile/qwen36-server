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
//! неиспользованный.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use qwen35_batch::real::model_weights::StateSnapshot;

/// Блок поиска = страница paged KV-пула (PAGE_SIZE = 64): снимок всё равно
/// режется страницами.
pub const BLOCK_TOKENS: usize = 64;

#[derive(Debug, Clone)]
pub struct PrefixHit {
    pub snap: StateSnapshot,
    /// Длина совпавшего префикса в токенах; кратна BLOCK_TOKENS не обязана быть —
    /// равна длине закешированного промпта.
    pub prefix_len: usize,
}

struct Entry {
    id: u64,
    /// Хеш блочной границы записи (по которому её находит find).
    key: u64,
    snap: StateSnapshot,
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
        debug_assert_eq!(snap.position, tokens.len(), "снимок не совпадает с токенами");
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
        while self.total_bytes.saturating_add(size_bytes) > self.budget_bytes && !self.lru.is_empty()
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
                size_bytes,
                tokens,
            },
        );
        true
    }

    /// Самый длинный закешированный префикс `tokens`. Кандидат — по хешам
    /// блочных границ от старших к младшим, решение — сверка токенов.
    /// Полное совпадение не возвращается: на primed-пути должен остаться
    /// хотя бы один токен prefill'а.
    pub fn find(&mut self, tokens: &[u32]) -> Option<PrefixHit> {
        let hashes = boundary_hashes(tokens);
        for &h in hashes.iter().rev() {
            let Some(ids) = self.buckets.get(&h).map(Vec::as_slice) else {
                continue;
            };
            // Длиннейшая запись корзины, чьи токены — реальный префикс.
            let mut best: Option<u64> = None;
            let mut best_len = 0usize;
            for &id in ids {
                let Some(e) = self.by_id.get(&id) else { continue };
                if e.tokens.len() > best_len
                    && e.tokens.len() < tokens.len()
                    && e.tokens.as_slice() == &tokens[..e.tokens.len()]
                {
                    best = Some(id);
                    best_len = e.tokens.len();
                }
            }
            if let Some(id) = best {
                self.touch(id);
                let e = &self.by_id[&id];
                return Some(PrefixHit {
                    snap: e.snap.clone(),
                    prefix_len: e.tokens.len(),
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
            .map(|e| e.tokens.iter().zip(tokens).take_while(|(a, b)| a == b).count())
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
        let hit = c.find(&next).expect("hit");
        assert_eq!(hit.prefix_len, 2 * BLOCK_TOKENS);
        assert_eq!(hit.snap.position, 2 * BLOCK_TOKENS);
    }

    #[test]
    fn exact_repeat_is_miss() {
        let mut c = PrefixCache::new(64);
        let cached = tokens_from(1, 2 * BLOCK_TOKENS);
        c.put(cached.clone(), snap_for(&cached, 4096));
        // Совпадение до последнего токена: primed-пути нужен ≥1 токен prefill'а.
        assert!(c.find(&cached).is_none());
    }

    #[test]
    fn divergent_prompt_is_miss() {
        let mut c = PrefixCache::new(64);
        let cached = tokens_from(1, 2 * BLOCK_TOKENS);
        c.put(cached, snap_for(&tokens_from(1, 2 * BLOCK_TOKENS), 4096));
        // Общий первый блок, расхождение во втором.
        let mut other = tokens_from(1, BLOCK_TOKENS);
        other.extend(tokens_from(7777, BLOCK_TOKENS));
        assert!(c.find(&other).is_none());
        // Расхождение в первом блоке.
        assert!(c.find(&tokens_from(500, 4 * BLOCK_TOKENS)).is_none());
        assert!(c.find(&tokens_from(1, BLOCK_TOKENS)).is_none());
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
        assert_eq!(c.find(&next).unwrap().prefix_len, 3 * BLOCK_TOKENS);
        // Третий блок другой → отпадает длинная, остаётся короткая.
        let mut next2 = short.clone();
        next2.extend(tokens_from(999, BLOCK_TOKENS));
        assert_eq!(c.find(&next2).unwrap().prefix_len, 2 * BLOCK_TOKENS);
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
        let hit = c.find(&next).expect("настоящая запись находится");
        assert_eq!(hit.snap.position, 2 * BLOCK_TOKENS);
        assert_eq!(hit.prefix_len, 2 * BLOCK_TOKENS);
        // Промпт «чужой» записи не получает её состояние (общий хеш, разное
        // содержимое) — conservative промах.
        let mut stranger_next = stranger.clone();
        stranger_next.extend(tokens_from(7000, 10));
        assert!(c.find(&stranger_next).is_none());
    }

    #[test]
    fn short_prompts_are_not_cached() {
        let mut c = PrefixCache::new(64);
        let small = tokens_from(1, 10);
        assert!(!c.put(small.clone(), snap_for(&small, 4096)));
        assert!(c.find(&small).is_none());
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
        assert!(c.find(&q1).is_none());
        let mut q2 = tokens_from(2, BLOCK_TOKENS);
        q2.extend(tokens_from(9000, BLOCK_TOKENS));
        assert!(c.find(&q2).is_some());
        let mut q4 = tokens_from(4, BLOCK_TOKENS);
        q4.extend(tokens_from(9000, BLOCK_TOKENS));
        assert!(c.find(&q4).is_some());
        // get(2) поднял его в LRU → следующим вытесняется start=3.
        let (t5, s5) = mk(5);
        assert!(c.put(t5, s5));
        let mut q3 = tokens_from(3, BLOCK_TOKENS);
        q3.extend(tokens_from(9000, BLOCK_TOKENS));
        assert!(c.find(&q3).is_none());
        assert!(c.find(&q2).is_some());
    }
}
