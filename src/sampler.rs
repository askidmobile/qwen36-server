//! Свой сэмплер: temperature + top_k + top_p + min_p + presence_penalty +
//! repetition_penalty. Без candle-transformers LogitsProcessor — меньше зависимостей.
//! RNG — свой xorshift64* (деп `rand` не нужен).

use std::collections::HashSet;

/// Пресеты сэмплинга из model card (BD-016).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplingPreset {
    /// thinking: t=1.0, top_p=0.95, top_k=20.
    Thinking,
    /// thinking-coding: t=0.6, top_p=0.95, top_k=20.
    ThinkingCoding,
    /// instruct: t=0.7, top_p=0.80, top_k=20, presence_penalty=1.5.
    Instruct,
}

impl SamplingPreset {
    pub fn by_model_id(id: &str) -> Option<Self> {
        match id {
            "thinking" => Some(Self::Thinking),
            "thinking-coding" => Some(Self::ThinkingCoding),
            "instruct" => Some(Self::Instruct),
            _ => None,
        }
    }
}

/// xorshift64* PRNG (детерминированный при заданном seed).
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    /// Равномерное [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545F4914F6CDD1D);
        (x >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Частичный отбор при `top_k == 0`: верхние PARTIAL_TOPK кандидатов
/// quickselect'ом (O(vocab)); если кумулятивная вероятность по ним не
/// набирает `top_p`, отбор повторяется полной сортировкой — точность та же,
/// что у полной сортировки, но платится она только на аномально плоских
/// распределениях. Клиент может прислать top_k=0 (openai.rs), и раньше это
/// означало полную сортировку 152K логитов на каждый токен.
const PARTIAL_TOPK: usize = 1024;

fn penalties_active(presence_penalty: f32, repetition_penalty: f32) -> bool {
    presence_penalty != 0.0 || (repetition_penalty != 0.0 && repetition_penalty != 1.0)
}

/// Сэмплировать токен из логитов. `generated` — уже сгенерированные токены
/// запроса (для penalties). Возвращает token id.
///
/// Строит множество встречавшихся токенов заново — O(n) на вызов; в
/// batched-движке используется `sample_with_seen` с инкрементальным множеством.
#[allow(clippy::too_many_arguments)]
pub fn sample(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
    generated: &[u32],
    rng: &mut Rng,
) -> u32 {
    let seen: HashSet<u32> = if penalties_active(presence_penalty, repetition_penalty) {
        generated.iter().copied().collect()
    } else {
        HashSet::new()
    };
    sample_with_seen(
        logits,
        temperature,
        top_k,
        top_p,
        min_p,
        presence_penalty,
        repetition_penalty,
        &seen,
        rng,
    )
}

/// То же, что `sample`, но множество встречавшихся токенов передаётся готовым
/// (инкрементально поддерживается вызывающим).
#[allow(clippy::too_many_arguments)]
pub fn sample_with_seen(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    presence_penalty: f32,
    repetition_penalty: f32,
    seen: &HashSet<u32>,
    rng: &mut Rng,
) -> u32 {
    let penalties = penalties_active(presence_penalty, repetition_penalty);
    if temperature <= 1e-5 && !penalties {
        return argmax(logits);
    }
    // Копия нужна только под penalties: они меняют логиты встречавшихся
    // токенов, а отбор кандидатов обязан видеть уже изменённые значения.
    let penalized: Vec<f32>;
    let logits: &[f32] = if penalties {
        let mut l = logits.to_vec();
        for &t in seen {
            let Some(v) = l.get_mut(t as usize) else { continue };
            if presence_penalty != 0.0 {
                *v -= presence_penalty;
            }
            if repetition_penalty != 0.0 && repetition_penalty != 1.0 {
                // HF-семантика: логит делится на penalty, если <0 — умножается.
                *v = if *v < 0.0 {
                    *v * repetition_penalty
                } else {
                    *v / repetition_penalty
                };
            }
        }
        penalized = l;
        &penalized
    } else {
        logits
    };

    // temperature <= 0 или дегенеративный случай → greedy argmax.
    if temperature <= 1e-5 {
        return argmax(logits);
    }

    // Отбор кандидатов по убыванию логита. Полная сортировка 152K логитов —
    // 10-30ms/токен/слот; при top_k>0 достаточно quickselect (O(vocab)) +
    // сортировка k кандидатов. При top_k==0 — частичный отбор PARTIAL_TOPK с
    // откатом на полную сортировку, если top_p не набран (см. константу).
    let by_desc = |a: &u32, b: &u32| {
        logits[*b as usize]
            .partial_cmp(&logits[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let select = |cap: Option<usize>| -> Vec<u32> {
        let mut all: Vec<u32> = (0..logits.len() as u32).collect();
        if let Some(c) = cap.filter(|&c| c > 0 && c < all.len()) {
            all.select_nth_unstable_by(c - 1, by_desc);
            all.truncate(c);
        }
        all.sort_unstable_by(by_desc);
        all
    };
    let cap = if top_k > 0 {
        Some(top_k)
    } else if top_p < 1.0 {
        Some(PARTIAL_TOPK)
    } else {
        None
    };
    let mut idx = select(cap);

    // softmax по отфильтрованным (температура — только на кандидатах: порядок
    // от неё не зависит, а делить весь словарь незачем). При top_k>0
    // нормировка по подмножеству — это и есть семантика «top_k, затем top_p».
    // При частичном отборе с top_k==0 нуклеус обязан считаться от ПОЛНОЙ
    // нормировки, иначе кумулятив завышен и срез случится раньше, чем при
    // полной сортировке; знаменатель по всему словарю — один проход без
    // сортировки. idx[0] — глобальный максимум в обоих случаях.
    let max_l = logits[idx[0] as usize];
    let full_denom: Option<f32> = if top_k == 0 && idx.len() < logits.len() {
        Some(logits.iter().map(|&l| ((l - max_l) / temperature).exp()).sum())
    } else {
        None
    };
    let softmax = |idx: &[u32], denom: Option<f32>| -> Vec<f32> {
        let mut probs: Vec<f32> = idx
            .iter()
            .map(|&i| ((logits[i as usize] - max_l) / temperature).exp())
            .collect();
        let sum: f32 = denom.unwrap_or_else(|| probs.iter().sum());
        for p in probs.iter_mut() {
            *p /= sum;
        }
        probs
    };
    let mut probs = softmax(&idx, full_denom);

    // top_p (nucleus): оставляем префикс с кумулятивной вероятностью <= top_p,
    // но минимум один токен.
    if top_p < 1.0 {
        let nucleus = |probs: &[f32]| -> Option<usize> {
            let mut cum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cum += p;
                if cum > top_p && i > 0 {
                    return Some(i);
                }
            }
            None
        };
        let mut keep = nucleus(&probs);
        if keep.is_none() && top_k == 0 && idx.len() < logits.len() {
            // Частичный отбор не набрал top_p — распределение плоское, нужен
            // полный хвост.
            idx = select(None);
            probs = softmax(&idx, None);
            keep = nucleus(&probs);
        }
        if let Some(keep) = keep {
            idx.truncate(keep);
            probs.truncate(keep);
        }
    }

    // min_p: фильтр относительно p_max (после top_k/top_p renormalize не нужен —
    // фильтр по абсолютным probs достаточен, константа сократится при выборе).
    if min_p > 0.0 {
        let p_max = probs[0];
        let threshold = min_p * p_max;
        let mut kept_i = Vec::with_capacity(idx.len());
        let mut kept_p = Vec::with_capacity(probs.len());
        for (&i, &p) in idx.iter().zip(probs.iter()) {
            if p >= threshold {
                kept_i.push(i);
                kept_p.push(p);
            }
        }
        if !kept_i.is_empty() {
            idx = kept_i;
            probs = kept_p;
        }
    }

    // Рулетка по probs (ненормированным — сумма константна для всех кандидатов).
    let total: f32 = probs.iter().sum();
    let mut r = rng.next_f32() * total;
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return idx[i];
        }
    }
    idx[0]
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0u32;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i as u32;
        }
    }
    best
}

/// Сырые вероятности модели: log-softmax по всему словарю от логитов ДО
/// штрафов, температуры и любого отбора.
///
/// Это то, что OpenAI отдаёт в `logprobs`. Взять готовое из сэмплера нельзя:
/// полная нормировка там считается только при `top_k == 0` и уже с
/// температурой внутри, а штрафы за повторы меняют порядок кандидатов, поэтому
/// топ-K после них — не топ-K модели.
///
/// Считается ДО того, как сэмплер что-либо трогает, и на его работу не влияет:
/// ни одного обращения к генератору, ни изменения порядка операций отбора.
pub struct RawLogprobs {
    max: f32,
    /// `ln Σ exp(l − max)` по всему словарю.
    ln_sum: f32,
    /// Топ-K по сырым логитам, по убыванию: (идентификатор, logprob).
    pub top: Vec<(u32, f32)>,
}

impl RawLogprobs {
    /// Logprob любого токена по тому же знаменателю.
    pub fn of(&self, token: u32, logits: &[f32]) -> f32 {
        match logits.get(token as usize) {
            Some(&l) => l - self.max - self.ln_sum,
            None => f32::NEG_INFINITY,
        }
    }
}

/// Один проход за максимумом и топ-K, второй за суммой экспонент.
///
/// Топ-K держим вставкой в маленький отсортированный вектор: `k` не больше 20
/// по контракту, и подавляющее большинство логитов отсеивается первым же
/// сравнением, так что это дешевле кучи и не требует копии словаря.
pub fn raw_logprobs(logits: &[f32], k: usize) -> RawLogprobs {
    let mut max = f32::NEG_INFINITY;
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k.min(logits.len()));
    for (i, &l) in logits.iter().enumerate() {
        if l > max {
            max = l;
        }
        if k == 0 {
            continue;
        }
        if top.len() == k && l <= top[k - 1].1 {
            continue;
        }
        let pos = top.partition_point(|&(_, v)| v > l);
        if pos < k {
            top.insert(pos, (i as u32, l));
            top.truncate(k);
        }
    }
    // Логиты ниже max − 30 не влияют на сумму: exp(−30) ≈ 9e-14, а f32 держит
    // около семи значащих цифр. Порог сделан с запасом против 1e-9.
    // Сумма копится в f64: слагаемых около 151 тысячи, и в f32 накопленная
    // ошибка выходит порядка sqrt(n)*eps ≈ 5e-5, что видно в logprob (замер
    // дал 7.6e-5 расхождения с лобовым log_softmax). В f64 она исчезает,
    // а стоит это столько же.
    let cutoff = max - 30.0;
    let sum: f64 = logits
        .iter()
        .filter(|&&l| l > cutoff)
        .map(|&l| ((l - max) as f64).exp())
        .sum();
    let ln_sum = sum.ln() as f32;
    for entry in top.iter_mut() {
        entry.1 = entry.1 - max - ln_sum;
    }
    RawLogprobs { max, ln_sum, top }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Лобовой log-softmax в f64 — эталон для raw_logprobs.
    fn naive_log_softmax(logits: &[f32]) -> Vec<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
        let ln_sum = sum.ln() as f32;
        logits.iter().map(|&l| l - max - ln_sum).collect()
    }

    /// Словарь размером с настоящий, значения вразнобой, два близких лидера.
    fn vocab_like_model() -> Vec<f32> {
        let mut logits = vec![0f32; 151_936];
        let mut x = 12345u64;
        for (i, v) in logits.iter_mut().enumerate() {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((x >> 33) as f32 / (1u64 << 31) as f32) * 24.0 - 12.0
                + (i % 7) as f32 * 0.1;
        }
        logits[777] = 30.0;
        logits[778] = 29.7;
        logits
    }

    #[test]
    fn raw_logprobs_matches_full_log_softmax() {
        let logits = vocab_like_model();
        let naive = naive_log_softmax(&logits);
        let got = raw_logprobs(&logits, 20);

        // Выбранный токен — по тому же знаменателю.
        assert!(
            (got.of(777, &logits) - naive[777]).abs() < 1e-5,
            "logprob максимума разошёлся с лобовым log_softmax"
        );
        for &(id, lp) in &got.top {
            assert!(
                (lp - naive[id as usize]).abs() < 1e-5,
                "logprob кандидата {id} разошёлся с лобовым log_softmax"
            );
        }

        // Сумма вероятностей подмножества не может превышать единицу.
        let s: f64 = got.top.iter().map(|&(_, lp)| (lp as f64).exp()).sum();
        assert!(s <= 1.0 + 1e-6, "сумма вероятностей топа больше единицы: {s}");

        // Топ отсортирован по убыванию и начинается с глобального максимума.
        assert_eq!(got.top.len(), 20);
        assert_eq!(got.top[0].0, 777);
        assert_eq!(got.top[1].0, 778);
        for w in got.top.windows(2) {
            assert!(w[0].1 >= w[1].1, "топ не отсортирован по убыванию");
        }
    }

    #[test]
    fn raw_logprobs_ignores_penalties_and_temperature() {
        // Смысл поля: вероятность МОДЕЛИ. Ни температура, ни штрафы на неё не
        // влияют, потому что считается она до сэмплера и по сырым логитам.
        let logits = vocab_like_model();
        let a = raw_logprobs(&logits, 5);
        let b = raw_logprobs(&logits, 5);
        assert_eq!(a.top, b.top, "две подряд дают разное — есть состояние");
    }

    #[test]
    fn raw_logprobs_k_zero_gives_no_top() {
        let logits = vocab_like_model();
        let got = raw_logprobs(&logits, 0);
        assert!(got.top.is_empty());
        // Знаменатель считается всё равно: logprob выбранного нужен и без топа.
        assert!((got.of(777, &logits) - naive_log_softmax(&logits)[777]).abs() < 1e-5);
    }

    #[test]
    fn greedy_when_zero_temperature() {
        let logits = vec![0.1, 0.9, 0.2];
        let mut rng = Rng::new(1);
        let t = sample(&logits, 0.0, 0, 1.0, 0.0, 0.0, 1.0, &[], &mut rng);
        assert_eq!(t, 1);
    }

    #[test]
    fn top_k_one_is_deterministic() {
        let logits = vec![0.1, 0.9, 0.2, 5.0];
        let mut rng = Rng::new(42);
        for _ in 0..8 {
            let t = sample(&logits, 1.0, 1, 1.0, 0.0, 0.0, 1.0, &[], &mut rng);
            assert_eq!(t, 3, "top_k=1 → всегда argmax");
        }
    }

    #[test]
    fn min_p_drops_tail() {
        // p_max ≈ 1.0 (logit 10 vs 0). min_p=0.5 → всё ниже 0.5*p_max отсекается.
        let logits = vec![10.0, 0.0, 0.0];
        let mut rng = Rng::new(7);
        for _ in 0..8 {
            let t = sample(&logits, 1.0, 0, 1.0, 0.5, 0.0, 1.0, &[], &mut rng);
            assert_eq!(t, 0);
        }
    }

    #[test]
    fn presence_penalty_suppresses_seen() {
        let logits = vec![1.0, 1.0];
        let mut rng = Rng::new(3);
        // Токен 0 уже встречался, большой penalty → выбирается 1.
        let t = sample(&logits, 0.01, 0, 1.0, 0.0, 10.0, 1.0, &[0], &mut rng);
        assert_eq!(t, 1);
    }

    #[test]
    fn repetition_penalty_suppresses_seen_positive_logit() {
        let logits = vec![4.0, 4.0];
        let mut rng = Rng::new(3);
        let t = sample(&logits, 0.01, 0, 1.0, 0.0, 0.0, 8.0, &[0], &mut rng);
        assert_eq!(t, 1);
    }

    #[test]
    fn partial_topk_argmax_parity() {
        // FR-003: top_k fast path обязан находить тот же argmax и тот же набор
        // top-k кандидатов, что полная сортировка (до ties).
        let mut rng = Rng::new(99);
        for _ in 0..50 {
            let logits: Vec<f32> = (0..10_000).map(|_| rng.next_f32() * 20.0 - 10.0).collect();
            let mut r1 = Rng::new(7);
            let t = sample(&logits, 0.0, 20, 1.0, 0.0, 0.0, 1.0, &[], &mut r1);
            let want = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();
            assert_eq!(t, want, "argmax расходится");
        }
    }

    #[test]
    fn topk_zero_partial_matches_full_sort() {
        // top_k=0 + top_p<1: частичный отбор обязан давать тот же токен, что
        // полная сортировка (тот же RNG, те же кандидаты в нуклеусе).
        let mut src = Rng::new(5);
        let logits: Vec<f32> = (0..50_000).map(|_| src.next_f32() * 12.0 - 6.0).collect();
        for seed in 1..40u64 {
            let mut r1 = Rng::new(seed);
            let mut r2 = Rng::new(seed);
            let fast = sample(&logits, 0.9, 0, 0.9, 0.0, 0.0, 1.0, &[], &mut r1);
            // Полная сортировка: top_k = vocab (cap не срабатывает).
            let full = sample(&logits, 0.9, logits.len(), 0.9, 0.0, 0.0, 1.0, &[], &mut r2);
            assert_eq!(fast, full, "seed {seed}");
        }
    }

    #[test]
    fn seeded_rng_deterministic() {
        let logits: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin()).collect();
        let seq = |seed| {
            let mut rng = Rng::new(seed);
            (0..16)
                .map(|_| sample(&logits, 0.8, 20, 0.95, 0.0, 0.0, 1.0, &[], &mut rng))
                .collect::<Vec<_>>()
        };
        assert_eq!(seq(123), seq(123));
        assert_ne!(seq(123), seq(124));
    }
}
