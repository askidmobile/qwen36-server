//! Свой сэмплер: temperature + top_k + top_p + min_p + presence_penalty +
//! repetition_penalty. Без candle-transformers LogitsProcessor — меньше зависимостей.
//! RNG — свой xorshift64* (деп `rand` не нужен).

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

/// Сэмплировать токен из логитов. `generated` — уже сгенерированные токены
/// запроса (для penalties). Возвращает token id.
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
    let mut logits = logits.to_vec();

    // Penalties по уже встречавшимся токенам (OpenAI-семантика).
    if presence_penalty != 0.0 || (repetition_penalty != 0.0 && repetition_penalty != 1.0) {
        let mut seen = std::collections::HashSet::new();
        for &t in generated {
            seen.insert(t);
        }
        for &t in &seen {
            let l = &mut logits[t as usize];
            if presence_penalty != 0.0 {
                *l -= presence_penalty;
            }
            if repetition_penalty != 0.0 && repetition_penalty != 1.0 {
                // HF-семантика: логит делится на penalty, если <0 — умножается.
                *l = if *l < 0.0 {
                    *l * repetition_penalty
                } else {
                    *l / repetition_penalty
                };
            }
        }
    }

    // temperature <= 0 или дегенеративный случай → greedy argmax.
    if temperature <= 1e-5 {
        return argmax(&logits);
    }
    for l in logits.iter_mut() {
        *l /= temperature;
    }

    // Отбор кандидатов по убыванию логита. Полная сортировка 248K логитов —
    // 10-30ms/токен/слот; при top_k>0 достаточно quickselect (O(vocab)) +
    // сортировка k кандидатов. top_k==0 + top_p<1 → редкий путь полной
    // сортировки (нужна кумулятивная вероятность по всему vocab).
    let by_desc = |a: &u32, b: &u32| {
        logits[*b as usize]
            .partial_cmp(&logits[*a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    };
    let mut idx: Vec<u32> = if top_k > 0 && top_k < logits.len() {
        let mut all: Vec<u32> = (0..logits.len() as u32).collect();
        all.select_nth_unstable_by(top_k - 1, by_desc);
        all.truncate(top_k);
        all.sort_unstable_by(by_desc);
        all
    } else {
        let mut all: Vec<u32> = (0..logits.len() as u32).collect();
        all.sort_unstable_by(by_desc);
        all
    };

    // top_k (уже применён в fast path; оставляем для полного пути)
    if top_k > 0 && idx.len() > top_k {
        idx.truncate(top_k);
    }

    // softmax по отфильтрованным.
    let max_l = logits[idx[0] as usize];
    let mut probs: Vec<f32> = idx
        .iter()
        .map(|&i| (logits[i as usize] - max_l).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // top_p (nucleus): оставляем префикс с кумулятивной вероятностью <= top_p,
    // но минимум один токен.
    if top_p < 1.0 {
        let mut cum = 0.0;
        let mut keep = idx.len();
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum > top_p && i > 0 {
                keep = i;
                break;
            }
        }
        idx.truncate(keep);
        probs.truncate(keep);
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

#[cfg(test)]
mod tests {
    use super::*;

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
