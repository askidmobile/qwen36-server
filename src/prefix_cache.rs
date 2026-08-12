//! Prefix cache (FR-T328): snapshot state после prefill → повторный prompt
//! без prefill. Уникально для hybrid DeltaNet моделей (llama.cpp их
//! не кэширует: «forcing full prompt re-processing due to recurrent memory»).
//!
//! Ключ: DefaultHasher по token ids. LRU по суммарному размеру snapshot'ов
//! (MiB). Эвикт: самый давно неиспользованный.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use qwen35_batch::real::model_weights::StateSnapshot;

struct Entry {
    snap: StateSnapshot,
    size_mib: usize,
    // Token IDs для защиты от hash collision: get сверяет полный prompt.
    tokens: Vec<u32>,
}

pub struct PrefixCache {
    map: HashMap<u64, Entry>,
    /// LRU: front = самый старый.
    lru: VecDeque<u64>,
    total_mib: usize,
    budget_mib: usize,
}

impl PrefixCache {
    pub fn new(budget_mib: usize) -> Self {
        Self {
            map: HashMap::new(),
            lru: VecDeque::new(),
            total_mib: 0,
            budget_mib,
        }
    }

    pub fn key_for(tokens: &[u32]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        tokens.hash(&mut h);
        h.finish()
    }

    pub fn get(&mut self, key: u64, tokens: &[u32]) -> Option<StateSnapshot> {
        let match_entry = self.map.get(&key).map(|e| e.tokens == tokens);
        if match_entry == Some(true) {
            self.touch(key);
            self.map.get(&key).map(|e| e.snap.clone())
        } else {
            None
        }
    }

    pub fn put(&mut self, key: u64, snap: StateSnapshot, tokens: Vec<u32>) {
        let size_mib = (snap.size_bytes() / 1024 / 1024).max(1);
        if size_mib > self.budget_mib {
            return; // snapshot больше всего бюджета — не кэшируем
        }
        if self.map.contains_key(&key) {
            self.total_mib -= self.map[&key].size_mib;
            self.map.remove(&key);
            self.lru.retain(|&k| k != key);
        }
        while self.total_mib + size_mib > self.budget_mib {
            let Some(oldest) = self.lru.pop_front() else { break };
            if let Some(e) = self.map.remove(&oldest) {
                self.total_mib -= e.size_mib;
            }
        }
        self.total_mib += size_mib;
        self.lru.push_back(key);
        self.map.insert(key, Entry { snap, size_mib, tokens });
    }

    fn touch(&mut self, key: u64) {
        self.lru.retain(|&k| k != key);
        self.lru.push_back(key);
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }
}

pub type SharedPrefixCache = Mutex<PrefixCache>;

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_snap(mib: usize) -> StateSnapshot {
        let floats = mib * 1024 * 1024 / 4;
        StateSnapshot {
            model_nonce: 1,
            position: 10,
            blocks: vec![
                qwen35_batch::real::model_weights::BlockStateSnap::DeltaNet(
                    qwen35_batch::real::model_weights::DeltaNetStateSnap {
                        conv_buf: vec![0.0; floats / 2],
                        ssm_state: vec![0.0; floats / 2],
                    },
                ),
            ],
        }
    }

    #[test]
    fn lru_eviction_by_budget() {
        let mut c = PrefixCache::new(3);
        c.put(1, fake_snap(1), vec![1]);
        c.put(2, fake_snap(1), vec![2]);
        c.put(3, fake_snap(1), vec![3]);
        assert_eq!(c.len(), 3);
        // 4-й не влезает → вытесняет самый старый (key 1).
        c.put(4, fake_snap(1), vec![4]);
        assert!(c.get(1, &[1]).is_none());
        assert!(c.get(2, &[2]).is_some());
        assert!(c.get(4, &[4]).is_some());
        // get(2) поднял его в LRU → следующим вытесняется key 3.
        c.put(5, fake_snap(1), vec![5]);
        assert!(c.get(3, &[3]).is_none());
        assert!(c.get(2, &[2]).is_some());
    }
}
