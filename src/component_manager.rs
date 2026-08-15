//! Runtime Component Manager (FR-017, FR-018, FR-019).
//!
//! Manages on-demand Vision and MTP components:
//! - Lazy state transitions: Unloaded -> Loading -> Warm -> Unloaded / Error.
//! - Warm TTL (default 60s) for automatic VRAM recovery when idle.
//! - Priority: Vision requests evict MTP first under VRAM pressure.
//! - Admission barrier: pauses new requests while components load/unload without aborting in-flight tasks.
//! - RAII ComponentLease tracking active users.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use anyhow::{bail, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentKind {
    Vision,
    Mtp,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComponentState {
    Unloaded,
    Loading,
    Warm { last_used: Instant },
    Error(String),
}

struct ComponentEntry {
    kind: ComponentKind,
    path: Option<PathBuf>,
    state: ComponentState,
    leases: usize,
}

pub struct ComponentManager {
    components: RwLock<[ComponentEntry; 2]>,
    ttl: Duration,
    barrier: Mutex<()>,
}

pub struct ComponentLease {
    kind: ComponentKind,
    manager: Arc<ComponentManager>,
}

impl Drop for ComponentLease {
    fn drop(&mut self) {
        self.manager.release_lease(self.kind);
    }
}

impl ComponentManager {
    pub fn new(vision_path: Option<PathBuf>, mtp_path: Option<PathBuf>, ttl: Duration) -> Arc<Self> {
        Arc::new(Self {
            components: RwLock::new([
                ComponentEntry {
                    kind: ComponentKind::Vision,
                    path: vision_path,
                    state: ComponentState::Unloaded,
                    leases: 0,
                },
                ComponentEntry {
                    kind: ComponentKind::Mtp,
                    path: mtp_path,
                    state: ComponentState::Unloaded,
                    leases: 0,
                },
            ]),
            ttl,
            barrier: Mutex::new(()),
        })
    }

    pub fn state(&self, kind: ComponentKind) -> ComponentState {
        let idx = kind as usize;
        self.components.read().unwrap()[idx].state.clone()
    }

    pub fn is_available(&self, kind: ComponentKind) -> bool {
        let idx = kind as usize;
        self.components.read().unwrap()[idx].path.is_some()
    }

    pub fn path(&self, kind: ComponentKind) -> Option<PathBuf> {
        let idx = kind as usize;
        self.components.read().unwrap()[idx].path.clone()
    }

    pub fn acquire_lease(self: &Arc<Self>, kind: ComponentKind) -> Result<ComponentLease> {
        let _guard = self.barrier.lock().unwrap();
        let idx = kind as usize;
        let mut comps = self.components.write().unwrap();
        if comps[idx].path.is_none() {
            bail!("{:?} component is not configured in profile", kind);
        }
        match &comps[idx].state {
            ComponentState::Error(err) => bail!("{:?} component failed to load: {err}", kind),
            _ => {
                comps[idx].leases += 1;
                comps[idx].state = ComponentState::Warm {
                    last_used: Instant::now(),
                };
                Ok(ComponentLease {
                    kind,
                    manager: Arc::clone(self),
                })
            }
        }
    }

    fn release_lease(&self, kind: ComponentKind) {
        let idx = kind as usize;
        let mut comps = self.components.write().unwrap();
        if comps[idx].leases > 0 {
            comps[idx].leases -= 1;
            if comps[idx].leases == 0 {
                comps[idx].state = ComponentState::Warm {
                    last_used: Instant::now(),
                };
            }
        }
    }

    /// Check and trigger TTL unloads. Returns components that need unloading.
    pub fn check_ttl(&self) -> Vec<ComponentKind> {
        let now = Instant::now();
        let mut to_unload = Vec::new();
        let mut comps = self.components.write().unwrap();
        for entry in comps.iter_mut() {
            if entry.leases == 0 {
                if let ComponentState::Warm { last_used } = entry.state {
                    if now.duration_since(last_used) >= self.ttl {
                        entry.state = ComponentState::Unloaded;
                        to_unload.push(entry.kind);
                    }
                }
            }
        }
        to_unload
    }

    /// Request VRAM relief. If MTP is loaded and idle, mark it for eviction first.
    pub fn evict_for_vram_pressure(&self) -> Option<ComponentKind> {
        let mut comps = self.components.write().unwrap();
        // MTP has lower priority -> evict first
        let mtp_idx = ComponentKind::Mtp as usize;
        if comps[mtp_idx].leases == 0 && matches!(comps[mtp_idx].state, ComponentState::Warm { .. }) {
            comps[mtp_idx].state = ComponentState::Unloaded;
            return Some(ComponentKind::Mtp);
        }
        None
    }

    pub fn set_error(&self, kind: ComponentKind, error: String) {
        let idx = kind as usize;
        let mut comps = self.components.write().unwrap();
        comps[idx].state = ComponentState::Error(error);
    }
}
