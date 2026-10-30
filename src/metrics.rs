//! Read-only request and host metrics for `/v1/status`.

use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Mutex, Once, RwLock,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::engine_types::GenerationUsage;

const RECENT_LIMIT: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RequestTotals {
    pub active_requests: usize,
    pub completed_requests: u64,
    pub failed_requests: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Includes live estimates while a request is streaming.
    pub output_tokens_live: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentRequest {
    pub started_at_unix_ms: u64,
    pub endpoint: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub duration_ms: u64,
    pub tokens_per_sec: f64,
    pub finish_reason: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuStatus {
    pub name: String,
    pub util_percent: Option<f64>,
    pub vram_used_mib: Option<u64>,
    pub vram_total_mib: Option<u64>,
    pub temp_c: Option<f64>,
    pub power_w: Option<f64>,
    pub power_limit_w: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SystemStatus {
    pub cpu_percent: f64,
    pub ram_used_mib: u64,
    pub ram_total_mib: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HostStatus {
    pub gpu: Option<GpuStatus>,
    pub system: Option<SystemStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStatus {
    pub backend: String,
    pub attention: String,
    pub prefill: String,
    pub paged_prefill: bool,
    pub cuda_graphs: bool,
}

impl RuntimeStatus {
    pub fn current() -> Self {
        if cfg!(feature = "cuda-turing") {
            Self {
                backend: "cuda-turing".into(),
                attention: "split_k_f16_f32_softmax".into(),
                prefill: "reference".into(),
                paged_prefill: false,
                cuda_graphs: false,
            }
        } else if cfg!(feature = "cuda") {
            Self {
                backend: "cuda".into(),
                attention: "flash_attention_2".into(),
                prefill: "flash_attention_2".into(),
                paged_prefill: true,
                cuda_graphs: true,
            }
        } else if cfg!(feature = "metal") {
            Self {
                backend: "metal".into(),
                attention: "metal".into(),
                prefill: "metal".into(),
                paged_prefill: false,
                cuda_graphs: false,
            }
        } else {
            Self {
                backend: "cpu".into(),
                attention: "cpu".into(),
                prefill: "cpu".into(),
                paged_prefill: false,
                cuda_graphs: false,
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusModel {
    pub id: String,
    pub quant: String,
    pub context_length: usize,
    pub slots: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusSnapshot {
    pub model: StatusModel,
    pub state: String,
    pub uptime_s: u64,
    pub runtime: RuntimeStatus,
    pub totals: RequestTotals,
    pub context_used: usize,
    pub in_flight: usize,
    pub queued: usize,
    pub recent_requests: Vec<RecentRequest>,
    pub gpu: Option<GpuStatus>,
    pub system: Option<SystemStatus>,
}

pub struct RequestMetrics {
    started: Instant,
    active: AtomicUsize,
    completed: AtomicU64,
    failed: AtomicU64,
    prompt_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    output_tokens_live: AtomicU64,
    recent: Mutex<VecDeque<RecentRequest>>,
}

impl Default for RequestMetrics {
    fn default() -> Self {
        Self {
            started: Instant::now(),
            active: AtomicUsize::new(0),
            completed: AtomicU64::new(0),
            failed: AtomicU64::new(0),
            prompt_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            output_tokens_live: AtomicU64::new(0),
            recent: Mutex::new(VecDeque::with_capacity(RECENT_LIMIT)),
        }
    }
}

impl RequestMetrics {
    pub fn begin(self: &Arc<Self>, endpoint: &'static str) -> RequestGuard {
        self.active.fetch_add(1, Ordering::Relaxed);
        RequestGuard {
            metrics: Arc::clone(self),
            endpoint,
            started: Instant::now(),
            started_at_unix_ms: unix_ms_now(),
            estimated_tokens: 0,
            finished: false,
        }
    }

    pub fn totals(&self) -> RequestTotals {
        RequestTotals {
            active_requests: self.active.load(Ordering::Relaxed),
            completed_requests: self.completed.load(Ordering::Relaxed),
            failed_requests: self.failed.load(Ordering::Relaxed),
            prompt_tokens: self.prompt_tokens.load(Ordering::Relaxed),
            completion_tokens: self.completion_tokens.load(Ordering::Relaxed),
            output_tokens_live: self.output_tokens_live.load(Ordering::Relaxed),
        }
    }

    pub fn recent(&self) -> Vec<RecentRequest> {
        self.recent
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    pub fn uptime(&self) -> Duration {
        self.started.elapsed()
    }

    fn push_recent(&self, request: RecentRequest) {
        let mut recent = self.recent.lock().unwrap_or_else(|e| e.into_inner());
        recent.push_front(request);
        recent.truncate(RECENT_LIMIT);
    }
}

pub struct RequestGuard {
    metrics: Arc<RequestMetrics>,
    endpoint: &'static str,
    started: Instant,
    started_at_unix_ms: u64,
    estimated_tokens: u64,
    finished: bool,
}

impl RequestGuard {
    /// Engine deltas are a useful live estimate until exact usage arrives in Done.
    pub fn delta(&mut self) {
        self.estimated_tokens = self.estimated_tokens.saturating_add(1);
        self.metrics
            .output_tokens_live
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn finish(&mut self, usage: &GenerationUsage, finish_reason: String) {
        if self.finished {
            return;
        }
        let actual = usage.completion_tokens as u64;
        if actual > self.estimated_tokens {
            self.metrics
                .output_tokens_live
                .fetch_add(actual - self.estimated_tokens, Ordering::Relaxed);
        } else if self.estimated_tokens > actual {
            saturating_sub(
                &self.metrics.output_tokens_live,
                self.estimated_tokens - actual,
            );
        }
        self.finished = true;
        self.metrics
            .prompt_tokens
            .fetch_add(usage.prompt_tokens as u64, Ordering::Relaxed);
        self.metrics
            .completion_tokens
            .fetch_add(actual, Ordering::Relaxed);
        self.metrics.completed.fetch_add(1, Ordering::Relaxed);
        self.decrement_active();
        self.push(usage.prompt_tokens as u64, actual, finish_reason, None);
    }

    pub fn fail(&mut self, error: impl Into<String>) {
        if self.finished {
            return;
        }
        if self.estimated_tokens > 0 {
            saturating_sub(&self.metrics.output_tokens_live, self.estimated_tokens);
        }
        self.finished = true;
        self.metrics.failed.fetch_add(1, Ordering::Relaxed);
        self.decrement_active();
        self.push(0, 0, "error".into(), Some(error.into()));
    }

    fn decrement_active(&self) {
        let _ = self
            .metrics
            .active
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                Some(n.saturating_sub(1))
            });
    }

    fn push(
        &self,
        prompt_tokens: u64,
        completion_tokens: u64,
        finish_reason: String,
        error: Option<String>,
    ) {
        let duration = self.started.elapsed();
        let duration_ms = duration.as_millis().min(u64::MAX as u128) as u64;
        let secs = duration.as_secs_f64();
        self.metrics.push_recent(RecentRequest {
            started_at_unix_ms: self.started_at_unix_ms,
            endpoint: self.endpoint.into(),
            prompt_tokens,
            completion_tokens,
            duration_ms,
            tokens_per_sec: if secs > 0.0 {
                completion_tokens as f64 / secs
            } else {
                0.0
            },
            finish_reason,
            error,
        });
    }
}

impl Drop for RequestGuard {
    fn drop(&mut self) {
        self.fail("request aborted");
    }
}

fn saturating_sub(value: &AtomicU64, amount: u64) {
    let _ = value.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
        Some(n.saturating_sub(amount))
    });
}

fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis()
        .min(u64::MAX as u128) as u64
}

/// Samples CPU/RAM and NVIDIA metrics once per second for status requests.
pub struct HostCollector {
    latest: RwLock<HostStatus>,
    started: Once,
}

impl Default for HostCollector {
    fn default() -> Self {
        Self {
            latest: RwLock::new(HostStatus::default()),
            started: Once::new(),
        }
    }
}

impl HostCollector {
    pub fn spawn(self: &Arc<Self>) {
        let this = Arc::clone(self);
        this.started.call_once(|| {
            let collector = Arc::clone(&this);
            tokio::spawn(async move {
                let mut system = sysinfo::System::new();
                system.refresh_cpu_usage();
                system.refresh_memory();
                loop {
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    system.refresh_cpu_usage();
                    system.refresh_memory();
                    let host = HostStatus {
                        gpu: gpu_status().await,
                        system: Some(SystemStatus {
                            cpu_percent: system.global_cpu_info().cpu_usage() as f64,
                            ram_used_mib: system.used_memory() / 1024 / 1024,
                            ram_total_mib: system.total_memory() / 1024 / 1024,
                        }),
                    };
                    *collector.latest.write().unwrap_or_else(|e| e.into_inner()) = host;
                }
            });
        });
    }

    pub fn snapshot(&self) -> HostStatus {
        self.latest
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}

async fn gpu_status() -> Option<GpuStatus> {
    use tokio::process::Command;

    let mut command = Command::new("nvidia-smi");
    command.kill_on_drop(true);
    command.args([
        "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
        "--format=csv,noheader,nounits",
    ]);
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    let output = tokio::time::timeout(Duration::from_millis(1800), command.output())
        .await
        .ok()?
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .to_string();
    let fields: Vec<_> = line.split(',').map(str::trim).collect();
    if fields.len() < 7 {
        return None;
    }
    Some(GpuStatus {
        name: fields[0].to_string(),
        util_percent: parse_num(fields[1]),
        vram_used_mib: parse_num::<f64>(fields[2]).map(|n| n as u64),
        vram_total_mib: parse_num::<f64>(fields[3]).map(|n| n as u64),
        temp_c: parse_num(fields[4]),
        power_w: parse_num(fields[5]),
        power_limit_w: parse_num(fields[6]),
    })
}

fn parse_num<T: std::str::FromStr>(value: &str) -> Option<T> {
    let value = value.trim();
    (!value.is_empty() && value != "[N/A]" && value != "N/A")
        .then(|| value.parse().ok())
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_metrics_keep_latest_first_and_exact_usage() {
        let metrics = Arc::new(RequestMetrics::default());
        let mut guard = metrics.begin("chat.completions");
        guard.delta();
        guard.finish(
            &GenerationUsage {
                prompt_tokens: 11,
                completion_tokens: 3,
                ..Default::default()
            },
            "stop".into(),
        );
        let totals = metrics.totals();
        assert_eq!(totals.active_requests, 0);
        assert_eq!(totals.completed_requests, 1);
        assert_eq!(totals.prompt_tokens, 11);
        assert_eq!(totals.completion_tokens, 3);
        assert_eq!(totals.output_tokens_live, 3);
        assert_eq!(metrics.recent()[0].finish_reason, "stop");
    }

    #[test]
    fn dropped_guard_is_recorded_as_safe_error() {
        let metrics = Arc::new(RequestMetrics::default());
        drop(metrics.begin("responses"));
        let totals = metrics.totals();
        assert_eq!(totals.active_requests, 0);
        assert_eq!(totals.failed_requests, 1);
        assert_eq!(
            metrics.recent()[0].error.as_deref(),
            Some("request aborted")
        );
    }
}
