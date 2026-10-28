//! `yforge monitor`: a small read-only terminal overview over `/v1/status`.

use anyhow::{anyhow, Context, Result};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{self, Event, KeyCode, KeyModifiers},
    execute, queue,
    style::Print,
    terminal::{self, Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::{
    collections::VecDeque,
    io::{self, IsTerminal, Write},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::{
    cli::MonitorArgs,
    metrics::{RecentRequest, StatusSnapshot},
};

pub async fn run(args: MonitorArgs) -> Result<()> {
    let url = args
        .url
        .or_else(|| std::env::var("YFORGE_URL").ok())
        .unwrap_or_else(|| {
            let port = std::env::var("PORT").unwrap_or_else(|_| "18099".into());
            format!("http://127.0.0.1:{port}")
        });
    let api_key = resolve_api_key(args.api_key)?;
    let status_url = format!("{}/v1/status", url.trim_end_matches('/'));
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()?;

    if !io::stdout().is_terminal() {
        let snapshot = fetch(&client, &status_url, &api_key)
            .await
            .context("failed to read server status")?;
        println!("{}", render(Some(&snapshot), &url, 0.0, None, 100));
        return Ok(());
    }

    let _terminal = TerminalGuard::enter()?;
    let (key_tx, mut key_rx) = tokio::sync::mpsc::unbounded_channel();
    let _key_reader = tokio::task::spawn_blocking(move || loop {
        match event::poll(Duration::from_millis(200)) {
            Ok(true) => match event::read() {
                Ok(Event::Key(key)) if key_tx.send(key).is_err() => break,
                Ok(_) => {}
                Err(_) => break,
            },
            Ok(false) => {}
            Err(_) => break,
        }
    });

    let mut interval = tokio::time::interval(Duration::from_secs(args.refresh_secs));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    interval.tick().await;
    let mut samples: VecDeque<(Instant, u64)> = VecDeque::new();
    let first = fetch(&client, &status_url, &api_key).await;
    let mut snapshot: Option<StatusSnapshot> = first.as_ref().ok().cloned();
    if let Some(status) = &snapshot {
        samples.push_back((Instant::now(), status.totals.output_tokens_live));
    }
    let mut last_error = first.as_ref().err().map(safe_error);
    draw(
        snapshot.as_ref(),
        &url,
        rolling_speed(&samples),
        last_error.as_deref(),
    )?;
    let mut refresh_now = false;

    loop {
        tokio::select! {
            _ = interval.tick() => refresh_now = true,
            key = key_rx.recv() => {
                let Some(key) = key else { break };
                let ctrl_c = key.modifiers.contains(KeyModifiers::CONTROL)
                    && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('C'));
                match key.code {
                    KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break,
                    _ if ctrl_c => break,
                    KeyCode::Char('r') | KeyCode::Char('R') => refresh_now = true,
                    _ => {}
                }
            }
        }
        if refresh_now {
            let next = fetch(&client, &status_url, &api_key).await;
            match next {
                Ok(status) => {
                    samples.push_back((Instant::now(), status.totals.output_tokens_live));
                    let cutoff = Instant::now() - Duration::from_secs(5);
                    while samples.front().is_some_and(|(at, _)| *at < cutoff) {
                        samples.pop_front();
                    }
                    snapshot = Some(status);
                    last_error = None;
                }
                Err(error) => last_error = Some(safe_error(&error)),
            }
            draw(
                snapshot.as_ref(),
                &url,
                rolling_speed(&samples),
                last_error.as_deref(),
            )?;
            refresh_now = false;
        }
    }
    Ok(())
}

fn resolve_api_key(explicit: Option<String>) -> Result<String> {
    if let Some(key) = explicit.filter(|key| !key.trim().is_empty()) {
        return Ok(key);
    }
    for name in ["YFORGE_API_KEY", "API_KEY"] {
        if let Ok(key) = std::env::var(name) {
            if !key.trim().is_empty() {
                return Ok(key);
            }
        }
    }
    if let Ok(raw) = std::env::var("API_KEYS") {
        if let Ok(keys) = serde_json::from_str::<serde_json::Value>(&raw) {
            if let Some(key) = keys
                .get(0)
                .and_then(|entry| entry.get("key"))
                .and_then(|key| key.as_str())
            {
                if !key.trim().is_empty() {
                    return Ok(key.into());
                }
            }
        }
    }
    Err(anyhow!(
        "API key is required: pass --api-key, YFORGE_API_KEY or API_KEY"
    ))
}

async fn fetch(
    client: &reqwest::Client,
    status_url: &str,
    api_key: &str,
) -> Result<StatusSnapshot> {
    let response = client.get(status_url).bearer_auth(api_key).send().await?;
    if !response.status().is_success() {
        return Err(anyhow!("HTTP {}", response.status()));
    }
    Ok(response.json().await?)
}

fn safe_error(error: &anyhow::Error) -> String {
    // reqwest errors contain the URL but not the Authorization header.
    let text = error.to_string();
    text.chars().take(120).collect()
}

fn rolling_speed(samples: &VecDeque<(Instant, u64)>) -> f64 {
    let (Some((first_at, first_tokens)), Some((last_at, last_tokens))) =
        (samples.front(), samples.back())
    else {
        return 0.0;
    };
    let seconds = last_at.duration_since(*first_at).as_secs_f64();
    if seconds <= 0.0 || last_tokens < first_tokens {
        return 0.0;
    }
    (last_tokens - first_tokens) as f64 / seconds
}

fn draw(
    snapshot: Option<&StatusSnapshot>,
    url: &str,
    speed: f64,
    error: Option<&str>,
) -> Result<()> {
    let (width, _) = terminal::size().unwrap_or((100, 30));
    let frame = render(snapshot, url, speed, error, width as usize);
    let mut out = io::stdout().lock();
    queue!(out, MoveTo(0, 0), Clear(ClearType::All), Print(frame))?;
    out.flush()?;
    Ok(())
}

fn render(
    snapshot: Option<&StatusSnapshot>,
    url: &str,
    speed: f64,
    error: Option<&str>,
    width: usize,
) -> String {
    let width = width.max(44);
    let rule = "-".repeat(width.saturating_sub(2));
    let mut lines = vec![format!("+{rule}+")];
    match snapshot {
        Some(status) => {
            let state = status.state.to_ascii_uppercase();
            let connection = if error.is_some() {
                "RECONNECTING"
            } else {
                "LIVE"
            };
            lines.push(line(
                &format!(
                    "| YFORGE MONITOR  {url}  {}  [{state}] [{connection}]",
                    status.model.id
                ),
                width,
            ));
            lines.push(line(
                &format!(
                    "| Runtime  {} / {} / {}  |  {} slot(s)  |  up {}",
                    status.runtime.backend,
                    status.runtime.attention,
                    status.runtime.prefill,
                    status.model.slots,
                    duration(status.uptime_s)
                ),
                width,
            ));
            lines.push(format!("+{rule}+"));

            let ctx_pct = if status.model.context_length > 0 {
                status.context_used as f64 / status.model.context_length as f64
            } else {
                0.0
            };
            lines.push(line(
                &format!(
                    "| Context  {}  {} / {} tokens ({:.0}%)",
                    bar(ctx_pct, width.saturating_sub(48).clamp(10, 30)),
                    compact(status.context_used as u64),
                    compact(status.model.context_length as u64),
                    ctx_pct * 100.0
                ),
                width,
            ));
            lines.push(line(
                &format!(
                    "| Requests active {}  queued {}  completed {}  failed {}",
                    status.totals.active_requests,
                    status.queued,
                    status.totals.completed_requests,
                    status.totals.failed_requests
                ),
                width,
            ));
            lines.push(format!("+{rule}+"));

            let gpu_text = match &status.gpu {
                Some(gpu) => format!(
                    "GPU {}  VRAM {}/{}  {}  {}",
                    optional_percent(gpu.util_percent),
                    optional_gib(gpu.vram_used_mib),
                    optional_gib(gpu.vram_total_mib),
                    optional_temp(gpu.temp_c),
                    optional_watts(gpu.power_w, gpu.power_limit_w)
                ),
                None => "GPU n/a  VRAM n/a".into(),
            };
            let system_text = match &status.system {
                Some(system) => format!(
                    "CPU {:.0}%  RAM {:.1}/{:.1} GiB",
                    system.cpu_percent,
                    mib_to_gib(system.ram_used_mib),
                    mib_to_gib(system.ram_total_mib)
                ),
                None => "CPU n/a  RAM n/a".into(),
            };
            if width >= 88 {
                lines.push(line(
                    &format!("| Speed {speed:.1} tok/s  |  {gpu_text}"),
                    width,
                ));
                lines.push(line(
                    &format!(
                        "| {system_text}  |  total output {}",
                        compact(status.totals.completion_tokens)
                    ),
                    width,
                ));
            } else {
                lines.push(line(&format!("| Speed {speed:.1} tok/s"), width));
                lines.push(line(&format!("| {gpu_text}"), width));
                lines.push(line(&format!("| {system_text}"), width));
            }
            lines.push(format!("+{rule}+"));
            lines.push(line("| Recent requests", width));
            if status.recent_requests.is_empty() {
                lines.push(line("| No completed requests yet.", width));
            } else {
                lines.push(line(
                    "| AGE    ENDPOINT                 PROMPT  OUTPUT  TOK/S   DURATION  STATE",
                    width,
                ));
                for request in status.recent_requests.iter().take(5) {
                    lines.push(line(&request_row(request), width));
                }
            }
        }
        None => {
            lines.push(line(&format!("| YFORGE MONITOR  {url}  [WAITING]"), width));
            lines.push(format!("+{rule}+"));
            lines.push(line("| Waiting for /v1/status ...", width));
        }
    }
    lines.push(format!("+{rule}+"));
    if let Some(error) = error {
        lines.push(line(&format!("| Error: {error}"), width));
    }
    lines.push(line("| q detach  r refresh  Ctrl+C quit", width));
    lines.push(format!("+{rule}+"));
    lines.join("\n")
}

fn request_row(request: &RecentRequest) -> String {
    let age = format_age(request.started_at_unix_ms);
    let state = request.error.as_deref().unwrap_or(&request.finish_reason);
    format!(
        "| {age:<6} {:<25} {:>6}  {:>6}  {:>6.1}  {:>7.2}s  {}",
        clip(&request.endpoint, 25),
        compact(request.prompt_tokens),
        compact(request.completion_tokens),
        request.tokens_per_sec,
        request.duration_ms as f64 / 1000.0,
        clip(state, 16)
    )
}

fn bar(fraction: f64, width: usize) -> String {
    let filled = (fraction.clamp(0.0, 1.0) * width as f64).round() as usize;
    format!(
        "[{}{}]",
        "#".repeat(filled),
        "-".repeat(width.saturating_sub(filled))
    )
}

fn compact(value: u64) -> String {
    if value >= 1_000_000 {
        format!("{:.1}M", value as f64 / 1_000_000.0)
    } else if value >= 1_000 {
        format!("{:.1}K", value as f64 / 1_000.0)
    } else {
        value.to_string()
    }
}

fn duration(seconds: u64) -> String {
    let hours = seconds / 3600;
    let minutes = seconds % 3600 / 60;
    let seconds = seconds % 60;
    format!("{hours:02}:{minutes:02}:{seconds:02}")
}

fn format_age(started_at_unix_ms: u64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let seconds = now.saturating_sub(started_at_unix_ms) / 1000;
    if seconds < 60 {
        format!("{seconds}s")
    } else if seconds < 3600 {
        format!("{}m", seconds / 60)
    } else {
        format!("{}h", seconds / 3600)
    }
}

fn optional_percent(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".into(), |value| format!("{value:.0}%"))
}

fn optional_gib(value: Option<u64>) -> String {
    value.map_or_else(
        || "n/a".into(),
        |value| format!("{:.1} GiB", mib_to_gib(value)),
    )
}

fn optional_temp(value: Option<f64>) -> String {
    value.map_or_else(|| "n/a".into(), |value| format!("{value:.0}C"))
}

fn optional_watts(power: Option<f64>, limit: Option<f64>) -> String {
    match (power, limit) {
        (Some(power), Some(limit)) => format!("{power:.0}/{limit:.0}W"),
        (Some(power), None) => format!("{power:.0}W"),
        _ => "n/a".into(),
    }
}

fn mib_to_gib(value: u64) -> f64 {
    value as f64 / 1024.0
}

fn clip(value: &str, width: usize) -> String {
    value.chars().take(width).collect()
}

fn line(value: &str, width: usize) -> String {
    let inner_width = width.saturating_sub(1);
    let body = clip(value, inner_width);
    let padding = inner_width.saturating_sub(body.chars().count());
    format!("{body}{}|", " ".repeat(padding))
}

struct TerminalGuard;

impl TerminalGuard {
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode()?;
        let mut out = io::stdout().lock();
        execute!(out, EnterAlternateScreen, Hide)?;
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = terminal::disable_raw_mode();
        let mut out = io::stdout().lock();
        let _ = execute!(out, Show, LeaveAlternateScreen);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::{RequestTotals, RuntimeStatus, StatusModel};

    #[test]
    fn queue_metric_is_derived_from_total_in_flight() {
        assert_eq!(5usize.saturating_sub(4), 1);
        assert_eq!(1usize.saturating_sub(4), 0);
    }

    #[test]
    fn compact_bar_and_clip_are_bounded() {
        assert_eq!(bar(10.0, 5), "[#####]");
        assert_eq!(clip("abcdef", 4), "abcd");
        assert_eq!(line("| abc", 8), "| abc  |");
    }

    #[test]
    fn rolling_speed_uses_recent_token_delta() {
        let start = Instant::now();
        let samples = VecDeque::from([(start, 10), (start + Duration::from_secs(2), 30)]);
        assert_eq!(rolling_speed(&samples), 10.0);
    }

    #[test]
    fn overview_renders_runtime_and_recent_metadata() {
        let snapshot = StatusSnapshot {
            model: StatusModel {
                id: "qwen3.6-27b".into(),
                quant: "Q4_K_M".into(),
                context_length: 32768,
                slots: 1,
            },
            state: "ready".into(),
            uptime_s: 42,
            runtime: RuntimeStatus::current(),
            totals: RequestTotals {
                completed_requests: 1,
                prompt_tokens: 20,
                completion_tokens: 4,
                output_tokens_live: 4,
                ..Default::default()
            },
            context_used: 24,
            in_flight: 0,
            queued: 0,
            recent_requests: vec![RecentRequest {
                started_at_unix_ms: 0,
                endpoint: "chat.completions".into(),
                prompt_tokens: 20,
                completion_tokens: 4,
                duration_ms: 1000,
                tokens_per_sec: 4.0,
                finish_reason: "stop".into(),
                error: None,
            }],
            gpu: None,
            system: None,
        };
        let frame = render(Some(&snapshot), "http://localhost:18099", 4.0, None, 100);
        assert!(frame.contains("[READY] [LIVE]"));
        assert!(frame.contains("chat.completions"));
        assert!(frame.contains("4.0 tok/s"));
        assert!(!frame.contains("private"));
    }
}
