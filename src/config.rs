//! Конфиг из env `QWEN36_*` (docs/engine-api.md §Config).

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

use crate::vram_plan;

#[derive(Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ApiKey {
    pub key: String,
    pub name: String,
}

impl std::fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ApiKey")
            .field("key", &"[REDACTED]")
            .field("name", &self.name)
            .finish()
    }
}

/// Дефолты сэмплинга — задаются в .env при запуске (QWEN36_*), меняются
/// через WebUI с сохранением обратно в .env. Код-дефолты = fallback,
/// если .env не задаёт значения.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SamplingDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub max_tokens: usize,
    pub thinking: bool,
    pub frequency_penalty: f32,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            max_tokens: 32768,
            thinking: true,
        }
    }
}

impl SamplingDefaults {
    fn from_env() -> Result<Self> {
        let d = Self::default();
        Ok(Self {
            temperature: parse_env("TEMPERATURE", d.temperature)?,
            top_p: parse_env("TOP_P", d.top_p)?,
            top_k: parse_env("TOP_K", d.top_k)?,
            min_p: parse_env("MIN_P", d.min_p)?,
            presence_penalty: parse_env("PRESENCE_PENALTY", d.presence_penalty)?,
            repetition_penalty: parse_env("REPETITION_PENALTY", d.repetition_penalty)?,
            max_tokens: parse_env("MAX_TOKENS", d.max_tokens)?,
            thinking: parse_env("THINKING", d.thinking)?,
            frequency_penalty: parse_env("FREQUENCY_PENALTY", d.frequency_penalty)?,
        })
    }
}

/// Пресет режима (instruct/thinking/thinking-coding). У разных моделей
/// оптимум разный — переопределяются через QWEN36_PRESETS (JSON) и WebUI.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SamplingPresetValues {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingPresetValues {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.80,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 1.5,
            repetition_penalty: 1.0,
        }
    }
}

pub type SamplingPresets = std::collections::HashMap<String, SamplingPresetValues>;

pub fn default_presets_for_model(model_name: &str) -> SamplingPresets {
    let lower = model_name.to_lowercase();
    let mut m = SamplingPresets::new();

    if lower.contains("ornith-1.5") || lower.contains("ornith_1.5") || lower.contains("ornith1.5") {
        // Рекомендованные параметры из официального model card Ornith-1.5:
        // - Thinking mode for general tasks: temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, rep_penalty=1.0
        // - Thinking mode for precise coding tasks (WebDev): temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, rep_penalty=1.0
        // - Instruct mode for general tasks: temp=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, rep_penalty=1.0
        // - Instruct mode for reasoning tasks: temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, rep_penalty=1.0
        m.insert(
            "thinking".into(),
            SamplingPresetValues {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 1.5,
                repetition_penalty: 1.0,
            },
        );
        m.insert(
            "thinking-coding".into(),
            SamplingPresetValues {
                temperature: 0.6,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
            },
        );
        m.insert(
            "instruct".into(),
            SamplingPresetValues {
                temperature: 0.7,
                top_p: 0.80,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 1.5,
                repetition_penalty: 1.0,
            },
        );
        m.insert(
            "instruct-reasoning".into(),
            SamplingPresetValues {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 1.5,
                repetition_penalty: 1.0,
            },
        );
    } else if lower.contains("gemma-4") || lower.contains("gemma_4") || lower.contains("gemma4") {
        // Gemma 4 E4B — reasoning model. temp=1.0 ломает transition
        // thought→final (модель —забывает” закрыть <|channel>thought и
        // пишет ответ внутри reasoning без <|channel>final). temp=0.7 stable.
        m.insert(
            "thinking".into(),
            SamplingPresetValues {
                temperature: 0.7,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
            },
        );
        m.insert(
            "thinking-coding".into(),
            SamplingPresetValues {
                temperature: 0.6,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
            },
        );
        m.insert(
            "instruct".into(),
            SamplingPresetValues {
                temperature: 0.7,
                top_p: 0.80,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 1.5,
                repetition_penalty: 1.0,
            },
        );
    } else {
        // Qwen3.8 / Qwen3.5 / Qwen3.6 / General LLM default:
        m.insert("instruct".into(), SamplingPresetValues::default());
        m.insert(
            "thinking".into(),
            SamplingPresetValues {
                temperature: 1.0,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
            },
        );
        m.insert(
            "thinking-coding".into(),
            SamplingPresetValues {
                temperature: 0.6,
                top_p: 0.95,
                top_k: 20,
                min_p: 0.0,
                presence_penalty: 0.0,
                repetition_penalty: 1.0,
            },
        );
    }
    m
}

pub fn default_presets() -> SamplingPresets {
    default_presets_for_model("")
}

/// QWEN36_PRESETS: JSON-объект {name: {temperature,...}} — частичный merge
/// поверх встроенных пресетов.
fn presets_from_env(model_path: &Path) -> Result<SamplingPresets> {
    let model_name = model_path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default();
    let mut presets = default_presets_for_model(&model_name);
    if let Some(raw) = get_env_var("PRESETS") {
        let overrides: SamplingPresets = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("PRESETS: невалидный JSON: {e}"))?;
        presets.extend(overrides);
    }
    Ok(presets)
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Manifest/current pointer (`PROFILE`), если задан.
    pub profile: Option<PathBuf>,
    pub resolved_profile: Option<std::sync::Arc<crate::profile::ResolvedProfile>>,
    pub model: PathBuf,
    pub host: String,
    pub port: u16,
    pub api_keys: Vec<ApiKey>,
    pub ctx: usize,
    pub slots: usize,
    pub kv_budget_mib: f64,
    pub kv_per_tok_mib: f64,
    pub prefix_cache_mib: usize,
    pub media_temp: PathBuf,
    pub sampling: SamplingDefaults,
    pub presets: SamplingPresets,
    pub env_file: PathBuf,
    /// Batch size для prefill (`BATCH_SIZE`, default 2048).
    pub batch_size: usize,
    /// CPU threads (`THREADS`, default = количество ядер).
    pub threads: usize,
    /// GPU layers to offload (`GPU_LAYERS`, default 999 = все).
    pub gpu_layers: usize,
    /// KV cache type (`KV_CACHE_TYPE`, default "q8_f16").
    pub kv_cache_type: String,
    /// Mmap (`MMAP`, default 1 = включен).
    pub mmap: bool,
    /// RoPE scale (`ROPE_SCALE`, default 1.0).
    pub rope_scale: f32,
    /// RoPE scale type (`ROPE_SCALE_TYPE`, default "none").
    pub rope_scale_type: String,
    /// Seed (`SEED`, default 0 = random).
    pub seed: u64,
    /// Context overflow policy (`CTX_OVERFLOW`, default "sliding_window").
    pub ctx_overflow: String,
    /// Frequency penalty (`FREQUENCY_PENALTY`, default 0.0).
    pub frequency_penalty: f32,
    /// Max queue length (`MAX_QUEUE`, default 64).
    pub max_queue: usize,
    /// Request timeout in seconds (`REQ_TIMEOUT`, default 600).
    pub req_timeout: u64,
    /// Flash Attention (`FLASH_ATTN`, default 1 = включен).
    pub flash_attn: bool,
}

impl Config {
    /// Загрузить `.env`, затем разобрать конфигурацию. Путь: `ENV_FILE`, иначе `.env`.
    pub fn load() -> Result<Self> {
        let path = get_env_var("ENV_FILE").unwrap_or_else(|| ".env".into());
        load_env_file(Path::new(&path))?;
        Self::from_env_with_path(PathBuf::from(path))
    }

    pub fn from_env() -> Result<Self> {
        Self::from_env_with_path(PathBuf::from(".env"))
    }

    fn from_env_with_path(env_file: PathBuf) -> Result<Self> {
        let api_keys_raw = get_env_var("API_KEYS")
            .ok_or_else(|| anyhow!("API_KEYS (или QWEN36_API_KEYS) обязателен: JSON-массив объектов key/name"))?;
        let api_keys = parse_api_keys(&api_keys_raw)?;
        let prefix_cache_mib = parse_env("PREFIX_CACHE_MIB", 0usize)?;
        if prefix_cache_mib != 0 {
            anyhow::bail!("prefix cache temporarily disabled (PREFIX_CACHE_MIB must be 0)");
        }
        let profile = get_env_var("PROFILE").map(PathBuf::from);
        let resolved_profile = profile
            .as_deref()
            .map(crate::profile::ResolvedProfile::load)
            .transpose()
            .with_context(|| {
                format!(
                    "PROFILE {}",
                    profile
                        .as_deref()
                        .unwrap_or_else(|| Path::new(""))
                        .display()
                )
            })?
            .map(std::sync::Arc::new);
        let model = match &resolved_profile {
            Some(profile) => profile.text_path.clone(),
            None => get_env_var("MODEL")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("models/qwen36-27b-q2_k_xl.gguf")),
        };
        let presets = presets_from_env(&model)?;
        let mut cfg = Self {
            profile,
            resolved_profile,
            model,
            host: get_env_var("HOST").unwrap_or_else(|| "0.0.0.0".into()),
            port: parse_env("PORT", 18099u16)?,
            api_keys,
            ctx: parse_env("CTX", 131072usize)?,
            slots: parse_env("SLOTS", 4usize)?,
            kv_budget_mib: 0.0,
            kv_per_tok_mib: 0.0,
            prefix_cache_mib,
            media_temp: get_env_var("MEDIA_TEMP")
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("yttri-media")),
            sampling: SamplingDefaults::from_env()?,
            presets,
            env_file,
            batch_size: parse_env("BATCH_SIZE", 2048usize)?,
            threads: parse_env("THREADS", num_cpus())?,
            gpu_layers: parse_env("GPU_LAYERS", 999usize)?,
            kv_cache_type: get_env_var("KV_CACHE_TYPE").unwrap_or_else(|| "q8_f16".into()),
            mmap: parse_env("MMAP", 1u8)? != 0,
            rope_scale: parse_env("ROPE_SCALE", 1.0f32)?,
            rope_scale_type: get_env_var("ROPE_SCALE_TYPE").unwrap_or_else(|| "none".into()),
            seed: parse_env("SEED", 0u64)?,
            ctx_overflow: get_env_var("CTX_OVERFLOW").unwrap_or_else(|| "sliding_window".into()),
            frequency_penalty: parse_env("FREQUENCY_PENALTY", 0.0f32)?,
            max_queue: parse_env("MAX_QUEUE", 64usize)?,
            req_timeout: parse_env("REQ_TIMEOUT", 600u64)?,
            flash_attn: parse_env("FLASH_ATTN", 1u8)? != 0,
        };
        cfg.apply_vram_plan()?;
        Ok(cfg)
    }

    /// FR-002 + dynamic KV (vLLM-style): ctx НЕ режется под worst-case
    /// (только до native модели); общий KV-бюджет enforce'ится движком
    /// (admission + очередь + force-finish самого длинного слота).
    /// Отключается QWEN36_NO_VRAM_PLAN=1. На macOS/CPU — без лимита.
    fn apply_vram_plan(&mut self) -> Result<()> {
        if std::env::var_os("QWEN36_NO_VRAM_PLAN").is_some() {
            return Ok(());
        }
        let Some(total) = vram_plan::total_vram_mib() else {
            return Ok(());
        };
        let fp = vram_plan::footprint_from_gguf(&self.model)?;
        let plan = vram_plan::compute_dynamic(total, &fp, self.ctx, self.slots)?;
        eprintln!("{}", plan.report);
        self.ctx = plan.ctx;
        self.slots = plan.slots;
        self.kv_budget_mib = plan.kv_budget_mib;
        self.kv_per_tok_mib = plan.kv_per_tok_mib;
        Ok(())
    }
}

fn parse_api_keys(raw: &str) -> Result<Vec<ApiKey>> {
    let keys: Vec<ApiKey> = serde_json::from_str(raw)
        .context("QWEN36_API_KEYS: ожидается JSON-массив объектов key/name")?;
    if keys.is_empty() {
        anyhow::bail!("QWEN36_API_KEYS: нужен хотя бы один ключ");
    }
    let mut names = std::collections::HashSet::new();
    let mut values = std::collections::HashSet::new();
    for entry in &keys {
        if entry.name.trim().is_empty() || entry.key.trim().is_empty() {
            anyhow::bail!("QWEN36_API_KEYS: key и name не могут быть пустыми");
        }
        if entry.name.trim() != entry.name || entry.key.trim() != entry.key {
            anyhow::bail!(
                "QWEN36_API_KEYS: key и name не должны начинаться/заканчиваться пробелами"
            );
        }
        if !names.insert(entry.name.as_str()) {
            anyhow::bail!("QWEN36_API_KEYS: повтор name {:?}", entry.name);
        }
        if !values.insert(entry.key.as_str()) {
            anyhow::bail!("QWEN36_API_KEYS: ключ {:?} указан повторно", entry.name);
        }
    }
    Ok(keys)
}

fn load_env_file(path: &Path) -> Result<()> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e).with_context(|| format!("read env file {}", path.display())),
    };
    for (idx, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line);
        let (name, raw_value) = line
            .split_once('=')
            .ok_or_else(|| anyhow!("{}:{}: ожидается NAME=value", path.display(), idx + 1))?;
        let name = name.trim();
        if !valid_env_name(name) {
            anyhow::bail!(
                "{}:{}: некорректное имя переменной {name:?}",
                path.display(),
                idx + 1
            );
        }
        if std::env::var_os(name).is_none() {
            let value = raw_value.trim();
            let value = if value.len() >= 2
                && ((value.starts_with('"') && value.ends_with('"'))
                    || (value.starts_with('\'') && value.ends_with('\'')))
            {
                &value[1..value.len() - 1]
            } else {
                value
            };
            std::env::set_var(name, value);
        }
    }
    Ok(())
}

fn valid_env_name(name: &str) -> bool {
    let mut bytes = name.bytes();
    matches!(bytes.next(), Some(b'A'..=b'Z' | b'a'..=b'z' | b'_'))
        && bytes.all(|b| b.is_ascii_alphanumeric() || b == b'_')
}

fn get_env_var(name: &str) -> Option<String> {
    std::env::var(name)
        .or_else(|_| std::env::var(format!("QWEN36_{name}")))
        .or_else(|_| std::env::var(format!("YTTRI_{name}")))
        .ok()
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match get_env_var(name) {
        Some(v) => v
            .parse()
            .map_err(|e: T::Err| anyhow!("{name}: некорректное значение {v:?}: {e}")),
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    // Тесты гоняют env процесса — сериализуем вручную через один тест.
    #[test]
    fn from_env_defaults_and_required_keys() {
        env::remove_var("QWEN36_API_KEYS");
        assert!(Config::from_env().is_err(), "без API keys старт запрещён");

        env::set_var(
            "QWEN36_API_KEYS",
            r#"[{"key":"k1","name":"primary"},{"key":"k2","name":"backup"}]"#,
        );
        env::remove_var("QWEN36_PROFILE");
        env::remove_var("QWEN36_MODEL");
        env::remove_var("QWEN36_HOST");
        env::remove_var("QWEN36_PORT");
        env::remove_var("QWEN36_CTX");
        env::remove_var("QWEN36_SLOTS");
        env::remove_var("QWEN36_MEDIA_TEMP");
        let c = Config::from_env().unwrap();
        assert!(c.profile.is_none());
        assert!(c.resolved_profile.is_none());
        assert_eq!(c.api_keys.len(), 2);
        assert_eq!(c.api_keys[0].name, "primary");
        assert_eq!(c.model, PathBuf::from("models/qwen36-27b-q2_k_xl.gguf"));
        assert_eq!(c.host, "0.0.0.0");
        assert_eq!(c.port, 8080);
        assert_eq!(c.ctx, 81920);
        assert_eq!(c.slots, 4);

        env::set_var("QWEN36_PORT", "9000");
        env::set_var("QWEN36_CTX", "4096");
        let c = Config::from_env().unwrap();
        assert_eq!(c.port, 9000);
        assert_eq!(c.ctx, 4096);

        for invalid in [
            "[]",
            r#"[{"key":"","name":"empty"}]"#,
            r#"[{"key":"same","name":"a"},{"key":"same","name":"b"}]"#,
            r#"[{"key":"a","name":"same"},{"key":"b","name":"same"}]"#,
            r#"[{"key":" spaced","name":"bad"}]"#,
        ] {
            env::set_var("QWEN36_API_KEYS", invalid);
            assert!(Config::from_env().is_err(), "{invalid}");
        }
        env::set_var("QWEN36_API_KEYS", r#"[{"key":"k1","name":"primary"}]"#);
        env::set_var("QWEN36_PORT", "x");
        assert!(Config::from_env().is_err());
        env::set_var("QWEN36_PORT", "8080");
        env::set_var("QWEN36_PREFIX_CACHE_MIB", "1");
        let err = Config::from_env().unwrap_err().to_string();
        assert!(err.contains("prefix cache temporarily disabled"));
        env::remove_var("QWEN36_PREFIX_CACHE_MIB");

        let path = std::env::temp_dir().join(format!("qwen36-env-{}.tmp", std::process::id()));
        std::fs::write(
            &path,
            "# comment\nQWEN36_API_KEYS='[{\"key\":\"file-key\",\"name\":\"file\"}]'\nQWEN36_PORT=9001\n",
        )
        .unwrap();
        env::remove_var("QWEN36_API_KEYS");
        env::set_var("QWEN36_PORT", "9002");
        load_env_file(&path).unwrap();
        assert_eq!(env::var("QWEN36_PORT").unwrap(), "9002");
        assert_eq!(
            parse_api_keys(&env::var("QWEN36_API_KEYS").unwrap()).unwrap()[0].name,
            "file"
        );
        env::remove_var("QWEN36_API_KEYS");
        env::remove_var("QWEN36_PORT");

        env::set_var("QWEN36_ENV_FILE", &path);
        env::set_var("QWEN36_NO_VRAM_PLAN", "1");
        let c = Config::load().unwrap();
        assert_eq!(c.api_keys[0].name, "file");
        assert_eq!(c.port, 9001);
        env::remove_var("QWEN36_ENV_FILE");
        env::remove_var("QWEN36_NO_VRAM_PLAN");
        env::remove_var("QWEN36_API_KEYS");
        env::remove_var("QWEN36_PORT");
        std::fs::remove_file(path).unwrap();
    }
}
