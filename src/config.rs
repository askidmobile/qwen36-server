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
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.80,
            top_k: 20,
            min_p: 0.0,
            presence_penalty: 1.5,
            repetition_penalty: 1.0,
            max_tokens: 32768,
            thinking: false,
        }
    }
}

impl SamplingDefaults {
    fn from_env() -> Result<Self> {
        let d = Self::default();
        Ok(Self {
            temperature: parse_env("QWEN36_TEMPERATURE", d.temperature)?,
            top_p: parse_env("QWEN36_TOP_P", d.top_p)?,
            top_k: parse_env("QWEN36_TOP_K", d.top_k)?,
            min_p: parse_env("QWEN36_MIN_P", d.min_p)?,
            presence_penalty: parse_env("QWEN36_PRESENCE_PENALTY", d.presence_penalty)?,
            repetition_penalty: parse_env("QWEN36_REPETITION_PENALTY", d.repetition_penalty)?,
            max_tokens: parse_env("QWEN36_MAX_TOKENS", d.max_tokens)?,
            thinking: parse_env("QWEN36_THINKING", d.thinking)?,
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
        // - General tasks: temp=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, rep_penalty=1.0
        // - Precise coding tasks: temp=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, rep_penalty=1.0
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
    if let Ok(raw) = std::env::var("QWEN36_PRESETS") {
        let overrides: SamplingPresets = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("QWEN36_PRESETS: невалидный JSON: {e}"))?;
        presets.extend(overrides);
    }
    Ok(presets)
}

pub struct Config {
    /// Manifest/current pointer (`QWEN36_PROFILE`), если задан.
    pub profile: Option<PathBuf>,
    /// Один раз validated profile; исключает повторное хеширование artifacts.
    pub resolved_profile: Option<std::sync::Arc<crate::profile::ResolvedProfile>>,
    /// Путь к GGUF (`QWEN36_MODEL`) или Text artifact resolved profile.
    pub model: PathBuf,
    /// Host (`QWEN36_HOST`, default 0.0.0.0).
    pub host: String,
    /// Port (`QWEN36_PORT`, default 8080).
    pub port: u16,
    /// Именованные API-ключи (`QWEN36_API_KEYS`, JSON-массив; обязателен).
    pub api_keys: Vec<ApiKey>,
    /// Контекст (`QWEN36_CTX`, default 81920).
    pub ctx: usize,
    /// Слоты (`QWEN36_SLOTS`, default 4).
    pub slots: usize,
    /// Общий KV-бюджет всех слотов (MiB). 0 = без лимита (macOS/CPU).
    pub kv_budget_mib: f64,
    /// MiB KV на токен на слот (для admission).
    pub kv_per_tok_mib: f64,
    /// Бюджет prefix cache (MiB). 0 = выключен (`QWEN36_PREFIX_CACHE_MIB`, default 0 (выключен: primed admit восстанавливает snapshot против stale state)).
    pub prefix_cache_mib: usize,
    /// Временный media root (`QWEN36_MEDIA_TEMP`). Limits фиксированы release contract.
    pub media_temp: PathBuf,
    /// Дефолты сэмплинга из .env (QWEN36_TEMPERATURE/.../QWEN36_MAX_TOKENS/...).
    pub sampling: SamplingDefaults,
    /// Пресеты режимов (QWEN36_PRESETS JSON, merge поверх встроенных).
    pub presets: SamplingPresets,
    /// Путь к .env (для записи «сохранить по умолчанию» из WebUI).
    pub env_file: PathBuf,
}

impl Config {
    /// Загрузить `.env`, затем разобрать конфигурацию. Уже заданные process env
    /// имеют приоритет. Путь: `QWEN36_ENV_FILE`, иначе `.env` в cwd.
    pub fn load() -> Result<Self> {
        let path = std::env::var("QWEN36_ENV_FILE").unwrap_or_else(|_| ".env".into());
        load_env_file(Path::new(&path))?;
        Self::from_env_with_path(PathBuf::from(path))
    }

    pub fn from_env() -> Result<Self> {
        Self::from_env_with_path(PathBuf::from(".env"))
    }

    fn from_env_with_path(env_file: PathBuf) -> Result<Self> {
        let api_keys =
            parse_api_keys(&std::env::var("QWEN36_API_KEYS").map_err(|_| {
                anyhow!("QWEN36_API_KEYS обязателен: JSON-массив объектов key/name")
            })?)?;
        let prefix_cache_mib = parse_env("QWEN36_PREFIX_CACHE_MIB", 0usize)?;
        if prefix_cache_mib != 0 {
            anyhow::bail!("prefix cache temporarily disabled (QWEN36_PREFIX_CACHE_MIB must be 0)");
        }
        let profile = std::env::var("QWEN36_PROFILE").ok().map(PathBuf::from);
        let resolved_profile = profile
            .as_deref()
            .map(crate::profile::ResolvedProfile::load)
            .transpose()
            .with_context(|| {
                format!(
                    "QWEN36_PROFILE {}",
                    profile
                        .as_deref()
                        .unwrap_or_else(|| Path::new(""))
                        .display()
                )
            })?
            .map(std::sync::Arc::new);
        let model = match &resolved_profile {
            Some(profile) => profile.text_path.clone(),
            None => std::env::var("QWEN36_MODEL")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("models/qwen36-27b-q2_k_xl.gguf")),
        };
        let mut cfg = Self {
            profile,
            resolved_profile,
            model,
            host: std::env::var("QWEN36_HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            port: parse_env("QWEN36_PORT", 8080u16)?,
            api_keys,
            ctx: parse_env("QWEN36_CTX", 81920usize)?,
            slots: parse_env("QWEN36_SLOTS", 4usize)?,
            kv_budget_mib: 0.0,
            kv_per_tok_mib: 0.0,
            prefix_cache_mib,
            media_temp: std::env::var("QWEN36_MEDIA_TEMP")
                .map(PathBuf::from)
                .unwrap_or_else(|_| std::env::temp_dir().join("qwen36-media")),
            sampling: SamplingDefaults::from_env()?,
            presets: presets_from_env(&model)?,
            env_file,
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

fn parse_env<T: std::str::FromStr>(name: &str, default: T) -> Result<T>
where
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(v) => v
            .parse()
            .map_err(|e: T::Err| anyhow!("{name}: некорректное значение {v:?}: {e}")),
        Err(_) => Ok(default),
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
