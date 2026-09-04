//! Конфиг из env: имена без префикса — `CTX`, `SLOTS`, `MODEL`
//! (docs/engine-api.md §Config).

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

/// Дефолты сэмплинга — задаются в .env при запуске, меняются
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
        let values = Self {
            temperature: parse_env("TEMPERATURE", d.temperature)?,
            top_p: parse_env("TOP_P", d.top_p)?,
            top_k: parse_env("TOP_K", d.top_k)?,
            min_p: parse_env("MIN_P", d.min_p)?,
            presence_penalty: parse_env("PRESENCE_PENALTY", d.presence_penalty)?,
            repetition_penalty: parse_env("REPETITION_PENALTY", d.repetition_penalty)?,
            max_tokens: parse_env("MAX_TOKENS", d.max_tokens)?,
            thinking: parse_env("THINKING", d.thinking)?,
            frequency_penalty: parse_env("FREQUENCY_PENALTY", d.frequency_penalty)?,
        };
        values.validate()?;
        Ok(values)
    }

    fn validate(&self) -> Result<()> {
        if !(self.temperature.is_finite() && self.temperature >= 0.0) {
            return Err(anyhow!(
                "TEMPERATURE должен быть конечным и неотрицательным"
            ));
        }
        if !(self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(anyhow!("TOP_P должен быть в интервале (0, 1]"));
        }
        if !(self.min_p.is_finite() && (0.0..=1.0).contains(&self.min_p)) {
            return Err(anyhow!("MIN_P должен быть в интервале [0, 1]"));
        }
        if !(self.presence_penalty.is_finite() && self.presence_penalty >= 0.0) {
            return Err(anyhow!(
                "PRESENCE_PENALTY должен быть конечным и неотрицательным"
            ));
        }
        if !(self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0) {
            return Err(anyhow!(
                "REPETITION_PENALTY должен быть конечным и положительным"
            ));
        }
        if self.max_tokens == 0 {
            return Err(anyhow!("MAX_TOKENS должен быть больше нуля"));
        }
        Ok(())
    }
}

/// Пресет режима. Встроенные значения выбираются по семейству модели;
/// пользовательские переопределения хранятся в MODEL_PRESETS.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

/// Явные env-переопределения сэмплинга поверх пресета карточки модели.
///
/// `None` — переменная не задана, действует пресет. Отдельный тип нужен именно
/// ради этого различия: `SamplingDefaults` подставляет значение по умолчанию и
/// «не задано» от «задано таким же» не отличает, поэтому его значения молча
/// проигрывали пресету у любой модели с карточкой.
#[derive(Debug, Clone)]
pub struct SamplingPolicy {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
    pub min_p: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub repetition_penalty: Option<f32>,
    /// Игнорировать сэмплинг из запроса клиента (`SAMPLING_LOCK`, по умолчанию
    /// включено). Клиент, приславший temperature 0, загонял модель в повтор
    /// собственных неудачных вызовов инструментов — замер 2026-08-30.
    pub lock: bool,
}

impl Default for SamplingPolicy {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            repetition_penalty: None,
            lock: true,
        }
    }
}

fn env_opt<T: std::str::FromStr>(name: &str) -> Result<Option<T>>
where
    T::Err: std::fmt::Display,
{
    match get_env_var(name).filter(|v| !v.trim().is_empty()) {
        Some(v) => v
            .parse()
            .map(Some)
            .map_err(|e: T::Err| anyhow!("{name}: некорректное значение {v:?}: {e}")),
        None => Ok(None),
    }
}

impl SamplingPolicy {
    pub fn from_env() -> Result<Self> {
        let lock = match get_env_var("SAMPLING_LOCK").as_deref().map(str::trim) {
            None | Some("") | Some("1") | Some("true") | Some("TRUE") => true,
            Some("0") | Some("false") | Some("FALSE") => false,
            Some(value) => {
                return Err(anyhow!(
                    "SAMPLING_LOCK: ожидалось 0/1/false/true, получено {value:?}"
                ))
            }
        };
        let policy = Self {
            temperature: env_opt("TEMPERATURE")?,
            top_p: env_opt("TOP_P")?,
            top_k: env_opt("TOP_K")?,
            min_p: env_opt("MIN_P")?,
            presence_penalty: env_opt("PRESENCE_PENALTY")?,
            repetition_penalty: env_opt("REPETITION_PENALTY")?,
            lock,
        };
        let probe = SamplingDefaults {
            temperature: policy.temperature.unwrap_or(0.0),
            top_p: policy.top_p.unwrap_or(1.0),
            top_k: policy.top_k.unwrap_or(0),
            min_p: policy.min_p.unwrap_or(0.0),
            presence_penalty: policy.presence_penalty.unwrap_or(0.0),
            repetition_penalty: policy.repetition_penalty.unwrap_or(1.0),
            ..SamplingDefaults::default()
        };
        probe.validate()?;
        Ok(policy)
    }

    /// Наложить заданные переменные на значения пресета.
    pub fn apply(&self, v: &mut SamplingPresetValues) {
        if let Some(x) = self.temperature {
            v.temperature = x;
        }
        if let Some(x) = self.top_p {
            v.top_p = x;
        }
        if let Some(x) = self.top_k {
            v.top_k = x;
        }
        if let Some(x) = self.min_p {
            v.min_p = x;
        }
        if let Some(x) = self.presence_penalty {
            v.presence_penalty = x;
        }
        if let Some(x) = self.repetition_penalty {
            v.repetition_penalty = x;
        }
    }

    /// Политика, эквивалентная полностью заполненным значениям из WebUI.
    /// `lock` сохраняется отдельно: endpoint дефолтов не управляет тем,
    /// разрешено ли клиентским запросам переопределять сэмплинг.
    pub fn from_defaults(defaults: &SamplingDefaults, lock: bool) -> Self {
        Self {
            temperature: Some(defaults.temperature),
            top_p: Some(defaults.top_p),
            top_k: Some(defaults.top_k),
            min_p: Some(defaults.min_p),
            presence_penalty: Some(defaults.presence_penalty),
            repetition_penalty: Some(defaults.repetition_penalty),
            lock,
        }
    }
}

pub type SamplingPresets = std::collections::HashMap<String, SamplingPresetValues>;
pub type ModelSamplingPresets = std::collections::HashMap<String, SamplingPresets>;

fn preset(
    temperature: f32,
    top_p: f32,
    top_k: usize,
    presence_penalty: f32,
) -> SamplingPresetValues {
    preset_rp(temperature, top_p, top_k, presence_penalty, 1.0)
}

/// То же, но с явным repetition_penalty. Нужен там, где карточка модели не
/// задаёт штраф, а без него генерация уходит в петлю: наблюдалось на Ornith
/// 1.5 — 15 000 токенов повтора после того, как модель заметила свою опечатку
/// и начала бесконечно себя перепроверять.
fn preset_rp(
    temperature: f32,
    top_p: f32,
    top_k: usize,
    presence_penalty: f32,
    repetition_penalty: f32,
) -> SamplingPresetValues {
    SamplingPresetValues {
        temperature,
        top_p,
        top_k,
        min_p: 0.0,
        presence_penalty,
        repetition_penalty,
    }
}

fn insert_modes(
    presets: &mut SamplingPresets,
    thinking: SamplingPresetValues,
    thinking_coding: SamplingPresetValues,
    instruct: SamplingPresetValues,
    instruct_reasoning: SamplingPresetValues,
) {
    presets.insert("thinking".into(), thinking);
    presets.insert("thinking-coding".into(), thinking_coding);
    presets.insert("instruct".into(), instruct);
    presets.insert("instruct-reasoning".into(), instruct_reasoning);
}

/// Стабильный ключ семейства для встроенных и пользовательских пресетов.
/// Квант, размер модели и оформление имени файла на выбор параметров не влияют.
pub fn sampling_family_for_model(model_name: &str) -> &'static str {
    let compact = model_name
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();

    if compact.contains("ornith15") {
        "ornith-1.5"
    } else if compact.contains("ornith") {
        "ornith"
    } else if compact.contains("gemma4") {
        "gemma-4"
    } else if compact.contains("qwen38") {
        "qwen-3.8"
    } else if compact.contains("qwen36") && (compact.contains("a3b") || compact.contains("moe")) {
        "qwen-3.6-moe"
    } else if compact.contains("qwen36") {
        "qwen-3.6"
    } else if compact.contains("qwen35") {
        "qwen-3.5"
    } else if compact.contains("qwen") {
        "qwen"
    } else {
        "generic"
    }
}

pub fn default_presets_for_model(model_name: &str) -> SamplingPresets {
    let mut m = SamplingPresets::new();
    match sampling_family_for_model(model_name) {
        "ornith-1.5" => {
            // Official Ornith 1.5 card: general 1.0/.95/20/presence 1.5;
            // precise coding .6/.95/20/presence 0.0; repetition_penalty 1.0.
            //
            // Штраф за повтор НЕ трогаем, хотя соблазн есть: пробовали 1.05
            // против зацикливания — сломало shell-команды. Штраф давит каждый
            // уже встреченный токен, а `$`, `{`, `}` в скрипте повторяются
            // десятки раз, и символ просто выпадает: `$NINEROUTER_KEY` стал
            // `NINEROUTER_KEY`, `'{"model":...}'` потерял скобки. От петель
            // защищает detect_loop в engine_batched, а не сэмплинг.
            let general = preset(1.0, 0.95, 20, 1.5);
            insert_modes(
                &mut m,
                general.clone(),
                preset(0.6, 0.95, 20, 0.0),
                general.clone(),
                general,
            );
        }
        "ornith" => {
            // Ornith 1.0 card recommends .6/.95/20 for normal serving.
            let general = preset(0.6, 0.95, 20, 0.0);
            insert_modes(
                &mut m,
                general.clone(),
                general.clone(),
                general.clone(),
                general,
            );
        }
        "gemma-4" => {
            // Official Gemma 4 generation_config has one sampling profile;
            // thinking is controlled by the chat template, not Qwen presets.
            let official = preset(1.0, 0.95, 64, 0.0);
            insert_modes(
                &mut m,
                official.clone(),
                official.clone(),
                official.clone(),
                official,
            );
        }
        "qwen-3.5" | "qwen-3.6-moe" => {
            // Qwen 3.5 and Qwen 3.6 MoE differ from dense 3.6/3.8:
            // general thinking uses presence 1.5.
            insert_modes(
                &mut m,
                preset(1.0, 0.95, 20, 1.5),
                preset(0.6, 0.95, 20, 0.0),
                preset(0.7, 0.80, 20, 1.5),
                preset(1.0, 0.95, 20, 1.5),
            );
        }
        "qwen-3.8" => {
            // Qwen 3.8 publishes thinking and instruct profiles only.
            let thinking = preset(1.0, 0.95, 20, 0.0);
            let instruct = preset(0.7, 0.80, 20, 1.5);
            insert_modes(
                &mut m,
                thinking.clone(),
                thinking,
                instruct.clone(),
                instruct,
            );
        }
        _ => {
            // Official Qwen 3.6 profiles and safe generic Qwen fallback.
            let instruct = preset(0.7, 0.80, 20, 1.5);
            insert_modes(
                &mut m,
                preset(1.0, 0.95, 20, 0.0),
                preset(0.6, 0.95, 20, 0.0),
                instruct.clone(),
                instruct,
            );
        }
    }
    m
}

pub fn default_presets() -> SamplingPresets {
    default_presets_for_model("")
}

/// PRESETS: legacy global override.
/// MODEL_PRESETS: {family: {mode: values}} — model-specific override.
pub fn presets_from_env(model_path: &Path) -> Result<SamplingPresets> {
    let model_name = model_path
        .file_name()
        .map(|n| n.to_string_lossy())
        .unwrap_or_default();
    let mut presets = default_presets_for_model(&model_name);
    if let Some(raw) = get_env_var("PRESETS") {
        let overrides: SamplingPresets =
            serde_json::from_str(&raw).map_err(|e| anyhow!("PRESETS: невалидный JSON: {e}"))?;
        presets.extend(overrides);
    }
    if let Some(raw) = get_env_var("MODEL_PRESETS") {
        let overrides: ModelSamplingPresets = serde_json::from_str(&raw)
            .map_err(|e| anyhow!("MODEL_PRESETS: невалидный JSON: {e}"))?;
        if let Some(model_overrides) = overrides.get(sampling_family_for_model(&model_name)) {
            presets.extend(model_overrides.clone());
        }
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
    /// Размещение маршрутизируемых экспертов (`MOE_EXPERTS`, FR-009):
    /// "vram" | "ram" | "auto" — та же переменная, что читает движок.
    pub moe_experts: String,
    pub prefix_cache_mib: usize,
    pub media_temp: PathBuf,
    pub sampling: SamplingDefaults,
    /// Явные env-переопределения поверх карточки модели и политика приоритета
    /// параметров из inference-запроса.
    pub sampling_policy: SamplingPolicy,
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
    /// Context overflow policy (`CTX_OVERFLOW`, default "error").
    pub ctx_overflow: String,
    /// Frequency penalty (`FREQUENCY_PENALTY`, default 0.0).
    pub frequency_penalty: f32,
    /// Max queue length (`MAX_QUEUE`, default 64).
    pub max_queue: usize,
    /// Request timeout in seconds (`REQ_TIMEOUT`, default 600).
    pub req_timeout: u64,
    /// Flash Attention (`FLASH_ATTN`, default 1 = включен).
    pub flash_attn: bool,
    /// Unsloth Studio backend URL (`STUDIO_URL`, default `http://127.0.0.1:8888`).
    pub studio_url: String,
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
        let api_keys_raw = get_env_var("API_KEYS").ok_or_else(|| {
            anyhow!("API_KEYS (или API_KEYS) обязателен: JSON-массив объектов key/name")
        })?;
        let api_keys = parse_api_keys(&api_keys_raw)?;
        let prefix_cache_mib = parse_env("PREFIX_CACHE_MIB", 0usize)?;
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
        let moe_experts = parse_moe_experts(std::env::var("MOE_EXPERTS").ok().as_deref())?;
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
            moe_experts,
            prefix_cache_mib,
            media_temp: get_env_var("MEDIA_TEMP")
                .map(PathBuf::from)
                .unwrap_or_else(|| std::env::temp_dir().join("yttri-media")),
            sampling: SamplingDefaults::from_env()?,
            sampling_policy: SamplingPolicy::from_env()?,
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
            ctx_overflow: get_env_var("CTX_OVERFLOW").unwrap_or_else(|| "error".into()),
            frequency_penalty: parse_env("FREQUENCY_PENALTY", 0.0f32)?,
            max_queue: parse_env("MAX_QUEUE", 64usize)?,
            req_timeout: parse_env("REQ_TIMEOUT", 600u64)?,
            flash_attn: parse_env("FLASH_ATTN", 1u8)? != 0,
            studio_url: get_env_var("STUDIO_URL").unwrap_or_else(|| "http://127.0.0.1:8888".into()),
        };
        cfg.apply_vram_plan()?;
        // Движок читает голое CTX, а сервер принимает ещё и CTX/YTTRI_CTX.
        // Без проброса окно страничного пула упиралось в собственный потолок
        // 32768: при CTX=131072 движок молча скользил окном вместо
        // длинного контекста. Кладём значение уже после apply_vram_plan — оно
        // может оказаться меньше запрошенного.
        std::env::set_var("CTX", cfg.ctx.to_string());
        Ok(cfg)
    }

    /// FR-002 + dynamic KV (vLLM-style): ctx НЕ режется под worst-case
    /// (только до native модели); общий KV-бюджет enforce'ится движком
    /// (admission + очередь + force-finish самого длинного слота).
    /// Отключается NO_VRAM_PLAN=1. На macOS/CPU — без лимита.
    fn apply_vram_plan(&mut self) -> Result<()> {
        if std::env::var_os("NO_VRAM_PLAN").is_some() {
            return Ok(());
        }
        let Some(total) = vram_plan::total_vram_mib() else {
            return Ok(());
        };
        let fp = vram_plan::footprint_from_gguf_with(&self.model, &self.moe_experts)?;
        let plan = vram_plan::compute_dynamic(total, &fp, self.ctx, self.slots, &self.moe_experts)?;
        eprintln!("{}", plan.report);
        self.ctx = plan.ctx;
        self.slots = plan.slots;
        self.kv_budget_mib = plan.kv_budget_mib;
        self.kv_per_tok_mib = plan.kv_per_tok_mib;
        Ok(())
    }
}

fn parse_api_keys(raw: &str) -> Result<Vec<ApiKey>> {
    let keys: Vec<ApiKey> =
        serde_json::from_str(raw).context("API_KEYS: ожидается JSON-массив объектов key/name")?;
    if keys.is_empty() {
        anyhow::bail!("API_KEYS: нужен хотя бы один ключ");
    }
    let mut names = std::collections::HashSet::new();
    let mut values = std::collections::HashSet::new();
    for entry in &keys {
        if entry.name.trim().is_empty() || entry.key.trim().is_empty() {
            anyhow::bail!("API_KEYS: key и name не могут быть пустыми");
        }
        if entry.name.trim() != entry.name || entry.key.trim() != entry.key {
            anyhow::bail!("API_KEYS: key и name не должны начинаться/заканчиваться пробелами");
        }
        if !names.insert(entry.name.as_str()) {
            anyhow::bail!("API_KEYS: повтор name {:?}", entry.name);
        }
        if !values.insert(entry.key.as_str()) {
            anyhow::bail!("API_KEYS: ключ {:?} указан повторно", entry.name);
        }
    }
    Ok(keys)
}

/// Разобрать MOE_EXPERTS (FR-009): vram|ram|auto, default "auto".
/// Мусорное значение — ошибка старта (fail-closed, FR-008).
pub fn parse_moe_experts(raw: Option<&str>) -> Result<String> {
    match raw {
        None | Some("") => Ok("auto".to_string()),
        Some(v @ ("vram" | "ram" | "auto")) => Ok(v.to_string()),
        Some(other) => Err(anyhow::anyhow!(
            "MOE_EXPERTS={other:?} не разобрать: ожидается vram|ram|auto (default auto)"
        )),
    }
}

/// Размещение из окружения для планера/диагностики (паникует на мусоре —
/// мусор ловится валидацией в Config::from_env раньше).
pub fn moe_placement_from_env() -> String {
    parse_moe_experts(std::env::var("MOE_EXPERTS").ok().as_deref())
        .expect("MOE_EXPERTS валидирован в Config::from_env")
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

/// Имя переменной без префикса. Единственная форма — сервер и движок читают
/// одно и то же имя.
fn get_env_var(name: &str) -> Option<String> {
    std::env::var(name).ok()
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

    fn assert_preset(
        presets: &SamplingPresets,
        name: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        presence_penalty: f32,
    ) {
        assert_preset_rp(
            presets,
            name,
            temperature,
            top_p,
            top_k,
            presence_penalty,
            1.0,
        )
    }

    /// Ornith 1.5 намеренно идёт со штрафом 1.05: карточка его не задаёт, а без
    /// него генерация зацикливалась (15 000 токенов повтора на самопроверке).
    #[allow(clippy::too_many_arguments)]
    fn assert_preset_rp(
        presets: &SamplingPresets,
        name: &str,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        presence_penalty: f32,
        repetition_penalty: f32,
    ) {
        let value = presets
            .get(name)
            .unwrap_or_else(|| panic!("missing preset {name}"));
        assert_eq!(value.temperature, temperature, "{name}.temperature");
        assert_eq!(value.top_p, top_p, "{name}.top_p");
        assert_eq!(value.top_k, top_k, "{name}.top_k");
        assert_eq!(value.min_p, 0.0, "{name}.min_p");
        assert_eq!(
            value.presence_penalty, presence_penalty,
            "{name}.presence_penalty"
        );
        assert_eq!(
            value.repetition_penalty, repetition_penalty,
            "{name}.repetition_penalty"
        );
    }

    #[test]
    fn sampling_family_ignores_quant_and_filename_separators() {
        assert_eq!(
            sampling_family_for_model("gemma-4-E4B-it-Q8_0.gguf"),
            "gemma-4"
        );
        assert_eq!(
            sampling_family_for_model("Ornith_1.5_9B_Q4_K_M.gguf"),
            "ornith-1.5"
        );
        assert_eq!(
            sampling_family_for_model("Qwen3.8-27B-UD-Q4_K_XL.gguf"),
            "qwen-3.8"
        );
        assert_eq!(
            sampling_family_for_model("Qwen3.5-4B-Q4_K_M.gguf"),
            "qwen-3.5"
        );
        assert_eq!(
            sampling_family_for_model("Qwen3.6-35B-A3B-Q4_K_M.gguf"),
            "qwen-3.6-moe"
        );
    }

    #[test]
    fn model_sampling_presets_match_official_recommendations() {
        let gemma = default_presets_for_model("gemma-4-E4B-it-Q8_0.gguf");
        for mode in [
            "thinking",
            "thinking-coding",
            "instruct",
            "instruct-reasoning",
        ] {
            assert_preset(&gemma, mode, 1.0, 0.95, 64, 0.0);
        }

        let ornith = default_presets_for_model("Ornith-1.5-9B-Q4_K_M.gguf");
        // repetition_penalty строго 1.0 как в карточке: 1.05 ломал shell-код.
        assert_preset(&ornith, "thinking", 1.0, 0.95, 20, 1.5);
        assert_preset(&ornith, "thinking-coding", 0.6, 0.95, 20, 0.0);
        assert_preset(&ornith, "instruct", 1.0, 0.95, 20, 1.5);
        assert_preset(&ornith, "instruct-reasoning", 1.0, 0.95, 20, 1.5);

        let qwen38 = default_presets_for_model("Qwen3.8-27B-Q8_0.gguf");
        assert_preset(&qwen38, "thinking", 1.0, 0.95, 20, 0.0);
        assert_preset(&qwen38, "thinking-coding", 1.0, 0.95, 20, 0.0);
        assert_preset(&qwen38, "instruct", 0.7, 0.80, 20, 1.5);
        assert_preset(&qwen38, "instruct-reasoning", 0.7, 0.80, 20, 1.5);

        let qwen36 = default_presets_for_model("Qwen3.6-27B-Q8_0.gguf");
        assert_preset(&qwen36, "thinking", 1.0, 0.95, 20, 0.0);
        assert_preset(&qwen36, "thinking-coding", 0.6, 0.95, 20, 0.0);
        assert_preset(&qwen36, "instruct", 0.7, 0.80, 20, 1.5);
        assert_preset(&qwen36, "instruct-reasoning", 0.7, 0.80, 20, 1.5);

        let qwen35 = default_presets_for_model("Qwen3.5-4B-Q4_K_M.gguf");
        assert_preset(&qwen35, "thinking", 1.0, 0.95, 20, 1.5);

        let qwen36_moe = default_presets_for_model("Qwen3.6-35B-A3B-Q4_K_M.gguf");
        assert_preset(&qwen36_moe, "thinking", 1.0, 0.95, 20, 1.5);
        assert_preset(&qwen36_moe, "thinking-coding", 0.6, 0.95, 20, 0.0);
    }

    // Тесты гоняют env процесса — сериализуем вручную через один тест.
    #[test]
    fn from_env_defaults_and_required_keys() {
        // Чистим окружение запуска: иначе API_KEYS оттуда отменяет проверку
        // «без ключей старт запрещён».
        for name in [
            "API_KEYS",
            "PROFILE",
            "MODEL",
            "HOST",
            "PORT",
            "CTX",
            "SLOTS",
            "MEDIA_TEMP",
            "PREFIX_CACHE_MIB",
        ] {
            env::remove_var(name);
        }
        env::remove_var("API_KEYS");
        assert!(Config::from_env().is_err(), "без API keys старт запрещён");

        env::set_var(
            "API_KEYS",
            r#"[{"key":"k1","name":"primary"},{"key":"k2","name":"backup"}]"#,
        );
        env::remove_var("PROFILE");
        env::remove_var("MODEL");
        env::remove_var("HOST");
        env::remove_var("PORT");
        env::remove_var("CTX");
        env::remove_var("SLOTS");
        env::remove_var("MEDIA_TEMP");
        let c = Config::from_env().unwrap();
        assert!(c.profile.is_none());
        assert!(c.resolved_profile.is_none());
        assert_eq!(c.api_keys.len(), 2);
        assert_eq!(c.api_keys[0].name, "primary");
        assert_eq!(c.model, PathBuf::from("models/qwen36-27b-q2_k_xl.gguf"));
        assert_eq!(c.host, "0.0.0.0");
        assert_eq!(c.port, 18099);
        assert_eq!(c.ctx, 131072);
        assert_eq!(c.slots, 4);

        env::set_var("PORT", "9000");
        env::set_var("CTX", "4096");
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
            env::set_var("API_KEYS", invalid);
            assert!(Config::from_env().is_err(), "{invalid}");
        }
        env::set_var("API_KEYS", r#"[{"key":"k1","name":"primary"}]"#);
        env::set_var("PORT", "x");
        assert!(Config::from_env().is_err());
        env::set_var("PORT", "8080");
        env::set_var("PREFIX_CACHE_MIB", "1");
        let c = Config::from_env().unwrap();
        assert_eq!(c.prefix_cache_mib, 1);
        env::remove_var("PREFIX_CACHE_MIB");

        let path = std::env::temp_dir().join(format!("qwen36-env-{}.tmp", std::process::id()));
        std::fs::write(
            &path,
            "# comment\nAPI_KEYS='[{\"key\":\"file-key\",\"name\":\"file\"}]'\nPORT=9001\n",
        )
        .unwrap();
        env::remove_var("API_KEYS");
        env::set_var("PORT", "9002");
        load_env_file(&path).unwrap();
        assert_eq!(env::var("PORT").unwrap(), "9002");
        assert_eq!(
            parse_api_keys(&env::var("API_KEYS").unwrap()).unwrap()[0].name,
            "file"
        );
        env::remove_var("API_KEYS");
        env::remove_var("PORT");

        env::set_var("ENV_FILE", &path);
        env::set_var("NO_VRAM_PLAN", "1");
        let c = Config::load().unwrap();
        assert_eq!(c.api_keys[0].name, "file");
        assert_eq!(c.port, 9001);
        env::remove_var("ENV_FILE");
        env::remove_var("NO_VRAM_PLAN");
        env::remove_var("API_KEYS");
        env::remove_var("PORT");
        std::fs::remove_file(path).unwrap();
    }
}
