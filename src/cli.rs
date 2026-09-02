//! Разбор командной строки `yforge` и мостик к движку.
//!
//! Три источника настроек, по убыванию приоритета (как у llama.cpp):
//! 1. флаги командной строки — `--ctx 131072`;
//! 2. переменные окружения процесса — `CTX=131072 yforge`;
//! 3. env-файл — `yforge --env prod.env`.
//!
//! Флаг кладёт значение в переменную окружения ДО чтения env-файла, а файл
//! пишет только незанятые имена, поэтому порядок соблюдается сам собой.
//!
//! Имена без префикса: `CTX`, а не `QWEN36_CTX`. Серверный `get_env_var`
//! понимает чистые имена сам, а вот движок в форке читает только `QWEN36_*` —
//! для него имена зеркалятся в [`mirror_engine_vars`] после чтения файла.

use clap::Parser;
use std::path::PathBuf;

/// Сервер локального инференса GGUF-моделей (Qwen 3.5/3.6/3.8, Ornith 1.x).
///
/// Параметры можно задать флагами, переменными окружения или env-файлом.
/// Полный список переменных — в README.md.
#[derive(Parser, Debug, Default)]
#[command(
    name = "yforge",
    version,
    about = "Yttri Forge — сервер локального инференса GGUF (OpenAI/Anthropic API + WebUI)",
    long_about = None,
    after_help = "ПРИМЕРЫ:\n  \
        yforge --model model.gguf --ctx 131072 --slots 1\n  \
        yforge --env prod.env\n  \
        yforge --env prod.env --ctx 65536      # флаг переопределяет файл\n  \
        CTX=65536 yforge --env prod.env        # окружение переопределяет файл\n\n\
        Логи идут в stdout/stderr — перенаправляйте обычными средствами оболочки."
)]
pub struct Cli {
    // ── Источник настроек ────────────────────────────────────────────────
    /// Env-файл с параметрами (NAME=value). Значения из него применяются
    /// последними: флаги и окружение процесса имеют приоритет.
    #[arg(long, value_name = "FILE")]
    pub env: Option<PathBuf>,

    // ── Модель ───────────────────────────────────────────────────────────
    /// Путь к GGUF-файлу модели.
    #[arg(short = 'm', long, value_name = "PATH")]
    pub model: Option<String>,

    /// Корневой каталог для сканирования моделей (горячая смена через API).
    #[arg(long, value_name = "DIR")]
    pub models_dir: Option<String>,

    /// Профиль модели (manifest.json), заменяет --model.
    #[arg(long, value_name = "PATH")]
    pub profile: Option<String>,

    // ── Сеть и доступ ────────────────────────────────────────────────────
    /// Адрес прослушивания.
    #[arg(long, value_name = "IP")]
    pub host: Option<String>,

    /// Порт прослушивания.
    #[arg(short = 'p', long, value_name = "PORT")]
    pub port: Option<u16>,

    /// API-ключи: JSON-массив `[{"key":"...","name":"..."}]`.
    #[arg(long, value_name = "JSON")]
    pub api_keys: Option<String>,

    /// Один API-ключ (короткая форма вместо JSON в --api-keys).
    #[arg(long, value_name = "KEY")]
    pub api_key: Option<String>,

    // ── Контекст и слоты ─────────────────────────────────────────────────
    /// Размер контекстного окна в токенах.
    #[arg(short = 'c', long, value_name = "N")]
    pub ctx: Option<usize>,

    /// Число параллельных слотов генерации.
    #[arg(short = 's', long, value_name = "N")]
    pub slots: Option<usize>,

    /// Потолок генерации по умолчанию, токенов.
    #[arg(short = 'n', long, value_name = "N")]
    pub max_tokens: Option<usize>,

    /// Таймаут запроса, секунд.
    #[arg(long, value_name = "SEC")]
    pub req_timeout: Option<u64>,

    /// Длина очереди запросов.
    #[arg(long, value_name = "N")]
    pub max_queue: Option<usize>,

    // ── Память и вычисления ──────────────────────────────────────────────
    /// Слоёв на GPU (999 = все).
    #[arg(long = "gpu-layers", visible_alias = "ngl", value_name = "N")]
    pub gpu_layers: Option<usize>,

    /// Тип страничного пула KV: q8 (вдвое меньше VRAM) или f16.
    ///
    /// Постоянный пул выделяется на всё окно контекста сразу при старте,
    /// поэтому это самая крупная статья VRAM после весов.
    #[arg(long, value_name = "TYPE", value_parser = ["q8", "f16"])]
    pub kv_pool: Option<String>,

    /// Тип временного batched-KV: q8 или q8_f16.
    #[arg(long, value_name = "TYPE", value_parser = ["q8", "q8_f16"])]
    pub kv_cache_type: Option<String>,

    /// Резерв VRAM под транзиенты, МиБ (не отдаётся пулу KV).
    #[arg(long, value_name = "MIB")]
    pub vram_headroom: Option<usize>,

    /// Кеш префикса промпта, МиБ системной памяти (0 = выключен).
    #[arg(long, value_name = "MIB")]
    pub prefix_cache: Option<usize>,

    /// CUDA-графы: 1 — включить (KV пишется прямо в пул, без двойного расхода).
    #[arg(long, value_name = "0|1")]
    pub cuda_graphs: Option<u8>,

    /// Flash Attention.
    #[arg(long = "flash-attn", visible_alias = "fa", value_name = "0|1")]
    pub flash_attn: Option<u8>,

    /// Потоков CPU.
    #[arg(short = 't', long, value_name = "N")]
    pub threads: Option<usize>,

    /// Размер батча префилла, токенов.
    #[arg(short = 'b', long, value_name = "N")]
    pub batch_size: Option<usize>,

    /// Держать все веса на GPU.
    #[arg(long, value_name = "0|1")]
    pub gpu_only: Option<u8>,

    // ── Спекулятивное декодирование (MTP) ────────────────────────────────
    /// Включить MTP (спекулятивное декодирование).
    #[arg(long, value_name = "0|1")]
    pub mtp: Option<u8>,

    /// Путь к GGUF головы MTP.
    #[arg(long, value_name = "PATH")]
    pub mtp_path: Option<String>,

    /// Шортлист словаря черновика MTP (файл с id токенов).
    #[arg(long, value_name = "FILE")]
    pub mtp_shortlist: Option<String>,

    /// Однопроходная проверка спекуляции.
    #[arg(long, value_name = "0|1")]
    pub verify_onepass: Option<u8>,

    // ── Сэмплинг ─────────────────────────────────────────────────────────
    /// Температура сэмплинга.
    #[arg(long = "temp", value_name = "F")]
    pub temperature: Option<f32>,

    /// Top-p (nucleus).
    #[arg(long, value_name = "F")]
    pub top_p: Option<f32>,

    /// Top-k.
    #[arg(long, value_name = "N")]
    pub top_k: Option<usize>,

    /// Min-p.
    #[arg(long, value_name = "F")]
    pub min_p: Option<f32>,

    /// Seed (0 = случайный).
    #[arg(long, value_name = "N")]
    pub seed: Option<u64>,

    /// Режим рассуждений по умолчанию.
    #[arg(long, value_name = "true|false")]
    pub thinking: Option<bool>,

    /// Держать серверную политику сэмплинга (1) или разрешить клиенту (0).
    #[arg(long, value_name = "0|1")]
    pub sampling_lock: Option<u8>,

    // ── Прочее ───────────────────────────────────────────────────────────
    /// URL бэкенда Unsloth Studio для проксирования WebUI.
    #[arg(long, value_name = "URL")]
    pub studio_url: Option<String>,

    /// Показать итоговую конфигурацию и выйти, не загружая модель.
    #[arg(long)]
    pub dry_run: bool,
}

impl Cli {
    /// Разложить флаги по переменным окружения.
    ///
    /// Вызывать ДО чтения env-файла: файл пишет только незанятые имена,
    /// поэтому уже выставленный флаг переживёт загрузку файла. Значения,
    /// пришедшие из окружения процесса, тоже не перетираются — флаг их
    /// перекрывает намеренно, это его приоритет.
    pub fn apply_to_env(&self) {
        // Флаг — верхний приоритет, поэтому он затирает и чистое имя, и
        // префиксную форму. Иначе `QWEN36_KV_POOL_Q8=1` из env-файла
        // переживал бы `--kv-pool f16`: зеркало не трогает уже занятое
        // `QWEN36_*`, и флаг молча терялся.
        let mut set = |name: &str, value: String| {
            std::env::set_var(name, &value);
            if ENGINE_VARS.contains(&name) {
                std::env::set_var(format!("QWEN36_{name}"), &value);
            }
        };

        if let Some(v) = &self.env {
            set("ENV_FILE", v.to_string_lossy().into_owned());
        }
        if let Some(v) = &self.model {
            set("MODEL", v.clone());
        }
        if let Some(v) = &self.models_dir {
            set("MODELS_DIR", v.clone());
        }
        if let Some(v) = &self.profile {
            set("PROFILE", v.clone());
        }
        if let Some(v) = &self.host {
            set("HOST", v.clone());
        }
        if let Some(v) = self.port {
            set("PORT", v.to_string());
        }
        if let Some(v) = &self.api_keys {
            set("API_KEYS", v.clone());
        }
        // Короткая форма ключа разворачивается в тот же JSON, что и --api-keys,
        // иначе пришлось бы городить второй путь разбора в config.rs.
        if let Some(v) = &self.api_key {
            let json = serde_json::json!([{ "key": v, "name": "default" }]);
            set("API_KEYS", json.to_string());
        }
        if let Some(v) = self.ctx {
            set("CTX", v.to_string());
        }
        if let Some(v) = self.slots {
            set("SLOTS", v.to_string());
        }
        if let Some(v) = self.max_tokens {
            set("MAX_TOKENS", v.to_string());
        }
        if let Some(v) = self.req_timeout {
            set("REQ_TIMEOUT", v.to_string());
        }
        if let Some(v) = self.max_queue {
            set("MAX_QUEUE", v.to_string());
        }
        if let Some(v) = self.gpu_layers {
            set("GPU_LAYERS", v.to_string());
        }
        // Пул KV задаётся типом, а движок читает булев флаг: f16 — это
        // «выключено», а не отдельный режим.
        if let Some(v) = &self.kv_pool {
            set("KV_POOL_Q8", if v == "q8" { "1" } else { "0" }.to_string());
        }
        if let Some(v) = &self.kv_cache_type {
            set("KV_CACHE_DTYPE", v.clone());
            set("KV_CACHE_TYPE", v.clone());
        }
        if let Some(v) = self.vram_headroom {
            set("VRAM_HEADROOM_MIB", v.to_string());
        }
        if let Some(v) = self.prefix_cache {
            set("PREFIX_CACHE_MIB", v.to_string());
        }
        if let Some(v) = self.cuda_graphs {
            set("CUDA_GRAPHS", v.to_string());
        }
        if let Some(v) = self.flash_attn {
            set("FLASH_ATTN", v.to_string());
        }
        if let Some(v) = self.threads {
            set("THREADS", v.to_string());
        }
        if let Some(v) = self.batch_size {
            set("BATCH_SIZE", v.to_string());
        }
        if let Some(v) = self.gpu_only {
            set("GPU_ONLY", v.to_string());
        }
        if let Some(v) = self.mtp {
            set("MTP", v.to_string());
        }
        if let Some(v) = &self.mtp_path {
            set("MTP_PATH", v.clone());
        }
        if let Some(v) = &self.mtp_shortlist {
            set("MTP_VOCAB_SHORTLIST", v.clone());
        }
        if let Some(v) = self.verify_onepass {
            set("VERIFY_ONEPASS", v.to_string());
        }
        if let Some(v) = self.temperature {
            set("TEMPERATURE", v.to_string());
        }
        if let Some(v) = self.top_p {
            set("TOP_P", v.to_string());
        }
        if let Some(v) = self.top_k {
            set("TOP_K", v.to_string());
        }
        if let Some(v) = self.min_p {
            set("MIN_P", v.to_string());
        }
        if let Some(v) = self.seed {
            set("SEED", v.to_string());
        }
        if let Some(v) = self.thinking {
            set("THINKING", v.to_string());
        }
        if let Some(v) = self.sampling_lock {
            set("SAMPLING_LOCK", v.to_string());
        }
        if let Some(v) = &self.studio_url {
            set("STUDIO_URL", v.clone());
        }
    }
}

/// Переменные, которые движок (yttri-forge) читает ТОЛЬКО с префиксом
/// `QWEN36_`. Сервер принимает их без префикса, поэтому имена зеркалятся.
///
/// Список явный, а не «скопировать всё окружение»: так видно, что именно
/// поддержано, и `QWEN36_PATH` не появляется рядом с системным `PATH`.
const ENGINE_VARS: &[&str] = &[
    // Модель, контекст, слоты
    "MODEL",
    "CTX",
    "SLOTS",
    "PORT",
    "TRACE",
    "REQ_TIMEOUT",
    "GPU_LAYERS",
    "GPU_ONLY",
    "MOE_BACKEND",
    "PREFILL_CHUNK",
    // KV-кеш и память
    "KV_POOL_Q8",
    "KV_CACHE_DTYPE",
    "KV_MIRROR_MIB",
    "KV_MIRROR_TOKENS",
    "KV_MIRROR_PREPARE",
    "KV_SCRATCH_TOKENS",
    "KV_Q8_ROUNDTRIP",
    "KEEP_SINGLE_KV",
    "VRAM_HEADROOM_MIB",
    "MEMPOOL_THRESHOLD_MIB",
    "MEMPOOL_RETAIN_MIB",
    "NO_MEMPOOL_RETAIN",
    "PREFIX_CACHE_MAX_TOKENS",
    "DEQUANT_CACHE",
    "EMB_GPU",
    // CUDA-графы
    "CUDA_GRAPHS",
    "GRAPH_MAX_B",
    "GRAPH_FAIL_CAP",
    "GRAPH_BLOCKS",
    "GRAPH_WINDOW",
    "GRAPH_MAX_LAYERS",
    "PGRAPH",
    "PGRAPH_LRU",
    "PGRAPH_MIN_T",
    "PGRAPH_Q8KV",
    "DGRAPH_LRU",
    // MTP
    "MTP",
    "MTP_ADAPTIVE",
    "MTP_GRAPH",
    "MTP_GRAPH_CHECK",
    "MTP_GRAPH_RECAPTURE",
    "MTP_PREDICT",
    "MTP_SHORTLIST_CHECK",
    "MTP_TIMING",
    "MTP_VOCAB_SHORTLIST",
    "MTP_VOCAB_TOP",
    "MTP_WIDTH",
    "MTP_DRAFT_LOG",
    "VERIFY_ONEPASS",
    "VERIFY_FUSED",
    // Ядра и вычисления
    "DISABLE_FLASH_PREFILL",
    "DISABLE_FUSED_PREFILL",
    "DISABLE_YTF16",
    "F16_FAST_ACC",
    "FA_GQA",
    "FA_SPLITS",
    "FORCE_DMMV",
    "FORCE_MMQ",
    "MMVQ_HOISTED",
    "ENABLE_MOE_GROUPED",
    "ENABLE_SPLITK_DECODE",
    "NO_FUSED_IN_PROJ",
    "DELTA_KERNEL",
    "DELTA_DECODE",
    "DELTA_V2",
    "DELTA_WARPS",
    "CHUNK_KERNEL",
    "YTF16_F16",
    "YTF16_MASK",
    "YTF16_AUDIT",
    // Диагностика
    "GPROF",
    "PHASE_PROF",
    "FA_DEBUG",
    "BISECT_HIDDEN",
    "DEBUG_ALLOC_MB",
    "NO_EVENT_TRACKING",
];

/// Скопировать чистые имена в `QWEN36_*` для движка.
///
/// Вызывать ПОСЛЕ загрузки env-файла: к этому моменту значения из всех трёх
/// источников уже лежат в окружении. Существующее `QWEN36_*` не трогаем —
/// старые конфиги продолжают работать как раньше.
///
/// Возвращает число проброшенных имён (для строки в логе).
pub fn mirror_engine_vars() -> usize {
    let mut n = 0;
    for name in ENGINE_VARS {
        let prefixed = format!("QWEN36_{name}");
        if std::env::var_os(&prefixed).is_some() {
            continue; // явный QWEN36_* имеет приоритет — обратная совместимость
        }
        if let Some(value) = std::env::var_os(name) {
            std::env::set_var(&prefixed, value);
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn kv_pool_f16_disables_q8_pool() {
        let cli = Cli {
            kv_pool: Some("f16".into()),
            ..Default::default()
        };
        cli.apply_to_env();
        assert_eq!(std::env::var("KV_POOL_Q8").unwrap(), "0");

        let cli = Cli {
            kv_pool: Some("q8".into()),
            ..Default::default()
        };
        cli.apply_to_env();
        assert_eq!(std::env::var("KV_POOL_Q8").unwrap(), "1");
    }

    #[test]
    fn api_key_shorthand_expands_to_json() {
        let cli = Cli {
            api_key: Some("secret-1".into()),
            ..Default::default()
        };
        cli.apply_to_env();
        let raw = std::env::var("API_KEYS").unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(parsed[0]["key"], "secret-1");
    }

    #[test]
    fn flag_overrides_prefixed_value_from_file() {
        // Флаг должен побеждать QWEN36_* из env-файла, иначе --kv-pool f16
        // не отключает пул, заданный как QWEN36_KV_POOL_Q8=1.
        std::env::set_var("QWEN36_KV_POOL_Q8", "1");
        let cli = Cli {
            kv_pool: Some("f16".into()),
            ..Default::default()
        };
        cli.apply_to_env();
        mirror_engine_vars();
        assert_eq!(std::env::var("QWEN36_KV_POOL_Q8").unwrap(), "0");
        std::env::remove_var("QWEN36_KV_POOL_Q8");
        std::env::remove_var("KV_POOL_Q8");
    }

    #[test]
    fn mirror_keeps_explicit_prefixed_value() {
        // Явный QWEN36_* побеждает: старые .env не должны менять поведение.
        std::env::set_var("QWEN36_DELTA_WARPS", "8");
        std::env::set_var("DELTA_WARPS", "4");
        mirror_engine_vars();
        assert_eq!(std::env::var("QWEN36_DELTA_WARPS").unwrap(), "8");

        std::env::remove_var("QWEN36_FA_SPLITS");
        std::env::set_var("FA_SPLITS", "2");
        mirror_engine_vars();
        assert_eq!(std::env::var("QWEN36_FA_SPLITS").unwrap(), "2");
    }
}
