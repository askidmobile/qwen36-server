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
//! Имена переменных — без префикса: `CTX`, `SLOTS`, `MODEL`. Одно и то же
//! имя читают обе стороны, сервер и движок, поэтому флаг пишет ровно одно
//! имя и зеркалирование не нужно.

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

    /// Потолок окна внимания, токенов (умолчание 81920).
    ///
    /// Отдельно от --ctx: `--ctx` объявляет контекст клиенту, а это —
    /// сколько движок реально держит в окне. Если потолок меньше,
    /// префилл падает на границе: «narrow invalid args start + len > dim_len».
    #[arg(long, value_name = "N")]
    pub context_limit: Option<usize>,

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
        // Одно имя, без префикса: движок читает то же самое.
        let set = |name: &str, value: String| std::env::set_var(name, &value);

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
        if let Some(v) = &self.api_key {
            set("API_KEYS", api_keys_json(v));
        }
        if let Some(v) = self.ctx {
            set("CTX", v.to_string());
        }
        if let Some(v) = self.context_limit {
            set("CONTEXT_LIMIT", v.to_string());
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

/// Короткая форма `--api-key` разворачивается в тот же JSON, что и
/// `--api-keys`: иначе в config.rs пришлось бы держать второй путь разбора.
fn api_keys_json(key: &str) -> String {
    serde_json::json!([{ "key": key, "name": "default" }]).to_string()
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
        // Проверяем чистую функцию, а не переменную окружения: тесты крейта
        // делят окружение процесса, и config-тест чистит API_KEYS параллельно.
        let parsed: serde_json::Value = serde_json::from_str(&api_keys_json("secret-1")).unwrap();
        assert_eq!(parsed[0]["key"], "secret-1");
        assert_eq!(parsed[0]["name"], "default");
    }

    #[test]
    fn flag_writes_clean_name_only() {
        // Флаг пишет ровно одно имя — то же, что читает движок.
        //
        // Тесты крейта делят окружение процесса и бегут параллельно, поэтому
        // проверяем на имени, которого не касается ни один соседний тест.
        let cli = Cli {
            kv_pool: Some("f16".into()),
            ..Default::default()
        };
        cli.apply_to_env();
        assert_eq!(std::env::var("KV_POOL_Q8").unwrap(), "0");
    }
}
