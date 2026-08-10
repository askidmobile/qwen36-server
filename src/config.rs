//! Конфиг из env `QWEN36_*` (docs/engine-api.md §Config).

use anyhow::{anyhow, Result};
use std::path::PathBuf;

use crate::vram_plan;

#[derive(Debug, Clone)]
pub struct Config {
    /// Путь к GGUF (`QWEN36_MODEL`).
    pub model: PathBuf,
    /// Host (`QWEN36_HOST`, default 0.0.0.0).
    pub host: String,
    /// Port (`QWEN36_PORT`, default 8080).
    pub port: u16,
    /// Master API key (`QWEN36_API_KEY`, обязателен).
    pub api_key: String,
    /// Контекст (`QWEN36_CTX`, default 81920).
    pub ctx: usize,
    /// Слоты (`QWEN36_SLOTS`, default 4).
    pub slots: usize,
}

impl Config {
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("QWEN36_API_KEY")
            .ok()
            .filter(|k| !k.is_empty())
            .ok_or_else(|| anyhow!("QWEN36_API_KEY обязателен (без него сервер не стартует)"))?;
        let mut cfg = Self {
            model: std::env::var("QWEN36_MODEL")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("models/qwen36-27b-q2_k_xl.gguf")),
            host: std::env::var("QWEN36_HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            port: parse_env("QWEN36_PORT", 8080u16)?,
            api_key,
            ctx: parse_env("QWEN36_CTX", 81920usize)?,
            slots: parse_env("QWEN36_SLOTS", 4usize)?,
        };
        cfg.apply_vram_plan()?;
        Ok(cfg)
    }

    /// FR-002: подгонка ctx/slots под VRAM карты (только CUDA-машины;
    /// на macOS/CPU total_vram_mib = None → планер не применяется).
    /// Отключается QWEN36_NO_VRAM_PLAN=1.
    fn apply_vram_plan(&mut self) -> Result<()> {
        if std::env::var_os("QWEN36_NO_VRAM_PLAN").is_some() {
            return Ok(());
        }
        let Some(total) = vram_plan::total_vram_mib() else {
            return Ok(());
        };
        let fp = vram_plan::footprint_from_gguf(&self.model)?;
        let plan = vram_plan::compute(total, &fp, self.ctx, self.slots)?;
        eprintln!("{}", plan.report);
        self.ctx = plan.ctx;
        self.slots = plan.slots;
        Ok(())
    }
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
    fn from_env_defaults_and_required_key() {
        env::remove_var("QWEN36_API_KEY");
        assert!(Config::from_env().is_err(), "без API key старт запрещён");

        env::set_var("QWEN36_API_KEY", "k");
        env::remove_var("QWEN36_MODEL");
        env::remove_var("QWEN36_HOST");
        env::remove_var("QWEN36_PORT");
        env::remove_var("QWEN36_CTX");
        env::remove_var("QWEN36_SLOTS");
        let c = Config::from_env().unwrap();
        assert_eq!(c.api_key, "k");
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

        env::set_var("QWEN36_PORT", "x");
        assert!(Config::from_env().is_err());
    }
}
