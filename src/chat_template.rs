//! Рендер официального Jinja chat_template из GGUF (как llama.cpp/vLLM).
//!
//! Модель обучена на конкретном формате (Qwen3.8 — Hermes <function=...>,
//! Qwen3.5/3.6 — JSON). Хардкод ChatML расходится с обучением → мусорные теги
//! вида `<tool_1>`. Рендерим родной шаблон из `tokenizer.chat_template`.
//!
//! minijinja-препроцессинг: Python-стиль `x.startswith('y')` → Jinja-тест
//! `x is startswith('y')` (тесты регистрируем сами). `raise_exception` —
//! функция-ошибка, как в llama.cpp.

use anyhow::{anyhow, Result};
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{json, Value};
use std::path::Path;

pub struct ChatTemplate {
    env: Environment<'static>,
}

fn preprocess(tpl: &str) -> String {
    tpl.replace(".startswith(", " is startswith(")
        .replace(".endswith(", " is endswith(")
}

impl ChatTemplate {
    /// Прочитать `tokenizer.chat_template` из GGUF-метаданных. None — модель
    /// без шаблона (тогда fallback на встроенный ChatML-билдер).
    pub fn from_gguf(path: &Path) -> Option<Self> {
        let mut f = std::fs::File::open(path).ok()?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut f).ok()?;
        let tpl = match content.metadata.get("tokenizer.chat_template") {
            Some(candle_core::quantized::gguf_file::Value::String(s)) => s.clone(),
            _ => return None,
        };
        let mut env = Environment::new();
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_test("startswith", |v: String, prefix: String| v.starts_with(&prefix));
        env.add_test("endswith", |v: String, suffix: String| v.ends_with(&suffix));
        // ponytail: leak источника шаблона — Environment<'static> требует 'static
        // источник. Шаблонов несколько штук за жизнь процесса, утечка ~10KB каждый.
        let src: &'static str = Box::leak(preprocess(&tpl).into_boxed_str());
        env.add_template("chat", src).ok()?;
        Some(Self { env })
    }

    /// Отрендерить промпт. messages — JSON-массив {role, content, tool_calls?};
    /// template сам ставит thinking-блок по enable_thinking и формат tool calls.
    pub fn render(
        &self,
        messages: &[crate::engine_types::ChatMessage],
        tools: Option<&Value>,
        enable_thinking: bool,
    ) -> Result<String> {
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| {
                let mut v = json!({
                    "role": m.role,
                    "content": m.text_content(),
                });
                if !m.tool_calls.is_empty() {
                    v["tool_calls"] = Value::Array(m.tool_calls.clone());
                }
                v
            })
            .collect();
        let mut ctx = json!({
            "messages": msgs,
            "add_generation_prompt": true,
            "enable_thinking": enable_thinking,
            "add_vision_id": false,
        });
        if let Some(t) = tools {
            if t.as_array().map(|a| !a.is_empty()).unwrap_or(false) {
                ctx["tools"] = t.clone();
            }
        }
        self.env
            .get_template("chat")
            .map_err(|e| anyhow!("chat template: {e}"))?
            .render(&ctx)
            .map_err(|e| anyhow!("chat template render: {e}"))
    }
}
