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
use minijinja::value::ValueKind;
use minijinja::{Environment, Error, ErrorKind};
use serde_json::{json, Value};
use std::path::Path;

pub struct ChatTemplate {
    env: Environment<'static>,
    bos_token: String,
    eos_token: String,
}

fn preprocess(tpl: &str) -> String {
    tpl.replace(".startswith(", " is startswith(")
        .replace(".endswith(", " is endswith(")
}

fn python_mapping_get(
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> Result<minijinja::Value, Error> {
    if value.kind() != ValueKind::Map || method != "get" || !(1..=2).contains(&args.len()) {
        return Err(Error::from(ErrorKind::UnknownMethod));
    }
    let found = value.get_item(&args[0])?;
    if found.is_undefined() && args.len() == 2 {
        Ok(args[1].clone())
    } else {
        Ok(found)
    }
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
        let token = |id_key: &str| -> String {
            let id = content
                .metadata
                .get(id_key)
                .and_then(|value| value.to_u32().ok())
                .unwrap_or(u32::MAX) as usize;
            content
                .metadata
                .get("tokenizer.ggml.tokens")
                .and_then(|value| value.to_vec().ok())
                .and_then(|tokens| tokens.get(id))
                .and_then(|value| value.to_string().ok())
                .cloned()
                .unwrap_or_default()
        };
        let bos_token = token("tokenizer.ggml.bos_token_id");
        let eos_token = token("tokenizer.ggml.eos_token_id");
        let mut env = Environment::new();
        env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
        env.add_function("raise_exception", |msg: String| -> Result<String, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_test("startswith", |v: String, prefix: String| {
            v.starts_with(&prefix)
        });
        env.add_test("endswith", |v: String, suffix: String| v.ends_with(&suffix));
        env.set_unknown_method_callback(|_, value, method, args| {
            python_mapping_get(value, method, args)
        });
        // ponytail: leak источника шаблона — Environment<'static> требует 'static
        // источник. Шаблонов несколько штук за жизнь процесса, утечка ~10KB каждый.
        let src: &'static str = Box::leak(preprocess(&tpl).into_boxed_str());
        env.add_template("chat", src).ok()?;
        Some(Self {
            env,
            bos_token,
            eos_token,
        })
    }

    /// Отрендерить промпт. messages — JSON-массив {role, content, tool_calls?};
    /// template сам ставит thinking-блок по enable_thinking и формат tool calls.
    /// reasoning_effort: None | "low" | "medium" | "xhigh" — шаблонные уровни.
    pub fn render(
        &self,
        messages: &[crate::engine_types::ChatMessage],
        tools: Option<&Value>,
        enable_thinking: bool,
        reasoning_effort: Option<&str>,
    ) -> Result<String> {
        // Шаблон знает только low/medium/xhigh; high = xhigh (дефолт шаблона).
        let reasoning_effort = reasoning_effort.map(|e| if e == "high" { "xhigh" } else { e });
        let msgs: Vec<Value> = messages
            .iter()
            .map(|m| {
                let mut v = json!({
                    "role": m.role,
                    // Для Jinja-шаблона: content — ТОЛЬКО текст (без thinking).
                    // Шаблон Ornith-1.5/Qwen3.8 сам встраивает reasoning_content
                    // из отдельного поля. Раньше мы клали его в content → двойной
                    // thinking-блок ломал модель (agent-loop зацикливался).
                    "content": m.text_content(),
                });
                if !m.tool_calls.is_empty() {
                    v["tool_calls"] = Value::Array(m.tool_calls.clone());
                }
                // preserve_thinking: reasoning_content — как отдельное поле
                // для шаблона (Qwen3.8/Ornith 1.5).
                if m.role == "assistant" {
                    if let Some(r) = &m.reasoning_content {
                        if !r.is_empty() {
                            v["reasoning_content"] = json!(r);
                        }
                    }
                }
                v
            })
            .collect();
        let mut ctx = json!({
            "messages": msgs,
            "add_generation_prompt": true,
            "enable_thinking": enable_thinking,
            "add_vision_id": false,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        });
        if let Some(effort) = reasoning_effort {
            ctx["reasoning_effort"] = json!(effort);
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_mapping_get_works_in_official_templates() {
        let mut env = Environment::new();
        env.set_unknown_method_callback(|_, value, method, args| {
            python_mapping_get(value, method, args)
        });
        env.add_template(
            "test",
            "{{ message.get('content') }}|{{ message.get('missing', 'x') }}",
        )
        .unwrap();
        let rendered = env
            .get_template("test")
            .unwrap()
            .render(json!({"message": {"content": "ok"}}))
            .unwrap();
        assert_eq!(rendered, "ok|x");
    }
}
