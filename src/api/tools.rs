//! Парсинг tool calls из вывода модели — порт боевого опыта Yttri
//! (local_llm/tool_calling.rs + prompts/parse_json_response).
//!
//! Два формата:
//! - JSON (Qwen3.5/3.6):  <tool_call>{"name":"f","arguments":{...}}</tool_call>
//! - Hermes (Qwen3.8):    <tool_call><function=f><parameter=x>v</parameter></function></tool_call>
//!
//! Маленькие модели ломают JSON: сырые переводы строк в значениях, висячие
//! запятые, лишний мусор после `}`, значения без кавычек. Лечим всё это.

use serde_json::Value;

/// (остальной текст, [(name, arguments_json_string)])
pub fn parse_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    let mut calls = Vec::new();
    let mut rest = String::new();
    let mut s = text;
    loop {
        // Толерантный поиск: модель на IQ2 искажает тег — `<tool_1>`,
        // `<toolcall>`, `<|tool_call>` — вместо каноничного `<tool_call>`.
        let Some((start, open_end)) = find_tool_tag(s, "<tool") else {
            rest.push_str(s);
            break;
        };
        rest.push_str(&s[..start]);
        let after = &s[open_end..];
        match find_tool_tag(after, "</tool") {
            Some((end, close_end)) => {
                let body = after[..end].trim();
                if let Some(call) = parse_block_body(body) {
                    calls.push(call);
                } else {
                    // Нераспарсенный блок — НЕ возвращаем разметку пользователю
                    // (Yttri: strip_tool_call_tags), выкидываем теги, текст тела оставляем.
                    if !body.is_empty() && !body.starts_with("<function=") && !body.starts_with('{') {
                        rest.push_str(body);
                    }
                }
                s = &after[close_end..];
            }
            None => {
                // Незакрытый тег: пробуем распарсить хвост (обрыв генерации).
                if let Some(call) = parse_block_body(after.trim()) {
                    calls.push(call);
                }
                break;
            }
        }
    }
    (rest.trim().to_string(), calls)
}

/// Найти тег по префиксу (`<tool` или `</tool`), вернуть (начало, конец_после_>`).
/// Терпит любое продолжение до `>`: `<tool_call>`, `<tool_1>`, `<|tool_call|>`, …
fn find_tool_tag(s: &str, prefix: &str) -> Option<(usize, usize)> {
    let mut from = 0;
    loop {
        let i = s[from..].find(prefix)? + from;
        // Следующий символ после префикса: не буква/цифра слова-тега — иначе это
        // `<tool_response>`/`<tools>` (не наш случай) — но `<tool_call>` разрешён.
        let tail = &s[i + prefix.len()..];
        let rel_close = tail.find('>')?;
        let name = &tail[..rel_close];
        let name_clean = name.trim_matches(|c: char| c == '|' || c == '_' || c == ' ');
        if matches!(name_clean, "call" | "calls" | "" | "1" | "2" | "3" | "4" | "5") {
            return Some((i, i + prefix.len() + rel_close + 1));
        }
        from = i + prefix.len() + rel_close + 1;
    }
}

fn parse_block_body(body: &str) -> Option<(String, String)> {
    if body.starts_with("<function=") {
        parse_hermes(body)
    } else {
        parse_json_call(body)
    }
}

/// Hermes: `<function=name>\n<parameter=k>\nvalue\n</parameter>…</function>`
fn parse_hermes(body: &str) -> Option<(String, String)> {
    let name_start = body.find("<function=")? + "<function=".len();
    let name_end = body[name_start..].find('>')? + name_start;
    let name = body[name_start..name_end].trim();
    if name.is_empty() {
        return None;
    }
    let mut args = serde_json::Map::new();
    let mut s = &body[name_end + 1..];
    while let Some(p) = s.find("<parameter=") {
        let k_start = p + "<parameter=".len();
        let k_end = s[k_start..].find('>')? + k_start;
        let key = s[k_start..k_end].trim().to_string();
        let after_open = &s[k_end + 1..];
        let (value, next) = match after_open.find("</parameter>") {
            Some(vend) => (
                after_open[..vend].trim_matches('\n').to_string(),
                &after_open[vend + "</parameter>".len()..],
            ),
            None => (after_open.trim().to_string(), ""),
        };
        // Значение может быть JSON (число/объект/массив/bool) или строка.
        let parsed = heal_json(&value)
            .and_then(|h| serde_json::from_str::<Value>(&h).ok())
            .unwrap_or(Value::String(value));
        args.insert(key, parsed);
        s = next;
        if s.is_empty() {
            break;
        }
    }
    Some((name.to_string(), Value::Object(args).to_string()))
}

/// JSON-формат с лечением (порт parse_json_response из Yttri).
fn parse_json_call(body: &str) -> Option<(String, String)> {
    let healed = heal_json(body)?;
    let v: Value = serde_json::from_str(&healed).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    if name.is_empty() {
        return None;
    }
    let args = match v.get("arguments") {
        Some(Value::String(s)) => {
            // arguments пришли строкой — лечим и разворачиваем
            match heal_json(s).and_then(|h| serde_json::from_str::<Value>(&h).ok()) {
                Some(inner) => inner.to_string(),
                None => s.clone(),
            }
        }
        Some(other) => other.to_string(),
        None => "{}".to_string(),
    };
    Some((name, args))
}

/// Лечение типичных поломок JSON от LLM. None — безнадёжно.
fn heal_json(raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    // 1. Прямой парсинг после санитизации.
    for candidate in candidates(trimmed) {
        let fixed = sanitize(&candidate);
        if serde_json::from_str::<Value>(&fixed).is_ok() {
            return Some(fixed);
        }
    }
    // 2. Обрезка хвоста: модель дописала мусор после последней `}` —
    //    откусываем посимвольно до ближайшей валидной `}` (Yttri, лог 14.08).
    let bytes = trimmed.as_bytes();
    let mut end = bytes.len();
    let min = end.saturating_sub(12);
    while end > min {
        end -= 1;
        if bytes[end] == b'}' {
            let sub = &trimmed[..=end];
            let fixed = sanitize(sub);
            if serde_json::from_str::<Value>(&fixed).is_ok() {
                return Some(fixed);
            }
        }
    }
    // 3. Значение без кавычек: {"name": fs_read, ...} → {"name": "fs_read", ...}
    let quoted = quote_bare_word_values(trimmed);
    if let Some(q) = quoted {
        let fixed = sanitize(&q);
        if serde_json::from_str::<Value>(&fixed).is_ok() {
            return Some(fixed);
        }
    }
    None
}

/// Варианты извлечения: как есть; из markdown-ограды; от первого { до последнего }.
fn candidates(raw: &str) -> Vec<String> {
    let mut out = vec![raw.to_string()];
    if raw.starts_with("```") && raw.ends_with("```") && raw.len() > 6 {
        let inner = &raw[3..raw.len() - 3];
        let body = match inner.split_once('\n') {
            Some((_lang, rest)) => rest,
            None => inner,
        };
        out.push(body.trim().to_string());
    }
    if let (Some(a), Some(b)) = (raw.find('{'), raw.rfind('}')) {
        if a < b {
            out.push(raw[a..=b].to_string());
        }
    }
    out
}

/// Санитизация: сырые control-символы в строках, trailing commas,
/// дозакрытие обрезанных скобок/строк.
fn sanitize(json: &str) -> String {
    let mut result = escape_control_chars_in_strings(json);
    loop {
        let before = result.clone();
        result = remove_trailing_comma(&result, '}');
        result = remove_trailing_comma(&result, ']');
        if result == before {
            break;
        }
    }
    // Дозакрытие: сначала строка (иначе скобки уедут внутрь неё).
    let (unterminated_at, mut openers) = scan_tail(&result);
    if let Some(pos) = unterminated_at {
        result.truncate(pos);
        result.push('"');
    }
    while let Some(op) = openers.pop() {
        result.push(if op == '{' { '}' } else { ']' });
    }
    result
}

/// Экранировать сырые \n \t \r внутри строковых литералов.
fn escape_control_chars_in_strings(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() + 16);
    let mut in_string = false;
    let mut escape_next = false;
    for &b in bytes {
        if escape_next {
            out.push(b);
            escape_next = false;
            continue;
        }
        match b {
            b'\\' if in_string => {
                out.push(b);
                escape_next = true;
            }
            b'"' => {
                out.push(b);
                in_string = !in_string;
            }
            b'\n' if in_string => out.extend_from_slice(b"\\n"),
            b'\r' if in_string => out.extend_from_slice(b"\\r"),
            b'\t' if in_string => out.extend_from_slice(b"\\t"),
            _ => out.push(b),
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn remove_trailing_comma(s: &str, closer: char) -> String {
    let bytes = s.as_bytes();
    let closer_b = closer as u8;
    let mut out = Vec::with_capacity(bytes.len());
    let mut in_string = false;
    let mut escape_next = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if escape_next {
            out.push(b);
            escape_next = false;
            i += 1;
            continue;
        }
        if in_string {
            out.push(b);
            if b == b'\\' {
                escape_next = true;
            } else if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push(b);
            i += 1;
            continue;
        }
        if b == b',' {
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == closer_b {
                i += 1; // пропускаем запятую
                continue;
            }
        }
        out.push(b);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Скан хвоста: (позиция незавершённой строки, стек незакрытых скобок по порядку открытия).
fn scan_tail(s: &str) -> (Option<usize>, Vec<char>) {
    let bytes = s.as_bytes();
    let mut in_string = false;
    let mut escape_next = false;
    let mut string_start = 0usize;
    let mut openers: Vec<char> = Vec::new();
    for (i, &b) in bytes.iter().enumerate() {
        if escape_next {
            escape_next = false;
            continue;
        }
        if in_string {
            if b == b'\\' {
                escape_next = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => {
                in_string = true;
                string_start = i;
            }
            b'{' | b'[' => openers.push(b as char),
            b'}' | b']' => {
                openers.pop();
            }
            _ => {}
        }
    }
    (
        if in_string { Some(string_start) } else { None },
        openers,
    )
}

/// `{"name": fs_read}` → `{"name": "fs_read"}` — голое слово после `:`.
fn quote_bare_word_values(s: &str) -> Option<String> {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() + 8);
    let mut in_string = false;
    let mut escape_next = false;
    let mut changed = false;
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if escape_next {
            out.push(b);
            escape_next = false;
            i += 1;
            continue;
        }
        if in_string {
            out.push(b);
            if b == b'\\' {
                escape_next = true;
            } else if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        if b == b'"' {
            in_string = true;
            out.push(b);
            i += 1;
            continue;
        }
        if b == b':' {
            out.push(b);
            i += 1;
            while i < bytes.len() && (bytes[i] as char).is_whitespace() {
                out.push(bytes[i]);
                i += 1;
            }
            if i < bytes.len()
                && bytes[i] != b'"'
                && bytes[i] != b'{'
                && bytes[i] != b'['
                && bytes[i] != b'-'
                && !bytes[i].is_ascii_digit()
            {
                // голое слово до , } ] или whitespace+delimiter
                let start = i;
                while i < bytes.len()
                    && !matches!(bytes[i], b',' | b'}' | b']' | b'\n' | b'\r')
                {
                    i += 1;
                }
                let word = s[start..i].trim_end();
                // true/false/null — валидны без кавычек
                if !matches!(word, "true" | "false" | "null") && !word.is_empty() {
                    out.push(b'"');
                    out.extend_from_slice(word.as_bytes());
                    out.push(b'"');
                    changed = true;
                } else {
                    out.extend_from_slice(s[start..i].as_bytes());
                }
                continue;
            }
            continue;
        }
        out.push(b);
        i += 1;
    }
    changed.then(|| String::from_utf8_lossy(&out).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_plain() {
        let (text, calls) = parse_tool_calls(
            "ок\n<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}</tool_call>",
        );
        assert_eq!(text, "ок");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "bash");
        assert!(calls[0].1.contains("ls"));
    }

    #[test]
    fn hermes_format() {
        let (text, calls) = parse_tool_calls(
            "<tool_call>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
        );
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "bash");
        assert!(calls[0].1.contains("ls -la"));
    }

    #[test]
    fn heal_raw_newline_in_string() {
        let (text, calls) = parse_tool_calls(
            "<tool_call>{\"name\": \"write\", \"arguments\": {\"content\": \"line1\nline2\"}}</tool_call>",
        );
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn heal_trailing_garbage() {
        let (_, calls) = parse_tool_calls(
            "<tool_call>{\"name\": \"f\", \"arguments\": {\"x\": 1}}})</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "f");
    }

    #[test]
    fn heal_trailing_comma() {
        let (_, calls) = parse_tool_calls(
            "<tool_call>{\"name\": \"f\", \"arguments\": {\"x\": 1,}}</tool_call>",
        );
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn heal_bare_word_value() {
        let (_, calls) = parse_tool_calls(
            "<tool_call>{\"name\": fs_read, \"arguments\": {}}</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "fs_read");
    }

    #[test]
    fn unclosed_tag() {
        let (_, calls) = parse_tool_calls(
            "text <tool_call>{\"name\": \"f\", \"arguments\": {\"x\": 1}}",
        );
        assert_eq!(calls.len(), 1);
    }

    #[test]
    fn tolerant_broken_open_tag() {
        // IQ2-модель написала <tool_1> вместо <tool_call> (live-наблюдение 27B IQ2_XXS).
        let (text, calls) = parse_tool_calls(
            "<tool_1>\n<function=bash>\n<parameter=command>\nls -la\n</parameter>\n</function>\n</tool_call>",
        );
        assert!(text.is_empty());
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "bash");
    }

    #[test]
    fn tool_response_not_eaten() {
        // <tool_response> — не наш тег, должен остаться текстом.
        let (text, calls) = parse_tool_calls("<tool_response>data</tool_response>");
        assert!(calls.is_empty());
        assert!(text.contains("tool_response"));
    }
}
