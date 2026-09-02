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

/// Печать сырого текста модели перед разбором (QWEN36_TOOLS_DEBUG=1).
/// Нужна, чтобы отличить обрезание в генерации от обрезания в разборе: без неё
/// оба выглядят одинаково — короткое значение в готовом вызове.
fn tools_debug() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("QWEN36_TOOLS_DEBUG").is_ok_and(|v| v != "0"))
}

/// (остальной текст, [(name, arguments_json_string)])
pub fn parse_tool_calls(text: &str) -> (String, Vec<(String, String)>) {
    parse_tool_calls_with_schema(text, None)
}

pub fn tools_enabled(tools: Option<&Value>) -> bool {
    tools
        .and_then(Value::as_array)
        .map(|tools| !tools.is_empty())
        .unwrap_or(false)
}

/// Не превращать незавершённую генерацию или обычный текст без объявленных
/// инструментов в исполняемый вызов.
pub fn parse_tool_calls_for_response(
    text: &str,
    finish_reason: &str,
    tools: Option<&Value>,
) -> (String, Vec<(String, String)>) {
    if finish_reason == "length" {
        if tools_enabled(tools) && find_tool_tag(text, "<tool").is_some() {
            eprintln!(
                "[tools] вызов не исполнен: генерация оборвалась по length до подтверждения"
            );
        }
        return (text.to_string(), Vec::new());
    }
    if !tools_enabled(tools) {
        return (text.to_string(), Vec::new());
    }
    parse_tool_calls_with_schema(text, tools)
}

/// То же, но с исходной JSON Schema инструментов.
///
/// Hermes передаёт каждое значение как текст между `<parameter>`-тегами.
/// Без схемы строка, похожая на JSON (`{"query": ...}`), раньше автоматически
/// превращалась в object. Для `bash.command` и `write.content` это меняло тип
/// уже после генерации модели, и клиент справедливо отвечал `must be string`.
/// Схема позволяет сохранить строковые параметры строками, не ломая числовые,
/// boolean и object-параметры остальных инструментов.
pub fn parse_tool_calls_with_schema(
    text: &str,
    tools: Option<&Value>,
) -> (String, Vec<(String, String)>) {
    if tools_debug() {
        eprintln!("[tools] сырой текст модели ({} байт): {text:?}", text.len());
    }
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
                if let Some(call) = parse_block_body(body, tools) {
                    if let Err(error) = validate_tool_call(&call, tools) {
                        eprintln!("[tools] невалидный вызов {}: {error}", call.0);
                        append_invalid_body(&mut rest, body);
                    } else {
                        calls.push(call);
                    }
                } else {
                    // Нераспарсенный блок не должен исчезать: это обычный
                    // видимый текст, а не подтверждённая команда клиенту.
                    append_invalid_body(&mut rest, body);
                }
                s = &after[close_end..];
            }
            None => {
                // Незакрытый тег: пробуем распарсить хвост (обрыв генерации).
                let body = after.trim();
                if let Some(call) = parse_block_body(body, tools) {
                    if let Err(error) = validate_tool_call(&call, tools) {
                        eprintln!("[tools] невалидный вызов {}: {error}", call.0);
                        append_invalid_body(&mut rest, body);
                    } else {
                        calls.push(call);
                    }
                } else {
                    append_invalid_body(&mut rest, body);
                }
                break;
            }
        }
    }
    (rest.trim().to_string(), calls)
}

fn append_invalid_body(rest: &mut String, body: &str) {
    if body.is_empty() {
        return;
    }
    if !rest.is_empty() && !rest.ends_with(char::is_whitespace) {
        rest.push('\n');
    }
    rest.push_str(body);
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

fn parse_block_body(body: &str, tools: Option<&Value>) -> Option<(String, String)> {
    if body.starts_with("<function=") {
        parse_hermes(body, tools)
    } else {
        parse_json_call(body)
    }
}

/// Hermes: `<function=name>\n<parameter=k>\nvalue\n</parameter>…</function>`
fn parse_hermes(body: &str, tools: Option<&Value>) -> Option<(String, String)> {
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
                strip_hermes_framing_newlines(&after_open[..vend]).to_string(),
                &after_open[vend + "</parameter>".len()..],
            ),
            None => (strip_one_leading_newline(after_open).to_string(), ""),
        };
        // В Hermes все параметры изначально текстовые. Тип восстанавливаем по
        // schema; эвристический JSON-разбор оставляем только когда схема не
        // требует строку (числа, bool, object и старые клиенты без tools).
        let parsed = if property_requires_string(tools, name, &key) {
            Value::String(value)
        } else {
            heal_json(&value)
                .and_then(|h| serde_json::from_str::<Value>(&h).ok())
                .unwrap_or(Value::String(value))
        };
        args.insert(key, parsed);
        s = next;
        if s.is_empty() {
            break;
        }
    }
    Some((name.to_string(), Value::Object(args).to_string()))
}

/// Hermes отделяет значение параметра одним переводом строки после открывающего
/// и перед закрывающим тегом. Удаляем только эти два framing-разделителя:
/// `trim_matches('\n')` раньше стирал намеренные пустые строки из `write.content`.
fn strip_hermes_framing_newlines(value: &str) -> &str {
    strip_one_trailing_newline(strip_one_leading_newline(value))
}

fn strip_one_leading_newline(value: &str) -> &str {
    value
        .strip_prefix("\r\n")
        .or_else(|| value.strip_prefix('\n'))
        .unwrap_or(value)
}

fn strip_one_trailing_newline(value: &str) -> &str {
    value
        .strip_suffix("\r\n")
        .or_else(|| value.strip_suffix('\n'))
        .unwrap_or(value)
}

fn tool_definition<'a>(tools: Option<&'a Value>, name: &str) -> Option<&'a Value> {
    tools?.as_array()?.iter().find(|tool| {
        tool.pointer("/function/name")
            .or_else(|| tool.get("name"))
            .and_then(Value::as_str)
            == Some(name)
    })
}

fn tool_parameters(tool: &Value) -> Option<&Value> {
    tool.pointer("/function/parameters")
        .or_else(|| tool.get("input_schema"))
        .or_else(|| tool.get("parameters"))
}

/// Проверить верхнеуровневый контракт вызова перед тем, как API отдаст его
/// агенту на исполнение. Это намеренно небольшой, консервативный subset JSON
/// Schema: object/array/scalar types, required, properties, items,
/// additionalProperties=false и anyOf/oneOf. Непонятные keywords не делают
/// корректный вызов невалидным.
fn validate_tool_call(call: &(String, String), tools: Option<&Value>) -> Result<(), String> {
    let Some(tool_list) = tools.and_then(Value::as_array) else {
        return Ok(()); // legacy parser без schema сохраняет прежний контракт
    };
    let Some(tool) = tool_definition(tools, &call.0) else {
        return Err(format!("инструмент {:?} не объявлен в запросе", call.0));
    };
    let arguments: Value = serde_json::from_str(&call.1)
        .map_err(|error| format!("arguments не являются JSON: {error}"))?;
    if !arguments.is_object() {
        return Err("arguments должны быть JSON-объектом".into());
    }
    if tool_list.is_empty() {
        return Err("список tools пуст".into());
    }
    if let Some(schema) = tool_parameters(tool) {
        validate_schema_value(&arguments, schema, "arguments")?;
    }
    Ok(())
}

fn validate_schema_value(value: &Value, schema: &Value, path: &str) -> Result<(), String> {
    if let Some(branches) = schema.get("anyOf").and_then(Value::as_array) {
        if !branches
            .iter()
            .any(|branch| validate_schema_value(value, branch, path).is_ok())
        {
            return Err(format!("{path} не соответствует ни одной ветке anyOf"));
        }
    }
    if let Some(branches) = schema.get("oneOf").and_then(Value::as_array) {
        let matches = branches
            .iter()
            .filter(|branch| validate_schema_value(value, branch, path).is_ok())
            .count();
        if matches != 1 {
            return Err(format!(
                "{path} должен соответствовать ровно одной ветке oneOf"
            ));
        }
    }

    if let Some(kind) = schema.get("type") {
        let accepted = match kind {
            Value::String(kind) => value_matches_type(value, kind),
            Value::Array(kinds) => kinds
                .iter()
                .filter_map(Value::as_str)
                .any(|kind| value_matches_type(value, kind)),
            _ => true,
        };
        if !accepted {
            return Err(format!("{path} имеет тип {}, несовместимый со schema", value_kind(value)));
        }
    }

    if let Value::Object(object) = value {
        if let Some(required) = schema.get("required").and_then(Value::as_array) {
            for key in required.iter().filter_map(Value::as_str) {
                if !object.contains_key(key) {
                    return Err(format!("{path}.{key}: отсутствует обязательное поле"));
                }
            }
        }
        let properties = schema.get("properties").and_then(Value::as_object);
        for (key, child) in object {
            if let Some(child_schema) = properties.and_then(|properties| properties.get(key)) {
                validate_schema_value(child, child_schema, &format!("{path}.{key}"))?;
            } else if schema.get("additionalProperties") == Some(&Value::Bool(false)) {
                return Err(format!("{path}.{key}: дополнительное поле запрещено"));
            }
        }
    }

    if let (Value::Array(items), Some(item_schema)) = (value, schema.get("items")) {
        for (index, item) in items.iter().enumerate() {
            validate_schema_value(item, item_schema, &format!("{path}[{index}]"))?;
        }
    }
    Ok(())
}

fn value_matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "null" => value.is_null(),
        "boolean" => value.is_boolean(),
        "object" => value.is_object(),
        "array" => value.is_array(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "string" => value.is_string(),
        _ => true,
    }
}

fn value_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(number) if number.is_i64() || number.is_u64() => "integer",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Найти JSON Schema аргументов и вернуть schema конкретного свойства.
/// Поддерживаются оба реально принимаемых API-формата:
/// OpenAI `function.parameters` и Anthropic `input_schema`.
fn property_schema<'a>(tools: Option<&'a Value>, name: &str, key: &str) -> Option<&'a Value> {
    tool_parameters(tool_definition(tools, name)?)?
        .get("properties")?
        .get(key)
}

fn property_requires_string(tools: Option<&Value>, name: &str, key: &str) -> bool {
    let Some(kind) = property_schema(tools, name, key).and_then(|schema| schema.get("type")) else {
        return false;
    };
    match kind {
        Value::String(kind) => kind == "string",
        Value::Array(kinds) => {
            let mut has_string = false;
            let only_string_or_null = kinds.iter().all(|kind| match kind.as_str() {
                Some("string") => {
                    has_string = true;
                    true
                }
                Some("null") => true,
                _ => false,
            });
            has_string && only_string_or_null
        }
        _ => false,
    }
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

    fn openai_tools_schema() -> Value {
        serde_json::json!([
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout": {"type": "integer"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                }
            }
        ])
    }

    #[test]
    fn hermes_json_looking_command_stays_string_by_schema() {
        let tools = openai_tools_schema();
        let (_, calls) = parse_tool_calls_with_schema(
            "<tool_call>\n<function=bash>\n<parameter=command>\n{\"model\":\"brave\",\"query\":\"test\"}\n</parameter>\n<parameter=timeout>\n30\n</parameter>\n</function>\n</tool_call>",
            Some(&tools),
        );
        let args: Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(
            args["command"],
            Value::String("{\"model\":\"brave\",\"query\":\"test\"}".into())
        );
        assert_eq!(args["timeout"], 30);
    }

    #[test]
    fn hermes_json_looking_write_content_stays_string_by_schema() {
        let tools = openai_tools_schema();
        let (_, calls) = parse_tool_calls_with_schema(
            "<tool_call>\n<function=write>\n<parameter=path>\n/tmp/body.json\n</parameter>\n<parameter=content>\n{\n  \"max_results\": 8\n}\n</parameter>\n</function>\n</tool_call>",
            Some(&tools),
        );
        let args: Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["path"], "/tmp/body.json");
        assert_eq!(args["content"], "{\n  \"max_results\": 8\n}");
    }

    #[test]
    fn json_tool_call_wrong_declared_type_is_not_executable() {
        let tools = openai_tools_schema();
        let (text, calls) = parse_tool_calls_with_schema(
            "<tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":{\"model\":\"brave\"}}}</tool_call>",
            Some(&tools),
        );
        assert!(calls.is_empty());
        assert!(text.contains("\"model\":\"brave\""));
    }

    #[test]
    fn missing_required_argument_is_not_executable() {
        let tools = openai_tools_schema();
        let (text, calls) = parse_tool_calls_with_schema(
            "<tool_call>{\"name\":\"write\",\"arguments\":{\"content\":\"body\"}}</tool_call>",
            Some(&tools),
        );
        assert!(calls.is_empty());
        assert!(text.contains("\"name\":\"write\""));
        assert!(text.contains("\"content\":\"body\""));
    }

    #[test]
    fn length_response_is_never_repaired_into_executable_call() {
        let tools = openai_tools_schema();
        let partial =
            "<tool_call>{\"name\":\"write\",\"arguments\":{\"content\":\"body\"";
        let (text, calls) = parse_tool_calls_for_response(partial, "length", Some(&tools));
        assert!(calls.is_empty());
        assert_eq!(text, partial);
    }

    #[test]
    fn tool_markup_without_tools_is_plain_text() {
        let output = "<tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":\"ls\"}}</tool_call>";
        let (text, calls) = parse_tool_calls_for_response(output, "stop", None);
        assert!(calls.is_empty());
        assert_eq!(text, output);
    }

    #[test]
    fn incompatible_argument_type_is_not_executable() {
        let tools = openai_tools_schema();
        let (text, calls) = parse_tool_calls_with_schema(
            "<tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":\"ls\",\"timeout\":\"soon\"}}</tool_call>",
            Some(&tools),
        );
        assert!(calls.is_empty());
        assert!(text.contains("\"timeout\":\"soon\""));
    }

    #[test]
    fn null_required_path_is_not_coerced_to_string() {
        let tools = openai_tools_schema();
        let (text, calls) = parse_tool_calls_with_schema(
            "<tool_call>{\"name\":\"write\",\"arguments\":{\"path\":null,\"content\":\"body\"}}</tool_call>",
            Some(&tools),
        );
        assert!(calls.is_empty());
        assert!(text.contains("\"path\":null"));
    }

    #[test]
    fn undeclared_tool_is_not_executable() {
        let tools = openai_tools_schema();
        let (text, calls) = parse_tool_calls_with_schema(
            "<tool_call>{\"name\":\"delete_everything\",\"arguments\":{}}</tool_call>",
            Some(&tools),
        );
        assert!(calls.is_empty());
        assert!(text.contains("delete_everything"));
    }

    #[test]
    fn hermes_removes_only_structural_boundary_newlines() {
        let tools = openai_tools_schema();
        let (_, calls) = parse_tool_calls_with_schema(
            "<tool_call>\n<function=write>\n<parameter=path>\n/tmp/a.css\n</parameter>\n<parameter=content>\nline1\n\n</parameter>\n</function>\n</tool_call>",
            Some(&tools),
        );
        let args: Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["content"], "line1\n");
    }

    #[test]
    fn anthropic_input_schema_preserves_string_argument() {
        let tools = serde_json::json!([{
            "name": "bash",
            "input_schema": {
                "type": "object",
                "properties": {"command": {"type": "string"}}
            }
        }]);
        let (_, calls) = parse_tool_calls_with_schema(
            "<tool_call><function=bash><parameter=command>{\"query\":\"test\"}</parameter></function></tool_call>",
            Some(&tools),
        );
        let args: Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(args["command"], "{\"query\":\"test\"}");
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
