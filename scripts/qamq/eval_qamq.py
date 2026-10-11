#!/usr/bin/env python3
"""Компактный harness для task-evals QAMQ (MMLU + GSM8K) поверх llama-server.

Без внешних зависимостей: только stdlib. Данные берутся через HF datasets-server,
инференс — через /v1/chat/completions запущенного llama-server.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

HF = "https://datasets-server.huggingface.co/rows"
LETTERS = ["A", "B", "C", "D"]


API_KEY = None
MODEL = "qwen3.5-9b-qamq"


def _headers(extra):
    h = dict(extra)
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def post_json(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers=_headers({"Content-Type": "application/json"}))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get_json(url, timeout=120):
    req = urllib.request.Request(url, headers=_headers({}))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def fetch_rows(dataset, config, split, n):
    rows, offset = [], 0
    while len(rows) < n:
        take = min(100, n - len(rows))
        url = f"{HF}?dataset={dataset}&config={config}&split={split}&offset={offset}&length={take}"
        d = get_json(url)
        batch = [x["row"] for x in d.get("rows", [])]
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
    return rows[:n]


def chat(base_url, messages, max_tokens, logprobs=False, top_logprobs=0):
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    return post_json(base_url.rstrip("/") + "/v1/chat/completions", payload)


def letter_logprobs(resp):
    """Собрать logprob для вариантов буквы из ответа (A-D) по top_logprobs первого токена."""
    try:
        content = resp["choices"][0]["logprobs"]["content"][0]
    except (KeyError, IndexError, TypeError):
        return None
    out = {}
    for cand in content.get("top_logprobs") or []:
        tok = (cand.get("token") or "").strip().upper()
        if tok in LETTERS and tok not in out:
            out[tok] = cand["logprob"]
    return out or None


def run_mmlu(base_url, n, verbose=False):
    rows = fetch_rows("cais/mmlu", "all", "test", n)
    correct, missing, t0 = 0, 0, time.time()
    for i, row in enumerate(rows):
        q = row["question"]
        choices = row["choices"]
        subject = str(row.get("subject", "general")).replace("_", " ")
        prompt = (
            f"The following are multiple choice questions (with answers) about {subject}.\n\n"
            f"{q}\n"
            + "\n".join(f"{LETTERS[j]}. {choices[j]}" for j in range(4))
            + "\n\nAnswer with a single letter: A, B, C, or D."
        )
        resp = chat(base_url, [{"role": "user", "content": prompt}], max_tokens=1,
                    logprobs=True, top_logprobs=20)
        lp = letter_logprobs(resp)
        if not lp:
            missing += 1
            pred = None
        else:
            pred = max(lp, key=lp.get)
        gold = LETTERS[row["answer"]]
        if pred == gold:
            correct += 1
        if verbose and i % 50 == 0:
            print(f"  mmlu {i}/{len(rows)} acc={correct/max(1,i+1):.3f}", flush=True)
    return {
        "task": "mmlu",
        "n": len(rows),
        "correct": correct,
        "acc": correct / len(rows) if rows else 0.0,
        "unscored": missing,
        "seconds": round(time.time() - t0, 1),
    }


NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def run_rummlu(base_url, n, verbose=False):
    """ruMMLU (MERA) — тот же single-token scoring, что и MMLU, но на русском."""
    rows = fetch_rows("ai-forever/MERA", "rummlu", "public_test", n)
    correct, missing, t0 = 0, 0, time.time()
    for i, row in enumerate(rows):
        ins = row["inputs"]
        prompt = (
            "Ответь на вопрос, выбрав один вариант. "
            "Запиши только букву правильного ответа: A, B, C или D.\n\n"
            f"{ins['text']}\n"
            f"A {ins['option_a']}\n"
            f"B {ins['option_b']}\n"
            f"C {ins['option_c']}\n"
            f"D {ins['option_d']}"
        )
        resp = chat(base_url, [{"role": "user", "content": prompt}], max_tokens=1,
                    logprobs=True, top_logprobs=20)
        lp = letter_logprobs(resp)
        if not lp:
            missing += 1
            pred = None
        else:
            pred = max(lp, key=lp.get)
        gold = str(row["outputs"]).strip().upper()
        if pred == gold:
            correct += 1
        if verbose and i % 50 == 0:
            print(f"  rummlu {i}/{len(rows)} acc={correct/max(1,i+1):.3f}", flush=True)
    return {
        "task": "rummlu",
        "n": len(rows),
        "correct": correct,
        "acc": correct / len(rows) if rows else 0.0,
        "unscored": missing,
        "seconds": round(time.time() - t0, 1),
    }


def extract_number(text):
    tail = text.split("####")[-1] if "####" in text else text
    nums = NUM_RE.findall(tail.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def run_gsm8k(base_url, n, max_tokens, verbose=False):
    rows = fetch_rows("openai/gsm8k", "main", "test", n)
    correct, unparsed, t0 = 0, 0, time.time()
    for i, row in enumerate(rows):
        prompt = (
            row["question"]
            + "\n\nSolve step by step. End the final line with: #### <answer>"
        )
        resp = chat(base_url, [{"role": "user", "content": prompt}], max_tokens=max_tokens)
        text = resp["choices"][0]["message"].get("content") or ""
        pred = extract_number(text)
        gold = extract_number(row["answer"])
        if pred is None:
            unparsed += 1
        if pred is not None and gold is not None and abs(pred - gold) < 1e-6:
            correct += 1
        if verbose and i % 25 == 0:
            print(f"  gsm8k {i}/{len(rows)} acc={correct/max(1,i+1):.3f}", flush=True)
    return {
        "task": "gsm8k",
        "n": len(rows),
        "correct": correct,
        "acc": correct / len(rows) if rows else 0.0,
        "unparsed": unparsed,
        "seconds": round(time.time() - t0, 1),
    }


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file from disk and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path"},
                    "max_lines": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email message.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": "Search a customer order database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string"},
                    "status": {"type": "string", "enum": ["open", "shipped", "cancelled"]},
                },
                "required": ["customer_id"],
            },
        },
    },
]

# (запрос, ожидаемое имя функции, обязательные ключи в аргументах)
TOOL_CASES = [
    ("Какая сейчас погода в Москве?", "get_weather", ["city"]),
    ("Прочитай файл /etc/hosts и покажи первые 10 строк.", "read_file",
     ["path", "max_lines"]),
    ("Отправь письмо на alice@example.com с темой 'Отчёт' и текстом 'Готово'.", "send_email",
     ["to", "subject", "body"]),
    ("Найди все отменённые заказы клиента C-4471.", "search_orders",
     ["customer_id", "status"]),
    ("What is the weather in Berlin right now?", "get_weather", ["city"]),
    ("Read /var/log/app.log, please.", "read_file", ["path"]),
    ("Email bob@example.org about the meeting tomorrow.", "send_email", ["to"]),
    ("Show open orders for customer 99812.", "search_orders", ["customer_id"]),
    ("Посмотри погоду в Санкт-Петербурге.", "get_weather", ["city"]),
    ("Открой C:\\logs\\today.txt и покажи 50 строк.", "read_file", ["path"]),
]


def run_toolcall(base_url, verbose=False):
    """Tool-calling: выбрать правильную функцию и заполнить обязательные аргументы."""
    ok_name, ok_args, n = 0, 0, len(TOOL_CASES)
    details, t0 = [], time.time()
    for i, (query, want_name, want_keys) in enumerate(TOOL_CASES):
        payload = {
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 256,
            "temperature": 0,
            "tools": TOOLS,
            "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = post_json(base_url.rstrip("/") + "/v1/chat/completions", payload)
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        name, args = None, {}
        if calls:
            name = calls[0].get("function", {}).get("name")
            raw = calls[0].get("function", {}).get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                args = {}
        name_hit = name == want_name
        keys_hit = name_hit and all(k in (args or {}) for k in want_keys)
        ok_name += int(name_hit)
        ok_args += int(keys_hit)
        details.append({"query": query, "want": want_name, "got": name,
                        "name_ok": name_hit, "args_ok": keys_hit})
        if verbose:
            print(f"  toolcall {i}: want={want_name} got={name} args_ok={keys_hit}",
                  flush=True)
    return {
        "task": "toolcall",
        "n": n,
        "name_acc": ok_name / n if n else 0.0,
        "name_and_args_acc": ok_args / n if n else 0.0,
        "seconds": round(time.time() - t0, 1),
        "details": details,
    }


NEEDLE_FACTS = [
    ("UV-7734", "пломба на щите учёта"),
    ("KM-2210", "сервисный ключ от кладовой"),
    ("TX-9031", "номер партии фильтров"),
    ("LP-4482", "код резервного генератора"),
    ("RD-6620", "маркировка кабельной трассы"),
]


def _filler(sentences, seed):
    """Детерминированный текст-наполнитель (без внешних данных)."""
    base = [
        "Отдел эксплуатации ведёт журнал наблюдений за оборудованием.",
        "Плановые обходы выполняются дважды в смену по утверждённому маршруту.",
        "Замечания фиксируются в электронном журнале с указанием времени.",
        "Подрядная организация согласует работы с дежурным инженером.",
        "Все измерения вносятся в ведомость без сокращений и правок.",
        "При отклонении параметров оформляется уведомление ответственному.",
    ]
    out = []
    for i in range(sentences):
        out.append(base[(i + seed) % len(base)])
    return " ".join(out)


def run_needle(base_url, context_lens, trials_per_len, chars_per_token=4.6, verbose=False):
    """Needle-in-a-haystack: спрятать факт в шум и спросить его дословно.

    chars_per_token — эмпирическая плотность русского текста (4.6 символа на
    токен, замерено на этом haystack). Целевой размер haystack в токенах × chars_per_token даёт длину в
    символах; сервер потом сообщает точное число в usage.prompt_tokens.
    """
    results = []
    for ctx in context_lens:
        correct, t0 = 0, time.time()
        for t in range(trials_per_len):
            code, what = NEEDLE_FACTS[(ctx + t) % len(NEEDLE_FACTS)]
            fact = f"Служебный идентификатор {code} закреплён за объектом: {what}."
            # позиция иглы внутри шума: чередуем начало/середину/конец
            pos_frac = [0.15, 0.5, 0.85][t % 3]
            total_chars = int(ctx * chars_per_token)
            n_sent = max(4, total_chars // 70)
            n_before = max(1, int(n_sent * pos_frac))
            n_after = max(1, n_sent - n_before)
            haystack = (_filler(n_before, ctx + t) + " " + fact + " "
                        + _filler(n_after, ctx + t + 7))
            prompt = (
                "Ниже служебный текст. Найди в нём служебный идентификатор и объект, "
                "за которым он закреплён, и ответь одной строкой в формате "
                "'КОД: объект'.\n\n" + haystack
            )
            resp = chat(base_url, [{"role": "user", "content": prompt}], max_tokens=48)
            actual = (resp.get("usage") or {}).get("prompt_tokens")
            text = (resp["choices"][0]["message"].get("content") or "").upper()
            if code in text and what.upper()[:8] in text:
                correct += 1
            if verbose:
                print(f"  needle ctx~{ctx} (prompt_tokens={actual}) t={t} "
                      f"pos={pos_frac} -> {code in text}", flush=True)
        results.append({
            "ctx_target": ctx, "trials": trials_per_len, "correct": correct,
            "acc": correct / trials_per_len if trials_per_len else 0.0,
            "prompt_tokens": actual,
            "seconds": round(time.time() - t0, 1),
        })
    return {"task": "needle", "results": results}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18099")
    ap.add_argument("--api-key", default=None,
                    help="Bearer-ключ для серверов с авторизацией (yforge)")
    ap.add_argument("--model", default=None, help="имя модели в запросе")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmlu-n", type=int, default=300)
    ap.add_argument("--gsm8k-n", type=int, default=131)
    ap.add_argument("--gsm8k-max-tokens", type=int, default=768)
    ap.add_argument("--needle-ctx", default="1024,4096,16384")
    ap.add_argument("--tasks", default="mmlu,gsm8k,rummlu,needle,toolcall")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    global API_KEY, MODEL
    API_KEY = a.api_key
    if a.model:
        MODEL = a.model

    results = {"label": a.label, "base_url": a.base_url, "tasks": {}}
    if "mmlu" in a.tasks:
        results["tasks"]["mmlu"] = run_mmlu(a.base_url, a.mmlu_n, a.verbose)
        print(json.dumps(results["tasks"]["mmlu"]), flush=True)
    if "gsm8k" in a.tasks:
        results["tasks"]["gsm8k"] = run_gsm8k(a.base_url, a.gsm8k_n, a.gsm8k_max_tokens, a.verbose)
        print(json.dumps(results["tasks"]["gsm8k"]), flush=True)
    if "rummlu" in a.tasks:
        results["tasks"]["rummlu"] = run_rummlu(a.base_url, a.mmlu_n, a.verbose)
        print(json.dumps(results["tasks"]["rummlu"]), flush=True)
    if "needle" in a.tasks:
        ctxs = [int(x) for x in str(a.needle_ctx).split(",") if x.strip()]
        results["tasks"]["needle"] = run_needle(a.base_url, ctxs, 3, verbose=a.verbose)
        print(json.dumps(results["tasks"]["needle"]), flush=True)
    if "toolcall" in a.tasks:
        results["tasks"]["toolcall"] = run_toolcall(a.base_url, a.verbose)
        tc = results["tasks"]["toolcall"]
        print(json.dumps({k: v for k, v in tc.items() if k != "details"}), flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("written " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
