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


def post_json(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get_json(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18099")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mmlu-n", type=int, default=300)
    ap.add_argument("--gsm8k-n", type=int, default=131)
    ap.add_argument("--gsm8k-max-tokens", type=int, default=768)
    ap.add_argument("--tasks", default="mmlu,gsm8k")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    results = {"label": a.label, "base_url": a.base_url, "tasks": {}}
    if "mmlu" in a.tasks:
        results["tasks"]["mmlu"] = run_mmlu(a.base_url, a.mmlu_n, a.verbose)
        print(json.dumps(results["tasks"]["mmlu"]), flush=True)
    if "gsm8k" in a.tasks:
        results["tasks"]["gsm8k"] = run_gsm8k(a.base_url, a.gsm8k_n, a.gsm8k_max_tokens, a.verbose)
        print(json.dumps(results["tasks"]["gsm8k"]), flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("written " + a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
