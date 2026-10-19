#!/usr/bin/env python3
"""Одинаковый HTTP-замер для yforge и ds4-dfm-rs на DGX Spark.

Каждый запрос уникален, если не включён repeat-режим. Токены берутся из
usage, а не из числа SSE-чанков. GB10 не отдаёт VRAM через nvidia-smi,
поэтому дополнительно пишем RSS/RSS_HWM процесса и MemAvailable хоста.
"""
import argparse
import csv
import hashlib
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

WORDS = ("observable deterministic throughput context memory kernel tensor matrix "
         "token stream profile compact stable cache engine request response local "
         "model inference compute sparse expert router attention layer residual "
         "latency prefix decode prefill benchmark hardware software system").split()
FIELDS = [
    "engine", "model", "phase", "target_context", "repeat", "ok", "error",
    "prompt_tokens", "completion_tokens", "cached_tokens", "ttft_s",
    "last_token_s", "prefill_tok_s", "decode_tok_s", "wall_s",
    "mem_available_before_mib", "mem_available_after_mib", "mem_available_min_mib",
    "proc_rss_mib", "proc_hwm_mib", "prompt_sha256", "output_text",
]


def mib(value):
    return round(value / (1024 * 1024), 3) if value is not None else None


def mem_available():
    with open("/proc/meminfo", encoding="ascii") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return None


def proc_memory(pid):
    rss = hwm = None
    with open(f"/proc/{pid}/status", encoding="ascii") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                hwm = int(line.split()[1]) * 1024
    return rss, hwm


class Sampler:
    def __init__(self, pid, interval=0.2):
        self.pid = pid
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.lock = threading.Lock()
        self.mem_min = mem_available()
        self.rss_max = 0
        self.hwm_max = 0

    def start(self):
        if self.pid:
            self.thread.start()

    def run(self):
        while not self.stop_event.wait(self.interval):
            mem = mem_available()
            rss, hwm = proc_memory(self.pid)
            with self.lock:
                if mem is not None:
                    self.mem_min = min(self.mem_min or mem, mem)
                self.rss_max = max(self.rss_max, rss or 0)
                self.hwm_max = max(self.hwm_max, hwm or 0)

    def stop(self):
        if self.pid:
            self.stop_event.set()
            self.thread.join(timeout=2)
        with self.lock:
            return self.mem_min, self.rss_max, self.hwm_max


def make_prompt(target, nonce, ordinal, prompt_file=None):
    if prompt_file:
        text = Path(prompt_file).read_text(encoding="utf-8", errors="replace")
        # Natural Italian prose is about 3.15 characters/token. Keep the
        # request below the model context and rotate the window per run.
        chars = min(max(512, int(target * 3.15)), len(text))
        offset = (ordinal * 104729 + sum(map(ord, nonce))) % max(1, len(text) - chars + 1)
        body = text[offset:offset + chars]
        return f"run-{nonce}-{ordinal}.\n{body}\nEND-{nonce}. Reply with one short sentence."
    # Fallback for synthetic microbenchmarks where no corpus is provided.
    count = max(1, int(target * 0.90))
    words = [f"run-{nonce}-{ordinal}"]
    words.extend(WORDS[(ordinal + i) % len(WORDS)] for i in range(count))
    words.append(f"end-{nonce}. Reply with one short sentence.")
    return " ".join(words)


def discover_model(base_url, api_key):
    request = urllib.request.Request(base_url.rstrip("/") + "/v1/models", headers=headers(api_key))
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read().decode())
    models = data.get("data") or []
    if not models:
        raise RuntimeError("empty /v1/models")
    return models[0]["id"]


def headers(api_key):
    result = {"Content-Type": "application/json"}
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    return result


def stream_chat(base_url, api_key, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers=headers(api_key),
    )
    started = time.monotonic()
    ttft = last_token = None
    prompt_tokens = completion_tokens = cached_tokens = 0
    text = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            usage = chunk.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens") or prompt_tokens
            completion_tokens = usage.get("completion_tokens") or completion_tokens
            details = usage.get("prompt_tokens_details") or {}
            cached_tokens = details.get("cached_tokens") or cached_tokens
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    now = time.monotonic() - started
                    ttft = now if ttft is None else ttft
                    last_token = now
                    if delta.get("content"):
                        text.append(piece)
    return {
        "ok": ttft is not None and completion_tokens > 0,
        "error": "" if ttft is not None and completion_tokens > 0 else "no generated tokens or usage",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "ttft_s": ttft,
        "last_token_s": last_token,
        "wall_s": time.monotonic() - started,
        "output_text": "".join(text)[:2000],
    }


def one_case(args, model, target, ordinal, prompt, phase):
    sampler = Sampler(args.pid)
    before = mem_available()
    sampler.start()
    try:
        result = stream_chat(args.base_url, args.api_key, model, prompt, args.max_tokens, args.timeout)
    except urllib.error.HTTPError as exc:
        result = {"ok": False, "error": f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:1000]}"}
    except Exception as exc:  # Network and server errors are measurements too.
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    mem_min, rss_max, hwm_max = sampler.stop()
    after = mem_available()
    ttft = result.get("ttft_s")
    last = result.get("last_token_s")
    prompt_tokens = result.get("prompt_tokens") or 0
    completion_tokens = result.get("completion_tokens") or 0
    result.update({
        "engine": args.engine,
        "model": model,
        "phase": phase,
        "target_context": target,
        "repeat": ordinal,
        "mem_available_before_mib": mib(before),
        "mem_available_after_mib": mib(after),
        "mem_available_min_mib": mib(mem_min),
        "proc_rss_mib": mib(rss_max),
        "proc_hwm_mib": mib(hwm_max),
        "prefill_tok_s": round(prompt_tokens / ttft, 3) if ttft and prompt_tokens else None,
        "decode_tok_s": round((completion_tokens - 1) / (last - ttft), 3)
                         if completion_tokens > 1 and last and ttft and last > ttft else None,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    })
    return {field: result.get(field) for field in FIELDS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18099")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--pid", type=int, default=0)
    parser.add_argument("--contexts", default="2048,8192,32768,65536,131072")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=65)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--repeat-context", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--prompt-file", default="")
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    model = args.model or discover_model(args.base_url, args.api_key)
    nonce = f"{int(time.time())}-{random.randrange(1_000_000_000)}"
    rows = []
    jsonl_path = args.outdir / f"{args.engine}.jsonl"
    csv_path = args.outdir / f"{args.engine}.csv"
    with jsonl_path.open("w", encoding="utf-8") as raw_log:
        for warmup in range(args.warmup):
            warmup_prompt = make_prompt(512, f"{nonce}-warmup", warmup, args.prompt_file)
            warmup_row = one_case(args, model, 512, warmup, warmup_prompt, "warmup")
            print(json.dumps(warmup_row, ensure_ascii=False), flush=True)
        for target in [int(v) for v in args.contexts.split(",") if v]:
            for repeat in range(1, args.repeats + 1):
                prompt = make_prompt(target, nonce, repeat, args.prompt_file)
                row = one_case(args, model, target, repeat, prompt, "context")
                rows.append(row)
                raw_log.write(json.dumps({"prompt_tokens_estimate": target, **row,
                                          "prompt": prompt if target <= 8192 else "<omitted>"},
                                         ensure_ascii=False) + "\n")
                raw_log.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.repeat_context and args.repeat_count:
            prompt = make_prompt(args.repeat_context, nonce, 900000, args.prompt_file)
            for repeat in range(1, args.repeat_count + 1):
                row = one_case(args, model, args.repeat_context, repeat, prompt, "engram-repeat")
                rows.append(row)
                raw_log.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_log.flush()
                print(json.dumps(row, ensure_ascii=False), flush=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for target in sorted({row["target_context"] for row in rows}):
        values = [row for row in rows if row["target_context"] == target and row["ok"]]
        if not values:
            continue
        prefill = statistics.median(r["prefill_tok_s"] for r in values if r["prefill_tok_s"])
        decode = statistics.median(r["decode_tok_s"] for r in values if r["decode_tok_s"])
        print(f"median target={target} prefill={prefill:.2f} decode={decode:.2f}", flush=True)


if __name__ == "__main__":
    main()
