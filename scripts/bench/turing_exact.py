#!/usr/bin/env python3
"""Exact 512-token prefill / 128-token decode benchmark for a running yforge."""
import argparse
import http.client
import json
import os
import statistics
import time
from urllib.parse import urlparse


def request(conn, key, payload):
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=json.dumps(payload, separators=(",", ":")),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    return conn.getresponse()


def nonstream(host, port, key, prompt, model):
    conn = http.client.HTTPConnection(host, port, timeout=600)
    started = time.perf_counter()
    response = request(
        conn,
        key,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": False,
            "temperature": 0,
            "max_tokens": 1,
            "stream": False,
        },
    )
    raw = response.read()
    elapsed = time.perf_counter() - started
    conn.close()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {raw[:300]!r}")
    return elapsed, json.loads(raw)["usage"]


def stream(host, port, key, prompt, model):
    conn = http.client.HTTPConnection(host, port, timeout=600)
    started = time.perf_counter()
    response = request(
        conn,
        key,
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": False,
            "temperature": 0,
            "max_tokens": 128,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:300]!r}")
    first = last = usage = None
    for line in response:
        if not line.startswith(b"data: "):
            continue
        data = line[6:].strip()
        if data == b"[DONE]":
            break
        event = json.loads(data)
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            if delta.get("content") or delta.get("reasoning_content"):
                now = time.perf_counter()
                first = first or now
                last = now
    ended = time.perf_counter()
    conn.close()
    if not first or not last or not usage:
        raise RuntimeError("stream ended without text or usage")
    return {
        "ttft_s": first - started,
        "decode_s": ended - first,
        "completion_tokens": usage["completion_tokens"],
        "prompt_tokens": usage["prompt_tokens"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:18199")
    parser.add_argument("--api-key", default=os.getenv("YFORGE_API_KEY", ""))
    parser.add_argument("--model", default="ornith-1.5-9b")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    if not args.api_key:
        parser.error("--api-key or YFORGE_API_KEY is required")
    parsed = urlparse(args.url)
    host, port = parsed.hostname, parsed.port or 80
    prompt = ("alpha " * 477) + "\nIgnore the repeated text. Count from 1 to 120 separated by commas. Do not explain."

    nonstream(host, port, args.api_key, prompt, args.model)  # graph capture / warmup
    prefill = []
    for _ in range(args.runs):
        elapsed, usage = nonstream(host, port, args.api_key, prompt, args.model)
        prefill.append(usage["prompt_tokens"] / elapsed)
    generation = []
    for _ in range(args.runs):
        result = stream(host, port, args.api_key, prompt, args.model)
        generation.append((result["completion_tokens"] - 1) / result["decode_s"])

    print(json.dumps({
        "prefill_tok_s": statistics.median(prefill),
        "generation_tok_s": statistics.median(generation),
        "prefill_runs": prefill,
        "generation_runs": generation,
    }, indent=2))


if __name__ == "__main__":
    main()
