#!/usr/bin/env python3
"""Метрики vLLM на той же методике, что ornith_ab.py: свежий промпт,
TTFT, префил t/s, декод t/s, VRAM. Сервер уже запущен."""
import json, os, random, re, statistics, subprocess, sys, time, urllib.request, urllib.error

PORT = int(os.environ.get("PORT", "18900"))
MODEL = "ornith"
CTX = int(os.environ.get("CTX", "32768"))
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "16384"))
MAXTOK = int(os.environ.get("MAXTOK", "64"))
REPS = int(os.environ.get("REPS", "4"))
VARIANT = os.environ.get("VARIANT", "vllm")


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(o.strip().splitlines()[0])


def stream(prompt, mt):
    b = json.dumps({"model": MODEL, "stream": True, "max_tokens": mt, "temperature": 0,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": prompt}]}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=b,
                               headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = tlast = None; ch = 0; ctok = 0; ptok = 0
    try:
        with urllib.request.urlopen(r, timeout=3600) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                p = line[5:].strip()
                if p == "[DONE]":
                    break
                try:
                    d = json.loads(p)
                except ValueError:
                    continue
                u = d.get("usage")
                if u:
                    ctok = u.get("completion_tokens") or ctok
                    ptok = u.get("prompt_tokens") or ptok
                for c in d.get("choices", []):
                    dl = c.get("delta") or {}
                    if dl.get("content") or dl.get("reasoning_content"):
                        n = time.time() - t0
                        if ttft is None:
                            ttft = n
                        tlast = n; ch += 1
    except Exception as e:
        return None, None, 0, 0, str(e)[:200], 0
    return ttft, tlast, ch, ptok, None, ctok


def fresh():
    words = int(PROMPT_TOKENS / 6.0)
    salt = random.randint(1, 10 ** 9)
    body = " ".join(f"w{w}" for w in range(words))
    return f"Session {salt}. Marker {salt}. " + body + " Summarize."


def main():
    base = vram()
    # прогрев
    c = fresh()
    ttft, tlast, ch, pt, err, ctok = stream(c, 16)
    if err:
        print("warmup err:", err); return 1
    pfs, decs, ttfts, ptoks = [], [], [], []
    for _ in range(REPS):
        c = fresh()
        ttft, tlast, ch, pt, err, ctok = stream(c, MAXTOK)
        if err or ttft is None:
            print(f"rep err: {err}"); continue
        ngen = ctok or ch
        span = max(1e-9, tlast - ttft)
        decs.append((ngen - 1) / span if ngen > 1 else 0.0)
        pfs.append(pt / ttft); ttfts.append(ttft); ptoks.append(pt)
    peak = vram()
    pf = statistics.median(pfs) if pfs else 0
    dc = statistics.median(decs) if decs else 0
    tt = statistics.median(ttfts) if ttfts else 0
    print("[%s] ctx=%d prompt=%s TTFT=%6.2fs prefill=%7.0ft/s decode=%6.1ft/s VRAM=%dMB"
          % (VARIANT, CTX, int(statistics.median(ptoks)) if ptoks else 0, tt, pf, dc, peak))
    return 0


if __name__ == "__main__":
    sys.exit(main())
