#!/usr/bin/env python3
"""Ornith-1.5-35B-A3B Q8_0 — A/B приёмка для yforge (MoE-путь).

Один вариант = набор env + один прогон. Печатает TTFT, префил t/s,
декод t/s и пиковую дельту VRAM относительно старта.
"""
import json, os, random, re, statistics, subprocess, sys, time, urllib.request, urllib.error

PORT = int(os.environ.get("PORT", "18820"))
GGUF = "/root/models/Ornith-1.5-35B-Q8_0.gguf"
MODEL_NAME = "ornith-1.5-35b"
CTX = int(os.environ.get("CTX", "32768"))
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "16384"))
MAXTOK = int(os.environ.get("MAXTOK", "48"))
REPS = int(os.environ.get("REPS", "3"))
VARIANT = os.environ.get("VARIANT", "base")
EXTRA = json.loads(os.environ.get("EXTRA", "{}"))

LOG = f"/root/logs/ab-{VARIANT}-{CTX}.log"


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return int(o.strip().splitlines()[0])


def free_port():
    o = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    for line in o.splitlines():
        if f":{PORT}" in line:
            for t in line.split():
                if t.startswith("pid="):
                    subprocess.run(["kill", "-9", t[4:].split(",")[0]], capture_output=True)
    time.sleep(3)


def wait_free():
    for _ in range(400):
        if vram() <= 400:
            return
        time.sleep(3)


def post(c, mt):
    b = json.dumps({"model": MODEL_NAME if ENGINE == "yforge" else "m", "stream": False, "max_tokens": mt, "temperature": 0,
                    "messages": [{"role": "user", "content": c}]}).encode()
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=b,
                               headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    try:
        with urllib.request.urlopen(r, timeout=3600) as resp:
            return resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.read().decode()
    except Exception as e:
        return json.dumps({"error": str(e)})


def stream(c, mt):
    b = json.dumps({"model": MODEL_NAME if ENGINE == "yforge" else "m", "stream": True, "max_tokens": mt, "temperature": 0,
                    "stream_options": {"include_usage": True},
                    "messages": [{"role": "user", "content": c}]}).encode()
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=b,
                               headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
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
                for cch in d.get("choices", []):
                    dl = cch.get("delta") or {}
                    if dl.get("content") or dl.get("reasoning_content"):
                        n = time.time() - t0
                        if ttft is None:
                            ttft = n
                        tlast = n; ch += 1
    except Exception as e:
        return None, None, 0, 0, str(e)[:200], 0
    return ttft, tlast, ch, ptok, None, ctok


ENGINE = os.environ.get("ENGINE", "yforge")


def launch():
    env = dict(os.environ)
    if ENGINE == "llamacpp":
        env.pop("EXTRA", None)
        cmd = ["/root/llama.cpp/build/bin/llama-server", "-m", GGUF, "-c", str(CTX),
               "-ngl", "999", "--host", "127.0.0.1", "--port", str(PORT),
               "--parallel", "1", "--api-key", "x", "--no-warmup", "-fa", "on",
               "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
        log = open(LOG, "wb")
        return subprocess.Popen(cmd, cwd="/root", stdout=log,
                                stderr=subprocess.STDOUT, env=env), log
    env.update({"MODEL": GGUF, "CTX": str(CTX), "CONTEXT_LIMIT": str(CTX), "SLOTS": "1",
                "PORT": str(PORT), "API_KEYS": '[{"key":"x","name":"m"}]', "GPU_ONLY": "1",
                "GPU_LAYERS": "999", "KV_POOL_Q8": "1", "KV_CACHE_TYPE": "q8",
                "CUDA_GRAPHS": "1", "PGRAPH": "on", "FLASH_ATTN": "1",
                "PREFIX_CACHE_MIB": "0", "SAMPLING_LOCK": "1", "RUST_LOG": "info",
                "MOE_EXPERTS": "auto"})
    env.update(EXTRA)
    log = open(LOG, "wb")
    return subprocess.Popen(["/root/target/release/yforge"], cwd="/root", stdout=log,
                            stderr=subprocess.STDOUT, env=env), log


def main():
    free_port(); wait_free()
    base = vram()
    proc, log = launch()
    ready = False
    for _ in range(1200):
        if proc.poll() is not None:
            break
        if '"choices"' in post("ping", 4):
            ready = True; break
        time.sleep(3)
    if not ready:
        tail = open(LOG, errors="replace").read()[-500:].replace("\n", " | ")
        print(f"[{VARIANT}] ctx={CTX} НЕ ПОДНЯЛСЯ: {tail}", flush=True)
        proc.kill(); proc.wait(); return 1

    ratio = 6.0
    ptok = 0
    c = None
    for _ in range(4):
        words = int(PROMPT_TOKENS / ratio)
        c = f"Session {random.randint(1, 10**9)}. " + " ".join(f"w{w}" for w in range(words)) + " Summarize."
        ttft, tlast, ch, _, err, ctok = stream(c, MAXTOK)
        if err is None and ttft is not None:
            break
        m = re.search(r"(\d+) tokens", err or "")
        if m:
            ratio = int(m.group(1)) / max(1, words); continue
        print(f"[{VARIANT}] ОШИБКА {(err or '')[:180]}", flush=True)
        proc.kill(); proc.wait(); return 1
    m = re.search(r'"prompt_tokens":\s*(\d+)', post(c, 1))
    if m:
        ptok = int(m.group(1))

    def fresh_prompt():
        # Уникальный маркер В НАЧАЛЕ: общий префикс у llama.cpp обрывается
        # сразу, и prompt cache не может дать фальшивый TTFT.
        words = int(PROMPT_TOKENS / 6.0)
        salt = random.randint(1, 10 ** 9)
        body = " ".join(f"w{w}" for w in range(words))
        return f"Session {salt}. Marker {salt}. " + body + " Summarize." 

    # Свежий промпт на каждый повтор: у llama.cpp включён prompt cache, и при
    # повторном одинаковом промпте его префилл почти ничего не считает —
    # сравнение на таком прогоне бессмысленно (замер 24k t/s против реальных 4.9k).
    pfs, decs, ttfts = [], [], []
    for rep in range(REPS):
        cr = fresh_prompt()
        ttft, tlast, ch, pt, err, ctok = stream(cr, MAXTOK)
        if err or ttft is None:
            print(f"[{VARIANT}] rep={rep}: ttft={ttft} chunks={ch} err={(err or '')[:160]}",
                  flush=True)
            continue
        ngen = ctok or ch
        span = max(1e-9, tlast - ttft)
        decs.append((ngen - 1) / span if ngen > 1 else 0.0)
        pfs.append(pt / ttft)
        ttfts.append(ttft)
    time.sleep(2)
    peak = vram() - base
    if pfs:
        pf = statistics.median(pfs[1:] if len(pfs) > 1 else pfs)
        dc = statistics.median(decs[1:] if len(decs) > 1 else decs)
        tt = statistics.median(ttfts[1:] if len(ttfts) > 1 else ttfts)
    else:
        pf = dc = tt = 0.0
    print(f"[{VARIANT}] ctx={CTX} prompt={ptok} TTFT={tt:6.2f}s prefill={pf:7.0f}t/s "
          f"decode={dc:6.1f}t/s dVRAM={peak}MB", flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
