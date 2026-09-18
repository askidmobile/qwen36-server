#!/usr/bin/env python3
"""Сравнение yforge и llama.cpp на RTX 4090 48GB, Qwen3.8-27B Q8_0.

Метрики: TTFT (префил), prefill tok/s, decode tok/s, пик VRAM, агрегатный
throughput на N слотов, MTP-специфика. Один протокол, один харнесс.
"""
import json, os, re, random, subprocess, sys, time, urllib.request, urllib.error

PORT = 18210
URL = f"http://localhost:{PORT}/v1/chat/completions"
GGUF = "/root/models/Qwen3.8-27B-Q8_0.gguf"
API_KEY = "x"
MODEL_ID = "m"
CTXS = [int(x) for x in os.environ.get("CTXS", "8192,32768,131072").split(",")]
SLOTS = int(os.environ.get("SLOTS", "1"))
TOKENS = int(os.environ.get("GEN", "65"))
REPEATS = int(os.environ.get("REPEATS", "2"))


def vram():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def wait_gpu_free(limit=2000, tries=120):
    for _ in range(tries):
        if vram() <= limit:
            return True
        time.sleep(3)
    return False


def free_port():
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f":{PORT}" in line:
            for tok in line.split():
                if tok.startswith("pid="):
                    subprocess.run(["kill", "-9", tok[4:].split(",")[0]],
                                   capture_output=True)
    for _ in range(30):
        out = subprocess.run(["ss", "-tln"], capture_output=True, text=True).stdout
        if f":{PORT}" not in out:
            return True
        time.sleep(2)
    return False


def launch(engine, ctx):
    env = dict(os.environ)
    log_path = f"/root/logs/cmp-{engine}-{ctx}.log"
    log = open(log_path, "wb")
    if engine == "yforge":
        env.update({
            "MODEL": GGUF, "CTX": str(ctx), "SLOTS": str(SLOTS), "PORT": str(PORT),
            "API_KEYS": '[{"key":"x","name":"m"}]',
            "GPU_ONLY": "1", "GPU_LAYERS": "999",
            "CONTEXT_LIMIT": str(ctx),
            "QK_INT8_PREFILL": os.environ.get("QK_INT8_PREFILL", "1"),
            "KV_POOL_Q8": os.environ.get("KV_POOL_Q8", "1"),
            "KV_CACHE_TYPE": os.environ.get("KV_CACHE_TYPE", "q8"),
            "CUDA_GRAPHS": "1", "PGRAPH": "on", "FLASH_ATTN": "1",
            "PREFIX_CACHE_MIB": "0",
            "SAMPLING_LOCK": "1",
            "RUST_LOG": "info",
        })
        cmd = ["/root/target/release/yforge"]
    elif engine == "llamacpp":
        cmd = ["/root/llama.cpp/build/bin/llama-server", "-m", GGUF,
               "-c", str(ctx), "-ngl", "999", "--host", "127.0.0.1",
               "--port", str(PORT), "--parallel", str(SLOTS), "--api-key", "x",
               "--no-warmup", "-fa", "on", "--cache-type-k", "q8_0",
               "--cache-type-v", "q8_0"]
    else:
        raise SystemExit(f"неизвестный движок: {engine}")
    return subprocess.Popen(cmd, cwd="/root", stdout=log, stderr=subprocess.STDOUT,
                            env=env), log_path


def discover_model():
    global MODEL_ID
    req = urllib.request.Request(f"http://localhost:{PORT}/v1/models",
                                 headers={"Authorization": f"Bearer {API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read().decode())
        data = d.get("data") or []
        if data:
            MODEL_ID = data[0]["id"]
    except Exception as e:
        print(f"model discovery failed: {e}", flush=True)
    print(f"model_id={MODEL_ID}", flush=True)


def post(content, max_tokens, timeout=1800):
    body = json.dumps({"model": MODEL_ID, "stream": False, "max_tokens": max_tokens,
                       "temperature": 0,
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode()
    except urllib.error.HTTPError as e:
        return e.read().decode()
    except Exception as e:
        return f'{{"error":"{e}"}}'


def stream(content, max_tokens, timeout=1800):
    body = json.dumps({"model": MODEL_ID, "stream": True, "max_tokens": max_tokens,
                       "temperature": 0, "stream_options": {"include_usage": True},
                       "messages": [{"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(URL, data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"})
    t0 = time.time(); ttft = tlast = None; nchunk = 0; ptok = ctok = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
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
                if u and u.get("prompt_tokens"):
                    ptok = u["prompt_tokens"]
                if u and u.get("completion_tokens"):
                    ctok = u["completion_tokens"]
                for ch in d.get("choices", []):
                    dl = ch.get("delta") or {}
                    if dl.get("content") or dl.get("reasoning_content"):
                        now = time.time() - t0
                        if ttft is None:
                            ttft = now
                        tlast = now; nchunk += 1
    except Exception as e:
        return None, None, 0, 0, str(e)[:200], 0
    return ttft, tlast, nchunk, ptok, None, ctok


def make_prompt(target_tokens, ratio=6.0):
    words = max(4, int(target_tokens / ratio))
    return (f"Session {random.randint(1, 10**9)}. "
            + " ".join(f"word{w}" for w in range(words))
            + " Summarize the pattern in one sentence.")


def run_engine(engine):
    base = vram()
    print(f"\n##### {engine} #####", flush=True)
    print(f"слотов: {SLOTS}, фон VRAM: {base} MB", flush=True)
    print(f"{'ctx':>8} {'tokens':>7} {'load_s':>7} {'vram':>7} {'ttft_s':>8} "
          f"{'pf_tok_s':>9} {'dec_tok_s':>9} {'gen':>5} {'chunks':>6} "
          f"{'pf_s':>6} {'dec_s':>6}", flush=True)
    for ctx in CTXS:
        free_port(); wait_gpu_free()
        t0 = time.time()
        proc, log_path = launch(engine, ctx)
        ready = False
        for _ in range(600):
            if proc.poll() is not None:
                break
            probe = post("ping", 4, timeout=900)
            if "model_not_loaded" in probe:
                discover_model()
                probe = post("ping", 4, timeout=900)
            if '"choices"' in probe:
                ready = True; break
            time.sleep(3)
        if not ready:
            tail = open(log_path, errors="replace").read()[-400:].replace("\n", " | ")
            print(f"{ctx:>8} НЕ ПОДНЯЛСЯ: {tail}", flush=True)
            proc.kill(); proc.wait(); wait_gpu_free(); continue
        load = round(time.time() - t0, 1)
        target = int(ctx * 0.85)
        ptok = 0; ttft = tlast = None; nchunk = 0; ctok = 0; ratio = 6.0
        for _ in range(4):
            content = make_prompt(target, ratio)
            ttft, tlast, nchunk, _, err, ctok = stream(content, TOKENS)
            if err is None and ttft is not None:
                m = re.search(r'"prompt_tokens":\s*(\d+)', post(content, 1))
                if m:
                    ptok = int(m.group(1)); break
            om = re.search(r"(\d+) tokens", err or "")
            if om:
                ratio = int(om.group(1)) / max(1, int(target / ratio)); continue
            print(f"{ctx:>8} ЗАПРОС УПАЛ: {(err or 'нет usage')[:150]}", flush=True)
            ptok = 0; break
        if not ptok or ttft is None:
            proc.kill(); proc.wait(); wait_gpu_free(); continue
        peak = vram()
        span = max(1e-9, tlast - ttft)
        ngen = ctok if ctok else nchunk
        dec = round((ngen - 1) / span, 1) if ngen > 1 else 0
        pf = round(ptok / ttft) if ttft > 0 else 0
        print(f"{ctx:>8} {ptok:>7} {load:>7} {peak - base:>7} {round(ttft,2):>8} "
              f"{pf:>9} {dec:>9} {ngen:>5} {nchunk:>6} "
              f"{round(ttft,2):>6} {round(span,2):>6}", flush=True)
        proc.kill(); proc.wait(); free_port(); wait_gpu_free()


for eng in sys.argv[1:] or ["yforge", "llamacpp"]:
    run_engine(eng)
print("\nDONE")
