#!/usr/bin/env python3
"""Внутрипроцессный детерминизм: один сервер, N одинаковых запросов подряд.
Отличает нестабильность ядра от нестабильности инициализации/загрузки."""
import json, os, subprocess, time, urllib.request, hashlib

PORT = int(os.environ.get("PORT", "18854"))
GGUF = "/root/models/Ornith-1.5-35B-Q8_0.gguf"
CTX = 32768
EXTRA = json.loads(os.environ.get("EXTRA", "{}"))
PROMPT = ("Session 313131. Marker 313131. " + " ".join(f"w{i}" for i in range(500))
          + " List the first five primes in order.")

def vram():
    o=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    return int(o.strip().splitlines()[0])

def free_port():
    o=subprocess.run(["ss","-tlnp"],capture_output=True,text=True).stdout
    for line in o.splitlines():
        if f":{PORT}" in line:
            for t in line.split():
                if t.startswith("pid="): subprocess.run(["kill","-9",t[4:].split(",")[0]],capture_output=True)
    time.sleep(3)

def gen(mt, cache_bust=""):
    p = cache_bust + PROMPT if cache_bust else PROMPT
    b=json.dumps({"model":"ornith-1.5-35b","max_tokens":mt,"temperature":0,
                  "messages":[{"role":"user","content":p}]}).encode()
    r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=b,
        headers={"Authorization":"Bearer x","Content-Type":"application/json"})
    with urllib.request.urlopen(r,timeout=3600) as resp: return json.loads(resp.read())

free_port()
for _ in range(200):
    if vram()<=400: break
    time.sleep(3)

env=dict(os.environ)
env.update({"MODEL":GGUF,"CTX":str(CTX),"CONTEXT_LIMIT":str(CTX),"SLOTS":"1","PORT":str(PORT),
    "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999","KV_POOL_Q8":"1",
    "KV_CACHE_TYPE":"q8","CUDA_GRAPHS":"1","PGRAPH":"on","FLASH_ATTN":"1","PREFIX_CACHE_MIB":"0",
    "SAMPLING_LOCK":"0","TEMPERATURE":"0","RUST_LOG":"error","MOE_EXPERTS":"auto","PREFILL_CHUNK":"4096"})
env.update(EXTRA)
log=open("/root/logs/det.log","wb")
p=subprocess.Popen(["/root/target/release/yforge"],cwd="/root",stdout=log,stderr=subprocess.STDOUT,env=env)
for _ in range(400):
    try: gen(1); break
    except Exception: time.sleep(3)
else:
    print("NOT READY"); p.kill(); raise SystemExit(1)

print("=== один процесс, один и тот же промпт 20 раз (temp=0) ===")
shas=[]
for i in range(20):
    d=gen(24)
    t=d["choices"][0]["message"]["content"]
    h=hashlib.sha256(t.encode()).hexdigest()[:12]
    shas.append(h)
    print(f"  run{i}: len={len(t)} sha={h}")
from collections import Counter
c=Counter(shas)
print("внутрипроцессно:", f"{len(c)} уникальных из {len(shas)}: {dict(c)}" if len(c)>1 else "IDENTICAL")

# тот же промпт, но через 30 с — исключаем прогрев
time.sleep(30)
shas2=[]
for i in range(3):
    d=gen(24)
    t=d["choices"][0]["message"]["content"]
    shas2.append(hashlib.sha256(t.encode()).hexdigest()[:12])
print("через 30 с:", "IDENTICAL" if len(set(shas2))==1 else f"DIFFER ({len(set(shas2))} уникальных)")
p.kill(); p.wait()
