#!/usr/bin/env python3
"""A/B сэмплера: один и тот же k прогоняется и последовательно, и параллельно."""
import json, os, random, statistics, subprocess, sys, time, urllib.request

PORT = int(os.environ.get("PORT", "18820"))
GGUF = "/root/models/Ornith-1.5-35B-Q8_0.gguf"
CTX = int(os.environ.get("CTX", "8192"))
PT = int(os.environ.get("PROMPT_TOKENS", "4096"))
MT = int(os.environ.get("MAXTOK", "96"))
REPS = int(os.environ.get("REPS", "2"))
EXTRA = json.loads(os.environ.get("EXTRA", "{}"))

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
def wait_free():
    for _ in range(400):
        if vram()<=400: return
        time.sleep(3)
def gen(c,mt):
    b=json.dumps({"model":"ornith-1.5-35b","max_tokens":mt,"temperature":0,
                  "messages":[{"role":"user","content":c}]}).encode()
    r=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=b,
        headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    return urllib.request.urlopen(r,timeout=3600).read().decode()
def stream(c,mt):
    b=json.dumps({"model":"ornith-1.5-35b","stream":True,"max_tokens":mt,"temperature":0,
                  "stream_options":{"include_usage":True},"messages":[{"role":"user","content":c}]}).encode()
    r=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=b,
        headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    t0=time.time(); ttft=tlast=None; ch=0; ctok=0
    with urllib.request.urlopen(r,timeout=3600) as resp:
        for raw in resp:
            line=raw.decode("utf-8","replace").strip()
            if not line.startswith("data:"): continue
            p=line[5:].strip()
            if p=="[DONE]": break
            try: d=json.loads(p)
            except ValueError: continue
            u=d.get("usage")
            if u: ctok=u.get("completion_tokens") or ctok
            for cch in d.get("choices",[]):
                dl=cch.get("delta") or {}
                if dl.get("content") or dl.get("reasoning_content"):
                    n=time.time()-t0
                    if ttft is None: ttft=n
                    tlast=n; ch+=1
    return ttft,tlast,ch,ctok
def run(k):
    env=dict(os.environ)
    env.update({"MODEL":GGUF,"CTX":str(CTX),"CONTEXT_LIMIT":str(CTX),"SLOTS":"1","PORT":str(PORT),
        "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999","KV_POOL_Q8":"1",
        "KV_CACHE_TYPE":"q8","CUDA_GRAPHS":"1","PGRAPH":"on","FLASH_ATTN":"1","PREFIX_CACHE_MIB":"0",
        "SAMPLING_LOCK":"0","RUST_LOG":"error","MOE_EXPERTS":"auto"})
    env.update(EXTRA)
    log=open(f"/root/logs/ab-kth-{k}.log","wb")
    p=subprocess.Popen(["/root/target/release/yforge"],cwd="/root",stdout=log,stderr=subprocess.STDOUT,env=env)
    for _ in range(400):
        try: gen("ping",1); break
        except Exception: time.sleep(3)
    else:
        print(f"k={k}: NOT READY"); p.kill(); return
    ds=[]
    for _ in range(REPS):
        salt=random.randint(1,10**9); words=int(PT/6.0)
        body=" ".join(f"w{w}" for w in range(words))
        ttft,tlast,ch,ctok=stream(f"Session {salt}. Marker {salt}. {body} Summarize.",MT)
        n=ctok or ch
        ds.append((n-1)/max(1e-9,tlast-ttft))
    p.kill(); p.wait(); free_port(); wait_free()
    print(f"[kth k={k:4d}] decode={statistics.median(ds):6.1f} t/s")

for k in (1,8,20,64,128,256,1024):
    run(k)
