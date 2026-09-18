#!/usr/bin/env python3
"""Пик VRAM при долгой генерации + разложение: load / после префила / пик декода."""
import json, os, random, re, subprocess, threading, time, urllib.request, urllib.error
PORT=18390
GGUF="/root/models/Qwen3.8-27B-Q8_0.gguf"; MODEL="m"
CTX=int(os.environ.get("CTX","8192")); GEN=int(os.environ.get("GEN","512"))
peak={"v":0}; stop={"v":False}

def vram():
    o=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    return int(o.strip().splitlines()[0])
def sampler():
    while not stop["v"]:
        try:
            v=vram()
            if v>peak["v"]: peak["v"]=v
        except Exception: pass
        time.sleep(0.25)
def wait_free():
    for _ in range(180):
        if vram()<=300: return
        time.sleep(3)
def free_port():
    o=subprocess.run(["ss","-tlnp"],capture_output=True,text=True).stdout
    for line in o.splitlines():
        if f":{PORT}" in line:
            for t in line.split():
                if t.startswith("pid="): subprocess.run(["kill","-9",t[4:].split(",")[0]],capture_output=True)
    time.sleep(3)
def launch(engine):
    env=dict(os.environ); log=open(f"/root/logs/pk-{engine}.log","wb")
    if engine=="yforge":
        env.update({"MODEL":GGUF,"CTX":str(CTX),"CONTEXT_LIMIT":str(CTX),"SLOTS":"1","PORT":str(PORT),
                    "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999",
                    "KV_POOL_Q8":"1","KV_CACHE_DTYPE":"q8","CUDA_GRAPHS":"1","PGRAPH":"on",
                    "FLASH_ATTN":"1","PREFIX_CACHE_MIB":"0","RUST_LOG":"info"})
        cmd=["/root/target/release/yforge"]
    else:
        cmd=["/root/llama.cpp/build/bin/llama-server","-m",GGUF,"-c",str(CTX),"-ngl","999",
             "--host","127.0.0.1","--port",str(PORT),"--parallel","1","--api-key","x","--no-warmup","-fa","on",
             "--cache-type-k","q8_0","--cache-type-v","q8_0"]
    return subprocess.Popen(cmd,cwd="/root",stdout=log,stderr=subprocess.STDOUT,env=env), log
def post(c, mt):
    b=json.dumps({"model":MODEL,"stream":False,"max_tokens":mt,"temperature":0,"messages":[{"role":"user","content":c}]}).encode()
    for _ in range(60):
        r=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=b,
                                 headers={"Content-Type":"application/json","Authorization":"Bearer x"})
        try:
            with urllib.request.urlopen(r,timeout=1800) as resp: return resp.read().decode()
        except urllib.error.HTTPError as e:
            body=e.read().decode()
            if "still loading" in body: time.sleep(5); continue
            return body
        except Exception as e: return f'{{"error":"{e}"}}'
    return '{"error":"loading"}'
def stream(c, mt):
    b=json.dumps({"model":MODEL,"stream":True,"max_tokens":mt,"temperature":0,
                  "stream_options":{"include_usage":True},"messages":[{"role":"user","content":c}]}).encode()
    t0=time.time(); ttft=tlast=None; ctok=0
    for _ in range(60):
        r=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=b,
                                 headers={"Content-Type":"application/json","Authorization":"Bearer x"})
        try:
            with urllib.request.urlopen(r,timeout=1800) as resp:
                for raw in resp:
                    line=raw.decode("utf-8","replace").strip()
                    if not line.startswith("data:"): continue
                    p=line[5:].strip()
                    if p=="[DONE]": break
                    try: d=json.loads(p)
                    except ValueError: continue
                    u=d.get("usage")
                    if u and u.get("completion_tokens"): ctok=u["completion_tokens"]
                    for cch in d.get("choices",[]):
                        dl=cch.get("delta") or {}
                        if dl.get("content") or dl.get("reasoning_content"):
                            n=time.time()-t0
                            if ttft is None: ttft=n
                            tlast=n
                return ttft,tlast,ctok,None
        except urllib.error.HTTPError as e:
            body=e.read().decode()
            if "still loading" in body: time.sleep(5); continue
            return None,None,0,body[:120]
        except Exception as e: return None,None,0,str(e)[:120]
    return None,None,0,"loading"
def run(engine):
    global MODEL
    MODEL="m"; free_port(); wait_free(); base=vram()
    proc,log=launch(engine)
    ok=False
    for _ in range(600):
        if proc.poll() is not None: break
        r=urllib.request.Request(f"http://localhost:{PORT}/v1/models",headers={"Authorization":"Bearer x"})
        try:
            with urllib.request.urlopen(r,timeout=20) as resp: d=json.loads(resp.read())
            if d.get("data"): MODEL=d["data"][0]["id"]; ok=True; break
        except Exception: pass
        time.sleep(3)
    if not ok: print(f"{engine}: НЕ ПОДНЯЛСЯ",flush=True); proc.kill(); proc.wait(); wait_free(); return
    time.sleep(2); load=vram()
    peak["v"]=0; stop["v"]=False
    th=threading.Thread(target=sampler,daemon=True); th.start()
    target=int(CTX*0.85); words=int(target/6)
    c=f"Session {random.randint(1,10**9)}. "+" ".join(f"word{w}" for w in range(words))+" Count from 1 to 300 slowly, one number per line."
    ttft,tlast,ctok,err=stream(c,GEN)
    time.sleep(2); stop["v"]=True; th.join(timeout=3)
    after=vram()
    dec=(ctok-1)/max(1e-9,tlast-ttft) if ctok and ctok>1 and ttft else 0
    print(f"{engine:9} ctx={CTX} load={load-base:>6} peak={peak['v']-base:>6} after={after-base:>6} "
          f"ttft={ttft if ttft else -1:6.2f}s dec={dec:5.1f}t/s gen={ctok} err={err}",flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()
run("llamacpp"); run("yforge")
print("DONE")
