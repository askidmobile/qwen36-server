#!/usr/bin/env python3
"""Финальная сверка в ПРОДОВОЙ конфигурации: окно 256K у обоих движков,
линейка промптов 0 / 8K / 32K / 128K / 256K. Все метрики."""
import json, os, random, re, subprocess, time, urllib.request, urllib.error
PORT=18690; GGUF="/root/models/Qwen3.8-27B-Q8_0.gguf"; MODEL="m"
CTX=262144
CASES=[0, 8192, 32768, 131072, 262144]
def vram():
    o=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    return int(o.strip().splitlines()[0])
def wait_free():
    for _ in range(300):
        if vram()<=300: return
        time.sleep(3)
def free_port():
    o=subprocess.run(["ss","-tlnp"],capture_output=True,text=True).stdout
    for line in o.splitlines():
        if f":{PORT}" in line:
            for t in line.split():
                if t.startswith("pid="): subprocess.run(["kill","-9",t[4:].split(",")[0]],capture_output=True)
    time.sleep(3)
def chat(c, mt):
    b=json.dumps({"model":MODEL,"stream":True,"max_tokens":mt,"temperature":0,
                  "stream_options":{"include_usage":True},"messages":[{"role":"user","content":c}]}).encode()
    t0=time.time(); ttft=tlast=None; ctok=0; ptok=0
    for _ in range(90):
        r=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=b,
                                 headers={"Content-Type":"application/json","Authorization":"Bearer x"})
        try:
            with urllib.request.urlopen(r,timeout=2400) as resp:
                for raw in resp:
                    line=raw.decode("utf-8","replace").strip()
                    if not line.startswith("data:"): continue
                    p=line[5:].strip()
                    if p=="[DONE]": break
                    try: d=json.loads(p)
                    except ValueError: continue
                    u=d.get("usage")
                    if u:
                        if u.get("completion_tokens"): ctok=u["completion_tokens"]
                        if u.get("prompt_tokens"): ptok=u["prompt_tokens"]
                    for cch in d.get("choices",[]):
                        dl=cch.get("delta") or {}
                        if dl.get("content") or dl.get("reasoning_content"):
                            n=time.time()-t0
                            if ttft is None: ttft=n
                            tlast=n
                return ttft,tlast,ctok,ptok,None
        except urllib.error.HTTPError as e:
            if "still loading" in e.read().decode(): time.sleep(5); continue
            return None,None,0,0,str(e)[:110]
        except Exception as e: return None,None,0,0,str(e)[:110]
    return None,None,0,0,"timeout"
def run(label, engine):
    global MODEL
    MODEL="m"; free_port(); wait_free(); base=vram()
    env=dict(os.environ); log=open(f"/root/logs/pf-{engine}.log","wb")
    if engine=="yforge":
        env.update({"MODEL":GGUF,"CTX":str(CTX),"CONTEXT_LIMIT":str(CTX),"SLOTS":"1","PORT":str(PORT),
                    "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999",
                    "KV_POOL_Q8":"1","KV_CACHE_TYPE":"q8","CUDA_GRAPHS":"1","PGRAPH":"on",
                    "FLASH_ATTN":"1","PREFIX_CACHE_MIB":"0","RUST_LOG":"info"})
        cmd=["/root/target/release/yforge"]
    else:
        cmd=["/root/llama.cpp/build/bin/llama-server","-m",GGUF,"-c",str(CTX),"-ngl","999",
             "--host","127.0.0.1","--port",str(PORT),"--parallel","1","--api-key","x","--no-warmup","-fa","on",
             "--cache-type-k","q8_0","--cache-type-v","q8_0"]
    proc=subprocess.Popen(cmd,cwd="/root",stdout=log,stderr=subprocess.STDOUT,env=env)
    ok=False
    for _ in range(900):
        if proc.poll() is not None: break
        r=urllib.request.Request(f"http://localhost:{PORT}/v1/models",headers={"Authorization":"Bearer x"})
        try:
            with urllib.request.urlopen(r,timeout=20) as resp: d=json.loads(resp.read())
            if d.get("data"): MODEL=d["data"][0]["id"]; ok=True; break
        except Exception: pass
        time.sleep(3)
    if not ok:
        print(f"{label}: НЕ ПОДНЯЛСЯ",flush=True); proc.kill(); proc.wait(); wait_free(); return
    time.sleep(3)
    print(f"--- {label} ---",flush=True)
    for want in CASES:
        if want == 0:
            c, tag = "hi", "short"
        else:
            words=max(1,int(want/6))
            c=f"Session {random.randint(1,10**9)}. "+" ".join(f"w{w}" for w in range(words))+" Summarize briefly."
            tag=f"{want//1024}K"
        t0=time.time(); ttft,tlast,ctok,ptok,err=chat(c,65); el=time.time()-t0
        time.sleep(3); used=vram()-base
        if err or not ttft:
            print(f"  {tag:>6}: ОШИБКА {err}",flush=True); continue
        dec=(ctok-1)/max(1e-9,tlast-ttft) if ctok and ctok>1 else 0
        pf=ptok/ttft if ttft else 0
        print(f"  {tag:>6}: prompt={ptok:>7} ttft={ttft:8.2f}s pf={pf:6.0f}t/s dec={dec:5.1f}t/s VRAM={used:>6}MB",flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()
run("llama.cpp (окно 256K)","llamacpp")
run("yforge (CTX=256K, прод)","yforge")
print("DONE")
