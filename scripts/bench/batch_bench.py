#!/usr/bin/env python3
"""Агрегатный throughput: N параллельных запросов на 8K/32K, yforge vs llama.cpp."""
import json, os, random, re, subprocess, sys, threading, time, urllib.request, urllib.error

PORT = 18230
URL = f"http://localhost:{PORT}/v1/chat/completions"
GGUF = "/root/models/Qwen3.8-27B-Q8_0.gguf"
MODEL_NAME = "m"
CONC = int(os.environ.get("CONC", "4"))
GEN = int(os.environ.get("GEN", "65"))

def vram():
    o=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    return int(o.strip().splitlines()[0])

def wait_free():
    for _ in range(120):
        if vram()<=2000: return
        time.sleep(3)

def free_port():
    o=subprocess.run(["ss","-tlnp"],capture_output=True,text=True).stdout
    for line in o.splitlines():
        if f":{PORT}" in line:
            for t in line.split():
                if t.startswith("pid="):
                    subprocess.run(["kill","-9",t[4:].split(",")[0]],capture_output=True)
    time.sleep(3)

def launch(engine, ctx, slots):
    env=dict(os.environ); log=open(f"/root/logs/batch-{engine}-{ctx}.log","wb")
    if engine=="yforge":
        env.update({"MODEL":GGUF,"CTX":str(ctx),"CONTEXT_LIMIT":str(ctx),"SLOTS":str(slots),"PORT":str(PORT),
                    "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999",
                    "KV_POOL_Q8":"1","KV_CACHE_TYPE":"q8","CUDA_GRAPHS":"1","PGRAPH":"on",
                    "FLASH_ATTN":"1","PREFIX_CACHE_MIB":"0","SAMPLING_LOCK":"1","RUST_LOG":"info"})
        cmd=["/root/target/release/yforge"]
    else:
        cmd=["/root/llama.cpp/build/bin/llama-server","-m",GGUF,"-c",str(ctx*slots),"-ngl","999",
             "--host","127.0.0.1","--port",str(PORT),"--parallel",str(slots),"--api-key","x",
             "--no-warmup","-fa","on","--cache-type-k","q8_0","--cache-type-v","q8_0"]
    return subprocess.Popen(cmd,cwd="/root",stdout=log,stderr=subprocess.STDOUT,env=env), log

def post(c, mt):
    b=json.dumps({"model":MODEL_NAME,"stream":False,"max_tokens":mt,"temperature":0,"messages":[{"role":"user","content":c}]}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    try:
        with urllib.request.urlopen(r,timeout=1800) as resp: return resp.read().decode()
    except urllib.error.HTTPError as e: return e.read().decode()
    except Exception as e: return f'{{"error":"{e}"}}'

def one(prompt, res, i):
    t0=time.time()
    body=json.dumps({"model":MODEL_NAME,"stream":True,"max_tokens":GEN,"temperature":0,
                     "stream_options":{"include_usage":True},"messages":[{"role":"user","content":prompt}]}).encode()
    r=urllib.request.Request(URL,data=body,headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    ttft=None; tlast=None; ctok=0
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
                for c in d.get("choices",[]):
                    dl=c.get("delta") or {}
                    if dl.get("content") or dl.get("reasoning_content"):
                        n=time.time()-t0
                        if ttft is None: ttft=n
                        tlast=n
    except Exception as e:
        res[i]=(None,None,0,str(e)[:120]); return
    res[i]=(ttft,tlast,ctok,None)

def run(engine, ctx, slots):
    global MODEL_NAME
    MODEL_NAME = "m"
    free_port(); wait_free(); base=vram()
    proc,log=launch(engine,ctx,slots)
    ready=False
    for _ in range(600):
        if proc.poll() is not None: break
        probe=post("ping",4)
        if '"choices"' in probe: ready=True; break
        if "model_not_loaded" in probe:
            req=urllib.request.Request(f"http://localhost:{PORT}/v1/models",headers={"Authorization":"Bearer x"})
            try:
                d=json.loads(urllib.request.urlopen(req,timeout=30).read())
                MODEL_NAME=d["data"][0]["id"]
                probe=post("ping",4)
                if '"choices"' in probe: ready=True; break
            except Exception: pass
        time.sleep(3)
    if not ready:
        print(f"{engine} ctx={ctx} slots={slots} НЕ ПОДНЯЛСЯ: {open(log.name,errors='replace').read()[-300:]}",flush=True)
        proc.kill(); proc.wait(); wait_free(); return
    print(f"{engine:10} ctx={ctx:>7} slots={slots} load={time.time()-base:.0f}s free={vram()<=2000}",flush=True)
    target=int(ctx*0.6)
    words=int(target/6)
    prompts=[f"Session {random.randint(1,10**9)}. "+" ".join(f"w{j}" for j in range(words))+" Summarize briefly." for _ in range(CONC)]
    res=[None]*CONC
    t0=time.time()
    ths=[threading.Thread(target=one,args=(prompts[i],res,i)) for i in range(CONC)]
    for t in ths: t.start()
    for t in ths: t.join()
    wall=time.time()-t0
    ok=[r for r in res if r and r[2]]
    agg=sum(r[2] for r in ok)/wall if wall>0 else 0
    ttfts=[r[0] for r in ok if r[0]]
    print(f"{engine:10} ctx={ctx:>7} slots={slots} wall={wall:.1f}s agg={agg:.1f}t/s "
          f"per_req_tps={[round((r[2]-1)/(r[1]-r[0]),1) if r[1] and r[0] else 0 for r in ok]} "
          f"ttft={[round(x,2) for x in ttfts]} vram={vram()-base}MB",flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()

for engine in ["yforge","llamacpp"]:
    for ctx in [int(x) for x in os.environ.get("CTXS","8192,32768").split(",")]:
        run(engine, ctx, CONC)
print("DONE")
