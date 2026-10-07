#!/usr/bin/env python3
"""A/B с ОДИНАКОВЫМ сэмплингом на обеих сторонах: temp=0.6, top_k=20, top_p=0.95."""
import json, os, random, re, statistics, subprocess, sys, time, urllib.request, urllib.error

PORT = int(os.environ.get("PORT", "18820"))
GGUF = "/root/models/Ornith-1.5-35B-Q8_0.gguf"
CTX = int(os.environ.get("CTX", "8192"))
PROMPT_TOKENS = int(os.environ.get("PROMPT_TOKENS", "4096"))
MAXTOK = int(os.environ.get("MAXTOK", "128"))
REPS = int(os.environ.get("REPS", "3"))
VARIANT = os.environ.get("VARIANT", "fair")
EXTRA = json.loads(os.environ.get("EXTRA", "{}"))
ENGINE = os.environ.get("ENGINE", "yforge")
LOG = f"/root/logs/ab-{VARIANT}-{CTX}.log"

def vram():
    o = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    return int(o.strip().splitlines()[0])

def free_port():
    o = subprocess.run(["ss","-tlnp"],capture_output=True,text=True).stdout
    for line in o.splitlines():
        if f":{PORT}" in line:
            for tok in line.split():
                if tok.startswith("pid="):
                    subprocess.run(["kill","-9",tok[4:].split(",")[0]],capture_output=True)
    time.sleep(3)

def wait_free():
    for _ in range(400):
        if vram() <= 400: return
        time.sleep(3)

def post(c, mt):
    b = json.dumps({"model":"ornith-1.5-35b" if ENGINE=="yforge" else "m","stream":False,"max_tokens":mt,
                     "temperature":0.6,"top_k":20,"top_p":0.95,"seed":7,
                     "messages":[{"role":"user","content":c}]}).encode()
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=b,
                               headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    try:
        with urllib.request.urlopen(r,timeout=3600) as resp: return resp.read().decode()
    except Exception as e: return json.dumps({"error":str(e)})

def stream(c, mt):
    b = json.dumps({"model":"ornith-1.5-35b" if ENGINE=="yforge" else "m","stream":True,"max_tokens":mt,
                    "temperature":0.6,"top_k":20,"top_p":0.95,"seed":7,
                    "stream_options":{"include_usage":True},
                    "messages":[{"role":"user","content":c}]}).encode()
    r = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=b,
                               headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    t0=time.time(); ttft=tlast=None; ch=0; ctok=0; ptok=0
    try:
        with urllib.request.urlopen(r,timeout=3600) as resp:
            for raw in resp:
                line=raw.decode("utf-8","replace").strip()
                if not line.startswith("data:"): continue
                p=line[5:].strip()
                if p=="[DONE]": break
                try: d=json.loads(p)
                except ValueError: continue
                u=d.get("usage")
                if u:
                    ctok=u.get("completion_tokens") or ctok
                    ptok=u.get("prompt_tokens") or ptok
                for cch in d.get("choices",[]):
                    dl=cch.get("delta") or {}
                    if dl.get("content") or dl.get("reasoning_content"):
                        n=time.time()-t0
                        if ttft is None: ttft=n
                        tlast=n; ch+=1
    except Exception as e:
        return None,None,0,0,str(e)[:200],0
    return ttft,tlast,ch,ptok,None,ctok

def launch():
    env=dict(os.environ)
    if ENGINE=="llamacpp":
        env.pop("EXTRA",None)
        cmd=["/root/llama.cpp/build/bin/llama-server","-m",GGUF,"-c",str(CTX),"-ngl","999",
             "--host","127.0.0.1","--port",str(PORT),"--parallel","1","--api-key","x","--no-warmup",
             "-fa","on","--cache-type-k","q8_0","--cache-type-v","q8_0"]
        return subprocess.Popen(cmd,cwd="/root",stdout=open(LOG,"wb"),stderr=subprocess.STDOUT,env=env)
    env.update({"MODEL":GGUF,"CTX":str(CTX),"CONTEXT_LIMIT":str(CTX),"SLOTS":"1","PORT":str(PORT),
                "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999",
                "KV_POOL_Q8":"1","KV_CACHE_TYPE":"q8","CUDA_GRAPHS":"1","PGRAPH":"on",
                "FLASH_ATTN":"1","PREFIX_CACHE_MIB":"0","SAMPLING_LOCK":"0","RUST_LOG":"info",
                "MOE_EXPERTS":"auto"})
    env.update(EXTRA)
    return subprocess.Popen(["/root/target/release/yforge"],cwd="/root",stdout=open(LOG,"wb"),stderr=subprocess.STDOUT,env=env)

def main():
    free_port(); wait_free(); base=vram()
    proc=launch()
    ready=False
    for _ in range(1200):
        if proc.poll() is not None: break
        if '"choices"' in post("ping",4): ready=True; break
        time.sleep(3)
    if not ready:
        print(f"[{VARIANT}] НЕ ПОДНЯЛСЯ"); proc.kill(); return 1
    def fresh():
        salt=random.randint(1,10**9)
        words=int(PROMPT_TOKENS/6.0)
        body=" ".join(f"w{w}" for w in range(words))
        return f"Session {salt}. Marker {salt}. {body} Summarize."
    pfs,decs,ttfts=[],[],[]
    for rep in range(REPS):
        cr=fresh()
        ttft,tlast,ch,pt,err,ctok=stream(cr,MAXTOK)
        if err or ttft is None:
            print(f"[{VARIANT}] rep={rep} err={(err or '')[:120]}"); continue
        ngen=ctok or ch
        span=max(1e-9,tlast-ttft)
        decs.append((ngen-1)/span if ngen>1 else 0.0)
        pfs.append(pt/ttft); ttfts.append(ttft)
    time.sleep(2); peak=vram()-base
    if pfs:
        pf=statistics.median(pfs[1:] or pfs); dc=statistics.median(decs[1:] or decs); tt=statistics.median(ttfts[1:] or ttfts)
    else: pf=dc=tt=0.0
    print(f"[{VARIANT}] ctx={CTX} prompt={pt} TTFT={tt:6.2f}s prefill={pf:7.0f}t/s decode={dc:6.1f}t/s dVRAM={peak}MB")
    proc.kill(); proc.wait(); free_port(); wait_free(); return 0

if __name__=="__main__": sys.exit(main())
