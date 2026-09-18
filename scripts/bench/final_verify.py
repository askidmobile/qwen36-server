#!/usr/bin/env python3
"""ФИНАЛЬНАЯ сверка. Продовая конфигурация: окно 256K у ОБОИХ движков.
Метрики за один прогон процесса:
  старт->первый токен, установившийся TTFT, префил/декод на 8/32/128/256K, VRAM.
Промпты уникальные -> кеш промпта ни у одного движка не помогает."""
import json, os, random, subprocess, time, urllib.request, urllib.error
PORT=18770; GGUF="/root/models/Qwen3.8-27B-Q8_0.gguf"; CTX=262144
CASES=[(8192,"8K"),(32768,"32K"),(131072,"128K"),(262144,"256K")]
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
    for _ in range(120):
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
                if ttft is None: return None,None,0,0,"no tokens"
                return ttft,tlast,ctok,ptok,None
        except urllib.error.HTTPError as e:
            bs=e.read().decode()
            if "still loading" in bs or e.code in (400,503): time.sleep(1); continue
            return None,None,0,0,f"{e.code} {bs[:80]}"
        except Exception as e:
            if "refused" in str(e): time.sleep(1); continue
            return None,None,0,0,str(e)[:90]
    return None,None,0,0,"timeout"
def run(label, engine):
    global MODEL
    MODEL = "m" if engine=="llamacpp" else "qwen3.8-27b"
    free_port(); wait_free(); base=vram()
    env=dict(os.environ); log=open(f"/root/logs/fv-{engine}.log","wb")
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
    t_proc=time.time()
    proc=subprocess.Popen(cmd,cwd="/root",stdout=log,stderr=subprocess.STDOUT,env=env)
    print(f"--- {label} ---",flush=True)
    # 1) старт -> первый токен (уникальный короткий промпт)
    ttft,tlast,ctok,ptok,err=chat(f"Start {random.randint(10**8,10**9)}: 2+2?",16)
    if err: print(f"  start: ОШИБКА {err}",flush=True); proc.kill(); proc.wait(); wait_free(); return
    start_first=time.time()-t_proc
    print(f"  {'start->1st':>10}: {start_first:6.2f}s",flush=True)
    # 2) установившийся TTFT (ещё 3 уникальных коротких)
    st=[]
    for i in range(3):
        a,_,_,_,e=chat(f"S{i}-{random.randint(10**8,10**9)}: 2+2?",16)
        if not e and a: st.append(a)
        time.sleep(0.5)
    med=sorted(st)[len(st)//2]
    print(f"  {'steady TTFT':>10}: {med:6.3f}s",flush=True)
    # 3) линейка контекстов
    for want,tag in CASES:
        words=max(1,int(want/6))
        c=f"Session {random.randint(10**8,10**9)}. "+" ".join(f"w{w}" for w in range(words))+" Summarize briefly."
        a,b2,ct,pk,e=chat(c,65)
        if e: print(f"  {tag:>10}: ОШИБКА {e}",flush=True); continue
        time.sleep(2); used=vram()-base
        dec=(ct-1)/max(1e-9,b2-a) if ct and ct>1 else 0
        pf=pk/a if a else 0
        print(f"  {tag:>10}: prompt={pk:>7} TTFT={a:8.2f}s pf={pf:6.0f}t/s dec={dec:5.1f}t/s VRAM={used:>6}MB",flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()
run("llama.cpp (окно 256K)","llamacpp")
run("yforge (CTX=256K, прод)","yforge")
print("DONE")
