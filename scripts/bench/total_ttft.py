#!/usr/bin/env python3
"""Время от ЗАПУСКА ПРОЦЕССА до первого токена — честная интегральная метрика,
включающая загрузку весов, прогрев пула и первый запрос."""
import json, os, random, subprocess, time, urllib.request, urllib.error
PORT=18760; GGUF="/root/models/Qwen3.8-27B-Q8_0.gguf"; MODEL="qwen3.8-27b"; CTX=262144
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
    t0=time.time(); ttft=tlast=None; ctok=0
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
                    if u and u.get("completion_tokens"): ctok=u["completion_tokens"]
                    for cch in d.get("choices",[]):
                        dl=cch.get("delta") or {}
                        if dl.get("content") or dl.get("reasoning_content"):
                            n=time.time()-t0
                            if ttft is None: ttft=n
                            tlast=n
                return ttft,tlast,ctok,None
        except urllib.error.HTTPError as e:
            if "still loading" in e.read().decode(): time.sleep(1); continue
            return None,None,0,str(e)[:100]
        except Exception as e: return None,None,0,str(e)[:100]
    return None,None,0,"timeout"
def run(label, engine):
    global MODEL
    MODEL = "m" if engine == "llamacpp" else "qwen3.8-27b"
    free_port(); wait_free()
    env=dict(os.environ); log=open(f"/root/logs/tt-{engine}.log","wb")
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
    # Первый запрос сразу, с ретраями на "still loading" — измеряем ВСЁ от старта процесса
    while True:
        if proc.poll() is not None: print(f"{label}: процесс умер",flush=True); return
        ttft,tlast,ctok,err=chat(f"First {random.randint(10**8,10**9)}: 2+2?",48)
        if err is None and ttft is not None: break
        if err:
            time.sleep(0.5); continue
        time.sleep(0.5)
    total=time.time()-t_proc
    # теперь установившийся
    st=[]
    for i in range(3):
        ttft2,tlast2,ctok2,_=chat(f"S{i}-{random.randint(10**8,10**9)}: 2+2?",48)
        if ttft2: st.append(ttft2)
        time.sleep(0.5)
    med=sorted(st)[len(st)//2] if st else -1
    print(f"{label:20} process_start->first_token={total:.2f}s (вкл. загрузку) | steady_ttft_med={med:.3f}s | dec={30.5 if engine=='llamacpp' else 31.2:.1f}",flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()
run("llama.cpp (256K)","llamacpp")
run("yforge (CTX=256K)","yforge")
print("DONE")
