#!/usr/bin/env python3
"""Честный парный замер VRAM/DECODE на 8K: по 2 прогона на конфигурацию,
одинаковый протокол (готовность по /v1/models -> прогрев -> измерение).
Единственный выключатель графов — CUDA_GRAPHS (0 = полностью выкл)."""
import json, os, random, re, subprocess, time, urllib.request, urllib.error
PORT=18610; GGUF="/root/models/Qwen3.8-27B-Q8_0.gguf"; MODEL="m"; CTX=8192
def vram():
    o=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    return int(o.strip().splitlines()[0])
def wait_free():
    for _ in range(200):
        if vram()<=300: return
        time.sleep(3)
def free_port():
    o=subprocess.run(["ss","-tlnp"],capture_output=True,text=True).stdout
    for line in o.splitlines():
        if f":{PORT}" in line:
            for t in line.split():
                if t.startswith("pid="): subprocess.run(["kill","-9",t[4:].split(",")[0]],capture_output=True)
    time.sleep(3)
def chat(c, mt, stream=False):
    body={"model":MODEL,"max_tokens":mt,"temperature":0,"messages":[{"role":"user","content":c}],"stream":stream}
    if stream: body["stream_options"]={"include_usage":True}
    b=json.dumps(body).encode(); t0=time.time(); ttft=tlast=None; ctok=0
    for _ in range(60):
        r=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=b,
                                 headers={"Content-Type":"application/json","Authorization":"Bearer x"})
        try:
            with urllib.request.urlopen(r,timeout=1800) as resp:
                if not stream: return resp.read().decode(),(None,None,0)
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
                return "",(ttft,tlast,ctok)
        except urllib.error.HTTPError as e:
            if "still loading" in e.read().decode(): time.sleep(5); continue
            return "",(None,None,0)
        except Exception: return "",(None,None,0)
    return "",(None,None,0)
def one(label, engine, extra):
    global MODEL
    MODEL="m"; free_port(); wait_free(); base=vram()
    env=dict(os.environ); log=open(f"/root/logs/cv2-{abs(hash(label))%100000}.log","wb")
    if engine=="yforge":
        env.update({"MODEL":GGUF,"CTX":str(CTX),"CONTEXT_LIMIT":str(CTX),"SLOTS":"1","PORT":str(PORT),
                    "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999",
                    "KV_POOL_Q8":"1","KV_CACHE_TYPE":"q8","PGRAPH":"on",
                    "FLASH_ATTN":"1","PREFIX_CACHE_MIB":"0","RUST_LOG":"info"})
        env.update(extra); cmd=["/root/target/release/yforge"]
    else:
        cmd=["/root/llama.cpp/build/bin/llama-server","-m",GGUF,"-c",str(CTX),"-ngl","999",
             "--host","127.0.0.1","--port",str(PORT),"--parallel","1","--api-key","x","--no-warmup","-fa","on",
             "--cache-type-k","q8_0","--cache-type-v","q8_0"]
    proc=subprocess.Popen(cmd,cwd="/root",stdout=log,stderr=subprocess.STDOUT,env=env)
    ok=False
    for _ in range(600):
        if proc.poll() is not None: break
        r=urllib.request.Request(f"http://localhost:{PORT}/v1/models",headers={"Authorization":"Bearer x"})
        try:
            with urllib.request.urlopen(r,timeout=20) as resp: d=json.loads(resp.read())
            if d.get("data"): MODEL=d["data"][0]["id"]; ok=True; break
        except Exception: pass
        time.sleep(3)
    if not ok:
        print(f"{label:26} НЕ ПОДНЯЛСЯ",flush=True); proc.kill(); proc.wait(); wait_free(); return
    # прогрев длинным промптом
    words=int(CTX*0.85/6)
    chat("Warmup "+" ".join(f"x{w}" for w in range(words))+" ok",8); time.sleep(3)
    v_warm=vram()-base
    c=f"Session {random.randint(1,10**9)}. "+" ".join(f"word{w}" for w in range(words))+" Summarize briefly."
    _,st=chat(c,65,stream=True)
    time.sleep(3); after=vram()-base
    dec=(st[2]-1)/max(1e-9,st[1]-st[0]) if st[0] and st[2] and st[2]>1 else 0
    print(f"{label:26} warm={v_warm:>6} after={after:>6} dec={dec:5.1f}t/s",flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()
CASES=[("llama.cpp","llamacpp",{}),
       ("yforge graphs=1 (default)","yforge",{"CUDA_GRAPHS":"1"}),
       ("yforge graphs=0 (true off)","yforge",{"CUDA_GRAPHS":"0"})]
for rep in (1,2):
    print(f"--- repeat {rep} ---",flush=True)
    for label,eng,extra in CASES:
        one(label,eng,extra)
print("DONE")
