#!/usr/bin/env python3
"""Старт процесса -> первый токен для обоих движков (Ornith-1.5-35B, окно 262144)."""
import json, os, subprocess, sys, time, urllib.request, urllib.error

PORT = 18870
GGUF = "/root/models/Ornith-1.5-35B-Q8_0.gguf"
CTX = 262144

def vram():
    o=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    return int(o.strip().splitlines()[0])

def kill_gpu():
    for p in subprocess.run(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],capture_output=True,text=True).stdout.split():
        subprocess.run(["kill","-9",p],capture_output=True)
    subprocess.run("pkill -9 -x yforge; pkill -9 -x llama-server",shell=True,capture_output=True)
    time.sleep(4)
    for _ in range(200):
        if vram()<=400: return
        time.sleep(3)

PROMPT = "Session 777777. " + " ".join(f"w{i}" for i in range(600)) + " Summarize."

def time_engine(env_extra, cmd, model):
    env=dict(os.environ); env.update(env_extra)
    t0=time.time()
    p=subprocess.Popen(cmd,cwd="/root",stdout=open("/root/logs/start.log","wb"),
                       stderr=subprocess.STDOUT,env=env)
    ready=None; first=None
    b=json.dumps({"model":model,"stream":True,"max_tokens":8,"temperature":0,
                  "stream_options":{"include_usage":True},
                  "messages":[{"role":"user","content":PROMPT}]}).encode()
    while time.time()-t0 < 900:
        r=urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",data=b,
            headers={"Content-Type":"application/json","Authorization":"Bearer x"})
        try:
            t_req=time.time()-t0
            with urllib.request.urlopen(r,timeout=900) as resp:
                for raw in resp:
                    line=raw.decode("utf-8","replace").strip()
                    if not line.startswith("data:"): continue
                    pl=line[5:].strip()
                    if pl=="[DONE]": break
                    try: d=json.loads(pl)
                    except ValueError: continue
                    for cch in d.get("choices",[]):
                        dl=cch.get("delta") or {}
                        if dl.get("content") or dl.get("reasoning_content"):
                            first=time.time()-t0
                            break
                    if first is not None: break
            if first is not None:
                ready=time.time()-t0
                break
        except Exception:
            time.sleep(2)
    p.kill(); p.wait(); kill_gpu()
    return ready, first

# yforge
yf_env={"MODEL":GGUF,"CTX":str(CTX),"CONTEXT_LIMIT":str(CTX),"SLOTS":"1","PORT":str(PORT),
        "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999","KV_POOL_Q8":"1",
        "KV_CACHE_TYPE":"q8","CUDA_GRAPHS":"1","PGRAPH":"on","FLASH_ATTN":"1","PREFIX_CACHE_MIB":"0",
        "SAMPLING_LOCK":"1","RUST_LOG":"warn","MOE_EXPERTS":"auto","PREFILL_CHUNK":"4096"}
kill_gpu()
r,f=time_engine(yf_env,["/root/target/release/yforge"],"ornith-1.5-35b")
print(f"yforge:   start->ready {r:.2f} s, start->first-token {f:.2f} s")

ll_env={}
kill_gpu()
cmd=["/root/llama.cpp/build/bin/llama-server","-m",GGUF,"-c",str(CTX),"-ngl","999",
     "--host","127.0.0.1","--port",str(PORT),"--parallel","1","--api-key","x","--no-warmup",
     "-fa","on","--cache-type-k","q8_0","--cache-type-v","q8_0"]
r,f=time_engine(ll_env,cmd,"m")
print(f"llama.cpp: start->ready {r:.2f} s, start->first-token {f:.2f} s")
