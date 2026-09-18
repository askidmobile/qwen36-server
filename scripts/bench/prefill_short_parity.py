#!/usr/bin/env python3
"""A/B коротких контекстов + проверка точности int8-QK на префиле (greedy parity)."""
import json, os, random, re, subprocess, time, urllib.request, urllib.error
PORT=18250; URL=f"http://localhost:{PORT}/v1/chat/completions"
GGUF="/root/models/Qwen3.8-27B-Q8_0.gguf"; MODEL="m"

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
                if t.startswith("pid="): subprocess.run(["kill","-9",t[4:].split(",")[0]],capture_output=True)
    time.sleep(3)
def launch(ctx, extra):
    env=dict(os.environ)
    env.update({"MODEL":GGUF,"CTX":str(ctx),"CONTEXT_LIMIT":str(ctx),"SLOTS":"1","PORT":str(PORT),
                "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999",
                "KV_POOL_Q8":"1","KV_CACHE_TYPE":"q8","CUDA_GRAPHS":"1","FLASH_ATTN":"1",
                "PGRAPH":"on","PREFIX_CACHE_MIB":"0","SAMPLING_LOCK":"1","RUST_LOG":"info"})
    env.update(extra)
    tag=extra.get("QK_INT8_PREFILL","0")
    log=open(f"/root/logs/psp-{tag}-{ctx}.log","wb")
    return subprocess.Popen(["/root/target/release/yforge"],cwd="/root",stdout=log,stderr=subprocess.STDOUT,env=env), log
def post(c, mt, temp=0):
    b=json.dumps({"model":MODEL,"stream":False,"max_tokens":mt,"temperature":temp,"messages":[{"role":"user","content":c}]}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    try:
        with urllib.request.urlopen(r,timeout=1800) as resp: return resp.read().decode()
    except urllib.error.HTTPError as e: return e.read().decode()
    except Exception as e: return f'{{"error":"{e}"}}'
def stream(c, mt, temp=0):
    b=json.dumps({"model":MODEL,"stream":True,"max_tokens":mt,"temperature":temp,
                  "stream_options":{"include_usage":True},"messages":[{"role":"user","content":c}]}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    t0=time.time(); ttft=tlast=None; ctok=0; text=[]
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
                    if dl.get("content") is not None: text.append(dl["content"])
                    if dl.get("content") or dl.get("reasoning_content"):
                        n=time.time()-t0
                        if ttft is None: ttft=n
                        tlast=n
    except Exception as e:
        return None,None,0,0,str(e)[:150],""
    return ttft,tlast,0,0,None,"".join(text)

def boot(ctx, extra):
    global MODEL
    MODEL="m"; free_port(); wait_free(); base=vram()
    proc,log=launch(ctx,extra)
    for _ in range(600):
        if proc.poll() is not None: break
        probe=post("ping",4)
        if '"choices"' in probe: return proc,log,base
        if "model_not_loaded" in probe:
            req=urllib.request.Request(f"http://localhost:{PORT}/v1/models",headers={"Authorization":"Bearer x"})
            try:
                MODEL=json.loads(urllib.request.urlopen(req,timeout=30).read())["data"][0]["id"]
                if '"choices"' in post("ping",4): return proc,log,base
            except Exception: pass
        time.sleep(3)
    return proc,log,base

def perf(label, ctx, extra):
    proc,log,base=boot(ctx,extra)
    target=int(ctx*0.85); ratio=6.0; ptok=0; ttft=tlast=None; ctok=0
    for _ in range(4):
        words=int(target/ratio)
        c=f"Session {random.randint(1,10**9)}. "+" ".join(f"word{w}" for w in range(words))+" Summarize the pattern in one sentence."
        ttft,tlast,_,_,err,_=stream(c,65)
        if err is None and ttft is not None:
            m=re.search(r'"prompt_tokens":\s*(\d+)',post(c,1))
            if m: ptok=int(m.group(1)); break
        om=re.search(r"(\d+) tokens",err or "")
        if om: ratio=int(om.group(1))/words; continue
        print(f"{label}: ОШИБКА {(err or '')[:100]}",flush=True); ptok=0; break
    if ptok and ttft:
        print(f"{label:34} ctx={ctx:>7} TTFT={ttft:7.2f}s pf={ptok/ttft:6.0f}t/s vram={vram()-base}MB",flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()

# --- перф на коротких ---
for ctx in [8192, 32768]:
    perf("QK_INT8_PREFILL=0", ctx, {"QK_INT8_PREFILL":"0"})
    perf("QK_INT8_PREFILL=1", ctx, {"QK_INT8_PREFILL":"1"})

# --- точность: greedy-выдача на 32K ---
PROMPT=("Session 4242. "+" ".join(f"word{w}" for w in range(4000))+
        " Ignore the noise. Reply with exactly this sentence: The quick brown fox jumps over the lazy dog. Then list the numbers 1 to 10 separated by commas.")
outs={}
for tag in ["0","1"]:
    proc,log,base=boot(32768,{"QK_INT8_PREFILL":tag,"SAMPLING_LOCK":"0","TEMPERATURE":"0"})
    _,_,_,_,err,text=stream(PROMPT,120,temp=0)
    outs[tag]=(text,err)
    print(f"parity[{tag}]: err={err} len={len(text)} head={text[:80]!r}",flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()
print("PARITY_IDENTICAL:", outs["0"][0]==outs["1"][0], flush=True)
print("DONE")
