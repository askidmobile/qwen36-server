#!/usr/bin/env python3
"""Решает судьбу префил-графов: скорость + VRAM на 8K..256K."""
import json, os, random, subprocess, time, urllib.request, urllib.error
PORT=18420; GGUF="/root/models/Qwen3.8-27B-Q8_0.gguf"; MODEL="m"
CTXS=[int(x) for x in os.environ.get("CTXS","8192,32768,131072").split(",")]

def vram():
    o=subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    return int(o.strip().splitlines()[0])
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
def run(label, ctx, extra):
    global MODEL
    MODEL="m"; free_port(); wait_free(); base=vram()
    env=dict(os.environ)
    env.update({"MODEL":GGUF,"CTX":str(ctx),"CONTEXT_LIMIT":str(ctx),"SLOTS":"1","PORT":str(PORT),
                "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999",
                "KV_POOL_Q8":"1","KV_CACHE_DTYPE":"q8","CUDA_GRAPHS":"1","PGRAPH":"on",
                "FLASH_ATTN":"1","PREFIX_CACHE_MIB":"0","SAMPLING_LOCK":"1","RUST_LOG":"info"})
    env.update(extra)
    log=open(f"/root/logs/pgc-{label.replace(' ','_')}-{ctx}.log","wb")
    proc=subprocess.Popen(["/root/target/release/yforge"],cwd="/root",stdout=log,stderr=subprocess.STDOUT,env=env)
    for _ in range(600):
        if proc.poll() is not None: break
        r=urllib.request.Request(f"http://localhost:{PORT}/v1/models",headers={"Authorization":"Bearer x"})
        try:
            with urllib.request.urlopen(r,timeout=20) as resp: d=json.loads(resp.read())
            if d.get("data"): MODEL=d["data"][0]["id"]; break
        except Exception: pass
        time.sleep(3)
    time.sleep(2)
    target=int(ctx*0.85); ratio=6.0; ptok=0; ttft=tlast=None; ctok=0; err=None
    for _ in range(4):
        words=int(target/ratio)
        c=f"Session {random.randint(1,10**9)}. "+" ".join(f"word{w}" for w in range(words))+" Summarize briefly."
        b=json.dumps({"model":MODEL,"stream":True,"max_tokens":65,"temperature":0,
                      "stream_options":{"include_usage":True},"messages":[{"role":"user","content":c}]}).encode()
        t0=time.time(); ttft=tlast=None; ctok=0; err=None
        for _ in range(60):
            rq=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=b,
                                      headers={"Content-Type":"application/json","Authorization":"Bearer x"})
            try:
                with urllib.request.urlopen(rq,timeout=1800) as resp:
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
                    break
            except urllib.error.HTTPError as e:
                body=e.read().decode()
                if "still loading" in body: time.sleep(5); continue
                err=body[:100]; break
            except Exception as e: err=str(e)[:100]; break
        if err is None and ttft is not None:
            b2=json.dumps({"model":MODEL,"stream":False,"max_tokens":1,"temperature":0,"messages":[{"role":"user","content":c}]}).encode()
            rq2=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=b2,
                                       headers={"Content-Type":"application/json","Authorization":"Bearer x"})
            try:
                with urllib.request.urlopen(rq2,timeout=900) as resp:
                    import re as _re
                    m=_re.search(r'"prompt_tokens":\s*(\d+)', resp.read().decode())
                    if m: ptok=int(m.group(1)); break
            except Exception: pass
        else:
            import re as _re
            om=_re.search(r"(\d+) tokens",err or "")
            if om: ratio=int(om.group(1))/words; continue
            break
    time.sleep(2); after=vram()
    if ptok and ttft:
        dec=(ctok-1)/max(1e-9,tlast-ttft) if ctok and ctok>1 else 0
        print(f"{label:26} ctx={ctx:>7} pf={ptok/ttft:5.0f}t/s TTFT={ttft:6.2f}s dec={dec:5.1f}t/s VRAM={after-base:>6}MB",flush=True)
    else:
        print(f"{label:26} ctx={ctx:>7} ОШИБКА {err}",flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()

for ctx in CTXS:
    run("prefill-graphs default", ctx, {})
    run("no prefill-graphs", ctx, {"PGRAPH_MIN_T":"999999"})
print("DONE")
