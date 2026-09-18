#!/usr/bin/env python3
"""A/B декода yforge: без MTP / с MTP (Q4_0-голова), на 8K и 32K."""
import json, os, re, random, signal, subprocess, sys, time, urllib.request, urllib.error

PORT = 18220
URL = f"http://localhost:{PORT}/v1/chat/completions"
GGUF = "/root/models/Qwen3.8-27B-Q8_0.gguf"
MTP = "/root/models/mtp.ytf"

def vram():
    out = subprocess.run(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],capture_output=True,text=True).stdout
    return int(out.strip().splitlines()[0])

def wait_free():
    for _ in range(120):
        if vram() <= 2000: return
        time.sleep(3)

def free_port():
    out = subprocess.run(["ss","-tlnp"],capture_output=True,text=True).stdout
    for line in out.splitlines():
        if f":{PORT}" in line:
            for tok in line.split():
                if tok.startswith("pid="):
                    subprocess.run(["kill","-9",tok[4:].split(",")[0]],capture_output=True)
    time.sleep(3)

def launch(ctx, mtp):
    env = dict(os.environ)
    env.update({"MODEL":GGUF,"CTX":str(ctx),"CONTEXT_LIMIT":str(ctx),"SLOTS":"1","PORT":str(PORT),
                "API_KEYS":'[{"key":"x","name":"m"}]',"GPU_ONLY":"1","GPU_LAYERS":"999",
                "KV_POOL_Q8":"1","KV_CACHE_TYPE":"q8","CUDA_GRAPHS":"1","PGRAPH":"on",
                "FLASH_ATTN":"1","PREFIX_CACHE_MIB":"0","SAMPLING_LOCK":"1","RUST_LOG":"info"})
    if mtp:
        env.update({"MTP":"1","MTP_PATH":MTP,"MTP_WIDTH":"2","MTP_ADAPTIVE":"1"})
    log = open(f"/root/logs/mtp-{ctx}-{'on' if mtp else 'off'}.log","wb")
    return subprocess.Popen(["/root/target/release/yforge"],cwd="/root",stdout=log,stderr=subprocess.STDOUT,env=env), log

def post(c, mt):
    b=json.dumps({"model":"qwen3.8-27b","stream":False,"max_tokens":mt,"temperature":0,"messages":[{"role":"user","content":c}]}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    try:
        with urllib.request.urlopen(r,timeout=1800) as resp: return resp.read().decode()
    except Exception as e: return f'{{"error":"{e}"}}'

def stream(c, mt):
    b=json.dumps({"model":"qwen3.8-27b","stream":True,"max_tokens":mt,"temperature":0,
                  "stream_options":{"include_usage":True},"messages":[{"role":"user","content":c}]}).encode()
    r=urllib.request.Request(URL,data=b,headers={"Content-Type":"application/json","Authorization":"Bearer x"})
    t0=time.time(); ttft=tlast=None; ch=0; ctok=0; ptok=0
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
                if u:
                    ptok=u.get("prompt_tokens",ptok); ctok=u.get("completion_tokens",ctok)
                for cch in d.get("choices",[]):
                    dl=cch.get("delta") or {}
                    if dl.get("content") or dl.get("reasoning_content"):
                        n=time.time()-t0
                        if ttft is None: ttft=n
                        tlast=n; ch+=1
    except Exception as e:
        return None,None,0,0,str(e)[:200],0
    return ttft,tlast,ch,ptok,None,ctok

def run(ctx, mtp):
    free_port(); wait_free()
    base=vram()
    proc,log=launch(ctx,mtp)
    ready=False
    for _ in range(600):
        if proc.poll() is not None: break
        if '"choices"' in post("ping",4): ready=True; break
        time.sleep(3)
    if not ready:
        tail=open(log.name,errors="replace").read()[-300:].replace("\n"," | ")
        print(f"  ctx={ctx} mtp={mtp} НЕ ПОДНЯЛСЯ: {tail}",flush=True)
        proc.kill(); proc.wait(); wait_free(); return
    load=time.time()
    words=int(ctx*0.85/6)
    content=f"Session {random.randint(1,10**9)}. "+" ".join(f"word{w}" for w in range(words))+" Summarize the pattern in one sentence."
    ttft,tlast,ch,ptok,err,ctok=stream(content,65)
    if err:
        print(f"  ctx={ctx} mtp={mtp} ОШИБКА: {err}",flush=True)
    else:
        ngen=ctok or ch; span=max(1e-9,tlast-ttft)
        dec=(ngen-1)/span
        peak=vram()
        print(f"  ctx={ctx} mtp={'ON' if mtp else 'OFF'} ttft={ttft:.2f}s pf={ptok/ttft:.0f}t/s dec={dec:.1f}t/s gen={ngen} chunks={ch} vram={peak-base}MB",flush=True)
        logtxt=open(log.name,errors="replace").read()
        m=re.findall(r"mtp[^\n]*",logtxt)
        if m: print("    "+m[-1][:180],flush=True)
    proc.kill(); proc.wait(); free_port(); wait_free()

for ctx in [int(x) for x in os.environ.get("CTXS","8192,32768").split(",")]:
    for mtp in [False, True]:
        run(ctx, mtp)
print("DONE")
