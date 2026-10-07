#!/usr/bin/env python3
"""Декод-нагрузка: КОРОТКИЙ промпт (малый префил) + много генерируемых токенов.
argv: port t_start t_end model mtok auth [words] [fixed]"""
import json, random, sys, time, urllib.request

PORT = int(sys.argv[1]); T_START = float(sys.argv[2]); T_END = float(sys.argv[3])
MODEL = sys.argv[4]; MT = int(sys.argv[5]); AUTH = sys.argv[6]
WORDS = int(sys.argv[7]) if len(sys.argv) > 7 else 120
FIXED = len(sys.argv) > 8 and sys.argv[8] == "fixed"

t0 = time.time(); last = 0
base = " ".join(f"w{i}" for i in range(WORDS)) + " Summarize the above in detail."
while True:
    el = time.time() - t0
    if el > T_END: break
    if el < T_START:
        time.sleep(0.3); continue
    p = base if FIXED else f"Session {random.randint(1,10**9)}. " + base
    b = json.dumps({"model": MODEL, "max_tokens": MT, "temperature": 0,
                    "messages": [{"role": "user", "content": p}]}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=b,
                               headers={"Authorization": f"Bearer {AUTH}",
                                        "Content-Type": "application/json"})
    try:
        d = json.loads(urllib.request.urlopen(r, timeout=900).read())
        last = (d.get("usage") or {}).get("completion_tokens", 0)
    except Exception as e:
        print("err", str(e)[:90], flush=True)
print("load done, last tok:", last, flush=True)
