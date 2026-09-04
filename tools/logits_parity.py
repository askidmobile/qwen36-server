#!/usr/bin/env python3
"""Паритет распределения следующего токена: наш сервер против llama.cpp.

Зачем: дефект подмены токенов (`inline-flex` -> `inline-rect`) виден на выдаче
только через трёхминутный реплей записанной сессии. Здесь тот же дефект
измеряется за две секунды — сравнением top-10 логпробов первого токена на
коротком промпте. Обе стороны детерминированы, разница воспроизводится точно.

Промпт обязан совпадать байт в байт, иначе числа бессмысленны: сверять через
PROMPT_DEBUG у нас и /apply-template у llama-server.

Эталон: llama-server из D:\\Projects\\yttri-inference\\llama-b10375 на том же
GGUF, порт 18098 закрыт файрволом -> ssh -L 18098:127.0.0.1:18098 yttri-win.

Замер 2026-09-04 (Ornith-1.5-9B Q4_K_M, промпт 117 байт): топ-1 сходится до
0.005, хвост расходится на 0.3-2.2 нат; под FORCE_DMMV=1 хуже (3.1).
"""

import json, sys, urllib.request

KEY = json.load(open('/Users/askid/.pi/agent/models.json'))['providers']['local-qwen']['apiKey']
OURS  = "http://192.168.2.89:18099/v1/chat/completions"
REF   = "http://127.0.0.1:18098/v1/chat/completions"

def ask(url, body, key=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)

def top(d, who):
    lp = d["choices"][0].get("logprobs")
    if not lp:
        print(f"  [{who}] logprobs нет:", json.dumps(d["choices"][0])[:200]); return None
    if not lp.get("content"):
        print(f"  [{who}] content пуст:", json.dumps(lp)[:200]); return None
    c = lp["content"][0]
    return [(t["token"], round(t["logprob"], 4)) for t in c.get("top_logprobs", [])]

prompt = sys.argv[1] if len(sys.argv) > 1 else "Write CSS for a flex row. Answer with code only."
body = {"model": "ornith-1.5-9b", "max_tokens": 1, "max_completion_tokens": 1, "temperature": 0,
        "logprobs": True, "top_logprobs": 10,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": prompt}]}

a = ask(OURS, body, KEY); b = ask(REF, dict(body, model="ornith"))
ta, tb = top(a, "наш"), top(b, "эталон")
print("наш   :", ta)
print("эталон:", tb)
if ta and tb:
    print("argmax совпал:", ta[0][0] == tb[0][0])
    da = dict(ta); db = dict(tb)
    common = set(da) & set(db)
    if common:
        print("макс |Δlogprob| по общим токенам: %.4f (общих %d из 10)" %
              (max(abs(da[t]-db[t]) for t in common), len(common)))
