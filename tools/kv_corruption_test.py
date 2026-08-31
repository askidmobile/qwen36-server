#!/usr/bin/env python3
"""KV-коррупция тест: многоходовая переписка с канарейками на длинном контексте.

Гипотеза: на длинном контексте (24K+, когда KV-пул под давлением) инференс
теряет/искажает ранний контекст — пробелы, чужие буквы, потерянные факты.

Каждый ход: паддинг (детерминированный «архивный текст» ~3K токенов) +
канарейка «записка №N: код СИГНАЛ-XXX-<seed>, ключ <6 цифр>». Финальный ход —
воспроизвести ВСЕ коды. Скоринг по маркеру СИГНАЛ-XXX и по 6-значному ключу.

Использование:
  python3 kv_corruption_test.py --host 127.0.0.1 --port 18099 --key <KEY> \
      --turns 8 --target-tokens 24000 --label q8 --out r-q8.json
"""

import argparse
import json
import random
import time
import urllib.request
import sys

TURN_TOPICS = [
    "кратко перескажи суть документа выше",
    "какие два вывода из документа самые важные? по строке на каждый",
    "какой термин из документа ты бы выделил? одним предложением почему",
    "противоречат ли друг другу части документа? кратко",
    "сформулируй один вопрос, который остался открытым после документа",
    "перескажи документ одним предложением",
    "что в документе относится к инфраструктуре? кратко",
    "какие числа фигурируют в документе? перечисли только их",
]


def make_padding(rng: random.Random, words: int, turn: int) -> str:
    """Детерминированный русский псевдо-текст ~words слов. Строки нумерованы,
    чтобы модель не склеивала их в бессмысленное повторение."""
    nouns = ["система", "узел", "канал", "отчёт", "протокол", "журнал", "модуль",
             "схема", "параметр", "компонент", "регламент", "инцидент", "метрика",
             "сервер", "очередь", "поток", "ресурс", "лимит", "бюджет", "квота"]
    verbs = ["проверен", "обновлён", "зафиксирован", "согласован", "задокументирован",
             "пересчитан", "перенастроен", "архивирован", "вынесен в план", "закрыт"]
    attrs = ["без замечаний", "с оговоркой", "по графику", "внепланово", "частично",
             "в полном объёме", "после ревизии", "до конца квартала"]
    lines = []
    w = 0
    i = 0
    while w < words:
        i += 1
        n1 = rng.choice(nouns)
        n2 = rng.choice(nouns)
        line = (f"Пункт {i}: {n1} {rng.choice(verbs)} {rng.choice(attrs)}; "
                f"{n2} учтён в реестре под номером {rng.randint(100, 999)}, "
                f"показатель {rng.randint(10, 99)}.{rng.randint(10, 99)} против "
                f"планового {rng.randint(10, 99)}.{rng.randint(10, 99)}. "
                f"Ответственный — отдел {rng.randint(1, 12)}, отметка {turn}-{i}.")
        lines.append(line)
        w += len(line.split())
    return "\n".join(lines)


def chat(host, port, key, messages, max_tokens=400, temperature=0.0, timeout=900, seed=42):
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "stream": False,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    dt = time.time() - t0
    content = body["choices"][0]["message"].get("content") or ""
    usage = body.get("usage", {})
    return content, usage, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--key", required=True)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pad-words", type=int, default=2200, help="слов паддинга на ход (~3K токенов)")
    ap.add_argument("--max-tokens", type=int, default=350)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    codes = [f"{rng.randint(100000, 999999)}" for _ in range(args.turns)]
    canaries = [f"СИГНАЛ-{i+1:03d}-{args.seed}" for i in range(args.turns)]

    messages = [
        {
            "role": "system",
            "content": (
                "Ты внимательный архивный ассистент. Пользователь присылает длинные "
                "документы; в конце каждого документа — «записка» с кодовым словом и "
                "шестизначным ключом. Запоминай их дословно. Отвечай кратко, без markdown."
            ),
        }
    ]

    log = {
        "label": args.label,
        "turns": args.turns,
        "seed": args.seed,
        "pad_words": args.pad_words,
        "temperature": args.temperature,
        "canaries": canaries,
        "codes": codes,
        "steps": [],
        "final": None,
        "score": None,
    }

    for i in range(args.turns):
        padding = make_padding(rng, args.pad_words, i + 1)
        user = (
            f"{padding}\n\n---\nЗаписка номер {i+1}: кодовое слово {canaries[i]}, "
            f"ключ доступа {codes[i]}. Далее: {TURN_TOPICS[i % len(TURN_TOPICS)]}."
        )
        messages.append({"role": "user", "content": user})
        content, usage, dt = chat(
            args.host, args.port, args.key,
            messages, max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        messages.append({"role": "assistant", "content": content})
        step = {
            "turn": i + 1,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "wall_s": round(dt, 1),
            "reply_head": content[:160],
        }
        log["steps"].append(step)
        print(
            f"[{args.label}] turn {i+1}/{args.turns}: "
            f"prompt={usage.get('prompt_tokens')} compl={usage.get('completion_tokens')} "
            f"{dt:.1f}s :: {content[:70]!r}",
            flush=True,
        )

    final_q = (
        "Перечисли ВСЕ записки из нашей беседы: для каждой — номер, кодовое слово "
        "и шестизначный ключ доступа, ровно как они были сообщены. По одной строке "
        "на записку, без пояснений."
    )
    messages.append({"role": "user", "content": final_q})
    final, usage, dt = chat(
        args.host, args.port, args.key,
        messages, max_tokens=800, temperature=0.0,
    )
    exact = sum(1 for c, k in zip(canaries, codes) if c in final and k in final)
    lost = [i + 1 for i, (c, k) in enumerate(zip(canaries, codes)) if c not in final and k not in final]
    distorted = [i + 1 for i, (c, k) in enumerate(zip(canaries, codes))
                 if (c not in final) != (k not in final)]
    log["final"] = {"reply": final, "wall_s": round(dt, 1),
                    "prompt_tokens": usage.get("prompt_tokens")}
    log["score"] = {
        "exact_hits": exact,
        "total": args.turns,
        "lost": lost,
        "distorted": distorted,
        "final_prompt_tokens": usage.get("prompt_tokens"),
    }
    print(
        f"[{args.label}] FINAL: {exact}/{args.turns} полных попаданий; "
        f"потеряно: {lost}; искажено: {distorted}; финальный промпт {usage.get('prompt_tokens')} tok",
        flush=True,
    )
    print("---- финальный ответ ----")
    print(final)

    out = args.out or f"kvtest-{args.label}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(f"[{args.label}] сохранено: {out}")


if __name__ == "__main__":
    sys.exit(main())
