import json, sqlite3, time

DB = "/workspace/owui-data/webui.db"
BASE = "deepseek-v4.1-flash-1017-uncensored-vq"

con = sqlite3.connect(DB)
cur = con.cursor()
cur.execute("SELECT id FROM user LIMIT 1")
uid = cur.fetchone()[0]
now = int(time.time())

def upsert(mid, name, params, description):
    meta = {"profileImageUrl": "", "description": description,
            "capabilities": {"citations": True, "code_interpreter": False},
            "suggestion_prompts": [], "tags": []}
    full = {"stream_response": True, "stream_options": {"include_usage": True}, **params}
    cur.execute("SELECT id FROM model WHERE id=?", (mid,))
    if cur.fetchone():
        cur.execute("UPDATE model SET name=?, base_model_id=?, params=?, meta=?, updated_at=? WHERE id=?",
                    (name, BASE, json.dumps(full), json.dumps(meta), now, mid))
        print("updated", mid)
    else:
        cur.execute("INSERT INTO model (id, user_id, base_model_id, name, params, meta, is_active, created_at, updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (mid, uid, BASE, name, json.dumps(full), json.dumps(meta), 1, now, now))
        print("inserted", mid)

upsert(f"{BASE}-chat", "DeepSeek V4.1 — чат (без рассуждений)",
       {"reasoning_effort": "none", "temperature": 1.0, "top_p": 1.0, "max_tokens": 4096},
       "Отвечает сразу, без длинных размышлений.")

upsert(f"{BASE}-reasoner", "DeepSeek V4.1 — reasoner (с рассуждениями)",
       {"reasoning_effort": "high", "temperature": 1.0, "top_p": 1.0, "max_tokens": 16384},
       "Думает перед ответом. Медленнее, лучше на сложном.")

con.commit()
for r in cur.execute("SELECT id, name, is_active FROM model"):
    print("  ", r)
con.close()
