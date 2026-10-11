#!/usr/bin/env python3
"""AWQ-подобный per-channel scaling для FFN: точный фолдинг без правок рантайма.

Для пары (ffn_up, ffn_down) в Qwen FFN считается как
    h = silu(gate(x)) * up(x);  y = down(h).
Если масштабировать входной канал i промежуточного слоя как
    up[i, :] /= s_i   и   down[:, i] *= s_i,
то произведение silu(gate_i)*up_i не меняется численно (в FP32/BF16 с точностью
округления), но распределение ошибки квантизации перераспределяется: каналы с
большой активацией получают относительно более точные веса down.

s берётся из imatrix (in_sum2 = сумма квадратов активаций по входному каналу
ffn_down), как в AWQ: s_i = (E[x_i^2])^(alpha/2), нормировано на геом. среднее.

Скрипт патчит копию BF16-GGUF на месте по data_offset: метаданные и все
остальные тензоры не трогаются.
"""
import argparse
import os
import shutil
import sys

import numpy as np
import gguf


def load_bf16(data: np.ndarray, n_elements: int) -> np.ndarray:
    """uint8-вью (n*2 байт) -> float32 через bf16-сдвиг."""
    raw = np.frombuffer(data.tobytes(), dtype=np.uint16, count=n_elements)
    u32 = raw.astype(np.uint32) << 16
    return u32.view(np.float32).copy()


def store_bf16(arr_f32: np.ndarray) -> bytes:
    """float32 -> bf16 (round-to-nearest-even через uint32 c округлением)."""
    u32 = arr_f32.astype(np.float32).view(np.uint32)
    bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    out = ((u32 + bias) >> np.uint32(16)).astype(np.uint16)
    return out.tobytes()


def fold_attention(mm, idx, stats, L, args):
    """Точный фолдинг масштаба в attention: v-проекция <-> attn_output.

    Attention для query-головы h читает V из kv-головы g = h // gqa. Если
    масштабировать выход V (канал g*head_dim+d) на s, а вход attn_output
    (канал h*head_dim+d) — на 1/s, результат не меняется. Статистика берётся из
    imatrix attn_output (вход = выход attention), агрегируется по query-головам,
    делящим одну kv-голову.
    """
    v_name = f"blk.{L}.attn_v.weight"
    o_name = f"blk.{L}.attn_output.weight"
    if v_name not in idx or o_name not in idx:
        return ""
    st = stats.get(o_name)
    if st is None:
        return " (attn: нет статистики)"
    v_t, o_t = idx[v_name], idx[o_name]
    v = load_bf16(np.frombuffer(v_t.data.tobytes(), dtype=np.uint8),
                  v_t.n_elements).reshape(v_t.shape[1], v_t.shape[0])
    o = load_bf16(np.frombuffer(o_t.data.tobytes(), dtype=np.uint8),
                  o_t.n_elements).reshape(o_t.shape[1], o_t.shape[0])
    n_in = o.shape[1]
    if st.shape[0] != n_in:
        return f" (attn: stats {st.shape[0]} != {n_in})"
    n_kv_out = v.shape[0]
    if n_in % n_kv_out != 0:
        return f" (attn: {n_in} не делится на {n_kv_out})"
    gqa = n_in // n_kv_out
    hd = n_kv_out // args.kv_heads if args.kv_heads else 256
    if n_kv_out % hd != 0:
        return f" (attn: {n_kv_out} не делится на hd={hd})"
    # in_sum2 по query-головам -> усредняем по gqa головам на kv-голову
    per_head = st.reshape(n_in // hd, hd)
    agg = per_head.reshape(n_kv_out // hd, gqa, hd).mean(axis=1)   # [kv_heads, hd]
    e = np.maximum(agg.reshape(-1), 1e-12)
    s_a = np.power(e, args.alpha / 2.0)
    s_a = s_a / np.exp(np.mean(np.log(s_a)))
    s_a = np.clip(s_a, args.smin, args.smax).astype(np.float32)

    v = v / s_a[:, None]                 # выход V scala по kv-каналу
    s_full = np.repeat(s_a.reshape(-1, hd), gqa, axis=0).reshape(-1)
    o = o * s_full[None, :]              # вход attn_output компенсирует

    mm[v_t.data_offset: v_t.data_offset + v_t.n_bytes] = np.frombuffer(
        store_bf16(v), dtype=np.uint8)
    mm[o_t.data_offset: o_t.data_offset + o_t.n_bytes] = np.frombuffer(
        store_bf16(o), dtype=np.uint8)
    return f", attn s in [{s_a.min():.3f}, {s_a.max():.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="исходный BF16 GGUF")
    ap.add_argument("--dst", required=True, help="куда положить патченную копию")
    ap.add_argument("--imatrix", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--smin", type=float, default=0.25)
    ap.add_argument("--smax", type=float, default=4.0)
    ap.add_argument("--attn", action="store_true",
                    help="фолдить также масштаб v<->attn_output")
    ap.add_argument("--kv-heads", type=int, default=4,
                    help="число kv-голов (для расчёта hd)")
    ap.add_argument("--blocks", default="all",
                    help="'all' или список слоёв через запятую, напр. 29,30,31")
    args = ap.parse_args()

    if not os.path.exists(args.dst):
        print(f"копирую {args.src} -> {args.dst} ...", flush=True)
        shutil.copyfile(args.src, args.dst)

    im = gguf.GGUFReader(args.imatrix)
    stats = {}
    for t in im.tensors:
        if t.name.endswith(".in_sum2"):
            base = t.name[: -len(".in_sum2")]
            stats[base] = np.frombuffer(t.data.tobytes(), dtype=np.float32).copy()
    print(f"imatrix: {len(stats)} тензоров со статистикой", flush=True)

    src = gguf.GGUFReader(args.src)
    idx = {t.name: t for t in src.tensors}
    layers = sorted({int(n.split(".")[1]) for n in idx if n.startswith("blk.")})
    if args.blocks != "all":
        want = {int(x) for x in args.blocks.split(",")}
        layers = [l for l in layers if l in want]

    patched = 0
    mm = np.memmap(args.dst, dtype=np.uint8, mode="r+")
    for L in layers:
        up_name = f"blk.{L}.ffn_up.weight"
        dn_name = f"blk.{L}.ffn_down.weight"
        if up_name not in idx or dn_name not in idx:
            continue
        in_sum2 = stats.get(dn_name)
        if in_sum2 is None:
            print(f"  blk.{L}: нет статистики в imatrix, пропуск")
            continue
        # s_i = (E[x_i^2])^(alpha/2), нормировка на геом. среднее
        e = np.maximum(in_sum2, 1e-12)
        s = np.power(e, args.alpha / 2.0)
        s = s / np.exp(np.mean(np.log(s)))
        s = np.clip(s, args.smin, args.smax).astype(np.float32)

        up_t, dn_t = idx[up_name], idx[dn_name]
        # GGUF хранит ne0 как контигуальную ось (вход линейного слоя).
        # Flat-буфер в C-order даёт матрицу [ne1, ne0] = [выход, вход]:
        #   up  : [intermediate, hidden]
        #   down: [hidden, intermediate]
        # Значит промежуточная ось — строки up и столбцы down.
        up = load_bf16(np.frombuffer(up_t.data.tobytes(), dtype=np.uint8),
                       up_t.n_elements).reshape(up_t.shape[1], up_t.shape[0])
        dn = load_bf16(np.frombuffer(dn_t.data.tobytes(), dtype=np.uint8),
                       dn_t.n_elements).reshape(dn_t.shape[1], dn_t.shape[0])
        if s.shape[0] != up.shape[0] or dn.shape[1] != up.shape[0]:
            print(f"  blk.{L}: несовпадение s={s.shape[0]} up={up.shape} dn={dn.shape}, пропуск")
            continue

        up = up / s[:, None]
        dn = dn * s[None, :]

        mm[up_t.data_offset: up_t.data_offset + up_t.n_bytes] = np.frombuffer(
            store_bf16(up), dtype=np.uint8)
        mm[dn_t.data_offset: dn_t.data_offset + dn_t.n_bytes] = np.frombuffer(
            store_bf16(dn), dtype=np.uint8)
        patched += 1
        note = ""
        if args.attn:
            note = fold_attention(mm, idx, stats, L, args)
        print(f"  blk.{L}: s in [{s.min():.3f}, {s.max():.3f}], патч ок{note}", flush=True)

    mm.flush()
    del mm
    print(f"готово: патчено {patched} слоёв -> {args.dst}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
