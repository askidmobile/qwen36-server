#!/usr/bin/env python3
"""Split a DeepSeek-V4.1 Flash GGUF allocation into dense and routed-expert bytes.

The tensor shapes here are not guesses: they were read out of the downloaded
IQ2_XXS shards (``gguf_audit.py --shapes``) and this model reproduces the real
backbone size to within shard-headers and padding. That matters because the
dense spine decides how much VRAM is left for the expert cache.

    python scripts/qamq/dense_budget.py ALLOCATION.txt [--real-bytes N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# GGML type -> (elements per block, bytes per block)
GGML_BLOCK = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2), "Q8_0": (32, 34), "Q5_K": (256, 176),
    "Q6_K": (256, 210), "Q4_K": (256, 144), "Q3_K": (256, 110), "Q2_K": (256, 84),
    "IQ1_M": (256, 56), "IQ1_S": (256, 50), "IQ2_XXS": (256, 66), "IQ2_XS": (256, 74),
    "IQ2_S": (256, 82), "IQ2_M": (256, 98), "IQ3_XXS": (256, 98), "IQ3_XS": (256, 110),
    "IQ3_S": (256, 110), "IQ4_NL": (32, 18), "IQ4_XS": (256, 136), "Q4_0": (32, 18),
}
H, I, E = 5120, 2304, 384
HIDDEN, HIDDEN_HEAD, Q_LORA, N_HEADS = 5120, 512, 1280, 64
HC = 24  # hyper-connection width
ENGRAM_WKV = (6144, 25600)
KV_SOURCES = (2, 8, 14, 20)
INDEX_SOURCES = (2, 8, 14, 20, 24, 28, 32, 36)
ENGRAM_LAYERS = (1, 14)


def shapes() -> dict[str, tuple[int, ...]]:
    """Exact tensor shapes of the llama.cpp deepseek41 layout."""
    s: dict[str, tuple[int, ...]] = {
        "token_embd.weight": (HIDDEN, 129280),
        "output.weight": (129280, HIDDEN),
        "output_norm.weight": (HIDDEN,),
    }
    for layer in range(40):
        p = f"blk.{layer}."
        s.update({
            p + "attn_q_a.weight": (HIDDEN, Q_LORA),
            p + "attn_q_b.weight": (Q_LORA, N_HEADS * HIDDEN_HEAD),
            p + "attn_kv.weight": (HIDDEN, HIDDEN_HEAD),
            p + "attn_output_a.weight": (4096, 1024, 8),
            p + "attn_output_b.weight": (8192, HIDDEN),
            p + "attn_norm.weight": (HIDDEN,),
            p + "attn_q_a_norm.weight": (Q_LORA,),
            p + "attn_kv_a_norm.weight": (HIDDEN_HEAD,),
            p + "attn_sinks.weight": (64,),
            p + "ffn_norm.weight": (HIDDEN,),
            p + "ffn_gate_inp.weight": (HIDDEN, E),
            p + "exp_probs_b.bias": (E,),
            p + "exp_probs_b_vl.bias": (E,),
            p + "ffn_gate_shexp.weight": (HIDDEN, I),
            p + "ffn_up_shexp.weight": (HIDDEN, I),
            p + "ffn_down_shexp.weight": (I, HIDDEN),
            p + "ffn_gate_exps.weight": (HIDDEN, E * I),
            p + "ffn_up_exps.weight": (HIDDEN, E * I),
            p + "ffn_down_exps.weight": (E * I, HIDDEN),
            p + "hc_attn_base.weight": (HC,),
            p + "hc_attn_fn.weight": (HIDDEN * 4, HC),
            p + "hc_attn_scale.weight": (3,),
            p + "hc_ffn_base.weight": (HC,),
            p + "hc_ffn_fn.weight": (HIDDEN * 4, HC),
            p + "hc_ffn_scale.weight": (3,),
        })
    for layer in KV_SOURCES:
        p = f"blk.{layer}."
        s[p + "attn_compressor_kv.weight"] = (HIDDEN, HIDDEN_HEAD)
        s[p + "attn_compressor_norm.weight"] = (HIDDEN_HEAD,)
        if layer != 20:  # ratio 2 pools, ratio 1 needs no gate
            s[p + "attn_compressor_gate.weight"] = (HIDDEN, HIDDEN_HEAD)
    for layer in INDEX_SOURCES:
        p = f"blk.{layer}."
        s[p + "indexer.attn_q_b.weight"] = (Q_LORA, 32 * 128)
        s[p + "indexer.proj.weight"] = (HIDDEN, 32)
        if layer in KV_SOURCES:  # only kv sources own the index keys
            s[p + "indexer.attn_k.weight"] = (HIDDEN_HEAD, 128)
            s[p + "indexer.k_norm.weight"] = (128,)
    for layer in ENGRAM_LAYERS:
        p = f"blk.{layer}."
        # the released GGUF stores this layer as gate (q*k folded) + wkv, not q/k
        s[p + "engram_gate.weight"] = (HIDDEN, 256)
        s[p + "engram_kv.weight"] = (HIDDEN, 256)
    return s


def nbytes(shape: tuple[int, ...], ggml_type: str) -> int:
    per_block, block_bytes = GGML_BLOCK[ggml_type]
    n = 1
    for d in shape:
        n *= d
    if shape[0] % per_block:
        raise ValueError(f"row {shape[0]} is not whole {ggml_type} blocks")
    return n // per_block * block_bytes


def read_allocation(path: Path) -> dict[str, str]:
    alloc: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, ggml_type = line.rpartition(": ")
        alloc[name] = ggml_type.strip()
    return alloc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("allocation", type=Path, nargs="+")
    parser.add_argument("--real-bytes", type=int, help="total size of the downloaded shards")
    args = parser.parse_args(argv)

    shape = shapes()
    status = 0
    for path in args.allocation:
        alloc = read_allocation(path)
        missing = [k for k in shape if k not in alloc]
        extra = [k for k in alloc if k not in shape and k != "per_layer_token_embd.weight"]
        print(f"== {path.name}")
        if missing or extra:
            print(f"   UNMATCHED missing={missing[:3]} extra={extra[:3]}")
            status = 1
            continue
        dense = sum(nbytes(shape[k], alloc[k]) for k in shape if "exps" not in k)
        experts = sum(nbytes(shape[k], alloc[k]) for k in shape if "exps" in k)
        print(f"   tensors matched    : {len(shape)}/{len(shape)}")
        print(f"   dense (non-expert) : {dense/1e9:8.3f} GB  {dense/1024**3:7.3f} GiB")
        print(f"   routed experts     : {experts/1e9:8.3f} GB  {experts/1024**3:7.3f} GiB")
        print(f"   total backbone     : {(dense+experts)/1e9:8.3f} GB  {(dense+experts)/1024**3:7.3f} GiB")
        if args.real_bytes:
            delta = dense + experts - args.real_bytes
            print(f"   real shards        : {args.real_bytes/1e9:8.3f} GB  delta {delta/1e6:+.1f} MB")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
