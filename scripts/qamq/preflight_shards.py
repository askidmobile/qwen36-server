#!/usr/bin/env python3
"""Check a DeepSeek-V4.1 GGUF folder against what the converter and runtime need.

The released builds before 2026-10-27 use four shards and the names
``engram_q``/``engram_k``/``engram_wkv``; llama.cpp b11870+ and the current Strata
read the six-shard layout with ``engram_gate``/``engram_kv``. Conversion rewrites
138 GB, so check first rather than discover a missing shard halfway through.

    python scripts/qamq/preflight_shards.py MODEL_DIR
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

from gguf_audit import read_names as read_header_names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dir")
    args = parser.parse_args(argv)

    shards = sorted(
        p for p in glob.glob(os.path.join(args.model_dir, "*.gguf"))
        if not p.endswith(".done")
    )
    if not shards:
        print("no .gguf shards found", file=sys.stderr)
        return 1

    names: set[str] = set()
    total = 0
    for path in shards:
        total += os.path.getsize(path)
        for name in read_header_names(path):
            names.add(name)
    print(f"shards {len(shards)}  bytes {total / 1e9:.2f} GB  tensors {len(names)}")

    old_layout = {n for n in names if "engram_wkv" in n or "engram_q" in n or "engram_k." in n}
    new_layout = {n for n in names if "engram_gate" in n or "engram_kv" in n}
    combined = "per_layer_token_embd.weight" in names

    print(f"  old engram names : {len(old_layout)}")
    print(f"  new engram names : {len(new_layout)}")
    print(f"  combined table   : {combined}")

    if old_layout and not combined:
        print("\nneeds convert_ds41_upstream.py (pre-2026-10-27 layout)")
        print(f"  free space needed: {total / 1e9:.0f} GB for the rewritten shards")
        layers = sorted({int(n.split(".")[1]) for n in old_layout})
        print(f"  engram layers in backbone: {layers}")
        return 2
    if combined and not old_layout:
        print("\nready: current layout, runtime can load it as-is")
        return 0
    print("\nmixed layout: inspect before running anything", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
