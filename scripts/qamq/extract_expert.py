#!/usr/bin/env python3
"""Slice one routed expert out of a real DeepSeek-V4.1 GGUF shard.

The expert tensors are ``{hidden, experts*intermediate}`` with experts
concatenated along dim 1, so expert ``e`` is a contiguous row range. Reading a
few megabytes instead of the 39 GB shard is what makes it possible to measure the
VQ width curve on real V4.1 weights instead of another model's FFN.

    python scripts/qamq/extract_expert.py MODEL_DIR --layer 20 --expert 7 \
        --tensor ffn_up_exps --out expert.npz
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
TYPE_BYTES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34),
    9: (32, 36), 10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176),
    14: (256, 210), 15: (256, 292), 16: (256, 66), 17: (256, 74), 18: (256, 98),
    19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82), 23: (256, 136),
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56),
    30: (1, 2), 34: (1, 1), 39: (32, 17),
}


def read_header(path: str) -> tuple[list[tuple[str, tuple[int, ...], int, int]], int, int]:
    """Tensor infos, data offset, and alignment of one GGUF file."""
    with open(path, "rb") as f:
        size = os.path.getsize(path)
        buf = f.read(min(size, 64 << 20))
    if buf[:4] != b"GGUF":
        raise ValueError(f"{path}: not a GGUF file")
    n_tensors, n_kv = struct.unpack_from("<QQ", buf, 8)
    off = 24
    alignment = 32

    def u(fmt):
        nonlocal off
        v, = struct.unpack_from(fmt, buf, off)
        off += struct.calcsize(fmt)
        return v

    def s():
        nonlocal off
        n = u("<Q")
        v = buf[off:off + n].decode("utf-8", "replace")
        off += n
        return v

    def skip(t, depth=0):
        nonlocal off
        if t == 8:
            s()
        elif t == 9:
            at = u("<I")
            n = u("<Q")
            for _ in range(n):
                skip(at, depth + 1)
        else:
            u(SCALAR[t])

    for _ in range(n_kv):
        key = s()
        t = u("<I")
        if key == "general.alignment" and t == 4:
            alignment = u("<I")
        else:
            skip(t)
    tensors = []
    for _ in range(n_tensors):
        name = s()
        nd = u("<I")
        dims = tuple(u("<Q") for _ in range(nd))
        ty = u("<I")
        toff = u("<Q")
        tensors.append((name, dims, ty, toff))
    data_start = (off + alignment - 1) // alignment * alignment
    return tensors, data_start, alignment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_dir")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--expert", type=int, default=7)
    parser.add_argument("--tensor", default="ffn_up_exps")
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    want = f"blk.{args.layer}.{args.tensor}.weight"
    for path in sorted(glob.glob(os.path.join(args.model_dir, "*-0000*-of-*.gguf"))):
        if path.endswith(".done"):
            continue
        tensors, data_start, alignment = read_header(path)
        info = next((t for t in tensors if t[0] == want), None)
        if info is None:
            continue
        name, dims, ty, toff = info
        per_block, block_bytes = TYPE_BYTES[ty]
        # ggml layout is (ne0=hidden, ne1=intermediate, ne2=experts), ne0 innermost,
        # so one expert is a contiguous (intermediate x hidden) slab
        if len(dims) == 3:
            hidden, inter, n_experts = dims
        elif len(dims) == 2:
            hidden = dims[0]
            n_experts = dims[1] // args.intermediate
            inter = args.intermediate
        else:
            raise SystemExit(f"{want}: unsupported shape {dims}")
        if not 0 <= args.expert < n_experts:
            raise SystemExit(f"expert {args.expert} outside 0..{n_experts - 1}")
        if hidden % per_block:
            raise SystemExit(f"row {hidden} is not whole {per_block}-element blocks")

        row_bytes = hidden // per_block * block_bytes
        offset = data_start + toff + args.expert * inter * row_bytes
        length = inter * row_bytes
        with open(path, "rb") as f:
            f.seek(offset)
            raw = f.read(length)
        if len(raw) != length:
            raise SystemExit(f"short read: {len(raw)} of {length} bytes")

        from gguf import quants
        dense = quants.dequantize(np.frombuffer(raw, dtype=np.uint8), ty)
        dense = dense.astype(np.float32)
        if dense.shape != (inter, hidden):
            dense = dense.reshape(inter, hidden)
        np.savez(
            args.out,
            weight=dense,
            meta=json.dumps({
                "tensor": want,
                "shard": os.path.basename(path),
                "data_offset": offset,
                "ggml_type": int(ty),
                "expert": args.expert,
                "experts_total": n_experts,
                "shape": list(dense.shape),
                "intermediate": inter,
                "hidden": hidden,
            }),
        )
        rms = float(np.sqrt((dense ** 2).mean()))
        print(f"{want} expert {args.expert}/{n_experts} {dense.shape} type {ty} rms {rms:.5f} -> {args.out}")
        return 0
    raise SystemExit(f"{want} not found in {args.model_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
