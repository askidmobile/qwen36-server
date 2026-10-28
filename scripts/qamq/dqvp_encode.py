#!/usr/bin/env python3
"""Encode a real matrix into a DQVP v4 pack at per-group VQ widths.

This closes the loop the allocator only predicts: quantize, pack, read back,
dequantize, and measure the error that actually came out. It works on any GGUF
tensor whose row length is a multiple of the VQ dimension, so a real FFN matrix
of another model can validate the pipeline before V4.1 weights are available.

    python scripts/qamq/dqvp_encode.py MODEL.gguf --tensor NAME \
        --groups 8 --plan 13,12,10,9 --out pack.bin

Each row group gets the next width from ``--plan`` (cyclic), which is how a
hot/warm/cold tier map looks once it is flattened onto real matrices.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import dqvp_pack as dq


def _assign(blocks: np.ndarray, codebook: np.ndarray, chunk: int = 1 << 16) -> np.ndarray:
    """Nearest-codebook assignment in row chunks, so 11.8M blocks do not need a
    (blocks x codes) distance matrix in memory at once."""
    out = np.empty(len(blocks), dtype=np.int64)
    cb2 = (codebook * codebook).sum(1)[None, :]
    for start in range(0, len(blocks), chunk):
        part = blocks[start:start + chunk]
        dist = (part * part).sum(1)[:, None] - 2 * part @ codebook.T + cb2
        out[start:start + len(part)] = dist.argmin(1)
    return out


def fit_codebook(blocks: np.ndarray, nc: int, iters: int, seed: int, sample: int = 1 << 18) -> np.ndarray:
    """k-means codebook over scale-normalized blocks; returns (1<<nc, dim) f32.

    The codebook is fitted on at most ``sample`` blocks but applied to all of them,
    which is what keeps a 2304x5120 expert tractable.
    """
    rng = np.random.default_rng(seed)
    k = 1 << nc
    if len(blocks) > sample:
        fit_blocks = blocks[rng.choice(len(blocks), size=sample, replace=False)]
    else:
        fit_blocks = blocks
    if len(fit_blocks) < k:
        raise ValueError(f"{nc} bits needs {k} blocks, only {len(fit_blocks)} available")
    codebook = fit_blocks[rng.choice(len(fit_blocks), size=k, replace=False)].copy()
    for _ in range(iters):
        assign = _assign(fit_blocks, codebook)
        counts = np.bincount(assign, minlength=k)
        sums = np.zeros_like(codebook)
        np.add.at(sums, assign, fit_blocks)
        used = counts > 0
        codebook[used] = sums[used] / counts[used][:, None]
    return codebook.astype(np.float32)


def decode(slots, codecs, rows: int, cols: int, dim: int, n_rows=None) -> np.ndarray:
    """Rebuild a matrix from container bytes alone (no residual state)."""
    dtype = {1: np.uint8, 2: np.float16, 4: np.float32}
    n_rows = n_rows or {slot.expert: 1 for slot in slots}
    out = np.empty((rows, cols), dtype=np.float32)
    blocks_per_row = cols // dim
    for slot in slots:
        codec = codecs[slot.codec_id]
        if codec.dim != dim:
            raise ValueError(f"codec dim {codec.dim} != {dim}")
        dtype_ = dtype[codec.elem_bytes]
        codebook = np.frombuffer(codec.codebook, dtype=dtype_).astype(np.float32)
        codebook = codebook.reshape(1 << codec.nc, dim)
        rows_in_slot = n_rows.get(slot.expert)
        if rows_in_slot is None:
            raise ValueError(f"slot {slot.expert} is not part of this matrix")
        idx = dq._unpack_indices(slot.indices, blocks_per_row * rows_in_slot, codec.nc)
        gains = np.frombuffer(slot.scales, dtype=np.float16).astype(np.float32)
        if len(gains) != rows_in_slot:
            raise ValueError(
                f"slot {slot.expert}: {len(gains)} gains for {rows_in_slot} rows"
            )
        block_rows = np.repeat(gains, blocks_per_row)[:, None]
        out[slot.expert : slot.expert + rows_in_slot] = (
            codebook[idx] * block_rows
        ).reshape(rows_in_slot, cols)
    return out


def load_matrix(path: str, name: str | None) -> tuple[np.ndarray, str]:
    """Read a GGUF tensor or a sliced-expert ``.npz`` written by extract_expert.py."""
    if path.endswith(".npz"):
        data = np.load(path)
        return data["weight"].astype(np.float32), name or f"{path}:{data['meta']}"
    import gguf
    from gguf import quants

    reader = gguf.GGUFReader(path)
    if name:
        tensor = next((t for t in reader.tensors if t.name == name), None)
        if tensor is None:
            raise SystemExit(f"{path}: no tensor {name}")
    else:
        candidates = [
            t for t in reader.tensors
            if len(t.shape) == 2 and int(min(t.shape)) >= 512 and int(np.prod(t.shape)) > 1 << 20
        ]
        if not candidates:
            raise SystemExit("no 2-D tensor large enough")
        tensor = max(candidates, key=lambda t: int(np.prod(t.shape)))
    return quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32), tensor.name


def encode(matrix: np.ndarray, plan: list[int], dim: int, iters: int, seed: int):
    """Split rows into len(plan) groups and VQ each at its width."""
    rows, cols = matrix.shape
    if cols % dim:
        raise ValueError(f"row length {cols} is not a multiple of vq_dim {dim}")
    if rows % len(plan):
        raise ValueError(f"{rows} rows do not split into {len(plan)} groups")
    per = rows // len(plan)
    blocks_per_row = cols // dim

    codecs, slots, recon = [], [], np.empty_like(matrix)
    for group, nc in enumerate(plan):
        r0, r1 = group * per, (group + 1) * per
        chunk = matrix[r0:r1].astype(np.float32)
        # one gain per row, as the allocator's byte model prices it: 2 bytes per
        # row instead of 2 bytes per 8-element block (which would cost 2 bpw)
        gains = (np.abs(chunk).max(axis=1, keepdims=True) / 3.0 + 1e-12).astype(np.float32)
        norm = chunk / gains
        blocks = norm.reshape(-1, dim)
        codebook = fit_codebook(blocks, nc, iters, seed + group)
        idx = _assign(blocks, codebook)
        recon[r0:r1] = (codebook[idx] * gains.repeat(blocks_per_row, axis=0)).reshape(chunk.shape)

        codec_id = len(codecs)
        codecs.append(dq.Codec(nc=nc, dim=dim, codebook=codebook.astype(np.float16).tobytes()))
        # one slot per expert-sized group: that is the granularity a real pack has,
        # so per-slot header cost stays negligible
        slots.append(
            dq.ExpertSlot(
                layer=0,
                expert=r0,
                role=0,
                codec_id=codec_id,
                indices=dq._pack_indices(idx.tolist(), nc, dim),
                scales=gains.astype(np.float16).tobytes(),
            )
        )
    return codecs, slots, recon


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model")
    parser.add_argument("--tensor", default=None)
    parser.add_argument("--plan", default="13,12,10,9")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--iters", type=int, default=6)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out", type=Path, default=Path("dqvp-pack.bin"))
    args = parser.parse_args(argv)

    matrix, label = load_matrix(args.model, args.tensor)
    plan = [int(v) for v in args.plan.split(",") if v.strip()]
    if not plan:
        raise SystemExit("--plan must list at least one width")
    while len(plan) > 1 and matrix.shape[0] % len(plan):
        plan = plan[:-1]
    if matrix.shape[0] % len(plan):
        raise SystemExit(f"{matrix.shape[0]} rows do not split into {len(plan)} groups")
    tensor = type("T", (), {"name": label})()

    codecs, slots, recon = encode(matrix, plan, args.dim, args.iters, args.seed)
    codecs, slots, recon = encode(matrix, plan, args.dim, args.iters, args.seed)
    n_rows = {slot.expert: matrix.shape[0] // len(plan) for slot in slots}
    accounting = dq.write_container(args.out, slots, codecs)
    back = dq.read_container(args.out)
    if len(back["slots"]) != len(slots):
        raise SystemExit("slot count changed on read-back")

    # Decode from the file, not from the in-memory reconstruction. f16 block
    # scales are the only difference, so the gap is a measured storage cost.
    decoded = decode(
        back["slots"],
        back["codecs"],
        matrix.shape[0],
        matrix.shape[1],
        args.dim,
        n_rows=n_rows,
    )
    err = float(((matrix - decoded) ** 2).sum() / (matrix ** 2).sum())
    err_mem = float(((matrix - recon) ** 2).sum() / (matrix ** 2).sum())
    scale_penalty = float(np.abs(decoded - recon).max())
    float_bytes = matrix.size * 4
    print(f"tensor {tensor.name} {matrix.shape} plan {plan}")
    print(f"  file-decode rel_err {err:.5f} (f32 scales would give {err_mem:.5f}, "
          f"f16 scale cost max {scale_penalty:.6g})")
    print(f"  payload {accounting['payload_bytes'] / 1e6:.2f} MB "
          f"vs float32 {float_bytes / 1e6:.2f} MB "
          f"= {32 * accounting['payload_bytes'] / float_bytes:.2f} bits/weight")
    print(f"  overhead {accounting['overhead_bytes'] / 1e6:.2f} MB, slots {accounting['slots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
