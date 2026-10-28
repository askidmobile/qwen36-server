#!/usr/bin/env python3
"""Measure 8-D VQ relative error per index width on real GGUF weights.

The DQVL index width (`nc`) is the only precision knob the YoungAi path exposes,
so the 13/11/10 or 13/10/9 tier plans need a measured curve, not a 2^(-b/4)
guess. This script dequantizes a real FFN-shaped tensor, re-quantizes sampled
8-element blocks with k-means codebooks at each width, and prints both the
absolute error and the ratio to a reference width.

    python scripts/qamq/vq_error.py MODEL.gguf [TENSOR] [--bits 8,9,10,11,12,13]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from gguf import GGUFReader, quants


def vq_error(matrix: np.ndarray, dim: int, nc: int, blocks: int, iters: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    flat = matrix.reshape(-1, dim)
    pick = rng.choice(len(flat), size=min(blocks, len(flat)), replace=False)
    sample = flat[pick].astype(np.float32)
    scale = np.abs(sample).max(axis=1, keepdims=True) / 3.0 + 1e-8
    norm = sample / scale
    k = 1 << nc
    if k > len(norm):
        raise ValueError(f"{nc} bits needs {k} blocks, only {len(norm)} sampled")
    codebook = norm[rng.choice(len(norm), size=k, replace=False)].copy()
    for _ in range(iters):
        dist = (norm * norm).sum(1)[:, None] - 2 * norm @ codebook.T + (codebook * codebook).sum(1)[None, :]
        assign = dist.argmin(1)
        counts = np.bincount(assign, minlength=k)
        sums = np.zeros_like(codebook)
        np.add.at(sums, assign, norm)
        used = counts > 0
        codebook[used] = sums[used] / counts[used][:, None]
    dist = (norm * norm).sum(1)[:, None] - 2 * norm @ codebook.T + (codebook * codebook).sum(1)[None, :]
    recon = codebook[dist.argmin(1)] * scale
    return float(((sample - recon) ** 2).sum() / (sample ** 2).sum())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model")
    parser.add_argument("tensor", nargs="?", default=None)
    parser.add_argument("--bits", default="8,9,10,11,12,13")
    parser.add_argument("--reference", type=int, default=12)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=30000)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args(argv)

    reader = GGUFReader(args.model)
    if args.tensor:
        tensor = next((t for t in reader.tensors if t.name == args.tensor), None)
        if tensor is None:
            raise SystemExit(f"{args.model}: no tensor {args.tensor}")
    else:
        candidates = [t for t in reader.tensors if len(t.shape) == 2]
        tensor = max(candidates, key=lambda t: int(np.prod(t.shape)))
    matrix = quants.dequantize(tensor.data, tensor.tensor_type).astype(np.float32)
    rms = float(np.sqrt((matrix**2).mean()))
    print(f"{tensor.name} {tuple(matrix.shape)} RMS {rms:.4f}", file=sys.stderr)

    widths = [int(v) for v in args.bits.split(",") if v.strip()]
    errors = {bits: vq_error(matrix, args.dim, bits, args.blocks, args.iters, args.seed) for bits in widths}
    base = errors.get(args.reference)
    print(f"{'bits':>5} {'rel_err':>10} {'vs_ref':>8}")
    for bits in widths:
        ratio = "" if base is None else f"{errors[bits] / base:8.2f}"
        print(f"{bits:>5} {errors[bits]:>10.6f} {ratio}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
