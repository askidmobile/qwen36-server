#!/usr/bin/env python3
"""VQ error curve on real DeepSeek-V4.1 experts.

`vq_error.py` works on GGUF tensors; this reads the sliced expert ``.npz`` files
that `extract_expert.py` writes, sweeps several experts, and prints the
routing-weighted table the allocator consumes. Runs offline on already-extracted
experts so the 39 GB shards never have to be streamed again.

    python scripts/qamq/vq_curve_v41.py expert-*.npz --bits 9,10,11,12,13
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from vq_error import vq_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("experts", nargs="+", type=Path)
    parser.add_argument("--bits", default="9,10,11,12,13")
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--blocks", type=int, default=40000)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--out", type=Path, help="write the measured curve as JSON")
    args = parser.parse_args(argv)

    widths = [int(v) for v in args.bits.split(",") if v.strip()]
    reference = 12 if 12 in widths else widths[-1]
    per_expert = []
    for path in args.experts:
        data = np.load(path)
        weight = data["weight"]
        meta = json.loads(str(data["meta"]))
        curve = {
            bits: vq_error(weight, args.dim, bits, args.blocks, args.iters, 3)
            for bits in widths
        }
        per_expert.append((meta, curve))
        shown = " ".join(f"{b}:{curve[b]:.5f}" for b in widths)
        print(
            f"{meta['tensor']} e{meta['expert']:>3} {tuple(meta['shape'])} "
            f"type{meta['ggml_type']:<3} {shown}",
            flush=True,
        )

    print(f"\nrelative to {reference} bits (min / median / max across experts):")
    summary = {}
    for bits in widths:
        ratios = sorted(c[bits] / c[reference] for _, c in per_expert)
        median = ratios[len(ratios) // 2]
        summary[str(bits)] = round(median, 4)
        print(f"  {bits:>2} bits  {ratios[0]:6.3f}  {median:6.3f}  {ratios[-1]:6.3f}")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "reference_bits": reference,
                    "dim": args.dim,
                    "blocks": args.blocks,
                    "experts": [
                        {"meta": m, "curve": {str(b): v for b, v in c.items()}}
                        for m, c in per_expert
                    ],
                    "median_ratio_to_reference": summary,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
