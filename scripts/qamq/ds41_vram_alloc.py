#!/usr/bin/env python3
"""Build a VRAM-aware expert allocation for DeepSeek-V4.1 Flash.

The input is routing hotness (JSON rows ``[layer, expert, count]`` or a
Strata/ds4 ``--dump-routing`` trace) and optional YoungAi ``--stats-out``
files measured at several VQ codebook widths.  The output is deliberately
small and concrete:

* ``L e nc`` rows for YoungAi's ``--vq-nc-table``;
* a ranked Strata ``STRP`` expert profile;
* a JSON report with byte accounting and predicted cache coverage.

Weight-only allocation is not trusted here. Existing DeepSeek measurements
show that expert weights are almost equally sensitive. Non-uniform precision
is selected only when the routing-weighted proxy beats the uniform baseline by
the requested gate. Measured YoungAi stats replace the theoretical 8-D VQ
curve; missing measurements stay explicit in the report. Otherwise every
expert gets the same width.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import struct
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

MIB = 1024 * 1024
PROFILE_MAGIC = b"STRP"
PROFILE_VERSION = 1


@dataclass(frozen=True)
class Shape:
    layers: int = 40
    experts: int = 384
    hidden: int = 5120
    intermediate: int = 2304
    vq_dim: int = 8
    codebook_bytes: int = 1
    payload_align: int = 16

    @property
    def expert_count(self) -> int:
        return self.layers * self.experts


@dataclass(frozen=True)
class Budgets:
    vram_mib: int = 12288
    dense_mib: int = 4608
    kv_mib: int = 256
    workspace_mib: int = 1024
    reserve_mib: int = 512
    ram_mib: int = 49152

    @property
    def vram_expert_bytes(self) -> int:
        return (
            max(
                0,
                self.vram_mib
                - self.dense_mib
                - self.kv_mib
                - self.workspace_mib
                - self.reserve_mib,
            )
            * MIB
        )

    @property
    def ram_expert_bytes(self) -> int:
        return self.ram_mib * MIB


@dataclass(frozen=True)
class ErrorPoint:
    error: float
    measured: bool


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def bits_of(nc: int) -> int:
    if nc < 2 or nc & (nc - 1):
        raise ValueError(f"codebook size must be a power of two >= 2, got {nc}")
    return nc.bit_length() - 1


def matrix_payload_bytes(
    rows: int, cols: int, bits: int, dim: int, align: int = 16
) -> int:
    """Bytes of one compressed matrix payload, including its DQVL-style header."""
    if rows <= 0 or cols <= 0 or cols % dim:
        raise ValueError(f"invalid matrix shape {rows}x{cols} for VQ dim {dim}")
    indices = cols // dim
    # DQVL v3 keeps a 12-bit main stream and, for 13 bits, a one-bit plane.
    # Other widths are packed directly by the proposed v4 encoder.
    if bits == 12:
        stream = rows * ((indices * 12 + 7) // 8)
    elif bits == 13:
        stream = rows * ((indices * 12 + 7) // 8 + (indices + 7) // 8)
    else:
        stream = rows * ((indices * bits + 7) // 8)
    raw = 32 + rows * 2 + stream + 8
    return _align(raw, align)


def expert_payload_bytes(bits: int, shape: Shape) -> int:
    """Three routed-expert matrices; the per-tier layer codebook is separate."""
    return (
        matrix_payload_bytes(
            shape.intermediate, shape.hidden, bits, shape.vq_dim, shape.payload_align
        )
        + matrix_payload_bytes(
            shape.intermediate, shape.hidden, bits, shape.vq_dim, shape.payload_align
        )
        + matrix_payload_bytes(
            shape.hidden, shape.intermediate, bits, shape.vq_dim, shape.payload_align
        )
    )


def codebook_bytes(bits: Sequence[int], shape: Shape) -> int:
    """Bytes of one nested codebook per layer, sized to the layer's widest tier."""
    if len(bits) != shape.expert_count:
        raise ValueError(
            f"expected {shape.expert_count} expert widths, got {len(bits)}"
        )
    total = 0
    for layer in range(shape.layers):
        widest = max(bits[layer * shape.experts : (layer + 1) * shape.experts])
        total += (1 << widest) * shape.vq_dim * shape.codebook_bytes
    return total


def total_model_bytes(bits: Sequence[int], shape: Shape) -> int:
    return sum(expert_payload_bytes(b, shape) for b in bits) + codebook_bytes(
        bits, shape
    )


def read_routing(path: Path, shape: Shape) -> list[int]:
    """Read JSON ranks (``[layer, expert, count]``) or a Strata routing trace."""
    raw = path.read_bytes()
    counts = [0] * shape.expert_count
    first = next((b for b in raw if b not in b" \t\r\n"), None)
    if first == ord("["):
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"{path}: routing JSON must be a list")
        for row in data:
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(f"{path}: routing rows must be [layer, expert, count]")
            layer, expert, count = (int(v) for v in row)
            _routing_index(layer, expert, count, path, shape)
            counts[layer * shape.experts + expert] += count
        return counts

    off = 0
    while off < len(raw):
        if off + 8 > len(raw):
            raise ValueError(f"{path}: trailing routing bytes at offset {off}")
        layer, k = struct.unpack_from("<ii", raw, off)
        off += 8
        if k < 0 or k > shape.experts:
            raise ValueError(f"{path}: invalid top-k {k} at offset {off - 8}")
        need = k * 8
        if off + need > len(raw):
            raise ValueError(f"{path}: truncated routing record at offset {off}")
        ids = struct.unpack_from(f"<{k}i", raw, off) if k else ()
        off += need  # ids followed by equally many f32 routing weights
        for expert in ids:
            _routing_index(layer, expert, 1, path, shape)
            counts[layer * shape.experts + expert] += 1
    return counts


def _routing_index(
    layer: int, expert: int, count: int, path: Path, shape: Shape
) -> None:
    if not (0 <= layer < shape.layers and 0 <= expert < shape.experts and count >= 0):
        raise ValueError(f"{path}: invalid routing row [{layer}, {expert}, {count}]")


def read_stats(
    paths: Iterable[Path | str],
) -> tuple[dict[tuple[int, int], dict[int, float]], int]:
    """Read YoungAi ``L e nc sse energy ...`` rows.

    Returns measured squared error keyed by (layer, expert, bits), plus a count
    of rows that were present but malformed and therefore ignored.
    """
    result: dict[tuple[int, int], dict[int, float]] = defaultdict(dict)
    skipped = 0
    for item in paths:
        path = Path(item)
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            try:
                layer, expert, nc = map(int, fields[:3])
                error = float(fields[3])
                bits = bits_of(nc)
            except (ValueError, IndexError):
                skipped += 1
                continue
            if error < 0 or not math.isfinite(error):
                raise ValueError(f"{path}:{line_no}: invalid error {error}")
            old = result[(layer, expert)].get(bits)
            if old is None or error < old:
                result[(layer, expert)][bits] = error
    return dict(result), skipped


def fill_error_curves(
    measured: Mapping[tuple[int, int], Mapping[int, float]],
    candidates: Sequence[int],
    shape: Shape,
    default_error: float,
) -> tuple[dict[int, list[dict[int, ErrorPoint]]], bool]:
    """Interpolate missing widths with the measured 8-D VQ 2^(-b/4) slope."""
    if default_error <= 0 or not math.isfinite(default_error):
        raise ValueError("default error must be positive and finite")
    curves: dict[int, list[dict[int, ErrorPoint]]] = {}
    estimated = False
    for layer in range(shape.layers):
        layer_curves: list[dict[int, ErrorPoint]] = []
        for expert in range(shape.experts):
            points = measured.get((layer, expert), {})
            if points:
                anchor_b = min(
                    points,
                    key=lambda b: (abs(b - (sum(candidates) / len(candidates))), b),
                )
                anchor_e = max(points[anchor_b], 1e-30)
                exact = points
            else:
                anchor_b = candidates[0]
                anchor_e = default_error
                exact = {}
                estimated = True
            curve: dict[int, ErrorPoint] = {}
            for bits in candidates:
                if bits in exact:
                    curve[bits] = ErrorPoint(exact[bits], True)
                else:
                    error = anchor_e * math.pow(2.0, -(bits - anchor_b) / 4.0)
                    curve[bits] = ErrorPoint(error, False)
                    estimated = True
            layer_curves.append(curve)
        curves[layer] = layer_curves
    return curves, estimated


def route_probabilities(counts: Sequence[int], shape: Shape) -> list[float]:
    probs = [0.0] * shape.expert_count
    for layer in range(shape.layers):
        start = layer * shape.experts
        total = sum(counts[start : start + shape.experts])
        if total:
            for expert in range(shape.experts):
                probs[start + expert] = counts[start + expert] / total
        else:
            for expert in range(shape.experts):
                probs[start + expert] = 1.0 / shape.experts
    return probs


def _uniform_width(
    candidates: Sequence[int], budget: int | None, reference: int, shape: Shape
) -> int:
    if reference not in candidates:
        raise ValueError(f"reference bit width {reference} is not in candidates")
    if budget is None:
        return reference
    fitting = [
        b
        for b in candidates
        if total_model_bytes([b] * shape.expert_count, shape) <= budget
    ]
    if not fitting:
        raise ValueError("budget cannot hold the smallest candidate width")
    return max(fitting)


def _allocate_nonuniform(
    curves: Mapping[int, Sequence[Mapping[int, ErrorPoint]]],
    probabilities: Sequence[float],
    candidates: Sequence[int],
    budget: int,
    shape: Shape,
) -> list[int]:
    """Greedy rate-distortion allocation under the uniform payload budget.

    Layer codebooks are excluded here: at most one per bit tier exists and
    their total is a few MiB.  Final accounting includes their exact bytes.
    """
    widths = [candidates[0]] * shape.expert_count
    used = sum(expert_payload_bytes(bits, shape) for bits in widths)
    if used > budget:
        raise ValueError("budget cannot hold the smallest candidate width")
    positions = {bits: i for i, bits in enumerate(candidates)}
    heap: list[tuple[float, int, int, int]] = []

    def push(index: int) -> None:
        layer, expert = divmod(index, shape.experts)
        here = widths[index]
        pos = positions[here]
        if pos + 1 >= len(candidates):
            return
        nxt = candidates[pos + 1]
        delta_bytes = expert_payload_bytes(nxt, shape) - expert_payload_bytes(
            here, shape
        )
        gain = probabilities[index] * (
            curves[layer][expert][here].error - curves[layer][expert][nxt].error
        )
        if gain > 0:
            score = math.inf if delta_bytes == 0 else gain / delta_bytes
            heapq.heappush(heap, (-score, index, here, nxt))

    for index in range(shape.expert_count):
        push(index)
    while heap:
        _, index, old, new = heapq.heappop(heap)
        if widths[index] != old:
            continue
        cost = expert_payload_bytes(new, shape) - expert_payload_bytes(old, shape)
        if used + cost > budget:
            continue
        widths[index] = new
        used += cost
        push(index)
    return widths


def _weighted_error(
    bits: Sequence[int],
    probabilities: Sequence[float],
    curves: Mapping[int, Sequence[Mapping[int, ErrorPoint]]],
    shape: Shape,
) -> float:
    return sum(
        probabilities[index] * curves[layer][expert][bits[index]].error
        for layer in range(shape.layers)
        for expert in range(shape.experts)
        for index in (layer * shape.experts + expert,)
    )


def _rank_for_cache(
    counts: Sequence[int], widths: Sequence[int], shape: Shape, slot_overhead: int
) -> list[dict]:
    ranked = []
    for index, (count, bits) in enumerate(zip(counts, widths)):
        layer, expert = divmod(index, shape.experts)
        size = expert_payload_bytes(bits, shape) + slot_overhead
        ranked.append(
            {
                "layer": layer,
                "expert": expert,
                "count": count,
                "bits": bits,
                "bytes": size,
            }
        )
    # Fractional-knapsack order: route hits won per cached byte.  Missing
    # experts are spread round-robin by layer rather than all sent to the tail.
    ranked.sort(
        key=lambda row: (
            -(row["count"] / row["bytes"]),
            -row["count"],
            row["layer"],
            row["expert"],
        )
    )
    return ranked


def _select_tiers(
    ranked: Sequence[dict], vram_budget: int, ram_budget: int
) -> tuple[list[dict], dict]:
    rows = [dict(row) for row in ranked]
    vram_used = ram_used = 0
    vram_hits = ram_hits = 0
    for row in rows:
        if vram_used + row["bytes"] <= vram_budget:
            row["tier"] = "vram"
            vram_used += row["bytes"]
            vram_hits += row["count"]
        elif ram_used + row["bytes"] <= ram_budget:
            row["tier"] = "ram"
            ram_used += row["bytes"]
            ram_hits += row["count"]
        else:
            row["tier"] = "ssd"
    total_picks = sum(row["count"] for row in rows)
    summary = {
        "vram_experts": sum(row["tier"] == "vram" for row in rows),
        "ram_experts": sum(row["tier"] == "ram" for row in rows),
        "ssd_experts": sum(row["tier"] == "ssd" for row in rows),
        "vram_used_bytes": vram_used,
        "ram_used_bytes": ram_used,
        "vram_coverage": vram_hits / total_picks if total_picks else None,
        "vram_ram_coverage": (vram_hits + ram_hits) / total_picks
        if total_picks
        else None,
    }
    return rows, summary


def write_nc_table(path: Path, widths: Sequence[int], shape: Shape) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("# L e nc; generated by ds41_vram_alloc.py\n")
        for index, bits in enumerate(widths):
            layer, expert = divmod(index, shape.experts)
            handle.write(f"{layer} {expert} {1 << bits}\n")


def write_strata_profile(path: Path, ranked: Sequence[dict], shape: Shape) -> None:
    if shape.layers > 65535 or shape.experts > 65535 or len(ranked) > 0xFFFFFFFF:
        raise ValueError("Strata profile dimensions exceed uint32/uint16 limits")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = [[-1] * shape.experts for _ in range(shape.layers)]
    for slot, row in enumerate(ranked):
        table[row["layer"]][row["expert"]] = slot
    with path.open("wb") as handle:
        handle.write(PROFILE_MAGIC)
        handle.write(
            struct.pack(
                "<5I",
                PROFILE_VERSION,
                shape.layers,
                shape.experts,
                len(ranked),
                len(ranked),
            )
        )
        for row in ranked:
            handle.write(struct.pack("<HH", row["layer"], row["expert"]))
        for layer in table:
            handle.write(struct.pack(f"<{shape.experts}i", *layer))


def _report(
    *,
    shape: Shape,
    budgets: Budgets,
    candidates: Sequence[int],
    counts: Sequence[int],
    probabilities: Sequence[float],
    curves: Mapping[int, Sequence[Mapping[int, ErrorPoint]]],
    widths: Sequence[int],
    uniform_bits: int,
    mode: str,
    improvement: float,
    gate: float,
    estimated: bool,
    skipped_stats: int,
    measured_experts: int,
    fully_measured_experts: int,
    stats_coverage: float,
    evidence_ready: bool,
    allow_estimated_tiers: bool,
    ranked: Sequence[dict],
    tiered: Sequence[dict],
    tier_summary: Mapping[str, object],
) -> dict:
    total_picks = sum(counts)
    coverage = tier_summary["vram_coverage"]
    total = total_model_bytes(widths, shape)
    histogram = {
        str(bits): sum(1 for value in widths if value == bits) for bits in candidates
    }
    return {
        "schema_version": 1,
        "shape": {
            "layers": shape.layers,
            "experts_per_layer": shape.experts,
            "experts_total": shape.expert_count,
            "hidden": shape.hidden,
            "intermediate": shape.intermediate,
            "vq_dim": shape.vq_dim,
        },
        "budget": {
            "vram_mib": budgets.vram_mib,
            "dense_mib": budgets.dense_mib,
            "kv_mib": budgets.kv_mib,
            "workspace_mib": budgets.workspace_mib,
            "reserve_mib": budgets.reserve_mib,
            "vram_expert_mib": round(budgets.vram_expert_bytes / MIB, 3),
            "fixed_codebook_bytes": codebook_bytes(widths, shape),
            "ram_mib": budgets.ram_mib,
        },
        "allocation": {
            "mode": mode,
            "uniform_bits": uniform_bits,
            "uniform_bytes": total_model_bytes(
                [uniform_bits] * shape.expert_count, shape
            ),
            "selected_bytes": total,
            "selected_mib": round(total / MIB, 3),
            "bits": list(widths),
            "histogram": histogram,
            "predicted_weighted_error": {
                "uniform": _weighted_error(
                    [uniform_bits] * shape.expert_count, probabilities, curves, shape
                ),
                "selected": _weighted_error(widths, probabilities, curves, shape),
                "candidate_improvement": improvement,
                "selected_improvement": improvement if mode == "nonuniform" else 0.0,
                "gate": gate,
            },
            "estimated_error_curve": estimated,
            "skipped_stats_rows": skipped_stats,
            "evidence": {
                "measured_experts": measured_experts,
                "fully_measured_experts": fully_measured_experts,
                "coverage": stats_coverage,
                "ready_for_nonuniform": evidence_ready,
                "allow_estimated_tiers": allow_estimated_tiers,
            },
            "safe_for_build": mode == "uniform" or evidence_ready,
            "runtime_note": (
                "current DQVL v3 GPU kernels cover 12/13-bit indices; other widths need a v4 kernel. "
                "Mixed widths also need a nested shared-codebook path, while YoungAi's current "
                "--vq-shared-cb expects one width per layer."
                if any(b not in (12, 13) for b in widths)
                or any(
                    len(
                        set(widths[layer * shape.experts : (layer + 1) * shape.experts])
                    )
                    > 1
                    for layer in range(shape.layers)
                )
                else "uniform v2/v3 widths are representable subject to normal codebook loading"
            ),
        },
        "routing": {
            "picks": total_picks,
            "hot_coverage": coverage,
            "vram_ram_coverage": tier_summary["vram_ram_coverage"],
            "vram_experts": tier_summary["vram_experts"],
            "ram_experts": tier_summary["ram_experts"],
            "ssd_experts": tier_summary["ssd_experts"],
            "vram_used_bytes": tier_summary["vram_used_bytes"],
            "ram_used_bytes": tier_summary["ram_used_bytes"],
            "ranking": tiered,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--routing",
        action="append",
        default=[],
        help="routing JSON or Strata trace (repeatable)",
    )
    parser.add_argument(
        "--stats",
        action="append",
        default=[],
        help="YoungAi --stats-out file (repeatable)",
    )
    parser.add_argument(
        "--candidates", default="10,11,12,13,14", help="comma-separated VQ index widths"
    )
    parser.add_argument(
        "--reference-bits",
        type=int,
        default=12,
        help="uniform width when --budget-gb is omitted",
    )
    parser.add_argument(
        "--budget-gb", type=float, help="total routed-expert storage budget"
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.02,
        help="minimum weighted-error gain over uniform",
    )
    parser.add_argument(
        "--min-stats-coverage",
        type=float,
        default=1.0,
        help="fraction of experts needing at least two measured widths for non-uniform tiers",
    )
    parser.add_argument(
        "--allow-estimated-tiers",
        action="store_true",
        help="allow the theoretical 8-D VQ curve without full activation calibration (experimental)",
    )
    parser.add_argument("--min-bits", type=int, default=10)
    parser.add_argument("--max-bits", type=int, default=14)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--hidden", type=int, default=5120)
    parser.add_argument("--intermediate", type=int, default=2304)
    parser.add_argument("--vq-dim", type=int, default=8)
    parser.add_argument("--codebook-bytes", type=int, default=1)
    parser.add_argument("--slot-overhead", type=int, default=0)
    parser.add_argument("--vram-mib", type=int, default=12288)
    parser.add_argument("--dense-mib", type=int, default=4608)
    parser.add_argument("--kv-mib", type=int, default=256)
    parser.add_argument("--workspace-mib", type=int, default=1024)
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--ram-mib", type=int, default=49152)
    parser.add_argument("--default-error", type=float, default=1.0)
    parser.add_argument("--out-table", type=Path)
    parser.add_argument("--out-profile", type=Path)
    parser.add_argument("--out-report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shape = Shape(
        layers=args.layers,
        experts=args.experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        vq_dim=args.vq_dim,
        codebook_bytes=args.codebook_bytes,
    )
    if (
        shape.layers <= 0
        or shape.experts <= 0
        or shape.hidden <= 0
        or shape.intermediate <= 0
    ):
        raise ValueError("shape values must be positive")
    if args.vq_dim <= 0 or args.codebook_bytes <= 0:
        raise ValueError("VQ dimension and codebook element size must be positive")
    candidates = sorted(
        {int(value) for value in args.candidates.split(",") if value.strip()}
    )
    if (
        not candidates
        or candidates[0] < args.min_bits
        or candidates[-1] > args.max_bits
    ):
        raise ValueError("candidate widths must lie within --min-bits..--max-bits")
    if any(bits < 1 or bits > 24 for bits in candidates):
        raise ValueError("candidate widths must be in 1..24")
    if args.min_improvement < 0:
        raise ValueError("--min-improvement must be non-negative")
    if not 0.0 <= args.min_stats_coverage <= 1.0:
        raise ValueError("--min-stats-coverage must be in [0, 1]")

    counts = [0] * shape.expert_count
    for path in map(Path, args.routing):
        for index, value in enumerate(read_routing(path, shape)):
            counts[index] += value
    probabilities = route_probabilities(counts, shape)
    measured, skipped = read_stats(args.stats)
    curves, estimated = fill_error_curves(
        measured, candidates, shape, args.default_error
    )

    budget = None if args.budget_gb is None else int(args.budget_gb * 1_000_000_000)
    uniform_bits = _uniform_width(candidates, budget, args.reference_bits, shape)
    uniform_widths = [uniform_bits] * shape.expert_count
    uniform_payload = sum(
        expert_payload_bytes(uniform_bits, shape) for _ in range(shape.expert_count)
    )
    uniform_total = uniform_payload + codebook_bytes(uniform_widths, shape)
    total_cap = uniform_total if budget is None else budget
    # Reserve a max-width nested codebook per layer so the selected mixed-tier
    # artifact cannot exceed the uniform budget after exact codebook accounting.
    max_codebooks = (
        shape.layers * (1 << candidates[-1]) * shape.vq_dim * shape.codebook_bytes
    )
    cap = total_cap - max_codebooks
    if cap <= 0:
        raise ValueError("budget is too small for the largest candidate codebook")
    proposed = _allocate_nonuniform(curves, probabilities, candidates, cap, shape)
    uniform_error = _weighted_error(uniform_widths, probabilities, curves, shape)
    proposed_error = _weighted_error(proposed, probabilities, curves, shape)
    improvement = (
        0.0 if uniform_error <= 0 else max(0.0, 1.0 - proposed_error / uniform_error)
    )
    fully_measured = sum(
        1 for points in measured.values() if len(set(points)) >= 2
    )
    stats_coverage = fully_measured / shape.expert_count
    evidence_ready = stats_coverage >= args.min_stats_coverage
    evidence_allowed = evidence_ready or args.allow_estimated_tiers
    mode = (
        "nonuniform"
        if evidence_allowed and improvement >= args.min_improvement
        else "uniform"
    )
    widths = proposed if mode == "nonuniform" else uniform_widths
    selected_bytes = total_model_bytes(widths, shape)
    if total_cap and selected_bytes > total_cap:
        raise ValueError(
            f"selected allocation {selected_bytes} exceeds budget {total_cap}"
        )

    budgets = Budgets(
        vram_mib=args.vram_mib,
        dense_mib=args.dense_mib,
        kv_mib=args.kv_mib,
        workspace_mib=args.workspace_mib,
        reserve_mib=args.reserve_mib,
        ram_mib=args.ram_mib,
    )
    ranked = _rank_for_cache(counts, widths, shape, args.slot_overhead)
    fixed_codebooks = codebook_bytes(widths, shape)
    vram_cache_budget = max(0, budgets.vram_expert_bytes - fixed_codebooks)
    tiered, tier_summary = _select_tiers(
        ranked, vram_cache_budget, budgets.ram_expert_bytes
    )
    report = _report(
        shape=shape,
        budgets=budgets,
        candidates=candidates,
        counts=counts,
        probabilities=probabilities,
        curves=curves,
        widths=widths,
        uniform_bits=uniform_bits,
        mode=mode,
        improvement=improvement,
        gate=args.min_improvement,
        estimated=estimated,
        skipped_stats=skipped,
        measured_experts=len(measured),
        fully_measured_experts=fully_measured,
        stats_coverage=stats_coverage,
        evidence_ready=evidence_ready,
        allow_estimated_tiers=args.allow_estimated_tiers,
        ranked=ranked,
        tiered=tiered,
        tier_summary=tier_summary,
    )

    if args.out_table:
        write_nc_table(args.out_table, widths, shape)
    if args.out_profile:
        write_strata_profile(args.out_profile, ranked, shape)
    if args.out_report:
        args.out_report.parent.mkdir(parents=True, exist_ok=True)
        args.out_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    if not any((args.out_table, args.out_profile, args.out_report)):
        json.dump(report, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        coverage = report["routing"]["hot_coverage"]
        coverage_text = "n/a" if coverage is None else f"{coverage:.1%}"
        print(
            f"{mode}: {report['allocation']['selected_mib']:.2f} MiB, "
            f"candidate error gain {improvement:.2%} "
            f"({'accepted' if mode == 'nonuniform' else 'rejected'}), "
            f"VRAM experts {tier_summary['vram_experts']}, coverage {coverage_text}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
