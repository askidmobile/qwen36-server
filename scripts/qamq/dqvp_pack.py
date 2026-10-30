#!/usr/bin/env python3
"""Reference container for DQVP v4: per-expert VQ codec, one index per expert.

Why a new container: GGUF stores all 384 routed experts of a layer in one tensor
with one ``ggml_type``, so per-expert width is not expressible. DQVP v4 carries a
per-expert codec id and byte range instead, which is what the allocator's
``L e nc`` map needs in order to become a file.

Layout (little endian)::

    header    magic "DQVP", version u32, layers u32, experts u32,
              vq_dim u32, codec_count u32, expert_count u32
    codecs    codec_count × (nc u8, dim u8, codebook_bytes u32, offset u64)
    codebooks concatenated (one nested codebook per layer, widest tier first)
    index     expert_count × (layer u16, expert u16, role u8, codec_id u8,
                               length u32, offset u64, n_scales u32, scale_offset u64)
    payload   VQ index streams, each ``ceil(blocks * nc / 8)`` bytes, 16-aligned,
              followed by that slot's f16 block scales

Block scales live in the container, not beside it: without them the index stream
is not invertible, and GGUF's per-block scale is exactly what makes its tensors
self-contained.

The role byte is 0=gate, 1=up, 2=down, 3=shared, so one expert entry per matrix.
Only index streams are packed here; the quantizer that produces them is a
separate step, because packing is what GGUF cannot express and therefore what
needs a reference implementation first.
"""
from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

MAGIC = b"DQVP"
VERSION = 4
ALIGN = 16
HEADER = struct.Struct("<4s6I")
CODEC = struct.Struct("<BBBBIQ")
ENTRY = struct.Struct("<HHBBIQII")
ROLES = ("gate", "up", "down", "shared")


@dataclass(frozen=True)
class Codec:
    nc: int
    dim: int
    codebook: bytes
    elem_bytes: int = 2  # f16 by default; 1 = E4M3, 4 = f32


@dataclass(frozen=True)
class ExpertSlot:
    layer: int
    expert: int
    role: int
    codec_id: int
    indices: bytes
    scales: bytes = b""  # f16 per VQ block; empty means scale-free codec


def _align(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def _pack_indices(indices, nc: int, dim: int) -> bytes:
    """Bit-pack 8-bit-truncatable index values at ``nc`` bits each."""
    bits = bytearray()
    acc = 0
    nbits = 0
    limit = 1 << nc
    for value in indices:
        if not 0 <= value < limit:
            raise ValueError(f"index {value} does not fit {nc} bits")
        acc |= value << nbits
        nbits += nc
        while nbits >= 8:
            bits.append(acc & 0xFF)
            acc >>= 8
            nbits -= 8
    if nbits:
        bits.append(acc & 0xFF)
    expected = (len(indices) * nc + 7) // 8
    if len(bits) != expected:
        raise AssertionError(f"packed {len(bits)} bytes, expected {expected}")
    return bytes(bits)


def _unpack_indices(blob: bytes, count: int, nc: int) -> list[int]:
    out = []
    acc = 0
    nbits = 0
    pos = 0
    mask = (1 << nc) - 1
    while len(out) < count:
        while nbits < nc:
            if pos >= len(blob):
                raise ValueError("index stream is shorter than the header claims")
            acc |= blob[pos] << nbits
            pos += 1
            nbits += 8
        out.append(acc & mask)
        acc >>= nc
        nbits -= nc
    return out


def write_container(path: Path, slots, codecs) -> dict:
    """Write the container and return the byte accounting actually used."""
    if not codecs:
        raise ValueError("at least one codec is required")
    if any(c.nc < 1 or c.nc > 24 for c in codecs):
        raise ValueError("codec widths must be in 1..24")
    for c in codecs:
        if c.dim <= 0 or c.elem_bytes <= 0:
            raise ValueError("codec dim and elem_bytes must be positive")
        if len(c.codebook) != (1 << c.nc) * c.dim * c.elem_bytes:
            raise ValueError(
                f"codec {c.nc}b needs {(1 << c.nc) * c.dim * c.elem_bytes} codebook "
                f"bytes, got {len(c.codebook)}"
            )

    expert_count = max(s.expert for s in slots) + 1 if slots else 0
    layers = max(s.layer for s in slots) + 1 if slots else 0
    dim = codecs[0].dim
    if any(c.dim != dim for c in codecs):
        raise ValueError("all codecs in one container must share vq_dim")

    offset = HEADER.size + CODEC.size * len(codecs)
    codebook_offsets, codebook_blob = [], bytearray()
    for codec in codecs:
        codebook_offsets.append(offset + len(codebook_blob))
        codebook_blob += codec.codebook
    index_offset = _align(offset + len(codebook_blob))

    rows = []
    payload_offset = _align(index_offset + ENTRY.size * len(slots))
    for slot in slots:
        idx_off = payload_offset
        payload_offset = _align(payload_offset + len(slot.indices))
        scale_off = payload_offset
        payload_offset = _align(payload_offset + len(slot.scales))
        rows.append((slot, idx_off, scale_off))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(HEADER.pack(MAGIC, VERSION, layers, expert_count, dim, len(codecs), len(slots)))
        for codec, off in zip(codecs, codebook_offsets):
            handle.write(CODEC.pack(codec.nc, codec.dim, codec.elem_bytes, 0, len(codec.codebook), off))
        handle.write(codebook_blob)
        handle.write(b"\0" * (index_offset - handle.tell()))
        for slot, idx_off, scale_off in rows:
            handle.write(
                ENTRY.pack(
                    slot.layer,
                    slot.expert,
                    slot.role,
                    slot.codec_id,
                    len(slot.indices),
                    idx_off,
                    len(slot.scales),
                    scale_off,
                )
            )
        handle.write(b"\0" * ((rows[0][1] if rows else index_offset + ENTRY.size * len(slots)) - handle.tell()))
        for slot, idx_off, scale_off in rows:
            handle.write(slot.indices)
            handle.write(b"\0" * (_align(len(slot.indices)) - len(slot.indices)))
            handle.write(slot.scales)
            handle.write(b"\0" * (_align(len(slot.scales)) - len(slot.scales)))
    size = path.stat().st_size
    payload = sum(len(s.indices) + len(s.scales) for s in slots)
    return {
        "bytes": size,
        "payload_bytes": payload,
        "index_bytes": sum(len(s.indices) for s in slots),
        "scale_bytes": sum(len(s.scales) for s in slots),
        "slots": len(rows),
        "codecs": len(codecs),
        "overhead_bytes": size - payload,
    }


def read_container(path: Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < HEADER.size:
        raise ValueError("file is shorter than the DQVP header")
    magic, version, layers, experts, dim, n_codecs, n_slots = HEADER.unpack_from(raw, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version {version}")
    if n_codecs == 0 or n_codecs > 32:
        raise ValueError("invalid codec count")
    if HEADER.size + CODEC.size * n_codecs > len(raw):
        raise ValueError("codec table runs past the end of the file")

    codecs = []
    for i in range(n_codecs):
        nc, cdim, elem, _pad, nbytes, off = CODEC.unpack_from(raw, HEADER.size + CODEC.size * i)
        if off + nbytes > len(raw):
            raise ValueError(f"codec {i} codebook runs past the end of the file")
        if nbytes != (1 << nc) * cdim * elem:
            raise ValueError(f"codec {i} codebook size does not match its width")
        codecs.append(Codec(nc=nc, dim=cdim, elem_bytes=elem, codebook=raw[off : off + nbytes]))

    index_offset = _align(HEADER.size + CODEC.size * n_codecs + sum(len(c.codebook) for c in codecs))
    if index_offset + ENTRY.size * n_slots > len(raw):
        raise ValueError("expert index runs past the end of the file")

    slots = []
    for i in range(n_slots):
        layer, expert, role, codec_id, length, off, n_scales, scale_off = ENTRY.unpack_from(
            raw, index_offset + ENTRY.size * i
        )
        if codec_id >= n_codecs:
            raise ValueError(f"slot {i} references codec {codec_id} of {n_codecs}")
        if length == 0 or off + length > len(raw):
            raise ValueError(f"slot {i} payload runs past the end of the file")
        if n_scales and scale_off + n_scales > len(raw):
            raise ValueError(f"slot {i} scales run past the end of the file")
        slots.append(
            ExpertSlot(
                layer=layer,
                expert=expert,
                role=role,
                codec_id=codec_id,
                indices=raw[off : off + length],
                scales=raw[scale_off : scale_off + n_scales],
            )
        )
    return {
        "layers": layers,
        "experts": experts,
        "vq_dim": dim,
        "codecs": codecs,
        "slots": slots,
    }


def self_check(path: Path, blocks: int = 64, seed: int = 11) -> dict:
    """Write a mixed-width container, read it back, and compare every index."""
    import random

    rng = random.Random(seed)
    codecs = [
        Codec(nc=nc, dim=8, elem_bytes=2,
              codebook=bytes(rng.randrange(256) for _ in range((1 << nc) * 8 * 2)))
        for nc in (9, 10, 12, 13)
    ]
    slots = []
    for layer in range(2):
        for expert in range(3):
            codec_id = (layer + expert) % len(codecs)
            nc = codecs[codec_id].nc
            values = [rng.randrange(1 << nc) for _ in range(blocks)]
            scales = struct.pack(f"<{blocks}H", *[rng.randrange(65536) for _ in range(blocks)])
            slots.append(
                ExpertSlot(
                    layer=layer,
                    expert=expert,
                    role=(layer + expert) % 3,
                    codec_id=codec_id,
                    indices=_pack_indices(values, nc, 8),
                    scales=scales,
                )
            )
    accounting = write_container(path, slots, codecs)
    got = read_container(path)
    if got["layers"] != 2 or got["experts"] != 3 or len(got["slots"]) != len(slots):
        raise AssertionError("container dimensions did not round-trip")
    for want, have in zip(slots, got["slots"]):
        if (want.layer, want.expert, want.role, want.codec_id) != (
            have.layer,
            have.expert,
            have.role,
            have.codec_id,
        ):
            raise AssertionError("expert index entry changed")
        nc = codecs[have.codec_id].nc
        if _unpack_indices(have.indices, blocks, nc) != _unpack_indices(want.indices, blocks, nc):
            raise AssertionError("index stream changed")
        if have.scales != want.scales:
            raise AssertionError("block scales changed")
    accounting["verified_slots"] = len(slots)
    return accounting


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--self-check", type=Path, metavar="PATH", help="write and verify a synthetic container")
    parser.add_argument("--inspect", type=Path, metavar="PATH", help="read a container and print its shape")
    args = parser.parse_args(argv)
    if args.self_check:
        info = self_check(args.self_check)
        print(
            f"ok: {info['verified_slots']} slots, {info['codecs']} codecs, "
            f"{info['bytes']} bytes, overhead {info['overhead_bytes']}"
        )
        return 0
    if args.inspect:
        got = read_container(args.inspect)
        widths = sorted({c.nc for c in got["codecs"]})
        print(
            f"DQVP v4 layers={got['layers']} experts={got['experts']} "
            f"dim={got['vq_dim']} slots={len(got['slots'])} widths={widths}"
        )
        return 0
    parser.print_help(sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
