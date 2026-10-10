#!/usr/bin/env python3
"""Build deterministic calibration/held-out text files for QAMQ.

Collects text/code from the repository and (optionally) an external engine
checkout, excludes secrets/build dirs, splits files deterministically by path
hash, and writes two UTF-8 corpora. No network access is used.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

EXCLUDE_PARTS = {
    ".git", "target", "node_modules", ".venv", "venv", "logs", "log",
    "models", "model", "archive", "artifacts", ".tmp-test", "qamq",
}
EXCLUDE_NAME_PARTS = (".env", "secret", "token", "key", "credential", "archive")
SUFFIXES = {".md", ".rs", ".py", ".ps1", ".sh", ".toml", ".json", ".yaml", ".yml"}
MAX_FILE_BYTES = 256 * 1024
DEFAULT_TOTAL_MIB = 4.0


def allowed(path: Path) -> bool:
    name = path.name.lower()
    if any(part.lower() in EXCLUDE_PARTS for part in path.parts):
        return False
    if any(part in name for part in EXCLUDE_NAME_PARTS):
        return False
    return path.suffix.lower() in SUFFIXES


def collect(roots: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in EXCLUDE_PARTS]
            for filename in filenames:
                path = Path(dirpath) / filename
                if not allowed(path):
                    continue
                try:
                    resolved = path.resolve()
                except OSError:
                    continue
                if resolved in seen:
                    continue
                seen.add(resolved)
                files.append(resolved)
    files.sort(key=lambda p: str(p))
    return files


def bucket(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).digest()
    return "heldout" if digest[0] % 5 == 0 else "calib"


def write_corpus(paths: list[Path], out: Path, limit_bytes: int) -> tuple[int, int]:
    total = 0
    used = 0
    with out.open("w", encoding="utf-8", errors="replace") as handle:
        for path in paths:
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if not raw or b"\x00" in raw[:4096]:
                continue
            text = raw[:MAX_FILE_BYTES].decode("utf-8", errors="replace")
            block = f"\n\n### FILE: {path}\n\n{text}\n"
            encoded = block.encode("utf-8")
            if total + len(encoded) > limit_bytes:
                break
            handle.write(block)
            total += len(encoded)
            used += 1
    return used, total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", default=[], help="source root (repeatable)")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--total-mib", type=float, default=DEFAULT_TOTAL_MIB)
    parser.add_argument("--engine-root", default="", help="optional external engine checkout")
    args = parser.parse_args()

    roots = [Path(r) for r in args.root]
    if args.engine_root:
        roots.append(Path(args.engine_root))
    files = collect(roots)
    calib = [p for p in files if bucket(p) == "calib"]
    heldout = [p for p in files if bucket(p) == "heldout"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_limit = int(args.total_mib * 1024 * 1024)
    calib_used, calib_bytes = write_corpus(calib, out_dir / "qamq-calib.txt", int(total_limit * 0.8))
    held_used, held_bytes = write_corpus(heldout, out_dir / "qamq-heldout.txt", int(total_limit * 0.2))
    for name, used, size in (
        ("qamq-calib.txt", calib_used, calib_bytes),
        ("qamq-heldout.txt", held_used, held_bytes),
    ):
        target = out_dir / name
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        print(f"{name}: files={used} bytes={size} sha256={digest}")
    print(f"source files: {len(files)} calib={len(calib)} heldout={len(heldout)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
