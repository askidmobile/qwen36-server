#!/usr/bin/env python3
"""Fail-closed Qwen3.5 source and converter-report auditor."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, NoReturn

SCHEMA = "qwen35-artifact-audit-v1"
PROCESSOR_SCHEMA = "qwen35-processor-parity-v1"
PROCESSOR_PROBE_SCHEMA = "qwen35-multimodal-probe-v1"
PROCESSOR_REFERENCE_SCHEMA = "qwen35-transformers-processor-reference-v1"
TRANSFORMERS_REVISION = "00e8e49eb3eda67290f635f6bdf59f236f6adf7e"
PROCESSOR_TOLERANCE = 1e-5
EXPECTED_COUNTS = {"text": 426, "vision": 297, "mtp": 15}
EXPECTED_TOTAL = sum(EXPECTED_COUNTS.values())


class AuditError(ValueError):
    pass


def fail(message: str) -> NoReturn:
    raise SystemExit(f"qwen35-artifacts: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot read JSON {path}: {error}") from error


def component_for_source(name: str) -> str:
    if name.startswith("model.visual."):
        return "vision"
    if name.startswith(("mtp.", "model.mtp.")):
        return "mtp"
    if name.startswith(("model.language_model.", "lm_head.")):
        return "text"
    raise AuditError(f"unknown source tensor prefix: {name}")


def read_weight_map(index_path: Path) -> dict[str, str]:
    document = load_json(index_path)
    if not isinstance(document, dict) or set(document) - {"metadata", "weight_map"}:
        raise AuditError("source index has unknown top-level fields")
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise AuditError("source index weight_map must be a non-empty object")
    result: dict[str, str] = {}
    for name, shard in weight_map.items():
        if not isinstance(name, str) or not name or not isinstance(shard, str) or not shard:
            raise AuditError("source index contains invalid tensor or shard name")
        if name in result:
            raise AuditError(f"duplicate source tensor: {name}")
        result[name] = shard
    return result


def audit_source(index_path: Path) -> dict[str, Any]:
    weight_map = read_weight_map(index_path)
    components: dict[str, list[str]] = defaultdict(list)
    for name in sorted(weight_map):
        components[component_for_source(name)].append(name)
    counts = {component: len(components.get(component, [])) for component in EXPECTED_COUNTS}
    if len(weight_map) != EXPECTED_TOTAL or counts != EXPECTED_COUNTS:
        raise AuditError(
            f"source inventory mismatch: total={len(weight_map)}, components={counts}; "
            f"expected total={EXPECTED_TOTAL}, components={EXPECTED_COUNTS}"
        )
    return {
        "path": str(index_path),
        "sha256": sha256(index_path),
        "total": len(weight_map),
        "counts": counts,
        "shards": sorted(set(weight_map.values())),
        "tensors": {component: components[component] for component in EXPECTED_COUNTS},
    }


def read_report(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"source", "output", "operation", "shape", "dtype"}
            if reader.fieldnames is None or set(reader.fieldnames) != required:
                raise AuditError(f"report {path} columns must be exactly {sorted(required)}")
            rows = list(reader)
    except OSError as error:
        raise AuditError(f"cannot read report {path}: {error}") from error
    if not rows:
        raise AuditError(f"report is empty: {path}")
    for row in rows:
        if any(not row[field] for field in ("source", "output", "operation", "shape", "dtype")):
            raise AuditError(f"report has empty field: {path}")
    return rows


def output_inventory(path: Path) -> dict[str, dict[str, Any]]:
    document = load_json(path)
    tensors = document.get("tensors") if isinstance(document, dict) else None
    if not isinstance(tensors, list):
        raise AuditError(f"GGUF inventory has no tensors array: {path}")
    outputs: dict[str, dict[str, Any]] = {}
    for tensor in tensors:
        if not isinstance(tensor, dict):
            raise AuditError(f"GGUF inventory tensor is not an object: {path}")
        name = tensor.get("name")
        if not isinstance(name, str) or not name:
            raise AuditError(f"GGUF inventory tensor has invalid name: {path}")
        if name in outputs:
            raise AuditError(f"duplicate GGUF tensor name {name!r}: {path}")
        outputs[name] = tensor
    return outputs


def audit_reports(
    source: dict[str, Any], report_specs: list[tuple[str, Path, Path | None]]
) -> list[dict[str, Any]]:
    source_components = {
        name: component
        for component, names in source["tensors"].items()
        for name in names
    }
    seen_sources: Counter[str] = Counter()
    source_operations: dict[str, list[str]] = defaultdict(list)
    seen_outputs: dict[str, str] = {}
    reports: list[dict[str, Any]] = []

    for expected_component, report_path, inventory_path in report_specs:
        if expected_component not in EXPECTED_COUNTS:
            raise AuditError(f"unknown report component: {expected_component}")
        rows = read_report(report_path)
        inventory = output_inventory(inventory_path) if inventory_path else None
        report_outputs: set[str] = set()
        component_sources: set[str] = set()
        for row in rows:
            source_name = row["source"]
            output_name = row["output"]
            actual_component = source_components.get(source_name)
            if actual_component is None:
                raise AuditError(f"report references unknown source tensor: {source_name}")
            if actual_component != expected_component:
                raise AuditError(
                    f"source tensor {source_name!r} belongs to {actual_component}, not {expected_component}"
                )
            prior_source = seen_outputs.get(output_name)
            if prior_source is not None:
                raise AuditError(
                    f"duplicate physical output {output_name!r}: {prior_source!r} and {source_name!r}"
                )
            seen_outputs[output_name] = source_name
            report_outputs.add(output_name)
            component_sources.add(source_name)
            seen_sources[source_name] += 1
            source_operations[source_name].append(row["operation"])
        expected_sources = set(source["tensors"][expected_component])
        if component_sources != expected_sources:
            missing = sorted(expected_sources - component_sources)
            extra = sorted(component_sources - expected_sources)
            raise AuditError(
                f"{expected_component} report source mismatch: missing={missing[:8]}, extra={extra[:8]}"
            )
        if inventory is not None:
            inventory_outputs = set(inventory)
            if report_outputs != inventory_outputs:
                missing = sorted(report_outputs - inventory_outputs)
                unreported = sorted(inventory_outputs - report_outputs)
                raise AuditError(
                    f"{expected_component} output mismatch: missing={missing[:8]}, unreported={unreported[:8]}"
                )
        reports.append({
            "component": expected_component,
            "path": str(report_path),
            "sha256": sha256(report_path),
            "source_count": len(component_sources),
            "output_count": len(report_outputs),
            "inventory": str(inventory_path) if inventory_path else None,
            "inventory_sha256": sha256(inventory_path) if inventory_path else None,
        })

    expected_all = set(source_components)
    seen_all = set(seen_sources)
    duplicate_identity = sorted(
        source_name
        for source_name, operations in source_operations.items()
        if len(operations) > 1 and all(operation == "identity" for operation in operations)
    )
    if seen_all != expected_all or duplicate_identity:
        raise AuditError(
            f"aggregate source coverage mismatch: missing={sorted(expected_all - seen_all)[:8]}, "
            f"unknown={sorted(seen_all - expected_all)[:8]}, duplicate_identity={duplicate_identity[:8]}"
        )
    return reports


def compare_processor(probe_path: Path, reference_path: Path, language: str) -> dict[str, Any]:
    if language not in {"en", "ru"}:
        raise AuditError("processor language must be en or ru")
    probe = load_json(probe_path)
    reference = load_json(reference_path)
    if not isinstance(probe, dict) or probe.get("schema_version") != PROCESSOR_PROBE_SCHEMA:
        raise AuditError("invalid Rust processor probe schema")
    if not isinstance(reference, dict) or reference.get("schema_version") != PROCESSOR_REFERENCE_SCHEMA:
        raise AuditError("invalid Transformers processor reference schema")
    for document, name in ((probe, "probe"), (reference, "reference")):
        if document.get("transformers_revision") != TRANSFORMERS_REVISION:
            raise AuditError(f"{name} Transformers revision mismatch")
        if document.get("kind") not in {"image", "video"}:
            raise AuditError(f"{name} media kind is invalid")
        if not isinstance(document.get("media"), dict) or not isinstance(document.get("prompt"), dict):
            raise AuditError(f"{name} processor result is malformed")
    if probe["kind"] != reference["kind"]:
        raise AuditError("processor media kind mismatch")

    exact_media_fields = ("grid_thw", "patch_shape", "visual_tokens", "frame_indices")
    for field in exact_media_fields:
        if probe["media"].get(field) != reference["media"].get(field):
            raise AuditError(f"processor media {field} mismatch")
    probe_timestamps = probe["media"].get("timestamps")
    reference_timestamps = reference["media"].get("timestamps")
    if not isinstance(probe_timestamps, list) or not isinstance(reference_timestamps, list):
        raise AuditError("processor timestamps are malformed")
    if len(probe_timestamps) != len(reference_timestamps) or any(
        not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
        or not float(left) == float(right)
        for left, right in zip(probe_timestamps, reference_timestamps)
    ):
        raise AuditError("processor timestamps mismatch")

    probe_values = probe["media"].get("patch_values")
    reference_values = reference["media"].get("patch_values")
    if not isinstance(probe_values, list) or not isinstance(reference_values, list):
        raise AuditError("processor patch values are malformed")
    if len(probe_values) != len(reference_values):
        raise AuditError("processor patch value count mismatch")
    max_abs = 0.0
    for index, (left, right) in enumerate(zip(probe_values, reference_values)):
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise AuditError(f"processor patch value {index} is not numeric")
        left = float(left)
        right = float(right)
        if not math.isfinite(left) or not math.isfinite(right):
            raise AuditError(f"processor patch value {index} is not finite")
        max_abs = max(max_abs, abs(left - right))
    if max_abs > PROCESSOR_TOLERANCE:
        raise AuditError(
            f"processor patch tolerance exceeded: max_abs={max_abs:.9g}, limit={PROCESSOR_TOLERANCE}"
        )

    exact_prompt_fields = (
        "input_ids",
        "mm_token_type_ids",
        "text_positions",
        "rope_positions",
        "decode_rope_delta",
    )
    for field in exact_prompt_fields:
        if probe["prompt"].get(field) != reference["prompt"].get(field):
            raise AuditError(f"processor prompt {field} mismatch")

    return {
        "schema_version": PROCESSOR_SCHEMA,
        "status": "pass",
        "language": language,
        "kind": probe["kind"],
        "transformers_revision": TRANSFORMERS_REVISION,
        "tolerance": PROCESSOR_TOLERANCE,
        "max_abs": max_abs,
        "patch_values": len(probe_values),
        "visual_tokens": probe["media"]["visual_tokens"],
        "probe": {"path": str(probe_path), "sha256": sha256(probe_path)},
        "reference": {"path": str(reference_path), "sha256": sha256(reference_path)},
    }


def parse_report_spec(value: str) -> tuple[str, Path, Path | None]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError("expected COMPONENT=REPORT.csv[,INSPECT.json]")
    paths = parts[1].split(",", 1)
    return parts[0], Path(paths[0]), Path(paths[1]) if len(paths) == 2 else None


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source-index", type=Path)
    mode.add_argument("--processor-probe", type=Path)
    parser.add_argument("--report", action="append", default=[], type=parse_report_spec)
    parser.add_argument("--processor-reference", type=Path)
    parser.add_argument("--language", choices=("en", "ru"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.processor_probe:
            if args.report:
                raise AuditError("--report is invalid in processor mode")
            if args.processor_reference is None or args.language is None:
                raise AuditError("processor mode requires --processor-reference and --language")
            result = compare_processor(args.processor_probe, args.processor_reference, args.language)
        else:
            if args.processor_reference is not None or args.language is not None:
                raise AuditError("processor arguments require --processor-probe")
            source = audit_source(args.source_index)
            reports = audit_reports(source, args.report) if args.report else []
            result = {
                "schema_version": SCHEMA,
                "status": "pass",
                "source": source,
                "reports": reports,
            }
    except AuditError as error:
        fail(str(error))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
