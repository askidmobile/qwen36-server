#!/usr/bin/env python3
"""Pinned Transformers Qwen3.5 processor oracle for decoded RGB fixtures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor

TRANSFORMERS_REVISION = "00e8e49eb3eda67290f635f6bdf59f236f6adf7e"
IMAGE_TOKEN_ID = 248056
VIDEO_TOKEN_ID = 248057


def sha256_f32(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f4").tobytes()).hexdigest()


def load_rgb(path: Path, width: int, height: int) -> np.ndarray:
    expected = width * height * 3
    data = path.read_bytes()
    if len(data) != expected:
        raise ValueError(f"RGB byte count mismatch for {path}: {len(data)} != {expected}")
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3).copy()


def build_positions(
    input_ids: list[int],
    mm_types: list[int],
    image_grids: list[list[int]],
    video_grids: list[list[int]],
) -> dict[str, Any]:
    expanded_video = [[1, grid[1], grid[2]] for grid in video_grids for _ in range(grid[0])]
    grids = {1: iter(image_grids), 2: iter(expanded_video)}
    rope = [[], [], []]
    current = 0
    index = 0
    while index < len(mm_types):
        kind = mm_types[index]
        end = index + 1
        while end < len(mm_types) and mm_types[end] == kind:
            end += 1
        if kind == 0:
            values = list(range(current, current + end - index))
            for dimension in rope:
                dimension.extend(values)
            current += end - index
        else:
            t, h, w = next(grids[kind])
            h //= 2
            w //= 2
            if t * h * w != end - index:
                raise ValueError("token/grid mismatch")
            for temporal in range(t):
                for row in range(h):
                    for column in range(w):
                        rope[0].append(current + temporal)
                        rope[1].append(current + row)
                        rope[2].append(current + column)
            current += max(h, w)
        index = end
    maximum = max(max(dimension) for dimension in rope) + 1 if input_ids else 0
    return {
        "text_positions": list(range(len(input_ids))),
        "rope_positions": rope,
        "decode_rope_delta": maximum - len(input_ids),
    }


def prompt_result(
    processor: Any,
    prompt: str,
    media_marker: dict[str, Any],
    media_kwargs: dict[str, Any],
) -> dict[str, Any]:
    messages = [{"role": "user", "content": [media_marker, {"type": "text", "text": prompt}]}]
    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    inputs = processor(text=[template], return_tensors="np", **media_kwargs)
    input_ids = inputs["input_ids"][0].tolist()
    assistant = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    no_think = processor.tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    input_ids.extend(assistant)
    input_ids.extend(no_think)
    mm_types = [1 if token == IMAGE_TOKEN_ID else 2 if token == VIDEO_TOKEN_ID else 0 for token in input_ids]
    image_grids = inputs.get("image_grid_thw")
    video_grids = inputs.get("video_grid_thw")
    positions = build_positions(
        input_ids,
        mm_types,
        [] if image_grids is None else image_grids.tolist(),
        [] if video_grids is None else video_grids.tolist(),
    )
    return {
        "input_ids": input_ids,
        "mm_token_type_ids": mm_types,
        **positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--kind", required=True, choices=("image", "video"))
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--rgb", required=True, action="append", type=Path)
    parser.add_argument("--indices")
    parser.add_argument("--source-fps", type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise SystemExit("dimensions must be positive")
    processor = AutoProcessor.from_pretrained(args.source, local_files_only=True)
    tokenizer_config = json.loads((args.source / "tokenizer_config.json").read_text(encoding="utf-8"))
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str):
        raise SystemExit("source tokenizer_config.json has no chat_template")
    processor.chat_template = chat_template
    processor.tokenizer.chat_template = chat_template
    frames = [load_rgb(path, args.width, args.height) for path in args.rgb]

    if args.kind == "image":
        if len(frames) != 1:
            raise SystemExit("image reference needs one RGB frame")
        values = processor.image_processor(images=frames, return_tensors="np")
        pixels = values["pixel_values"]
        grids = values["image_grid_thw"]
        prompt = prompt_result(
            processor,
            args.prompt,
            {"type": "image"},
            {"images": frames},
        )
        timestamps: list[float] = []
        indices: list[int] = []
    else:
        if not args.indices or not args.source_fps or not math.isfinite(args.source_fps) or args.source_fps <= 0:
            raise SystemExit("video reference needs indices and positive source FPS")
        indices = [int(value) for value in args.indices.split(",")]
        if len(indices) != len(frames):
            raise SystemExit("video frame/index count mismatch")
        video = np.stack(frames)
        # Processor API expects metadata to preserve source-frame timestamps.
        from transformers.video_utils import VideoMetadata

        metadata = VideoMetadata(
            total_num_frames=max(indices) + 1,
            fps=args.source_fps,
            duration=(max(indices) + 1) / args.source_fps,
            video_backend="reference",
            frames_indices=indices,
        )
        template = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": args.prompt}]}],
            tokenize=False,
            add_generation_prompt=False,
        )
        processed = processor(
            text=[template],
            videos=[video],
            video_metadata=[metadata],
            return_tensors="np",
            do_sample_frames=False,
        )
        pixels = processed["pixel_values_videos"]
        grids = processed["video_grid_thw"]
        timestamps = processor._calculate_timestamps(indices, args.source_fps, 2)
        input_ids = processed["input_ids"][0].tolist()
        # Processor replacement wraps all temporal groups in original template's
        # outer vision markers. Qwen3.5 contract keeps one marker pair per group.
        vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        first_video = input_ids.index(VIDEO_TOKEN_ID)
        outer_start = max(index for index in range(first_video) if input_ids[index] == vision_start_id)
        if outer_start > 0 and vision_start_id in input_ids[:outer_start]:
            input_ids.pop(max(index for index in range(outer_start) if input_ids[index] == vision_start_id))
        for index in range(len(input_ids) - 1):
            if input_ids[index : index + 2] == [vision_end_id, vision_end_id]:
                input_ids.pop(index)
                break
        input_ids.extend(processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))
        input_ids.extend(processor.tokenizer.encode("<think>\n\n</think>\n\n", add_special_tokens=False))
        mm_types = [2 if token == VIDEO_TOKEN_ID else 0 for token in input_ids]
        prompt = {
            "input_ids": input_ids,
            "mm_token_type_ids": mm_types,
            **build_positions(input_ids, mm_types, [], grids.tolist()),
        }

    result = {
        "schema_version": "qwen35-transformers-processor-reference-v1",
        "transformers_revision": TRANSFORMERS_REVISION,
        "kind": args.kind,
        "media": {
            "grid_thw": grids[0].tolist(),
            "patch_shape": list(pixels.shape),
            "patch_sha256": sha256_f32(pixels),
            "patch_values": pixels.reshape(-1).tolist(),
            "visual_tokens": int(np.prod(grids[0]) // 4),
            "frame_indices": indices,
            "timestamps": timestamps,
        },
        "prompt": prompt,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
