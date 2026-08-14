#!/usr/bin/env python3
"""Generate compact deterministic English/Russian multimodal fixtures.

Requires ImageMagick 7 and FFmpeg 8. Model weights are never read. Output is
small and safe to generate on development machines.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, NoReturn

SCHEMA = "qwen35-multimodal-fixtures-v1"
FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)
ICC_CANDIDATES = (
    Path("/System/Library/ColorSync/Profiles/Display P3.icc"),
    Path("/usr/share/color/icc/colord/DisplayP3.icc"),
    Path(r"C:\Windows\System32\spool\drivers\color\DisplayP3.icc"),
)
FIXED_MTIME = 946684800


def fail(message: str) -> NoReturn:
    raise SystemExit(f"generate_multimodal_fixtures: {message}")


def tool(name: str) -> str:
    path = shutil.which(name)
    if not path:
        fail(f"required tool not found: {name}")
    return path


def run(args: list[str]) -> None:
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def normalize_file(path: Path, magick: str | None = None) -> None:
    if magick is not None:
        run([magick, str(path), "-strip", "-define", "png:exclude-chunks=date,time", str(path)])
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_version(path: str) -> str:
    completed = subprocess.run(
        [path, "-version"], check=True, text=True, encoding="utf-8", errors="replace",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    return completed.stdout.splitlines()[0].strip()


def first_file(candidates: tuple[Path, ...], label: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    fail(f"{label} not found")


def image(magick: str, font: Path, output: Path, lines: list[str], *, chart: bool = False) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    args = [magick, "-size", "640x384", "xc:white", "-font", str(font)]
    if chart:
        args += [
            "-stroke", "#1f2937", "-strokewidth", "3",
            "-draw", "line 90,310 570,310 line 90,55 90,310",
            "-fill", "#2563eb", "-stroke", "none",
            "-draw", "rectangle 140,230 220,310 rectangle 280,170 360,310 rectangle 420,90 500,310",
        ]
    y = 60
    for index, line in enumerate(lines):
        size = "34" if index == 0 else "26"
        args += ["-fill", "#111827", "-pointsize", size, "-annotate", f"+44+{y}", line]
        y += 52
    run(args + [str(output)])
    normalize_file(output, magick)


def multi_image(magick: str, font: Path, output: Path, label: str, shape: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    draw = "circle 320,205 320,100" if shape == "circle" else "rectangle 220,100 420,300"
    run([
        magick, "-size", "640x384", "xc:white", "-font", str(font),
        "-fill", "#111827", "-pointsize", "30", "-annotate", "+36+48", label,
        "-fill", "#f59e0b", "-stroke", "#92400e", "-strokewidth", "5", "-draw", draw,
        str(output),
    ])
    normalize_file(output, magick)


def transparent_alpha(magick: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        magick, "-size", "320x192", "xc:none", "-fill", "rgba(220,38,38,0.5)",
        "-draw", "rectangle 30,30 290,162", str(output),
    ])
    normalize_file(output, magick)


def icc_image(magick: str, profile: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        magick, "-size", "320x192", "xc:#16a34a", "-profile", str(profile),
        "-define", "png:exclude-chunks=date,time", str(output),
    ])
    os.utime(output, (FIXED_MTIME, FIXED_MTIME))


def add_exif_orientation(path: Path, orientation: int) -> None:
    data = path.read_bytes()
    if not data.startswith(b"\xff\xd8") or not 1 <= orientation <= 8:
        fail("invalid JPEG or EXIF orientation")
    tiff = b"MM\x00*\x00\x00\x00\x08\x00\x01\x01\x12\x00\x03\x00\x00\x00\x01" + orientation.to_bytes(2, "big") + b"\x00\x00\x00\x00\x00\x00"
    payload = b"Exif\x00\x00" + tiff
    path.write_bytes(data[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + data[2:])


def oriented_jpeg(magick: str, font: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        magick, "-size", "384x240", "xc:white", "-font", str(font), "-fill", "#111827",
        "-pointsize", "32", "-annotate", "+35+75", "UP / ВВЕРХ", str(output),
    ])
    add_exif_orientation(output, 6)
    os.utime(output, (FIXED_MTIME, FIXED_MTIME))


def static_gif(magick: str, font: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        magick, "-size", "320x192", "xc:white", "-font", str(font), "-fill", "#111827",
        "-pointsize", "28", "-annotate", "+30+100", "STATIC / СТАТИКА", str(output),
    ])
    normalize_file(output, magick)


def animated_gif(magick: str, font: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, color in enumerate(("#dc2626", "#16a34a", "#2563eb", "#f59e0b"), start=1):
        frame = output.parent / f".frame-{index}.png"
        image(magick, font, frame, [f"FRAME / КАДР {index}"])
        run([magick, str(frame), "-fill", color, "-draw", "circle 320,240 320,170", str(frame)])
        frames.append(frame)
    run([magick, "-delay", "50", "-loop", "0", *map(str, frames), str(output)])
    normalize_file(output, magick)
    for frame in frames:
        frame.unlink()


def video(ffmpeg: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x192:d=1:r=4",
        "-f", "lavfi", "-i", "color=c=blue:s=320x192:d=1:r=4",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0,format=yuv420p[v]",
        "-map", "[v]", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1",
        "-metadata", "creation_time=2000-01-01T00:00:00Z", str(output),
    ])
    os.utime(output, (FIXED_MTIME, FIXED_MTIME))


def fixture_specs() -> list[dict[str, Any]]:
    return [
        {"id": "en-ocr", "language": "en", "file": "en/ocr.png", "kind": "image", "category": "ocr", "prompt": "Transcribe all text exactly.", "oracle": {"type": "exact_text", "value": "INVOICE A-17\nTOTAL 42.50 USD\nOPTION B"}},
        {"id": "en-chart", "language": "en", "file": "en/chart.png", "kind": "image", "category": "chart", "prompt": "Which quarter has the highest value? Answer with quarter and value.", "oracle": {"type": "exact_fields", "fields": {"quarter": "Q3", "value": "30"}}},
        {"id": "en-spatial", "language": "en", "file": "en/spatial.png", "kind": "image", "category": "spatial", "prompt": "Where is the blue square relative to the red circle?", "oracle": {"type": "required_forbidden", "required": ["right"], "forbidden": ["left"]}},
        {"id": "en-multi-a", "language": "en", "file": "en/multi-a.png", "kind": "image", "category": "multi_image", "group": "en-multi", "prompt": "Which image contains a circle?", "oracle": {"type": "exact_option", "value": "A"}},
        {"id": "en-multi-b", "language": "en", "file": "en/multi-b.png", "kind": "image", "category": "multi_image", "group": "en-multi"},
        {"id": "ru-ocr", "language": "ru", "file": "ru/ocr.png", "kind": "image", "category": "ocr", "prompt": "Перепиши весь текст без изменений.", "oracle": {"type": "exact_text", "value": "СЧЁТ № Й-17\nИТОГО 42,50 ₽\nВАРИАНТ Б"}},
        {"id": "ru-chart", "language": "ru", "file": "ru/chart.png", "kind": "image", "category": "chart", "prompt": "Какой квартал имеет наибольшее значение? Ответь кварталом и числом.", "oracle": {"type": "exact_fields", "fields": {"quarter": "3 квартал", "value": "30"}}},
        {"id": "ru-spatial", "language": "ru", "file": "ru/spatial.png", "kind": "image", "category": "spatial", "prompt": "Где синий квадрат относительно красного круга?", "oracle": {"type": "required_forbidden", "required": ["справа"], "forbidden": ["слева"]}},
        {"id": "ru-multi-a", "language": "ru", "file": "ru/multi-a.png", "kind": "image", "category": "multi_image", "group": "ru-multi", "prompt": "На каком изображении круг? Ответь буквой.", "oracle": {"type": "exact_option", "value": "А"}},
        {"id": "ru-multi-b", "language": "ru", "file": "ru/multi-b.png", "kind": "image", "category": "multi_image", "group": "ru-multi"},
        {"id": "mixed-caption", "language": "mixed", "file": "shared/caption.png", "kind": "image", "category": "caption", "prompt": "Briefly describe this image / Кратко опиши изображение.", "oracle": {"type": "required_forbidden", "required": ["yellow", "triangle"], "forbidden": ["circle", "blue"]}},
        {"id": "mixed-alpha", "language": "mixed", "file": "shared/alpha.png", "kind": "image", "category": "alpha", "prompt": "Describe the visible color after alpha compositing on white.", "oracle": {"type": "required_forbidden", "required": ["red", "pink"], "forbidden": ["transparent"]}},
        {"id": "mixed-icc", "language": "mixed", "file": "shared/display-p3.png", "kind": "image", "category": "icc", "prompt": "Name the dominant color after conversion to sRGB.", "oracle": {"type": "required_forbidden", "required": ["green"], "forbidden": ["red", "blue"]}},
        {"id": "mixed-orientation", "language": "mixed", "file": "shared/orientation.jpg", "kind": "image", "category": "exif", "prompt": "Is the text upright after orientation handling?", "oracle": {"type": "exact_boolean", "value": True}},
        {"id": "mixed-static-gif", "language": "mixed", "file": "shared/static.gif", "kind": "image", "category": "gif_static", "oracle": {"type": "exact_kind", "value": "image"}},
        {"id": "mixed-animated-gif", "language": "mixed", "file": "shared/animated.gif", "kind": "video", "category": "gif_animated", "oracle": {"type": "exact_kind", "value": "video"}},
        {"id": "en-video", "language": "en", "file": "en/temporal.mp4", "kind": "video", "category": "temporal", "prompt": "Which color appears first and which appears second?", "oracle": {"type": "exact_sequence", "value": ["red", "blue"]}},
        {"id": "ru-video", "language": "ru", "file": "ru/temporal.mp4", "kind": "video", "category": "temporal", "prompt": "Какой цвет появляется первым, а какой вторым?", "oracle": {"type": "exact_sequence", "value": ["красный", "синий"]}},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/multimodal"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    magick = tool("magick")
    ffmpeg = tool("ffmpeg")
    font = first_file(FONT_CANDIDATES, "Arial/DejaVu Sans font")
    profile = first_file(ICC_CANDIDATES, "Display P3 ICC profile")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()) and not args.force:
        fail(f"output is not empty: {root}; pass --force")
    if root.exists() and args.force:
        shutil.rmtree(root)
    root.mkdir(parents=True)

    image(magick, font, root / "en/ocr.png", ["INVOICE A-17", "TOTAL 42.50 USD", "OPTION B"])
    image(magick, font, root / "ru/ocr.png", ["СЧЁТ № Й-17", "ИТОГО 42,50 ₽", "ВАРИАНТ Б"])
    image(magick, font, root / "en/chart.png", ["QUARTERS: Q1=10 Q2=20 Q3=30"], chart=True)
    image(magick, font, root / "ru/chart.png", ["КВАРТАЛЫ: 1=10 2=20 3=30"], chart=True)
    for lang in ("en", "ru"):
        spatial = root / f"{lang}/spatial.png"
        image(magick, font, spatial, ["RED CIRCLE     BLUE SQUARE" if lang == "en" else "КРАСНЫЙ КРУГ     СИНИЙ КВАДРАТ"])
        run([magick, str(spatial), "-fill", "#dc2626", "-draw", "circle 210,245 210,180", "-fill", "#2563eb", "-draw", "rectangle 400,180 530,310", str(spatial)])
        normalize_file(spatial, magick)
        multi_image(magick, font, root / f"{lang}/multi-a.png", "IMAGE A" if lang == "en" else "ИЗОБРАЖЕНИЕ А", "circle")
        multi_image(magick, font, root / f"{lang}/multi-b.png", "IMAGE B" if lang == "en" else "ИЗОБРАЖЕНИЕ Б", "square")
    caption = root / "shared/caption.png"
    image(magick, font, caption, ["YELLOW TRIANGLE / ЖЁЛТЫЙ ТРЕУГОЛЬНИК"])
    run([magick, str(caption), "-fill", "#facc15", "-stroke", "#854d0e", "-strokewidth", "4", "-draw", "polygon 320,120 190,320 450,320", str(caption)])
    normalize_file(caption, magick)
    transparent_alpha(magick, root / "shared/alpha.png")
    icc_image(magick, profile, root / "shared/display-p3.png")
    oriented_jpeg(magick, font, root / "shared/orientation.jpg")
    static_gif(magick, font, root / "shared/static.gif")
    animated_gif(magick, font, root / "shared/animated.gif")
    video(ffmpeg, root / "en/temporal.mp4")
    shutil.copyfile(root / "en/temporal.mp4", root / "ru/temporal.mp4")
    os.utime(root / "ru/temporal.mp4", (FIXED_MTIME, FIXED_MTIME))

    fixtures = fixture_specs()
    for fixture in fixtures:
        path = root / fixture["file"]
        if not path.is_file():
            fail(f"fixture not generated: {path}")
        fixture["bytes"] = path.stat().st_size
        fixture["sha256"] = sha256(path)

    manifest = {
        "schema_version": SCHEMA,
        "generator_sha256": sha256(Path(__file__).resolve()),
        "tools": {
            "imagemagick": command_version(magick),
            "ffmpeg": command_version(ffmpeg),
            "font_file": font.name,
            "font_sha256": sha256(font),
            "icc_file": profile.name,
            "icc_sha256": sha256(profile),
        },
        "fixtures": fixtures,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"generated {len(fixtures)} fixtures in {root}")


if __name__ == "__main__":
    main()
