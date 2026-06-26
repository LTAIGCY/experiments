#!/usr/bin/env python3
"""Prepare generated image-to-video outputs for FID/FVD evaluation.

Outputs:
- generated_clips/<condition>/<clip_id>/frame_0000.png ... frame_0048.png
- generated_frames/all/<clip_id>_frame_0000.png ... for aggregate FID
- manifests/generated_clips_manifest.csv
- manifests/generated_frames_manifest.csv
- manifests/build_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import imageio_ffmpeg
from PIL import Image


DEFAULT_SOURCES_JSON = Path("configs/generated_video_sources.example.json")


@dataclass(frozen=True)
class VideoItem:
    condition: str
    source_path: Path
    source_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prepared/generated_videos_49f"),
    )
    parser.add_argument("--frames-per-clip", type=int, default=49)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--fid-frame-mode",
        choices=("all", "uniform16"),
        default="all",
    )
    parser.add_argument(
        "--sources-json",
        type=Path,
        default=DEFAULT_SOURCES_JSON,
        help="JSON object mapping condition names to source video directories.",
    )
    parser.add_argument(
        "--pad-short-clips",
        action="store_true",
        help="Pad clips shorter than frames-per-clip by duplicating the last extracted frame.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_sources(sources_json: Path | None) -> dict[str, Path]:
    if sources_json is None:
        raise ValueError("Please provide --sources-json.")
    if not sources_json.exists():
        raise FileNotFoundError(
            f"Source config not found: {sources_json}. "
            "Copy configs/generated_video_sources.example.json and edit the paths."
        )
    data = json.loads(sources_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{sources_json} must contain a JSON object.")
    return {str(condition): Path(str(root)) for condition, root in data.items()}


def discover_videos(sources: dict[str, Path]) -> list[VideoItem]:
    items: list[VideoItem] = []
    for condition, root in sources.items():
        if not root.exists():
            raise FileNotFoundError(root)
        videos = sorted(root.glob("*.mp4"))
        if len(videos) != 40:
            raise RuntimeError(f"Expected 40 mp4 files for {condition}, found {len(videos)} in {root}")
        for idx, path in enumerate(videos):
            items.append(VideoItem(condition=condition, source_path=path, source_index=idx))
    return items


def fid_indices(frame_count: int, mode: str) -> list[int]:
    if mode == "all":
        return list(range(frame_count))
    if frame_count <= 16:
        return list(range(frame_count))
    return sorted({round(i * (frame_count - 1) / 15) for i in range(16)})


def prepare_output(output_root: Path, overwrite: bool) -> None:
    output_root = output_root.resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} exists. Re-run with --overwrite.")
        allowed_names = {"prepared", "outputs"}
        if not any(parent.name in allowed_names for parent in output_root.parents):
            raise RuntimeError(f"Refusing to delete output outside a prepared/outputs directory: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "generated_clips").mkdir(parents=True, exist_ok=True)
    (output_root / "generated_frames" / "all").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)


def extract_clip(
    ffmpeg: str,
    video: Path,
    clip_dir: Path,
    frames_per_clip: int,
    fps: int,
    width: int,
    height: int,
    pad_short_clips: bool,
) -> int:
    clip_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = str(clip_dir / "frame_%04d.png")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-vf",
        f"fps={fps},scale={width}:{height}:flags=lanczos,format=rgb24",
        "-frames:v",
        str(frames_per_clip),
        "-start_number",
        "0",
        output_pattern,
    ]
    subprocess.run(cmd, check=True)
    frames = sorted(clip_dir.glob("frame_*.png"))
    padded_frames = 0
    if len(frames) < frames_per_clip and pad_short_clips and frames:
        last_frame = frames[-1]
        for frame_index in range(len(frames), frames_per_clip):
            shutil.copy2(last_frame, clip_dir / f"frame_{frame_index:04d}.png")
            padded_frames += 1
        frames = sorted(clip_dir.glob("frame_*.png"))
    if len(frames) != frames_per_clip:
        raise RuntimeError(f"{video} produced {len(frames)} frames, expected {frames_per_clip}")
    with Image.open(frames[0]) as im:
        if im.size != (width, height):
            raise RuntimeError(f"{frames[0]} has size {im.size}, expected {(width, height)}")
        if im.mode != "RGB":
            raise RuntimeError(f"{frames[0]} has mode {im.mode}, expected RGB")
    return padded_frames


def hardlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    sources = load_sources(args.sources_json)
    items = discover_videos(sources)
    prepare_output(output_root, args.overwrite)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    fid_idx = set(fid_indices(args.frames_per_clip, args.fid_frame_mode))

    clip_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    link_counts: dict[str, int] = {}
    padded_clip_count = 0
    total_padded_frames = 0

    for global_idx, item in enumerate(items):
        clip_id = f"gen_{item.condition}_{item.source_index:03d}"
        clip_dir = output_root / "generated_clips" / item.condition / clip_id
        padded_frames = extract_clip(
            ffmpeg,
            item.source_path,
            clip_dir,
            args.frames_per_clip,
            args.fps,
            args.width,
            args.height,
            args.pad_short_clips,
        )
        if padded_frames:
            padded_clip_count += 1
            total_padded_frames += padded_frames
        frame_paths = [clip_dir / f"frame_{i:04d}.png" for i in range(args.frames_per_clip)]
        for frame_index, src in enumerate(frame_paths):
            if frame_index not in fid_idx:
                continue
            dst = output_root / "generated_frames" / "all" / f"{clip_id}_frame_{frame_index:04d}.png"
            method = hardlink_or_copy(src, dst)
            link_counts[method] = link_counts.get(method, 0) + 1
            frame_rows.append(
                {
                    "frame_id": f"{clip_id}_frame_{frame_index:04d}",
                    "clip_id": clip_id,
                    "condition": item.condition,
                    "frame_index": frame_index,
                    "source_video": str(item.source_path),
                    "prepared_path": str(dst),
                }
            )
        clip_rows.append(
            {
                "clip_id": clip_id,
                "condition": item.condition,
                "source_index": item.source_index,
                "source_video": str(item.source_path),
                "frame_count": args.frames_per_clip,
                "fps": args.fps,
                "width": args.width,
                "height": args.height,
                "padded_frames": padded_frames,
                "prepared_clip_dir": str(clip_dir),
                "frame_paths": ";".join(str(p) for p in frame_paths),
            }
        )

    manifests = output_root / "manifests"
    write_csv(
        manifests / "generated_clips_manifest.csv",
        [
            "clip_id",
            "condition",
            "source_index",
            "source_video",
            "frame_count",
            "fps",
            "width",
            "height",
            "padded_frames",
            "prepared_clip_dir",
            "frame_paths",
        ],
        clip_rows,
    )
    write_csv(
        manifests / "generated_frames_manifest.csv",
        ["frame_id", "clip_id", "condition", "frame_index", "source_video", "prepared_path"],
        frame_rows,
    )
    summary = {
        "output_root": str(output_root),
        "video_count": len(items),
        "clip_count": len(clip_rows),
        "fid_frame_count": len(frame_rows),
        "frames_per_clip": args.frames_per_clip,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "fid_frame_mode": args.fid_frame_mode,
        "pad_short_clips": args.pad_short_clips,
        "padded_clip_count": padded_clip_count,
        "total_padded_frames": total_padded_frames,
        "ffmpeg": ffmpeg,
        "source_dirs": {k: str(v) for k, v in sources.items()},
        "link_counts": link_counts,
    }
    (manifests / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
