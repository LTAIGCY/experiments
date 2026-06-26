#!/usr/bin/env python3
"""Compute optical-flow based Motion Score for prepared video clips.

Motion Score is the mean magnitude of dense optical flow between adjacent
frames. Higher values indicate stronger frame-to-frame motion.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("clips_root", type=Path)
    parser.add_argument("--frames-per-clip", type=int, default=49)
    parser.add_argument(
        "--pair-count",
        type=int,
        default=None,
        help="Number of adjacent frame pairs to evaluate from the start of each clip.",
    )
    parser.add_argument("--resize-width", type=int, default=256)
    parser.add_argument("--resize-height", type=int, default=144)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/motion_score_result.json"))
    parser.add_argument("--clip-csv", type=Path, default=Path("outputs/motion_score_clips.csv"))
    parser.add_argument("--pair-csv", type=Path, default=Path("outputs/motion_score_pairs.csv"))
    return parser.parse_args()


def discover_clips(root: Path, frames_per_clip: int) -> list[tuple[str, Path]]:
    clips: list[tuple[str, Path]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_dir()):
        frames = sorted(path.glob("frame_*.png"))
        if len(frames) >= frames_per_clip:
            condition = path.parent.name if path.parent != root else "all"
            clips.append((condition, path))
    return clips


def load_gray(path: Path, resize_width: int, resize_height: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return cv2.resize(image, (resize_width, resize_height), interpolation=cv2.INTER_AREA)


def flow_magnitude(prev_gray: np.ndarray, next_gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    return float(np.mean(magnitude))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    pair_count = args.pair_count if args.pair_count is not None else args.frames_per_clip - 1
    if pair_count < 1 or pair_count > args.frames_per_clip - 1:
        raise ValueError(f"pair-count must be in [1, {args.frames_per_clip - 1}], got {pair_count}")

    clips = discover_clips(args.clips_root, args.frames_per_clip)
    if not clips:
        raise RuntimeError(f"No clips with at least {args.frames_per_clip} frames found in {args.clips_root}")

    pair_rows: list[dict[str, object]] = []
    clip_rows: list[dict[str, object]] = []
    by_condition: dict[str, list[float]] = defaultdict(list)

    for condition, clip_dir in tqdm(clips, desc="clips"):
        frame_paths = sorted(clip_dir.glob("frame_*.png"))
        grays = [
            load_gray(path, args.resize_width, args.resize_height)
            for path in frame_paths[: pair_count + 1]
        ]
        pair_values = []
        for pair_index in range(pair_count):
            value = flow_magnitude(grays[pair_index], grays[pair_index + 1])
            pair_values.append(value)
            pair_rows.append(
                {
                    "condition": condition,
                    "clip_id": clip_dir.name,
                    "pair_index": pair_index,
                    "frame_a": pair_index,
                    "frame_b": pair_index + 1,
                    "motion_score": value,
                }
            )
        clip_mean = float(np.mean(pair_values))
        clip_std = float(np.std(pair_values))
        by_condition[condition].append(clip_mean)
        clip_rows.append(
            {
                "condition": condition,
                "clip_id": clip_dir.name,
                "pair_count": pair_count,
                "motion_score_mean": clip_mean,
                "motion_score_std": clip_std,
            }
        )

    pair_values_all = np.array([float(row["motion_score"]) for row in pair_rows], dtype=np.float64)
    clip_values_all = np.array([float(row["motion_score_mean"]) for row in clip_rows], dtype=np.float64)
    condition_summary = {
        condition: {
            "clip_count": len(values),
            "mean_of_clip_means": float(np.mean(values)),
            "std_of_clip_means": float(np.std(values)),
        }
        for condition, values in sorted(by_condition.items())
    }
    result = {
        "metric": "Motion Score",
        "definition": "Mean Farneback optical-flow magnitude between adjacent frames.",
        "value": float(np.mean(pair_values_all)),
        "mean_of_clip_means": float(np.mean(clip_values_all)),
        "std_of_clip_means": float(np.std(clip_values_all)),
        "clips_root": str(args.clips_root),
        "clip_count": len(clips),
        "frames_per_clip": args.frames_per_clip,
        "pairs_per_clip": pair_count,
        "pair_count": len(pair_rows),
        "resize": [args.resize_width, args.resize_height],
        "unit": "pixels/frame at resized resolution",
        "direction": "higher is more dynamic",
        "condition_summary": condition_summary,
        "clip_csv": str(args.clip_csv),
        "pair_csv": str(args.pair_csv),
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(
        args.clip_csv,
        ["condition", "clip_id", "pair_count", "motion_score_mean", "motion_score_std"],
        clip_rows,
    )
    write_csv(
        args.pair_csv,
        ["condition", "clip_id", "pair_index", "frame_a", "frame_b", "motion_score"],
        pair_rows,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
