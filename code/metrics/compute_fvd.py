#!/usr/bin/env python3
"""Compute FVD using TFHub DeepMind I3D Kinetics-400 activations.

Inputs are prepared clip folders:
- real_clips/<clip_id>/frame_0000.png ... frame_0048.png
- generated_clips/<condition>/<clip_id>/frame_0000.png ... frame_0048.png

The script reads clips, resizes frames to 224x224, extracts the TFHub I3D
default 400-d activation for each clip, and computes Frechet distance between
the two activation distributions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import linalg

import tensorflow as tf
import tensorflow_hub as hub


I3D_TFHUB_URL = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("real_clips", type=Path)
    parser.add_argument("generated_clips", type=Path)
    parser.add_argument("--frames-per-clip", type=int, default=49)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--output-json", type=Path, default=Path("outputs/fvd_result.json"))
    parser.add_argument("--feature-dir", type=Path, default=Path("outputs/fvd_features"))
    parser.add_argument("--real-cache-name", default="real_orfd_i3d_tfhub_400d.npz")
    parser.add_argument("--generated-cache-name", default="generated_i3d_tfhub_400d.npz")
    parser.add_argument("--force", action="store_true", help="Recompute features even if cached .npz files exist.")
    return parser.parse_args()


def discover_real_clips(root: Path, frames_per_clip: int) -> list[Path]:
    clips = sorted([p for p in root.iterdir() if p.is_dir()])
    return [p for p in clips if len(sorted(p.glob("frame_*.png"))) == frames_per_clip]


def discover_generated_clips(root: Path, frames_per_clip: int) -> list[Path]:
    clips: list[Path] = []
    for condition_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        clips.extend(sorted([p for p in condition_dir.iterdir() if p.is_dir()]))
    return [p for p in clips if len(sorted(p.glob("frame_*.png"))) == frames_per_clip]


def load_clip(clip_dir: Path, frames_per_clip: int, resize: int) -> np.ndarray:
    frame_paths = sorted(clip_dir.glob("frame_*.png"))
    if len(frame_paths) != frames_per_clip:
        raise RuntimeError(f"{clip_dir} has {len(frame_paths)} frames, expected {frames_per_clip}")
    frames = []
    for path in frame_paths:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((resize, resize), Image.Resampling.BILINEAR)
            frames.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(frames, axis=0)


def batched(items: list[Path], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def extract_features(
    clip_dirs: list[Path],
    cache_path: Path,
    frames_per_clip: int,
    batch_size: int,
    resize: int,
    model,
    force: bool,
) -> np.ndarray:
    if cache_path.exists() and not force:
        data = np.load(cache_path)
        return data["features"]
    features = []
    signature = model.signatures["default"]
    for batch_index, batch_dirs in enumerate(batched(clip_dirs, batch_size), start=1):
        videos = np.stack([load_clip(d, frames_per_clip, resize) for d in batch_dirs], axis=0)
        outputs = signature(rgb_input=tf.convert_to_tensor(videos, dtype=tf.float32))
        feats = outputs["default"].numpy().astype(np.float64)
        features.append(feats)
        print(f"extracted {min(batch_index * batch_size, len(clip_dirs))}/{len(clip_dirs)} clips", flush=True)
    arr = np.concatenate(features, axis=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=arr, clip_dirs=np.array([str(p) for p in clip_dirs]))
    return arr


def frechet_distance(features1: np.ndarray, features2: np.ndarray) -> float:
    mu1 = np.mean(features1, axis=0)
    mu2 = np.mean(features2, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        eps = 1e-6
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            max_imag = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component in sqrtm result is too large: {max_imag}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def main() -> None:
    args = parse_args()
    real_clips = discover_real_clips(args.real_clips, args.frames_per_clip)
    generated_clips = discover_generated_clips(args.generated_clips, args.frames_per_clip)
    if not real_clips:
        raise RuntimeError(f"No real clips found in {args.real_clips}")
    if not generated_clips:
        raise RuntimeError(f"No generated clips found in {args.generated_clips}")
    print(f"real_clips={len(real_clips)} generated_clips={len(generated_clips)}", flush=True)

    model = hub.load(I3D_TFHUB_URL)
    real_features = extract_features(
        real_clips,
        args.feature_dir / args.real_cache_name,
        args.frames_per_clip,
        args.batch_size,
        args.resize,
        model,
        args.force,
    )
    generated_features = extract_features(
        generated_clips,
        args.feature_dir / args.generated_cache_name,
        args.frames_per_clip,
        args.batch_size,
        args.resize,
        model,
        args.force,
    )
    fvd = frechet_distance(real_features, generated_features)
    result = {
        "metric": "FVD",
        "value": fvd,
        "real_clips": str(args.real_clips),
        "generated_clips": str(args.generated_clips),
        "real_clip_count": len(real_clips),
        "generated_clip_count": len(generated_clips),
        "frames_per_clip": args.frames_per_clip,
        "resize": args.resize,
        "batch_size": args.batch_size,
        "feature_extractor": "TFHub DeepMind I3D Kinetics-400 default output",
        "feature_dim": int(real_features.shape[1]),
        "real_feature_cache": str(args.feature_dir / args.real_cache_name),
        "generated_feature_cache": str(args.feature_dir / args.generated_cache_name),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
