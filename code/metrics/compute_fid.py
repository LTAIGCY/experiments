#!/usr/bin/env python3
"""Compute FID with pytorch-fid feature extraction and SciPy-compatible sqrtm.

This avoids the scipy.linalg.sqrtm(..., disp=False) API incompatibility in
newer SciPy versions while keeping the standard pytorch-fid Inception model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import linalg

from pytorch_fid.fid_score import calculate_activation_statistics
from pytorch_fid.inception import InceptionV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("real_frames", type=Path)
    parser.add_argument("generated_frames", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dims", type=int, default=2048)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/fid_result.json"),
    )
    return parser.parse_args()


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    diff = mu1 - mu2
    eps = 1e-6
    cov_product = sigma1.dot(sigma2)
    covmean = linalg.sqrtm(cov_product)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            max_imag = np.max(np.abs(covmean.imag))
            raise ValueError(f"Imaginary component in sqrtm result is too large: {max_imag}")
        covmean = covmean.real

    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def count_images(path: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts)


def image_files(path: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [str(p) for p in sorted(path.iterdir()) if p.is_file() and p.suffix.lower() in exts]


def main() -> None:
    args = parse_args()
    for path in [args.real_frames, args.generated_frames]:
        if not path.exists():
            raise FileNotFoundError(path)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx]).to(args.device)

    real_files = image_files(args.real_frames)
    generated_files = image_files(args.generated_frames)
    real_count = len(real_files)
    generated_count = len(generated_files)
    print(f"real_frames={real_count} generated_frames={generated_count}", flush=True)

    m1, s1 = calculate_activation_statistics(
        real_files, model, args.batch_size, args.dims, args.device
    )
    m2, s2 = calculate_activation_statistics(
        generated_files, model, args.batch_size, args.dims, args.device
    )
    fid = frechet_distance(m1, s1, m2, s2)

    result = {
        "metric": "FID",
        "value": fid,
        "real_frames": str(args.real_frames),
        "generated_frames": str(args.generated_frames),
        "real_frame_count": real_count,
        "generated_frame_count": generated_count,
        "dims": args.dims,
        "batch_size": args.batch_size,
        "device": args.device,
        "implementation": "pytorch-fid InceptionV3 activations + SciPy sqrtm compatible Frechet distance",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
