from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image
from tqdm import tqdm


TARGET_W = 1280
TARGET_H = 720
TARGET_RATIO = TARGET_W / TARGET_H


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bottom_biased_crop_box(width: int, height: int) -> tuple[int, int, int, int]:
    ratio = width / height
    if abs(ratio - TARGET_RATIO) < 1e-4:
        return 0, 0, width, height

    if ratio > TARGET_RATIO:
        # Too wide: crop left/right, keep full vertical ground context.
        crop_w = round(height * TARGET_RATIO)
        left = max(0, (width - crop_w) // 2)
        return left, 0, left + crop_w, height

    # Too tall: crop height with a strong bottom bias to remove sky while
    # preserving the road/ground region for terrain-weather generation.
    crop_h = round(width / TARGET_RATIO)
    top = max(0, height - crop_h)
    return 0, top, width, top + crop_h


def process_one(src: Path, dst: Path) -> dict:
    with Image.open(src) as im:
        im = im.convert("RGB")
        orig_w, orig_h = im.size
        box = bottom_biased_crop_box(orig_w, orig_h)
        cropped = im.crop(box)
        resized = cropped.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst, format="PNG", optimize=True)

    return {
        "orig_width": orig_w,
        "orig_height": orig_h,
        "crop_left": box[0],
        "crop_top": box[1],
        "crop_right": box[2],
        "crop_bottom": box[3],
        "output_width": TARGET_W,
        "output_height": TARGET_H,
        "output_sha1": sha1_file(dst),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=Path("manifests/generation_references_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("prepared/generation_references_1280x720"))
    parser.add_argument("--output-manifest", type=Path, default=Path("manifests/generation_references_1280x720_manifest.csv"))
    parser.add_argument("--output-report", type=Path, default=Path("outputs/generation_references_1280x720_summary.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    with args.source_manifest.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for row in tqdm(rows, desc="bottom-crop refs"):
        src = Path(row["source_path"])
        image_id = row.get("image_id") or f"reference_{len(out_rows):04d}"
        dst = args.output_dir / f"{image_id}_1280x720.png"
        info = process_one(src, dst)
        out_row = dict(row)
        out_row.update(info)
        out_row["processed_path"] = str(dst)
        out_row["crop_rule"] = "center_crop_width_or_bottom_aligned_height_to_16_9_then_resize_1280x720"
        out_rows.append(out_row)

    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with args.output_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    summary = {
        "input_manifest": str(args.source_manifest),
        "output_dir": str(args.output_dir),
        "output_manifest": str(args.output_manifest),
        "count": len(out_rows),
        "target_size": [TARGET_W, TARGET_H],
        "crop_rule": "If image is wider than 16:9, crop left/right centered. If taller than 16:9, bottom-align crop to remove sky and preserve ground.",
    }
    args.output_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
