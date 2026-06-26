from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


CLASSES = ["rainy_muddy", "snowy_offroad", "dusty_sandy"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_class_name(name: str, classes: list[str]) -> str | None:
    low = name.lower()
    for cls in classes:
        if low == cls or low.startswith(cls):
            return cls
    return None


def discover_inputs(args) -> dict[str, Path]:
    mapping = {cls: None for cls in args.classes}
    aliases = {
        "rainy_muddy": args.rainy_dir,
        "snowy_offroad": args.snowy_dir,
        "dusty_sandy": args.dusty_dir,
        "foggy_offroad": args.foggy_dir,
    }
    for cls, value in aliases.items():
        if cls in mapping and value:
            mapping[cls] = Path(value)
    if args.input_root:
        root = Path(args.input_root)
        for child in root.iterdir():
            if child.is_dir():
                cls = normalize_class_name(child.name, args.classes)
                if cls is not None and mapping[cls] is None:
                    mapping[cls] = child
    missing = [cls for cls, path in mapping.items() if path is None or not path.exists()]
    if missing and not args.allow_missing_classes:
        raise FileNotFoundError(f"Missing generated input folders for: {missing}")
    return {cls: path for cls, path in mapping.items() if path is not None}


def image_stats(path: Path) -> dict:
    with Image.open(path) as pil:
        rgb = pil.convert("RGB")
        arr = np.asarray(rgb)
        small = rgb.resize((256, 144))
        phash = str(imagehash.phash(small))
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    brightness = float(arr.mean())
    saturation = float(hsv[:, :, 1].mean())
    highlight_ratio = float((gray > 245).mean())
    dark_ratio = float((gray < 8).mean())
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    r = arr[:, :, 0].astype(np.float32)
    g = arr[:, :, 1].astype(np.float32)
    b = arr[:, :, 2].astype(np.float32)
    pink_ratio = float(((r > 150) & (b > 120) & (g < 120) & ((r - g) > 35)).mean())
    return {
        "width": int(arr.shape[1]),
        "height": int(arr.shape[0]),
        "brightness": brightness,
        "saturation": saturation,
        "highlight_ratio": highlight_ratio,
        "dark_ratio": dark_ratio,
        "laplacian_var": laplacian_var,
        "pink_ratio": pink_ratio,
        "phash": phash,
    }


def reject_reason(stats: dict) -> str:
    if stats["highlight_ratio"] > 0.20 or stats["brightness"] > 225:
        return "overexposed"
    if stats["dark_ratio"] > 0.35 or stats["brightness"] < 25:
        return "too_dark"
    if stats["laplacian_var"] < 18:
        return "blur_or_low_detail"
    if stats["pink_ratio"] > 0.08:
        return "pink_or_purple_color_cast"
    if stats["saturation"] < 8:
        return "very_low_saturation"
    return ""


def hamming_hex(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def select_class(cls: str, input_dir: Path, target: int, min_phash_distance: int) -> tuple[list[dict], list[dict]]:
    accepted: list[dict] = []
    rejected: list[dict] = []
    seen_hashes: list[str] = []
    files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    for path in files:
        try:
            stats = image_stats(path)
            reason = reject_reason(stats)
            if not reason:
                near_dup = any(hamming_hex(stats["phash"], h) < min_phash_distance for h in seen_hashes)
                if near_dup:
                    reason = "near_duplicate"
            row = {"class": cls, "source_path": str(path), **stats}
            if reason:
                row["reject_reason"] = reason
                rejected.append(row)
                continue
            row["quality_score"] = (
                stats["laplacian_var"] / 100.0
                - stats["highlight_ratio"] * 10.0
                - stats["pink_ratio"] * 12.0
                - abs(stats["brightness"] - 125.0) / 80.0
            )
            accepted.append(row)
            seen_hashes.append(stats["phash"])
        except Exception as exc:
            rejected.append({"class": cls, "source_path": str(path), "reject_reason": f"read_error:{exc}"})

    accepted = sorted(accepted, key=lambda x: x["quality_score"], reverse=True)
    selected = accepted[:target]
    for row in accepted[target:]:
        row["reject_reason"] = "above_target_count"
        rejected.append(row)
    return selected, rejected


def make_preview(rows: list[dict], out_path: Path, title: str, max_images: int = 80) -> None:
    sample = rows[:max_images]
    if not sample:
        return
    tiles = []
    for row in sample:
        im = Image.open(row.get("output_path", row["source_path"])).convert("RGB")
        im.thumbnail((180, 110))
        tile = Image.new("RGB", (190, 140), "white")
        tile.paste(im, ((190 - im.width) // 2, 4))
        ImageDraw.Draw(tile).text((4, 118), Path(row["source_path"]).stem[:28], fill="black")
        tiles.append(tile)
    cols = 5
    sheet = Image.new("RGB", (cols * 190, ((len(tiles) + cols - 1) // cols) * 140 + 30), "white")
    ImageDraw.Draw(sheet).text((8, 8), title, fill="black")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols) * 190, 30 + (i // cols) * 140))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default=None, help="Root with class subfolders.")
    parser.add_argument("--rainy-dir", default=None)
    parser.add_argument("--snowy-dir", default=None)
    parser.add_argument("--dusty-dir", default=None)
    parser.add_argument("--foggy-dir", default=None)
    parser.add_argument("--classes", nargs="+", default=CLASSES)
    parser.add_argument("--output-root", type=Path, default=Path("prepared/generated_curated"))
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests"))
    parser.add_argument("--report-root", type=Path, default=Path("outputs"))
    parser.add_argument("--preview-root", type=Path, default=Path("outputs/review_sheets"))
    parser.add_argument("--output-name", default="ours_firstframes_auto")
    parser.add_argument("--target-per-class", type=int, default=400)
    parser.add_argument("--min-phash-distance", type=int, default=5)
    parser.add_argument("--allow-missing-classes", action="store_true")
    args = parser.parse_args()

    inputs = discover_inputs(args)
    out_root = args.output_root / args.output_name
    manifest_path = args.manifest_root / f"generated_curated_{args.output_name}_manifest.csv"
    rejected_path = args.manifest_root / f"generated_curated_{args.output_name}_rejected.csv"
    summary_path = args.report_root / f"generated_curated_{args.output_name}_summary.json"
    preview_root = args.preview_root / f"generated_curated_{args.output_name}"

    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    selected_all: list[dict] = []
    rejected_all: list[dict] = []
    for cls, input_dir in inputs.items():
        selected, rejected = select_class(cls, input_dir, args.target_per_class, args.min_phash_distance)
        class_out = out_root / cls
        class_out.mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(selected):
            dst = class_out / f"{cls}_gen_{i:04d}.png"
            with Image.open(row["source_path"]) as im:
                im.convert("RGB").save(dst)
            row["output_path"] = str(dst)
            row["output_sha1"] = sha1_file(dst)
            selected_all.append(row)
        rejected_all.extend(rejected)
        make_preview(selected, preview_root / f"{cls}_selected_preview.png", f"{args.output_name} / {cls}")

    pd.DataFrame(selected_all).to_csv(manifest_path, index=False)
    pd.DataFrame(rejected_all).to_csv(rejected_path, index=False)
    summary = {
        "output_root": str(out_root),
        "manifest": str(manifest_path),
        "rejected_manifest": str(rejected_path),
        "target_per_class": args.target_per_class,
        "selected_by_class": pd.DataFrame(selected_all).groupby("class").size().to_dict() if selected_all else {},
        "rejected_by_reason": pd.DataFrame(rejected_all).groupby("reject_reason").size().to_dict() if rejected_all else {},
        "note": "Generated images are intended for train augmentation only. Do not add them to val/test.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
