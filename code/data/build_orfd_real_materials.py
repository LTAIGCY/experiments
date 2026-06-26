#!/usr/bin/env python3
"""Build ORFD real-material folders for FID/FVD evaluation.

The script reads only ORFD RGB `image_data` directories under training/testing,
selects non-overlapping 49-frame clips, and materializes a convenient folder
layout using hardlinks by default so the raw dataset remains untouched.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SequenceInfo:
    split: str
    sequence_id: str
    image_dir: Path
    frames: tuple[Path, ...]


@dataclass(frozen=True)
class ClipCandidate:
    split: str
    sequence_id: str
    image_dir: Path
    source_start_index: int
    frames: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 49-frame ORFD real clip/frame package for FID/FVD."
    )
    parser.add_argument(
        "--orfd-root",
        type=Path,
        default=Path("data/raw/ORFD/ORFD_Dataset_ZIP"),
        help="Root containing ORFD training/testing directories.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prepared/orfd_real_49f_160"),
        help="Output package directory.",
    )
    parser.add_argument("--clip-count", type=int, default=160)
    parser.add_argument("--frames-per-clip", type=int, default=49)
    parser.add_argument(
        "--fid-frame-mode",
        choices=("all", "uniform16"),
        default="all",
        help="Frames to expose in real_frames/all for FID.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and recreate output-root if it already exists.",
    )
    return parser.parse_args()


def discover_sequences(orfd_root: Path) -> list[SequenceInfo]:
    sequences: list[SequenceInfo] = []
    for split in ("training", "testing"):
        split_root = orfd_root / split
        if not split_root.exists():
            continue
        for image_dir in sorted(split_root.glob(r"*\*\image_data")):
            frames = tuple(sorted(image_dir.glob("*.png")))
            if not frames:
                continue
            sequences.append(
                SequenceInfo(
                    split=split,
                    sequence_id=image_dir.parent.name,
                    image_dir=image_dir,
                    frames=frames,
                )
            )
    return sequences


def make_candidates(
    sequences: Iterable[SequenceInfo], frames_per_clip: int
) -> dict[str, list[ClipCandidate]]:
    by_sequence: dict[str, list[ClipCandidate]] = defaultdict(list)
    for seq in sequences:
        for start in range(0, len(seq.frames) - frames_per_clip + 1, frames_per_clip):
            frames = seq.frames[start : start + frames_per_clip]
            by_sequence[seq.sequence_id].append(
                ClipCandidate(
                    split=seq.split,
                    sequence_id=seq.sequence_id,
                    image_dir=seq.image_dir,
                    source_start_index=start,
                    frames=frames,
                )
            )
    return dict(by_sequence)


def select_round_robin(
    by_sequence: dict[str, list[ClipCandidate]], clip_count: int
) -> list[ClipCandidate]:
    selected: list[ClipCandidate] = []
    sequence_ids = sorted(by_sequence)
    positions = {sequence_id: 0 for sequence_id in sequence_ids}
    while len(selected) < clip_count:
        made_progress = False
        for sequence_id in sequence_ids:
            pos = positions[sequence_id]
            candidates = by_sequence[sequence_id]
            if pos >= len(candidates):
                continue
            selected.append(candidates[pos])
            positions[sequence_id] += 1
            made_progress = True
            if len(selected) >= clip_count:
                break
        if not made_progress:
            break
    if len(selected) < clip_count:
        raise RuntimeError(
            f"Only {len(selected)} clips are available, fewer than requested {clip_count}."
        )
    return selected


def safe_prepare_output(output_root: Path, overwrite: bool) -> None:
    output_root = output_root.resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists. Re-run with --overwrite to replace it."
            )
        if not any(parent.name in {"prepared", "outputs"} for parent in output_root.parents):
            raise RuntimeError(f"Refusing to delete output outside a prepared/outputs directory: {output_root}")
        if "orfd" not in output_root.name.lower():
            raise RuntimeError(f"Refusing to delete unexpected output directory: {output_root}")
        shutil.rmtree(output_root)
    (output_root / "real_clips").mkdir(parents=True, exist_ok=True)
    (output_root / "real_frames" / "all").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)


def materialize_link(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            os.symlink(src, dst)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def fid_indices(frame_count: int, mode: str) -> list[int]:
    if mode == "all":
        return list(range(frame_count))
    if mode == "uniform16":
        if frame_count <= 16:
            return list(range(frame_count))
        return sorted({round(i * (frame_count - 1) / 15) for i in range(16)})
    raise ValueError(mode)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_package(args: argparse.Namespace) -> dict[str, object]:
    orfd_root = args.orfd_root.resolve()
    output_root = args.output_root.resolve()
    if not orfd_root.exists():
        raise FileNotFoundError(orfd_root)

    sequences = discover_sequences(orfd_root)
    if not sequences:
        raise RuntimeError(f"No ORFD image_data directories found under {orfd_root}")

    by_sequence = make_candidates(sequences, args.frames_per_clip)
    total_candidates = sum(len(v) for v in by_sequence.values())
    selected = select_round_robin(by_sequence, args.clip_count)

    safe_prepare_output(output_root, args.overwrite)

    link_counts: Counter[str] = Counter()
    clip_rows: list[dict[str, object]] = []
    frame_rows: list[dict[str, object]] = []
    sequence_rows: list[dict[str, object]] = []
    label_template_rows: list[dict[str, object]] = []

    selected_by_sequence = Counter(c.sequence_id for c in selected)
    representative_frame_by_sequence: dict[str, Path] = {}
    for clip in selected:
        representative_frame_by_sequence.setdefault(
            clip.sequence_id, clip.frames[len(clip.frames) // 2]
        )
    all_candidates_by_sequence = {k: len(v) for k, v in by_sequence.items()}
    for seq in sequences:
        sequence_rows.append(
            {
                "source_dataset": "ORFD",
                "split": seq.split,
                "sequence_id": seq.sequence_id,
                "image_dir": str(seq.image_dir),
                "rgb_frame_count": len(seq.frames),
                "available_nonoverlap_49f_clips": all_candidates_by_sequence.get(seq.sequence_id, 0),
                "selected_49f_clips": selected_by_sequence.get(seq.sequence_id, 0),
                "condition": "orfd_unlabeled_aggregate",
                "manual_condition_note": "Label manually for per-condition FID/FVD.",
            }
        )
        label_template_rows.append(
            {
                "sequence_id": seq.sequence_id,
                "split": seq.split,
                "selected_49f_clips": selected_by_sequence.get(seq.sequence_id, 0),
                "representative_frame": str(
                    representative_frame_by_sequence.get(seq.sequence_id, seq.frames[len(seq.frames) // 2])
                ),
                "condition": "",
                "allowed_conditions": "foggy_offroad|snowy_offroad|rainy_muddy|dry_dusty_like_offroad|exclude",
                "notes": "",
            }
        )

    fid_frame_indices = fid_indices(args.frames_per_clip, args.fid_frame_mode)
    for clip_number, clip in enumerate(selected, start=1):
        clip_id = f"orfd_clip_{clip_number:04d}"
        clip_dir = output_root / "real_clips" / clip_id
        first_frame = clip.frames[0]
        last_frame = clip.frames[-1]
        for frame_index, src in enumerate(clip.frames):
            dst = clip_dir / f"frame_{frame_index:04d}.png"
            link_counts[materialize_link(src, dst)] += 1
            if frame_index in fid_frame_indices:
                fid_dst = (
                    output_root
                    / "real_frames"
                    / "all"
                    / f"{clip_id}_frame_{frame_index:04d}.png"
                )
                link_counts[materialize_link(src, fid_dst)] += 1
                frame_rows.append(
                    {
                        "frame_id": f"{clip_id}_frame_{frame_index:04d}",
                        "clip_id": clip_id,
                        "source_dataset": "ORFD",
                        "split": clip.split,
                        "sequence_id": clip.sequence_id,
                        "condition": "orfd_unlabeled_aggregate",
                        "frame_index": frame_index,
                        "source_path": str(src),
                        "prepared_path": str(fid_dst),
                    }
                )
        clip_rows.append(
            {
                "clip_id": clip_id,
                "source_dataset": "ORFD",
                "split": clip.split,
                "sequence_id": clip.sequence_id,
                "condition": "orfd_unlabeled_aggregate",
                "start_frame": clip.source_start_index,
                "frame_count": args.frames_per_clip,
                "first_source_frame": str(first_frame),
                "last_source_frame": str(last_frame),
                "prepared_clip_dir": str(clip_dir),
                "frame_paths": ";".join(str(clip_dir / f"frame_{i:04d}.png") for i in range(args.frames_per_clip)),
            }
        )

    manifests = output_root / "manifests"
    write_csv(
        manifests / "real_clips_manifest.csv",
        [
            "clip_id",
            "source_dataset",
            "split",
            "sequence_id",
            "condition",
            "start_frame",
            "frame_count",
            "first_source_frame",
            "last_source_frame",
            "prepared_clip_dir",
            "frame_paths",
        ],
        clip_rows,
    )
    write_csv(
        manifests / "real_frames_manifest.csv",
        [
            "frame_id",
            "clip_id",
            "source_dataset",
            "split",
            "sequence_id",
            "condition",
            "frame_index",
            "source_path",
            "prepared_path",
        ],
        frame_rows,
    )
    write_csv(
        manifests / "sequence_summary.csv",
        [
            "source_dataset",
            "split",
            "sequence_id",
            "image_dir",
            "rgb_frame_count",
            "available_nonoverlap_49f_clips",
            "selected_49f_clips",
            "condition",
            "manual_condition_note",
        ],
        sequence_rows,
    )
    write_csv(
        manifests / "sequence_condition_label_template.csv",
        [
            "sequence_id",
            "split",
            "selected_49f_clips",
            "representative_frame",
            "condition",
            "allowed_conditions",
            "notes",
        ],
        label_template_rows,
    )

    summary = {
        "source_dataset": "ORFD",
        "orfd_root": str(orfd_root),
        "output_root": str(output_root),
        "frames_per_clip": args.frames_per_clip,
        "requested_clip_count": args.clip_count,
        "selected_clip_count": len(selected),
        "fid_frame_mode": args.fid_frame_mode,
        "fid_frame_count": len(frame_rows),
        "sequence_count": len(sequences),
        "source_rgb_frame_count": sum(len(s.frames) for s in sequences),
        "available_nonoverlap_49f_clips": total_candidates,
        "condition": "orfd_unlabeled_aggregate",
        "link_counts": dict(link_counts),
        "note": (
            "This package intentionally uses only ORFD training/testing image_data RGB frames. "
            "dense_depth, sparse_depth, label, and calib files are excluded."
        ),
    }
    (manifests / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "README.md").write_text(
        "\n".join(
            [
                "# ORFD FID/FVD Real Materials",
                "",
                "This package contains real ORFD RGB material prepared for aggregate FID/FVD.",
                "",
                "- `real_clips/`: 160 clips, each with 49 PNG frames.",
                "- `real_frames/all/`: FID frame folder derived from the same selected clips.",
                "- `manifests/real_clips_manifest.csv`: clip-level provenance.",
                "- `manifests/real_frames_manifest.csv`: FID frame-level provenance.",
                "- `manifests/sequence_summary.csv`: source sequence counts and selection counts.",
                "- `manifests/sequence_condition_label_template.csv`: manual labeling template for per-condition splits.",
                "- `manifests/build_summary.json`: build settings and counts.",
                "",
                "The `condition` field is `orfd_unlabeled_aggregate`; manually label sequences",
                "before using this package for per-condition FID/FVD.",
                "",
                "Raw data is not modified. Files are materialized with hardlinks when possible.",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = build_package(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
