#!/usr/bin/env python3
"""Build GUI-AiF-compatible splits from RICO Widget Captioning parquet files.

Supports train/val/test splits independently so the full 48K dataset can be
processed while respecting the original split boundaries.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RICO Widget Captioning parquet files to GUI-AiF JSONL format."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw_hf/rootsautomation_RICO-WidgetCaptioning/data"),
        help="Directory containing the parquet files.",
    )
    parser.add_argument(
        "--train-pattern",
        default="train-*.parquet",
        help="Glob pattern for training split parquet files.",
    )
    parser.add_argument(
        "--val-pattern",
        default="val-*.parquet",
        help="Glob pattern for validation split parquet files.",
    )
    parser.add_argument(
        "--test-pattern",
        default="test-*.parquet",
        help="Glob pattern for test split parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed/rico_widget_full"),
    )
    parser.add_argument("--train-size", type=int, default=None,
                        help="Max training samples (default: all).")
    parser.add_argument("--val-size", type=int, default=None,
                        help="Max validation samples (default: all).")
    parser.add_argument("--test-size", type=int, default=None,
                        help="Max test samples (default: all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-splits", nargs="*", choices=["train", "val", "test"],
                        default=[], help="Splits to skip entirely.")
    return parser.parse_args()


def load_rows(raw_dir: Path, pattern: str) -> list[dict]:
    rows: list[dict] = []
    files = sorted(raw_dir.glob(pattern))
    if not files:
        print(f"  [WARN] No files matched pattern '{pattern}' in {raw_dir}")
        return rows
    for parquet_path in files:
        table = pq.read_table(parquet_path)
        rows.extend(table.to_pylist())
    print(f"  Loaded {len(rows)} rows from {len(files)} file(s) matching '{pattern}'")
    return rows


def pick_caption(captions: list[str], rng: random.Random) -> str:
    captions = [c.strip() for c in captions if c and c.strip()]
    return rng.choice(captions) if captions else ""


def convert_row(row: dict, split: str, index: int, output_dir: Path,
                rng: random.Random, source_name: str) -> dict | None:
    image_payload = row.get("image") or {}
    image_bytes = image_payload.get("bytes")
    if not image_bytes:
        return None

    instruction = pick_caption(row.get("captions") or [], rng)
    if not instruction:
        return None

    rela_box = row.get("bbox") or []
    if len(rela_box) != 4:
        return None

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = [float(v) for v in rela_box]
    abs_box = [
        round(x1 * width, 2),
        round(y1 * height, 2),
        round(x2 * width, 2),
        round(y2 * height, 2),
    ]

    image_dir = output_dir / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"rico_{split}_{index:06d}.jpg"
    image.save(image_path, format="JPEG", quality=95)

    return {
        "image_path": str(image_path.resolve()),
        "image_url": row.get("file_name", ""),
        "height": height,
        "width": width,
        "instruction": instruction,
        "rela_box": [x1, y1, x2, y2],
        "abs_box": abs_box,
        "domain": "mobile",
        "source": source_name,
        "screenId": row.get("screenId"),
        "app_package_name": row.get("app_package_name"),
    }


def process_split(rows: list[dict], split_name: str, max_size: int | None,
                  output_dir: Path, rng: random.Random, source_name: str) -> list[dict]:
    """Convert rows for a single split, optionally sampling down to max_size."""
    if max_size is not None and max_size < len(rows):
        rng.shuffle(rows)
        rows = rows[:max_size]

    records: list[dict] = []
    for row in rows:
        record = convert_row(row, split_name, len(records), output_dir, rng, source_name)
        if record is not None:
            records.append(record)
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_yaml(path: Path, jsonl_path: Path) -> None:
    path.write_text(f"datasets:\n  - json_path: {jsonl_path.resolve()}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_configs = [
        ("train", args.train_pattern, args.train_size),
        ("val",   args.val_pattern,   args.val_size),
        ("test",  args.test_pattern,  args.test_size),
    ]

    source_name = "RICO-WidgetCaptioning"

    for split_name, pattern, max_size in split_configs:
        if split_name in args.skip_splits:
            print(f"\n=== {split_name.upper()}: SKIPPED ===")
            continue

        print(f"\n=== {split_name.upper()} (pattern='{pattern}', max={max_size or 'all'}) ===")
        rows = load_rows(args.raw_dir, pattern)
        if not rows:
            print(f"  No data for {split_name}, skipping.")
            continue

        records = process_split(rows, split_name, max_size, output_dir, rng, source_name)
        if not records:
            print(f"  No valid records produced for {split_name}.")
            continue

        jsonl_path = output_dir / f"{split_name}.jsonl"
        yaml_path = output_dir / f"dataset_{split_name}.yaml"
        write_jsonl(jsonl_path, records)
        write_yaml(yaml_path, jsonl_path)
        print(f"  Written: {jsonl_path} ({len(records)} samples)")
        print(f"  YAML:    {yaml_path}")


if __name__ == "__main__":
    main()
