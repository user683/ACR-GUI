#!/usr/bin/env python3
"""Convert ShowUI-web + ShowUI-desktop to GUI-AiF JSONL format.

Reads from locally downloaded ModelScope data.
Web: parquet + extracted images. Desktop: parquet with embedded images.
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw_hf"

WEB_PARQUET_GLOB = str(DATA_DIR / "showlab_ShowUI-web" / "*.parquet")
WEB_IMAGES_DIR = DATA_DIR / "modelscope_full" / "images"
DESKTOP_PARQUET_GLOB = str(DATA_DIR / "modelscope_desktop_full" / "data" / "*.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ShowUI local data to GUI-AiF JSONL format."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("processed/showui"),
                        help="Output directory.")
    parser.add_argument("--web-parquet", type=str, default=WEB_PARQUET_GLOB,
                        help="Glob for web parquet files.")
    parser.add_argument("--web-images", type=str, default=str(WEB_IMAGES_DIR),
                        help="Directory containing extracted web images.")
    parser.add_argument("--desktop-parquet", type=str, default=DESKTOP_PARQUET_GLOB,
                        help="Glob for desktop parquet files.")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--web-only", action="store_true")
    parser.add_argument("--desktop-only", action="store_true")
    return parser.parse_args()


def build_image_index(images_root: Path) -> dict[str, Path]:
    """Scan image directory and build a mapping from filename to path."""
    idx: dict[str, Path] = {}
    if not images_root.exists():
        print(f"  [WARN] Image dir not found: {images_root}")
        return idx
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            # Key by relative path within images_root
            rel = str(p.relative_to(images_root))
            idx[rel] = p
    print(f"  Indexed {len(idx)} images from {images_root}")
    return idx


def process_web_dataset(parquet_glob: str, images_root: str,
                        output_dir: Path) -> list[dict]:
    """Read web parquet, explode sequences, match images."""
    print("  Building image index ...")
    image_idx = build_image_index(Path(images_root))

    files = sorted(glob.glob(parquet_glob))
    if not files:
        print(f"  [ERROR] No parquet files found: {parquet_glob}")
        return []
    print(f"  Reading {len(files)} parquet file(s) ...")
    table = pq.read_table(files)
    rows = table.to_pylist()
    print(f"  Loaded {len(rows)} screenshot rows")

    total_elements = 0
    missing_img = 0
    records: list[dict] = []

    for row_idx, row in enumerate(rows):
        image_url = row.get("image_url", "")
        if not image_url:
            missing_img += 1
            continue

        img_path = image_idx.get(image_url)
        if img_path is None:
            # Try with/without leading components
            alt_key = "/".join(image_url.split("/")[1:]) if "/" in image_url else image_url
            img_path = image_idx.get(alt_key)
        if img_path is None:
            missing_img += 1
            continue

        instructions = row.get("instruction") or []
        bboxes = row.get("bbox") or []
        types = row.get("type") or []

        for i, (instr, bbox) in enumerate(zip(instructions, bboxes)):
            if not instr or not bbox or len(bbox) != 4:
                continue
            elem_type = types[i] if i < len(types) else ""
            x1, y1, x2, y2 = [float(v) for v in bbox]
            records.append({
                "image_path": str(img_path.resolve()),
                "image_url": image_url,
                "instruction": instr.strip(),
                "rela_box": [x1, y1, x2, y2],
                "element_type": elem_type,
                "domain": "web",
                "source": "ShowUI-web",
                "screenId": Path(image_url).stem,
                "app_package_name": "",
            })

        if (row_idx + 1) % 2000 == 0:
            print(f"    {row_idx + 1} screenshots, {len(records)} elements")

    if missing_img:
        print(f"  [WARN] {missing_img} rows without matching image")
    print(f"  Total: {len(records)} element records")
    return records


def process_desktop_dataset(parquet_glob: str, output_dir: Path) -> list[dict]:
    """Read desktop parquet with embedded images."""
    files = sorted(glob.glob(parquet_glob))
    if not files:
        print(f"  [ERROR] No parquet files: {parquet_glob}")
        return []
    print(f"  Reading {len(files)} parquet files ...")
    table = pq.read_table(files)
    rows = table.to_pylist()
    print(f"  Loaded {len(rows)} rows")

    image_dir = output_dir / "images" / "desktop"
    image_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for idx, row in enumerate(rows):
        image_payload = row.get("image") or {}
        image_bytes = image_payload.get("bytes") if isinstance(image_payload, dict) else None
        if not image_bytes:
            continue

        bbox = row.get("bbox") or []
        if len(bbox) != 4:
            continue

        instruction = (row.get("instruction") or "").strip()
        if not instruction:
            continue

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = image.size
        x1, y1, x2, y2 = [float(v) for v in bbox]

        if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0 and (x2 - x1) <= 1.0:
            rela_box = [x1, y1, x2, y2]
            abs_box = [round(x1 * width, 2), round(y1 * height, 2),
                       round(x2 * width, 2), round(y2 * height, 2)]
        else:
            rela_box = [round(x1 / width, 4), round(y1 / height, 4),
                        round(x2 / width, 4), round(y2 / height, 4)]
            abs_box = [x1, y1, x2, y2]

        img_path = image_dir / f"desktop_{idx:06d}.jpg"
        image.save(img_path, format="JPEG", quality=95)

        elem_type = row.get("type", "")
        image_url = row.get("image_url", "")

        records.append({
            "image_path": str(img_path.resolve()),
            "image_url": image_url,
            "height": height,
            "width": width,
            "instruction": instruction,
            "rela_box": rela_box,
            "abs_box": abs_box,
            "domain": "desktop",
            "source": "ShowUI-desktop",
            "screenId": f"desktop_{idx:06d}",
            "app_package_name": elem_type,
        })

    print(f"  Total: {len(records)} valid records")
    return records


def finalize_web_records(records: list[dict]) -> list[dict]:
    """Load web images to compute dimensions and abs_box."""
    result: list[dict] = []
    img_size_cache: dict[str, tuple[int, int]] = {}

    for rec in records:
        img_path = rec["image_path"]
        if img_path not in img_size_cache:
            try:
                img = Image.open(img_path)
                img_size_cache[img_path] = img.size
            except Exception:
                continue
        width, height = img_size_cache[img_path]

        x1, y1, x2, y2 = rec["rela_box"]
        abs_box = [round(x1 * width, 2), round(y1 * height, 2),
                   round(x2 * width, 2), round(y2 * height, 2)]

        result.append({
            "image_path": img_path,
            "image_url": rec.get("image_url", ""),
            "height": height,
            "width": width,
            "instruction": rec["instruction"],
            "rela_box": rec["rela_box"],
            "abs_box": abs_box,
            "domain": "web",
            "source": "ShowUI-web",
            "screenId": rec.get("screenId", ""),
            "app_package_name": rec.get("element_type", ""),
        })

    return result


def split_records(records: list[dict], val_ratio: float, test_ratio: float,
                  rng: random.Random) -> tuple[list[dict], list[dict], list[dict]]:
    rng.shuffle(records)
    n = len(records)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    return records[test_n + val_n:], records[test_n:test_n + val_n], records[:test_n]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_yaml(path: Path, jsonl_path: Path) -> None:
    path.write_text(f"datasets:\n  - json_path: {jsonl_path.resolve()}\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_web: list[dict] = []
    all_desktop: list[dict] = []

    if not args.desktop_only:
        print("\n" + "=" * 60)
        print("=== ShowUI-Web ===")
        raw = process_web_dataset(args.web_parquet, args.web_images, output_dir)
        if raw:
            print("  Computing image dimensions ...")
            all_web = finalize_web_records(raw)
            print(f"  Finalized: {len(all_web)} records")

    if not args.web_only:
        print("\n" + "=" * 60)
        print("=== ShowUI-Desktop ===")
        all_desktop = process_desktop_dataset(args.desktop_parquet, output_dir)

    all_records = all_web + all_desktop
    print(f"\n{'=' * 60}")
    print(f"=== Combined: {len(all_records)} records ===")
    print(f"  Web: {len(all_web)}, Desktop: {len(all_desktop)}")

    if not all_records:
        print("No records. Aborting.")
        return

    train, val, test = split_records(all_records, args.val_ratio, args.test_ratio, rng)

    for split_name, records in [("train", train), ("val", val), ("test", test)]:
        jsonl_path = output_dir / f"{split_name}.jsonl"
        yaml_path = output_dir / f"dataset_{split_name}.yaml"
        write_jsonl(jsonl_path, records)
        write_yaml(yaml_path, jsonl_path)
        web_n = sum(1 for r in records if r["domain"] == "web")
        dsk_n = sum(1 for r in records if r["domain"] == "desktop")
        print(f"  {split_name}: {len(records)} records (web={web_n}, desktop={dsk_n})")


if __name__ == "__main__":
    main()
