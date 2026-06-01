#!/usr/bin/env python3
"""Download and process ShowUI-web and ShowUI-desktop datasets from ModelScope.

Usage:
    # Full pipeline (download + process)
    python gui_data_process/scripts/download_and_prepare_showui.py --all

    # Download only
    python gui_data_process/scripts/download_and_prepare_showui.py --download-only

    # Process only (data already downloaded)
    python gui_data_process/scripts/download_and_prepare_showui.py --process-only

    # Web/Desktop only
    python gui_data_process/scripts/download_and_prepare_showui.py --all --web-only
    python gui_data_process/scripts/download_and_prepare_showui.py --all --desktop-only
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import shutil
import time
from pathlib import Path

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.resolve().parent  # gui_data_process/
DATA_DIR = ROOT_DIR / "data" / "raw_hf"
PROCESSED_DIR = ROOT_DIR / "processed"

# HF mirror for faster download in China
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

# ModelScope datasets
WEB_DATASET = "showlab/ShowUI-web"
DESKTOP_DATASET = "showlab/ShowUI-desktop"

# Output names
WEB_PARQUET_DIR = DATA_DIR / "showlab_ShowUI-web"
DESKTOP_PARQUET_DIR = DATA_DIR / "modelscope_desktop_full"
WEB_IMAGES_DIR = DATA_DIR / "modelscope_full" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare ShowUI datasets."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Download AND process data.")
    group.add_argument("--download-only", action="store_true", help="Only download source data.")
    group.add_argument("--process-only", action="store_true", help="Only process already-downloaded data.")

    parser.add_argument("--web-only", action="store_true")
    parser.add_argument("--desktop-only", action="store_true")

    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DIR / "showui")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--endpoint", type=str, default=HF_ENDPOINT,
                        help="HuggingFace / ModelScope endpoint. "
                             "Use https://hf-mirror.com for China mainland.")
    return parser.parse_args()


# ============================================================
#  Download helpers
# ============================================================

def download_from_modelscope(dataset_id: str, local_dir: Path, endpoint: str) -> bool:
    """Download a dataset from ModelScope to local_dir."""
    print(f"\n[DOWNLOAD] {dataset_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (look for parquet or images)
    existing = (list(local_dir.rglob("*.parquet")) or
                list(local_dir.rglob("*.png")) or
                list(local_dir.rglob("*.jpg")))
    if existing:
        print(f"  [SKIP] {len(existing)} files already exist. Delete if you want re-download.")
        return True

    try:
        from modelscope.hub.api import HubApi
        api = HubApi()
        # snapshot_download returns the local path
        api.snapshot_download(
            dataset_id,
            cache_dir=str(local_dir.parent),
            local_dir=str(local_dir),
        )
        print(f"  [OK] Downloaded to {local_dir}")
        return True
    except ImportError:
        print("  [FALLBACK] modelscope not installed, trying huggingface_hub...")
        return _download_from_hf(dataset_id, local_dir, endpoint)


def _download_from_hf(repo_id: str, local_dir: Path, endpoint: str) -> bool:
    """Fallback: download via huggingface_hub with mirror endpoint."""
    try:
        from huggingface_hub import snapshot_download
        os.environ["HF_ENDPOINT"] = endpoint
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_dir),
            resume_download=True,
        )
        print(f"  [OK] Downloaded to {local_dir}")
        return True
    except Exception as e:
        print(f"  [ERROR] Download failed: {e}")
        return False


def download_web_images(dataset_id: str, local_dir: Path, endpoint: str) -> bool:
    """Download only the web screenshot images (large: ~45 GB)."""
    print(f"\n[DOWNLOAD IMAGES] {dataset_id} -> {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)

    existing_imgs = list(local_dir.rglob("*.png")) + list(local_dir.rglob("*.jpg"))
    if existing_imgs:
        print(f"  [SKIP] {len(existing_imgs)} images already exist.")
        return True

    print("  [WARN] This will download ~45 GB of images. This may take a while...")
    start = time.time()
    res = download_from_modelscope(dataset_id, local_dir, endpoint)
    if res:
        elapsed = time.time() - start
        print(f"  [OK] Images downloaded in {elapsed/60:.1f} minutes")
    return res


# ============================================================
#  Process helpers — adapted from prepare_showui.py
# ============================================================

def build_image_index(images_root: Path) -> dict[str, Path]:
    """Scan image directory and build filename -> path mapping."""
    idx: dict[str, Path] = {}
    if not images_root.exists():
        print(f"    [WARN] Image dir not found: {images_root}")
        return idx
    for p in images_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            rel = str(p.relative_to(images_root))
            idx[rel] = p
    return idx


def process_web(parquet_glob: str, images_root: Path, output_dir: Path) -> list[dict]:
    """Read web parquet metadata, explode UI-element sequences, match images."""
    print("    Building image index ...")
    image_idx = build_image_index(images_root)
    print(f"    Indexed {len(image_idx)} images")

    files = sorted(glob.glob(parquet_glob))
    if not files:
        print(f"    [ERROR] No parquet files found: {parquet_glob}")
        return []

    import pyarrow.parquet as pq
    print(f"    Reading {len(files)} parquet file(s) ...")
    table = pq.read_table(files)
    rows = table.to_pylist()
    print(f"    Loaded {len(rows)} screenshot rows")

    missing_img = 0
    records: list[dict] = []

    for row_idx, row in enumerate(rows):
        image_url = row.get("image_url", "")
        if not image_url:
            missing_img += 1
            continue

        img_path = image_idx.get(image_url)
        if img_path is None:
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
                "app_package_name": elem_type,
            })

        if (row_idx + 1) % 5000 == 0:
            print(f"      {row_idx + 1} screenshots processed, {len(records)} elements")

    if missing_img:
        print(f"    [WARN] {missing_img} rows without matching image")
    print(f"    Web: {len(records)} element records from {len(rows)} screenshots")
    return records


def finalize_web(records: list[dict]) -> list[dict]:
    """Load web images to compute dimensions and absolute bboxes."""
    from PIL import Image
    result: list[dict] = []
    img_cache: dict[str, tuple[int, int]] = {}
    for rec in records:
        img_path = rec["image_path"]
        if img_path not in img_cache:
            try:
                img = Image.open(img_path)
                img_cache[img_path] = img.size
            except Exception:
                continue
        width, height = img_cache[img_path]
        x1, y1, x2, y2 = rec["rela_box"]
        result.append({
            "image_path": img_path,
            "image_url": rec.get("image_url", ""),
            "height": height, "width": width,
            "instruction": rec["instruction"],
            "rela_box": rec["rela_box"],
            "abs_box": [round(x1 * width, 2), round(y1 * height, 2),
                        round(x2 * width, 2), round(y2 * height, 2)],
            "domain": "web", "source": "ShowUI-web",
            "screenId": rec.get("screenId", ""),
            "app_package_name": rec.get("element_type", ""),
        })
    return result


def process_desktop(parquet_glob: str, output_dir: Path) -> list[dict]:
    """Read desktop parquet with embedded image bytes, extract to disk."""
    import pyarrow.parquet as pq
    from PIL import Image

    files = sorted(glob.glob(parquet_glob))
    if not files:
        print(f"    [ERROR] No parquet files: {parquet_glob}")
        return []
    print(f"    Reading {len(files)} parquet files ...")
    table = pq.read_table(files)
    rows = table.to_pylist()
    print(f"    Loaded {len(rows)} rows")

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

        # Distinguish relative (0-1) vs absolute (pixel) bboxes
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

        records.append({
            "image_path": str(img_path.resolve()),
            "image_url": row.get("image_url", ""),
            "height": height, "width": width,
            "instruction": instruction,
            "rela_box": rela_box, "abs_box": abs_box,
            "domain": "desktop", "source": "ShowUI-desktop",
            "screenId": f"desktop_{idx:06d}",
            "app_package_name": row.get("type", ""),
        })

    print(f"    Desktop: {len(records)} valid records")
    return records


def split_and_write(records: list[dict], output_dir: Path,
                    val_ratio: float, test_ratio: float, seed: int) -> None:
    """Shuffle, split into train/val/test, and write JSONL + YAML."""
    rng = random.Random(seed)
    rng.shuffle(records)

    n = len(records)
    test_n = int(n * test_ratio)
    val_n = int(n * val_ratio)
    splits = {
        "train": records[test_n + val_n:],
        "val": records[test_n:test_n + val_n],
        "test": records[:test_n],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        jsonl_path = output_dir / f"{name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        yaml_path = output_dir / f"dataset_{name}.yaml"
        yaml_path.write_text(f"datasets:\n  - json_path: {jsonl_path.resolve()}\n")

        web_n = sum(1 for r in recs if r["domain"] == "web")
        dsk_n = sum(1 for r in recs if r["domain"] == "desktop")
        print(f"    {name}: {len(recs)} records (web={web_n}, desktop={dsk_n})")
        print(f"      -> {jsonl_path}")


# ============================================================
#  Main
# ============================================================

def main() -> None:
    args = parse_args()

    do_download = args.all or args.download_only
    do_process = args.all or args.process_only
    do_web = not args.desktop_only
    do_desktop = not args.web_only

    # --- Download ---
    if do_download:
        print("=" * 60)
        print("=== DOWNLOAD ===")
        if do_web:
            # Web metadata (parquet files, ~16 MB)
            download_from_modelscope(WEB_DATASET, WEB_PARQUET_DIR, args.endpoint)

            # Web images (~45 GB)
            # Note: ModelScope datasets include images by default.
            # If images are missing, extract them:
            web_images_parsed = WEB_PARQUET_DIR / "showui_web_images"
            if not (WEB_IMAGES_DIR.exists() and any(WEB_IMAGES_DIR.rglob("*.png"))):
                print(f"\n[INFO] Checking for images in {WEB_PARQUET_DIR}...")
                pngs = list(WEB_PARQUET_DIR.rglob("*.png"))
                if pngs:
                    print(f"  Found {len(pngs)} PNGs, copying to {WEB_IMAGES_DIR}...")
                    WEB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
                    for fn in pngs:
                        dest = WEB_IMAGES_DIR / fn.relative_to(WEB_PARQUET_DIR)
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(fn, dest)
                else:
                    # Try downloading from modelscope_full location
                    print("  [INFO] Attempting ModelScope download for full images...")
                    download_web_images("showlab/ShowUI-web", WEB_IMAGES_DIR, args.endpoint)

        if do_desktop:
            download_from_modelscope(DESKTOP_DATASET, DESKTOP_PARQUET_DIR, args.endpoint)

    # --- Process ---
    if do_process:
        print("\n" + "=" * 60)
        print("=== PROCESS ===")
        output_dir = args.output_dir.resolve()
        all_records: list[dict] = []

        if do_web:
            web_parquet_glob = str(WEB_PARQUET_DIR / "*.parquet")
            images_root = WEB_IMAGES_DIR if WEB_IMAGES_DIR.exists() else WEB_PARQUET_DIR
            print(f"\n--- ShowUI-Web ---")
            print(f"    Parquet glob: {web_parquet_glob}")
            print(f"    Images dir:   {images_root}")
            raw = process_web(web_parquet_glob, images_root, output_dir)
            if raw:
                print("    Computing absolute bboxes ...")
                web_records = finalize_web(raw)
                print(f"    Finalized: {len(web_records)} web records")
                all_records.extend(web_records)

        if do_desktop:
            desktop_parquet_glob = str(DESKTOP_PARQUET_DIR / "data" / "*.parquet")
            print(f"\n--- ShowUI-Desktop ---")
            print(f"    Parquet glob: {desktop_parquet_glob}")
            desktop_records = process_desktop(desktop_parquet_glob, output_dir)
            all_records.extend(desktop_records)

        print(f"\n{'=' * 60}")
        print(f"=== TOTAL: {len(all_records)} records ===")
        if not all_records:
            print("[ERROR] No records produced. Check download paths.")
            return

        split_and_write(all_records, output_dir, args.val_ratio, args.test_ratio, args.seed)
        print(f"\n[DONE] Output: {output_dir}")


if __name__ == "__main__":
    main()
