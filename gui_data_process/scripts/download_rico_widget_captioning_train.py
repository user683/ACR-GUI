#!/usr/bin/env python3
"""Download the train split of rootsautomation/RICO-WidgetCaptioning."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "rootsautomation/RICO-WidgetCaptioning"
DEFAULT_OUTPUT_DIR = Path("data/raw_hf/rootsautomation_RICO-WidgetCaptioning")
DEFAULT_PATTERNS = ("data/train-*.parquet",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only the train split files from rootsautomation/RICO-WidgetCaptioning."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store the downloaded files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Dataset revision, branch, or commit SHA to download. Default: main",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even when they already exist locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
        allow_patterns=list(DEFAULT_PATTERNS),
        token=args.token,
        force_download=args.force,
    )

    downloaded = sorted(Path(local_path).glob("data/train-*.parquet"))
    if not downloaded:
        raise RuntimeError(
            "No train split files were found after download. "
            "Check whether the dataset file pattern has changed."
        )

    print(f"Downloaded {len(downloaded)} train split file(s) to {Path(local_path).resolve()}")
    for path in downloaded:
        print(path)


if __name__ == "__main__":
    main()
