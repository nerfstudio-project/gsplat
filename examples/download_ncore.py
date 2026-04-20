# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download an NCore v4 clip from HuggingFace.

Downloads all files for a single NCore v4 clip from the gated HuggingFace
dataset.  Requires accepting the license and providing an API token.

The downloaded clip can be used directly with av_trainer.py by passing
the meta JSON path:

    python av_trainer.py --scene /path/to/clip/pai_*.json --max-steps 15000

Usage:
    # Download clip (requires HF_TOKEN env var or --hf-token)
    HF_TOKEN=hf_... python download_ncore.py \\
        --clip-id 004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6

    # Download to specific directory
    HF_TOKEN=hf_... python download_ncore.py \\
        --clip-id 004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6 \\
        --output-dir ./ncore_data
"""

from __future__ import annotations

import argparse
import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

HF_DATASET_ID = "nvidia/PhysicalAI-Autonomous-Vehicles-NCore"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_DATASET_ID}/resolve/main"


def _hf_download_file(url: str, dest: str, token: str) -> None:
    """Download a single file from HuggingFace with auth."""
    if not url.startswith("https://"):
        raise ValueError(f"Only HTTPS URLs are allowed, got: {url}")
    req = Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(req, timeout=300) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(
                            f"\r    {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB "
                            f"({pct:.0f}%)",
                            end="",
                            flush=True,
                        )
            if total:
                print()
    except (URLError, OSError) as e:
        raise RuntimeError(f"Failed to download {url} -> {dest}: {e}") from e


def _hf_list_clip_files(clip_id: str, token: str) -> list[dict]:
    """List files in an NCore clip on HuggingFace."""
    url = (
        f"https://huggingface.co/api/datasets/{HF_DATASET_ID}"
        f"/tree/main/clips/{clip_id}"
    )
    req = Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except (URLError, OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to list files for clip {clip_id}: {e}") from e


def download_clip(clip_id: str, token: str, output_dir: str) -> str:
    """Download all files for an NCore clip from HuggingFace.
    Returns the clip directory path."""
    print(f"Listing files for clip {clip_id}...")
    files = _hf_list_clip_files(clip_id, token)
    print(f"Found {len(files)} files")

    clip_dir = os.path.join(output_dir, clip_id)
    os.makedirs(clip_dir, exist_ok=True)

    total_size = sum(f.get("size", 0) for f in files)
    print(f"Total download: {total_size / 1e9:.2f} GB")

    meta_json = None
    path_prefix = f"clips/{clip_id}/"
    for i, f in enumerate(files):
        # Preserve nested structure to avoid basename collisions.
        rel_path = f["path"].split(path_prefix, 1)[-1]
        dest = os.path.join(clip_dir, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)

        if rel_path.endswith(".json"):
            meta_json = dest

        if os.path.exists(dest) and os.path.getsize(dest) == f.get("size", -1):
            print(f"  [{i+1}/{len(files)}] {rel_path} (cached)")
            continue

        size_mb = f.get("size", 0) / 1e6
        print(f"  [{i+1}/{len(files)}] {rel_path} ({size_mb:.1f} MB)")
        url = f"{HF_BASE_URL}/{f['path']}"
        _hf_download_file(url, dest, token)

    print(f"\nClip downloaded to {clip_dir}")
    if meta_json:
        print(f"Meta JSON: {meta_json}")
        print("\nTo train:")
        print(
            f"  python av_trainer.py --scene {meta_json} "
            f"--max-steps 15000 --mcmc --sh-degree 3"
        )
    else:
        print(f"Warning: no meta JSON (.json) file found in clip {clip_id}")
    return clip_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download an NCore v4 clip from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Accept the license at:
  https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore

Get a token at:
  https://huggingface.co/settings/tokens
""",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default="004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6",
        help="NCore clip UUID (default: 004c2001-5fc3-43b1-a4d8-bfb0bbb9fdc6)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ncore_data",
        help="directory to save downloaded files (default: ./ncore_data)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    args = parser.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        parser.error(
            "HuggingFace token required. Set HF_TOKEN env var or pass --hf-token.\n"
            "Get a token at https://huggingface.co/settings/tokens\n"
            "Accept the dataset license at "
            "https://huggingface.co/datasets/nvidia/"
            "PhysicalAI-Autonomous-Vehicles-NCore"
        )

    download_clip(args.clip_id, token, args.output_dir)


if __name__ == "__main__":
    main()
