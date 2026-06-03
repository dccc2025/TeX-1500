#!/usr/bin/env python3
"""Download TeX-UNet weights from Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download released TeX-UNet weights.")
    parser.add_argument("--repo-id", default="dccc2025/TeX-1500-baselines")
    parser.add_argument("--filename", default="tex_unet_v2/final.pt")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tex_unet_v2.pt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    downloaded = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(Path(downloaded).read_bytes())
    print(f"Saved {args.repo_id}/{args.filename} to {args.output}")


if __name__ == "__main__":
    main()
