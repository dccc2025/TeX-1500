#!/usr/bin/env python3
"""Download TeX-UNet weights from Hugging Face Hub."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

DEFAULT_FILES = (
    "model.safetensors",
    "config.json",
    "inference_config.yaml",
    "normalization.json",
    "metrics.json",
)


def parse_args() -> argparse.Namespace:
<<<<<<< HEAD
    parser = argparse.ArgumentParser(description="Download released TeX-UNet weights.")
    parser.add_argument("--repo-id", default="dccc2025/TeX-UNet")
    parser.add_argument("--filename", default="tex_unet_v2_darpa/model.safetensors")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/tex_unet_v2_darpa.safetensors"))
=======
    parser = argparse.ArgumentParser(description="Download released TeX-UNet files.")
    parser.add_argument("--repo-id", default="dccc2025/TeX-UNet")
    parser.add_argument("--variant", default="tex_unet_v2_darpa")
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_FILES),
        help="Files to download from the variant directory.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help=(
            "Compatibility mode: download one exact HF path, "
            "e.g. tex_unet_v2_darpa/model.safetensors."
        ),
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("checkpoints/hf/TeX-UNet"),
        help="Local root that mirrors the HF model repository layout.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Compatibility mode: output path for --filename.",
    )
>>>>>>> d26300b (final version)
    return parser.parse_args()


def download_one(*, repo_id: str, filename: str, revision: str | None, output: Path) -> Path:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded, output)
    return output


def main() -> None:
    args = parse_args()
    if args.filename or args.output:
        filename = args.filename or f"{args.variant}/model.safetensors"
        output = args.output or args.local_dir / filename
        saved = download_one(
            repo_id=args.repo_id,
            filename=filename,
            revision=args.revision,
            output=output,
        )
        print(f"Saved {args.repo_id}/{filename} to {saved}")
        return

    saved_files = []
    for name in args.files:
        filename = name if "/" in name else f"{args.variant}/{name}"
        saved_files.append(
            download_one(
                repo_id=args.repo_id,
                filename=filename,
                revision=args.revision,
                output=args.local_dir / filename,
            )
        )

    print(f"Saved {len(saved_files)} file(s) from {args.repo_id}/{args.variant}:")
    for path in saved_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
