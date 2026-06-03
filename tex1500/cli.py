"""Command line interface for TeX-UNet inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from .inference import InferenceConfig, load_model, predict_scene, save_prediction
from .io import load_hsi_scene, parse_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TeX-UNet inference on one HSI scene.")
    parser.add_argument("--input", type=Path, required=True, help="Input .mat, .npy, or .npz HSI file.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="TeX-UNet checkpoint path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for predicted TeX files.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML inference config.")
    parser.add_argument("--model-config", type=Path, default=None, help="Optional JSON/YAML model config.")

    parser.add_argument("--hsi-key", default=None)
    parser.add_argument("--wavelength-key", default=None)
    parser.add_argument("--good-band-key", default=None)
    parser.add_argument("--wavelengths", default=None, help="Comma list or path to wavelength vector.")
    parser.add_argument("--good-band-indices", default=None, help="Comma list or path to valid band indices.")
    parser.add_argument("--channel-axis", choices=("auto", "0", "-1"), default="auto")
    parser.add_argument("--good-band-index-base", choices=("auto", "zero", "one"), default="auto")

    parser.add_argument("--num-bands", type=int, default=None)
    parser.add_argument("--min-band-coverage", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--spatial-mode", choices=("tiled", "full"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-png", action="store_true")
    return parser.parse_args()


def _read_mapping(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    return yaml.safe_load(text) or {}


def _build_inference_config(args: argparse.Namespace) -> InferenceConfig:
    payload = _read_mapping(args.config)
    if "inference" in payload:
        payload = payload["inference"] or {}
    allowed = set(asdict(InferenceConfig()).keys())
    config_values = {key: value for key, value in payload.items() if key in allowed}
    for key in (
        "num_bands",
        "min_band_coverage",
        "patch_size",
        "stride",
        "batch_size",
        "precision",
        "device",
        "spatial_mode",
        "seed",
    ):
        value = getattr(args, key)
        if value is not None:
            config_values[key] = value
    if args.no_png:
        config_values["save_png"] = False
    return InferenceConfig(**config_values)


def main() -> None:
    args = parse_args()
    config = _build_inference_config(args)
    model_config = _read_mapping(args.model_config)

    hsi, wavelength, good_idx = load_hsi_scene(
        args.input,
        hsi_key=args.hsi_key,
        wavelength_key=args.wavelength_key,
        good_band_key=args.good_band_key,
        channel_axis=args.channel_axis,
        good_band_index_base=args.good_band_index_base,
    )
    cli_wavelength = parse_vector(args.wavelengths)
    if cli_wavelength is not None:
        wavelength = cli_wavelength
    cli_good = parse_vector(args.good_band_indices)
    if cli_good is not None:
        good_idx = cli_good.astype("int64")

    device = torch.device(config.device)
    model = load_model(args.checkpoint, device=device, model_config=model_config)
    prediction = predict_scene(
        model,
        hsi,
        wavelength_um=wavelength,
        good_band_indices=good_idx,
        config=config,
    )
    save_prediction(prediction, args.output_dir, save_png=config.save_png)
    print(f"Saved TeX prediction to {args.output_dir}")
    print(
        "Shapes: "
        f"T={prediction.temperature_norm.shape}, "
        f"e={prediction.emissivity_norm.shape}, "
        f"X={prediction.texture_norm.shape}"
    )


if __name__ == "__main__":
    main()
