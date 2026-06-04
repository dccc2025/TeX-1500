"""Full-scene TeX-UNet inference."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import TeXUNet, TeXUNetConfig
from .normalization import TeXUNetNormalizer


@dataclass(frozen=True)
class InferenceConfig:
    num_bands: int = 64
    min_band_coverage: int = 5
    patch_size: int = 224
    stride: int = 112
    spatial_mode: str = "tiled"
    blend_window: str = "raised_cosine"
    blend_min_weight: float = 0.4
    blend_gaussian_sigma: float = 0.25
    precision: str = "bf16"
    seed: int = 20260527
    batch_size: int = 16
    full_pad_multiple: int = 16
    device: str = "cuda"
    save_png: bool = True


@dataclass(frozen=True)
class Prediction:
    temperature_norm: np.ndarray
    temperature_kelvin: np.ndarray
    emissivity_norm: np.ndarray
    texture_norm: np.ndarray
    wavelength_um: np.ndarray
    band_indices: np.ndarray


def model_config_from_mapping(mapping: dict[str, Any] | None) -> TeXUNetConfig:
    if not mapping:
        return TeXUNetConfig()
    if isinstance(mapping.get("model_config"), dict):
        mapping = mapping["model_config"]
    valid_names = {field.name for field in fields(TeXUNetConfig)}
    return TeXUNetConfig(**{key: value for key, value in mapping.items() if key in valid_names})


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def require_cuda_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve and validate the CUDA device required by this release."""

    resolved = torch.device(device or "cuda")
    if resolved.type != "cuda":
        raise ValueError(
            "TeX-UNet inference is GPU-only for this release; pass --device cuda "
            "or --device cuda:<index>."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in this PyTorch environment. Install a CUDA-enabled "
            "PyTorch build before running TeX-UNet inference."
        )
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise ValueError(
            f"Requested CUDA device index {resolved.index}, but only "
            f"{torch.cuda.device_count()} device(s) are visible."
        )
    return resolved


def load_model(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device | None = None,
    model_config: dict[str, Any] | TeXUNetConfig | None = None,
) -> TeXUNet:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    device = require_cuda_device(device)

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state_dict = load_file(str(path), device="cpu")
        cfg = (
            model_config
            if isinstance(model_config, TeXUNetConfig)
            else model_config_from_mapping(model_config)
        )
    else:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
            raw_config = checkpoint.get("model_config") or model_config
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            raw_config = checkpoint.get("model_config") or model_config
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
            raw_config = model_config
        else:
            raise TypeError(f"Unsupported checkpoint payload type: {type(checkpoint)!r}")
        cfg = (
            raw_config
            if isinstance(raw_config, TeXUNetConfig)
            else model_config_from_mapping(raw_config)
        )

    model = TeXUNet(cfg)
    model.load_state_dict(_strip_module_prefix(state_dict), strict=True)
    model.to(device)
    model.eval()
    return model


def amp_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError("precision must be 'fp32', 'bf16', or 'fp16'")


def tile_starts(size: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if size <= patch_size:
        return [0]
    starts = list(range(0, size - patch_size + 1, stride))
    last = size - patch_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def make_blend_window(
    patch_size: int,
    *,
    kind: str,
    min_weight: float,
    gaussian_sigma: float,
) -> np.ndarray:
    if kind == "uniform":
        window = np.ones((patch_size, patch_size), dtype=np.float32)
    elif kind in {"raised_cosine", "hann"}:
        one_d = np.hanning(patch_size).astype(np.float32)
        window = np.outer(one_d, one_d)
        if kind == "raised_cosine":
            window = min_weight + (1.0 - min_weight) * window
    elif kind == "gaussian":
        coords = np.linspace(-1.0, 1.0, patch_size, dtype=np.float32)
        yy, xx = np.meshgrid(coords, coords, indexing="ij")
        sigma = max(float(gaussian_sigma), 1e-6)
        window = np.exp(-0.5 * (xx * xx + yy * yy) / (sigma * sigma)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported blend window: {kind}")
    if kind != "raised_cosine" and min_weight > 0:
        window = np.maximum(window, min_weight)
    return np.ascontiguousarray(window.astype(np.float32, copy=False))


def make_spectral_subsets(
    *,
    good_count: int,
    num_bands: int,
    min_coverage: int,
    wavelength_um: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if good_count < num_bands:
        raise ValueError(f"Only {good_count} valid bands are available, but num_bands={num_bands}")
    if min_coverage <= 0:
        raise ValueError("min_band_coverage must be positive")
    if good_count == num_bands and min_coverage == 1:
        return [np.arange(good_count, dtype=np.int64)]

    coverage = np.zeros(good_count, dtype=np.int32)
    subsets: list[np.ndarray] = []
    max_passes = int(math.ceil(min_coverage * good_count / num_bands)) + min_coverage + 32
    while int(coverage.min()) < min_coverage:
        undercovered = np.flatnonzero(coverage < min_coverage)
        if undercovered.size >= num_bands:
            local_idx = rng.choice(undercovered, size=num_bands, replace=False)
        else:
            local = [int(v) for v in undercovered]
            remaining = np.setdiff1d(
                np.arange(good_count, dtype=np.int64),
                np.asarray(local, dtype=np.int64),
                assume_unique=False,
            )
            weights = 1.0 / (coverage[remaining].astype(np.float64) + 1.0)
            weights /= weights.sum()
            extra = rng.choice(remaining, size=num_bands - len(local), replace=False, p=weights)
            local_idx = np.asarray([*local, *[int(v) for v in extra]], dtype=np.int64)
        order = np.argsort(wavelength_um[local_idx], kind="stable")
        local_idx = np.asarray(local_idx[order], dtype=np.int64)
        subsets.append(local_idx)
        coverage[local_idx] += 1
        if len(subsets) > max_passes:
            raise RuntimeError(
                f"Failed to reach min_band_coverage={min_coverage}; "
                f"current minimum coverage is {int(coverage.min())}."
            )
    return subsets


def _pad_to_minimum(hsi: np.ndarray, *, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w, _ = hsi.shape
    pad_h = max(0, patch_size - h)
    pad_w = max(0, patch_size - w)
    if pad_h == 0 and pad_w == 0:
        return hsi, (h, w)
    mode = "reflect" if h > 1 and w > 1 else "edge"
    padded = np.pad(hsi, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return np.ascontiguousarray(padded), (h, w)


def _pad_to_multiple(hsi: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w, _ = hsi.shape
    if multiple <= 1:
        return hsi, (h, w)
    target_h = int(math.ceil(h / multiple) * multiple)
    target_w = int(math.ceil(w / multiple) * multiple)
    pad_h = target_h - h
    pad_w = target_w - w
    if pad_h == 0 and pad_w == 0:
        return hsi, (h, w)
    mode = "reflect" if h > 1 and w > 1 else "edge"
    padded = np.pad(hsi, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
    return np.ascontiguousarray(padded), (h, w)


def _run_full_image(
    hsi_sel: np.ndarray,
    selected_wavelength_um: np.ndarray,
    *,
    model: TeXUNet,
    device: torch.device,
    normalizer: TeXUNetNormalizer,
    config: InferenceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded, original_hw = _pad_to_multiple(hsi_sel, config.full_pad_multiple)
    hsi_norm = normalizer.normalize_hsi_np(padded)
    hsi_tensor = torch.from_numpy(np.moveaxis(hsi_norm, -1, 0)[None]).to(device=device)
    wav_tensor = torch.from_numpy(selected_wavelength_um[None].astype(np.float32)).to(device=device)
    dtype = amp_dtype(config.precision)
    amp_enabled = device.type == "cuda" and config.precision != "fp32"
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=amp_enabled):
        pred = model(hsi_tensor, wav_tensor)
    h, w = original_hw
    pred_t = pred["T"].detach().float().cpu().numpy()[0, 0, :h, :w]
    pred_e = pred["e"].detach().float().cpu().numpy()[0].transpose(1, 2, 0)[:h, :w]
    pred_x = pred["X"].detach().float().cpu().numpy()[0, 0, :h, :w]
    return pred_t, pred_e, pred_x


def _run_tiled(
    hsi_sel: np.ndarray,
    selected_wavelength_um: np.ndarray,
    *,
    model: TeXUNet,
    device: torch.device,
    normalizer: TeXUNetNormalizer,
    config: InferenceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded, original_hw = _pad_to_minimum(hsi_sel, patch_size=config.patch_size)
    height, width, _ = padded.shape
    y_starts = tile_starts(height, config.patch_size, config.stride)
    x_starts = tile_starts(width, config.patch_size, config.stride)
    coords = [(y0, x0) for y0 in y_starts for x0 in x_starts]
    blend = make_blend_window(
        config.patch_size,
        kind=config.blend_window,
        min_weight=config.blend_min_weight,
        gaussian_sigma=config.blend_gaussian_sigma,
    )

    pred_t_full = np.zeros((height, width), dtype=np.float32)
    pred_x_full = np.zeros((height, width), dtype=np.float32)
    pred_e_full = np.zeros((height, width, config.num_bands), dtype=np.float32)
    count_full = np.zeros((height, width), dtype=np.float32)

    dtype = amp_dtype(config.precision)
    amp_enabled = device.type == "cuda" and config.precision != "fp32"
    for start in range(0, len(coords), config.batch_size):
        batch_coords = coords[start : start + config.batch_size]
        batch_tiles = []
        for y0, x0 in batch_coords:
            tile = padded[y0 : y0 + config.patch_size, x0 : x0 + config.patch_size]
            tile = normalizer.normalize_hsi_np(tile)
            batch_tiles.append(np.ascontiguousarray(np.moveaxis(tile, -1, 0)))
        hsi_tensor = torch.from_numpy(np.stack(batch_tiles, axis=0)).to(device=device)
        wav_tensor = torch.from_numpy(
            np.stack([selected_wavelength_um.astype(np.float32) for _ in batch_coords], axis=0)
        ).to(device=device)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=amp_enabled,
        ):
            pred = model(hsi_tensor, wav_tensor)

        pred_t = pred["T"].detach().float().cpu().numpy()[:, 0]
        pred_e = pred["e"].detach().float().cpu().numpy().transpose(0, 2, 3, 1)
        pred_x = pred["X"].detach().float().cpu().numpy()[:, 0]
        for idx, (y0, x0) in enumerate(batch_coords):
            y1 = y0 + config.patch_size
            x1 = x0 + config.patch_size
            pred_t_full[y0:y1, x0:x1] += pred_t[idx] * blend
            pred_x_full[y0:y1, x0:x1] += pred_x[idx] * blend
            pred_e_full[y0:y1, x0:x1] += pred_e[idx] * blend[:, :, None]
            count_full[y0:y1, x0:x1] += blend

    count = np.maximum(count_full, 1e-6)
    pred_t_full /= count
    pred_x_full /= count
    pred_e_full /= count[:, :, None]
    h, w = original_hw
    return pred_t_full[:h, :w], pred_e_full[:h, :w], pred_x_full[:h, :w]


def predict_scene(
    model: TeXUNet,
    hsi_hwc: np.ndarray,
    *,
    wavelength_um: np.ndarray | None = None,
    good_band_indices: np.ndarray | None = None,
    config: InferenceConfig | None = None,
    normalizer: TeXUNetNormalizer | None = None,
) -> Prediction:
    config = config or InferenceConfig()
    normalizer = normalizer or TeXUNetNormalizer()
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("TeX-UNet inference is GPU-only; the model must be on a CUDA device.")
    hsi = np.asarray(hsi_hwc, dtype=np.float32)
    if hsi.ndim != 3:
        raise ValueError(f"hsi_hwc must be [H,W,C], got {hsi.shape}")
    full_c = hsi.shape[-1]
    good_idx = (
        np.arange(full_c, dtype=np.int64)
        if good_band_indices is None
        else np.asarray(good_band_indices, dtype=np.int64)
    )
    if wavelength_um is None:
        wavelength_um = np.linspace(
            normalizer.wavelength.min_value,
            normalizer.wavelength.max_value,
            full_c,
            dtype=np.float32,
        )
    wavelength_um = np.asarray(wavelength_um, dtype=np.float32).squeeze().reshape(-1)
    if wavelength_um.size == full_c:
        wav_good = wavelength_um[good_idx]
    elif wavelength_um.size == good_idx.size:
        wav_good = wavelength_um
    else:
        raise ValueError(
            f"wavelength length {wavelength_um.size} does not match full bands {full_c} "
            f"or selected bands {good_idx.size}"
        )

    hsi_good = hsi[..., good_idx]
    rng = np.random.default_rng(config.seed)
    subsets = make_spectral_subsets(
        good_count=good_idx.size,
        num_bands=config.num_bands,
        min_coverage=config.min_band_coverage,
        wavelength_um=wav_good,
        rng=rng,
    )

    height, width, _ = hsi_good.shape
    t_acc = np.zeros((height, width), dtype=np.float32)
    x_acc = np.zeros((height, width), dtype=np.float32)
    e_acc = np.zeros((height, width, good_idx.size), dtype=np.float32)
    e_count = np.zeros(good_idx.size, dtype=np.float32)

    runner = _run_full_image if config.spatial_mode == "full" else _run_tiled
    if config.spatial_mode not in {"full", "tiled"}:
        raise ValueError("spatial_mode must be 'full' or 'tiled'")

    for local_idx in subsets:
        selected_hsi = hsi_good[..., local_idx]
        selected_wav = wav_good[local_idx]
        pred_t, pred_e, pred_x = runner(
            selected_hsi,
            selected_wav,
            model=model,
            device=device,
            normalizer=normalizer,
            config=config,
        )
        t_acc += pred_t
        x_acc += pred_x
        e_acc[..., local_idx] += pred_e
        e_count[local_idx] += 1.0

    t_norm = t_acc / float(len(subsets))
    x_norm = x_acc / float(len(subsets))
    e_norm = e_acc / np.maximum(e_count, 1.0)[None, None, :]
    return Prediction(
        temperature_norm=t_norm.astype(np.float32, copy=False),
        temperature_kelvin=normalizer.denormalize_temperature_np(t_norm),
        emissivity_norm=e_norm.astype(np.float32, copy=False),
        texture_norm=x_norm.astype(np.float32, copy=False),
        wavelength_um=wav_good.astype(np.float32, copy=False),
        band_indices=good_idx.astype(np.int64, copy=False),
    )


def save_prediction(
    prediction: Prediction,
    output_dir: str | Path,
    *,
    save_png: bool = True,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    import scipy.io as sio

    sio.savemat(
        output / "prediction.mat",
        {
            "T_norm": prediction.temperature_norm,
            "T_kelvin": prediction.temperature_kelvin,
            "e_norm": prediction.emissivity_norm,
            "X_norm": prediction.texture_norm,
            "wavelength_um": prediction.wavelength_um,
            "band_indices": prediction.band_indices,
        },
    )

    if save_png:
        _save_pngs(prediction, output)


def _save_pngs(prediction: Prediction, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_image(array: np.ndarray, name: str, cmap: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4))
        finite = array[np.isfinite(array)]
        if finite.size:
            lo, hi = np.percentile(finite, [1.0, 99.0])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                array = np.clip(array, lo, hi)
        im = ax.imshow(array, cmap=cmap, aspect="auto")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.savefig(output / name, bbox_inches="tight", dpi=300)
        plt.close(fig)

    save_image(prediction.temperature_kelvin, "T.png", "hot")
    mid = prediction.emissivity_norm.shape[-1] // 2
    save_image(prediction.emissivity_norm[..., mid], "emissivity_midband.png", "viridis")
    save_image(prediction.texture_norm, "texture.png", "gray")
