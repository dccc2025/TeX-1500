"""Input loading helpers for calibrated HSI scenes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio

HSI_KEYS = ("denoised_hsi_original", "denoised_hsi", "hsi", "HSI", "radiance", "cube")
WAVELENGTH_KEYS = ("working_wav", "hsi_wav", "wavelength", "wavelength_um", "wav", "lambda")
GOOD_BAND_KEYS = ("good_band_indices", "good_bands", "band_indices", "valid_band_indices")


def public_keys(mapping: dict[str, Any]) -> list[str]:
    return [key for key in mapping if not key.startswith("__")]


def first_present(
    mapping: dict[str, Any],
    candidates: tuple[str, ...],
    *,
    explicit_key: str | None = None,
) -> np.ndarray | None:
    if explicit_key:
        if explicit_key not in mapping:
            raise KeyError(f"Key {explicit_key!r} not found; available={public_keys(mapping)}")
        return np.asarray(mapping[explicit_key])
    for key in candidates:
        if key in mapping:
            return np.asarray(mapping[key])
    return None


def ensure_hwc(array: np.ndarray, *, channel_axis: str = "auto") -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"HSI input must be 3D, got shape {arr.shape}")

    if channel_axis == "-1":
        return np.ascontiguousarray(arr)
    if channel_axis == "0":
        return np.ascontiguousarray(np.moveaxis(arr, 0, -1))
    if channel_axis != "auto":
        raise ValueError("channel_axis must be 'auto', '0', or '-1'")

    first, second, third = arr.shape
    last_looks_channel = third <= 512 and (third <= max(first, second) or third >= max(first, second))
    first_looks_channel = first <= 512 and (first < min(second, third) or first >= max(second, third))
    if last_looks_channel and not first_looks_channel:
        return np.ascontiguousarray(arr)
    if first_looks_channel and not last_looks_channel:
        return np.ascontiguousarray(np.moveaxis(arr, 0, -1))
    if last_looks_channel and first_looks_channel:
        if third > first:
            return np.ascontiguousarray(arr)
        if first > third:
            return np.ascontiguousarray(np.moveaxis(arr, 0, -1))
    if last_looks_channel:
        return np.ascontiguousarray(arr)
    raise ValueError(
        f"Cannot infer channel axis for HSI shape {arr.shape}; pass --channel-axis 0 or -1."
    )


def normalize_good_band_indices(
    raw: np.ndarray | None,
    full_c: int,
    *,
    index_base: str = "auto",
) -> np.ndarray:
    if raw is None:
        return np.arange(full_c, dtype=np.int64)
    idx = np.asarray(raw).squeeze()
    if idx.dtype == np.bool_:
        idx = np.flatnonzero(idx.reshape(-1))
    else:
        idx = idx.reshape(-1)
        if not np.issubdtype(idx.dtype, np.integer):
            if not np.all(np.isfinite(idx)) or not np.allclose(idx, np.round(idx)):
                raise ValueError("good band indices must be integer-valued")
        idx = idx.astype(np.int64)
    if idx.size == 0:
        raise ValueError("good band index list is empty")
    if index_base == "one":
        idx = idx - 1
    elif index_base == "auto" and int(idx.min()) >= 1 and int(idx.max()) == full_c:
        idx = idx - 1
    elif index_base != "zero" and index_base != "auto":
        raise ValueError("index_base must be 'auto', 'zero', or 'one'")
    if int(idx.min()) < 0 or int(idx.max()) >= full_c:
        raise ValueError(
            f"good band indices out of range for {full_c} bands: "
            f"min={int(idx.min())}, max={int(idx.max())}"
        )
    return np.unique(idx).astype(np.int64, copy=False)


def load_vector_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path)).squeeze()
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
        key = payload.files[0]
        return np.asarray(payload[key]).squeeze()
    text = path.read_text(encoding="utf-8").replace(",", " ")
    return np.asarray([float(item) for item in text.split()], dtype=np.float32)


def parse_vector(text_or_path: str | None) -> np.ndarray | None:
    if not text_or_path:
        return None
    path = Path(text_or_path)
    if path.is_file():
        return load_vector_file(path)
    return np.asarray([float(item.strip()) for item in text_or_path.split(",") if item.strip()])


def load_hsi_scene(
    path: str | Path,
    *,
    hsi_key: str | None = None,
    wavelength_key: str | None = None,
    good_band_key: str | None = None,
    channel_axis: str = "auto",
    good_band_index_base: str = "auto",
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mat":
        payload = sio.loadmat(path)
        hsi = first_present(payload, HSI_KEYS, explicit_key=hsi_key)
        if hsi is None:
            raise KeyError(f"No HSI key found in {path}; available={public_keys(payload)}")
        wavelength = first_present(payload, WAVELENGTH_KEYS, explicit_key=wavelength_key)
        good_raw = first_present(payload, GOOD_BAND_KEYS, explicit_key=good_band_key)
    elif suffix == ".npz":
        npz = np.load(path)
        payload = {key: npz[key] for key in npz.files}
        hsi = first_present(payload, HSI_KEYS, explicit_key=hsi_key)
        if hsi is None and npz.files:
            hsi = np.asarray(npz[npz.files[0]])
        if hsi is None:
            raise KeyError(f"No HSI array found in {path}")
        wavelength = first_present(payload, WAVELENGTH_KEYS, explicit_key=wavelength_key)
        good_raw = first_present(payload, GOOD_BAND_KEYS, explicit_key=good_band_key)
    elif suffix == ".npy":
        hsi = np.load(path)
        wavelength = None
        good_raw = None
    else:
        raise ValueError(f"Unsupported input suffix {suffix!r}; use .mat, .npy, or .npz")

    hsi_hwc = ensure_hwc(hsi, channel_axis=channel_axis)
    good_idx = normalize_good_band_indices(
        good_raw,
        hsi_hwc.shape[-1],
        index_base=good_band_index_base,
    )
    if wavelength is not None:
        wavelength = np.asarray(wavelength, dtype=np.float32).squeeze().reshape(-1)
    return hsi_hwc, wavelength, good_idx
