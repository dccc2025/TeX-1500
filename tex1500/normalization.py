"""Normalization used by the released TeX-UNet baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValueRange:
    min_value: float
    max_value: float

    @property
    def span(self) -> float:
        span = self.max_value - self.min_value
        if span <= 0:
            raise ValueError(f"Invalid value range: {self}")
        return span

    def normalize_np(self, value: np.ndarray, *, clip: bool = True) -> np.ndarray:
        x = np.asarray(value, dtype=np.float32)
        if clip:
            x = np.clip(x, self.min_value, self.max_value)
        return ((x - self.min_value) / self.span).astype(np.float32, copy=False)

    def denormalize_np(self, value: np.ndarray) -> np.ndarray:
        return (np.asarray(value, dtype=np.float32) * self.span + self.min_value).astype(
            np.float32,
            copy=False,
        )


@dataclass(frozen=True)
class TeXUNetNormalizer:
    hsi: ValueRange = ValueRange(0.0, 25.0)
    temperature: ValueRange = ValueRange(243.0, 343.0)
    wavelength: ValueRange = ValueRange(6.0, 14.0)
    minmax_percentile_low: float = 1.0
    minmax_percentile_high: float = 99.0
    eps: float = 1e-6

    def normalize_hsi_np(self, value: np.ndarray) -> np.ndarray:
        return self.hsi.normalize_np(value, clip=True)

    def normalize_temperature_np(self, value: np.ndarray) -> np.ndarray:
        x = np.asarray(value, dtype=np.float32)
        x = np.clip(x, self.temperature.min_value, self.temperature.max_value)
        finite = x[np.isfinite(x)]
        if finite.size:
            lo, hi = np.percentile(
                finite,
                [self.minmax_percentile_low, self.minmax_percentile_high],
            )
            lo = max(float(lo), self.temperature.min_value)
            hi = min(float(hi), self.temperature.max_value)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                x = np.clip(x, lo, hi)
        return self.temperature.normalize_np(x, clip=True)

    def denormalize_temperature_np(self, value: np.ndarray) -> np.ndarray:
        return self.temperature.denormalize_np(value)
