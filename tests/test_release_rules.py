import numpy as np
import pytest
import torch

from tex1500 import InferenceConfig, normalizer_from_mapping, require_cuda_device
from tex1500.inference import Prediction, save_prediction


def test_inference_config_defaults_to_cuda() -> None:
    assert InferenceConfig().device == "cuda"


def test_cpu_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="GPU-only"):
        require_cuda_device("cpu")


def test_missing_cuda_does_not_fall_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        require_cuda_device("cuda")


def test_hf_normalization_mapping() -> None:
    normalizer = normalizer_from_mapping(
        {
            "hsi": {"min": 0.0, "max": 25.0},
            "temperature": {
                "min_K": 243.0,
                "max_K": 343.0,
                "percentile_low": 1.0,
                "percentile_high": 99.0,
            },
            "wavelength": {"min_um": 6.0, "max_um": 14.0},
            "eps": 1e-6,
        }
    )
    assert normalizer.hsi.max_value == 25.0
    assert normalizer.temperature.min_value == 243.0
    assert normalizer.wavelength.max_value == 14.0


def test_save_prediction_writes_only_mat_and_pngs(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prediction = Prediction(
        temperature_norm=np.zeros((2, 3), dtype=np.float32),
        temperature_kelvin=np.full((2, 3), 300.0, dtype=np.float32),
        emissivity_norm=np.zeros((2, 3, 4), dtype=np.float32),
        texture_norm=np.ones((2, 3), dtype=np.float32),
        wavelength_um=np.linspace(8.0, 12.0, 4, dtype=np.float32),
        band_indices=np.arange(4, dtype=np.int64),
    )

    def fake_save_pngs(_prediction: Prediction, output) -> None:
        for name in ("T.png", "emissivity_midband.png", "texture.png"):
            (output / name).write_bytes(b"png")

    monkeypatch.setattr("tex1500.inference._save_pngs", fake_save_pngs)
    save_prediction(prediction, tmp_path)

    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "T.png",
        "emissivity_midband.png",
        "prediction.mat",
        "texture.png",
    ]
