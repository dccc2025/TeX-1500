import pytest
import torch

from tex1500 import InferenceConfig, normalizer_from_mapping, require_cuda_device


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
