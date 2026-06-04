# Data Format

This repository reads the TeX-1500 Hugging Face dataset format used by
`jialelin2007/TeX-1500`.

## Current HF Layout

The current public HF dataset release contains one preview sample:

```text
data/sample_0001/
  hsi.mat
  hsi_noisy.mat
  T.mat
  e.mat
  X.mat
  previews/
    hsi_band.png
    T.png
    e.png
    X.png
metadata/
  dataset_summary.json
  sample_manifest.jsonl
docs/
  DATA_FORMAT.md
```

The paper benchmark contains 1,522 samples; future HF releases can add more
sample directories using the same file conventions.

## `hsi.mat`

The inference CLI consumes `hsi.mat` directly.

| Variable | Shape in `sample_0001` | Dtype | Meaning |
|---|---:|---|---|
| `denoised_hsi_original` | `[260, 1500, 256]` | `float32` | Calibrated denoised LWIR HSI cube in `[H, W, C]` order. |
| `working_wav` | `[1, 256]` | `float32` | Calibrated wavelength positions in micrometers. |
| `hsi_wav` | `[1, 256]` | `float64` | Original HSI wavelength grid. |
| `good_band_indices` | `[1, 230]` | `int64` | Zero-indexed valid spectral bands. |
| `calibrated_sky` | `[1, 256]` | `float64` | Calibrated sky estimate. |
| `observed_sky` | `[1, 256]` | `float64` | Observed sky estimate. |
| `srf` | `[1, 1]` | `float64` | Sensor-response metadata placeholder. |

The loader searches for HSI keys in this order:

```text
denoised_hsi_original, denoised_hsi, hsi, HSI, radiance, cube
```

It searches wavelength keys in this order:

```text
working_wav, hsi_wav, wavelength, wavelength_um, wav, lambda
```

It searches valid-band keys in this order:

```text
good_band_indices, good_bands, band_indices, valid_band_indices
```

Explicit keys can be passed with `--hsi-key`, `--wavelength-key`, and
`--good-band-key`.

## Labels

The label files are used for evaluation or visualization, not for basic
inference:

| File | Variable | Shape in `sample_0001` | Meaning |
|---|---|---:|---|
| `T.mat` | `T` | `[260, 1500]` | Temperature field in Kelvin. |
| `e.mat` | `e` | `[260, 1500, 256]` | Spectral emissivity field. |
| `X.mat` | `X` | `[260, 1500]` | Scalar texture field. |

TeX-UNet predicts normalized `e` and `X`; temperature is also saved as both
normalized `T_norm` and denormalized `T_kelvin`.

## Running On A Sample

```bash
tex1500-infer \
  --input data/hf/TeX-1500/data/sample_0001/hsi.mat \
  --checkpoint checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/model.safetensors \
  --model-config checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/config.json \
  --normalization-config checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/normalization.json \
  --config checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/inference_config.yaml \
  --output-dir outputs/sample_0001_darpa \
  --device cuda:0
```

The CLI accepts `.mat`, `.npz`, and `.npy` inputs. Arrays may be `[H, W, C]` or
`[C, H, W]`; use `--channel-axis -1` or `--channel-axis 0` if a cropped sample
is ambiguous. This release is GPU-only; CPU inference is intentionally rejected.

## Outputs

```text
T_norm.npy
T_kelvin.npy
emissivity_norm.npy
texture_norm.npy
prediction.npz
prediction.mat
T.png
emissivity_midband.png
texture.png
```

PNG files are previews only. Use `.npy`, `.npz`, or `.mat` outputs for numeric
evaluation.
