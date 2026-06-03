# Data Format

The inference CLI expects one calibrated HSI scene per file.

## Supported Files

- `.mat`
- `.npz`
- `.npy`

The HSI array should be either `[H, W, C]` or `[C, H, W]`. Use
`--channel-axis -1` or `--channel-axis 0` if automatic inference is ambiguous.

## Common Keys

For `.mat` and `.npz` files, the loader searches these keys:

| Field | Keys |
|---|---|
| HSI cube | `denoised_hsi_original`, `denoised_hsi`, `hsi`, `HSI`, `radiance`, `cube` |
| Wavelengths | `working_wav`, `hsi_wav`, `wavelength`, `wavelength_um`, `wav`, `lambda` |
| Valid bands | `good_band_indices`, `good_bands`, `band_indices`, `valid_band_indices` |

Explicit keys can be passed with `--hsi-key`, `--wavelength-key`, and
`--good-band-key`.

## Wavelengths

Wavelengths should be in micrometers and should either have length `C` for the
full cube or length equal to the selected valid-band count. If no wavelengths
are provided, the CLI falls back to a linear 6--14 um grid. That fallback is
only for smoke testing; real TeX-1500 inference should use calibrated
wavelength positions.

## Outputs

The prediction directory contains NumPy arrays, a compressed `.npz`, an optional
MATLAB `.mat`, and optional PNG previews:

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

Temperature is denormalized with the training range `[243 K, 343 K]`. Emissivity
and texture are saved in the normalized TeX-UNet target space.
