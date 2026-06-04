# TeX-1500

**GPU-only inference code for the TeX-1500 TeX-UNet baseline.**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2606.03806-b31b1b)](https://arxiv.org/abs/2606.03806)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d)](https://huggingface.co/datasets/jialelin2007/TeX-1500)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-Hugging%20Face-ffcc4d)](https://huggingface.co/dccc2025/TeX-UNet)
[![Code](https://img.shields.io/badge/Code-GitHub-111111)](https://github.com/dccc2025/TeX-1500)

TeX-1500 is a paired LWIR hyperspectral benchmark for temperature `T`,
emissivity `e`, and scalar texture `X` decomposition. This repository is an
inference-only GitHub release for the TeX-UNet baseline. It includes the model
architecture, HSI loaders, tiled/full-scene inference, output writers, compact
examples, and the architecture figure below.

The dataset and pretrained checkpoints are hosted externally:

- Dataset: https://huggingface.co/datasets/jialelin2007/TeX-1500
- Checkpoints: https://huggingface.co/dccc2025/TeX-UNet
- Paper: https://arxiv.org/abs/2606.03806

![TeX-UNet architecture](assets/tex_unet_architecture.jpg)

## Release Scope

This GitHub repository intentionally keeps only code, lightweight configs,
documentation, tests, and one image asset:

```text
assets/tex_unet_architecture.jpg
```

The repository does not store training code, raw data, checkpoints, generated
outputs, optimizer states, experiment logs, or extra paper figures. Local
directories such as `data/`, `checkpoints/`, and `outputs/` are ignored by Git
and are intended for downloaded Hugging Face files and local predictions.

## GPU-Only Rule

This release supports CUDA inference only. CPU inference is not a supported
runtime path. The CLI and Python API reject `--device cpu` and fail if
`torch.cuda.is_available()` is false.

Install a CUDA-enabled PyTorch build before installing the package. The locked
release uses torch 2.7.1. The example below uses the official CUDA 12.8 wheel
index:

```bash
git clone https://github.com/dccc2025/TeX-1500.git
cd TeX-1500

uv venv .venv --python 3.10
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.7.1
uv pip install --python .venv/bin/python -e .
source .venv/bin/activate

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled torch is required.")
print("gpu:", torch.cuda.get_device_name(0))
PY
```

If CUDA 12.8 is not appropriate for your machine, use the official PyTorch CUDA
index that matches your driver. Do not install from the CPU wheel index.

## Download Files

The dataset is gated on Hugging Face. Log in and accept the dataset conditions
before downloading:

```bash
hf auth login
```

If your network has problems with Xet-backed downloads, disabling Xet can make
the Hugging Face CLI more predictable:

```bash
export HF_HUB_DISABLE_XET=1
```

Download the DARPA TeX-UNet checkpoint and metadata:

```bash
python scripts/download_weights.py --variant tex_unet_v2_darpa
```

This creates:

```text
checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/
  model.safetensors
  config.json
  inference_config.yaml
  normalization.json
  metrics.json
```

To download the FTIR few-shot checkpoint instead:

```bash
python scripts/download_weights.py --variant tex_unet_v2_ftir_fewshot
```

Download the current public dataset preview sample:

```bash
hf download jialelin2007/TeX-1500 \
  data/sample_0001/hsi.mat \
  data/sample_0001/T.mat \
  data/sample_0001/e.mat \
  data/sample_0001/X.mat \
  metadata/sample_manifest.jsonl \
  metadata/dataset_summary.json \
  docs/DATA_FORMAT.md \
  --repo-type dataset \
  --local-dir data/hf/TeX-1500
```

## Single-HSI Inference

Run TeX-UNet on one HSI scene:

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

When `config.json`, `inference_config.yaml`, and `normalization.json` are in
the same directory as `model.safetensors`, the CLI can discover them
automatically. The explicit flags above are shown for reproducibility.

For a custom single HSI file, pass a `.mat`, `.npz`, or `.npy` input. Arrays may
be `[H, W, C]` or `[C, H, W]`; use `--channel-axis -1` or `--channel-axis 0` if
the shape is ambiguous. For `.mat` and `.npz`, the loader searches common keys:

```text
HSI keys: denoised_hsi_original, denoised_hsi, hsi, HSI, radiance, cube
Wavelength keys: working_wav, hsi_wav, wavelength, wavelength_um, wav, lambda
Valid-band keys: good_band_indices, good_bands, band_indices, valid_band_indices
```

You can override them explicitly:

```bash
tex1500-infer \
  --input path/to/custom_hsi.mat \
  --hsi-key denoised_hsi_original \
  --wavelength-key working_wav \
  --good-band-key good_band_indices \
  --checkpoint checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/model.safetensors \
  --model-config checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/config.json \
  --normalization-config checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/normalization.json \
  --config checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/inference_config.yaml \
  --output-dir outputs/custom_darpa \
  --device cuda:0
```

The default model samples 64 valid bands per pass. The input must contain at
least 64 valid bands after `good_band_indices` filtering.

Expected outputs:

```text
outputs/sample_0001_darpa/
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

`T_kelvin` is denormalized with the released model normalization. `e` and `X`
are saved in the normalized target space used by the baseline.

## Checkpoints

| HF directory | Training scope | Intended use |
|---|---|---|
| `tex_unet_v2_darpa` | DARPA IH training split | DARPA IH test and FTIR zero-shot transfer |
| `tex_unet_v2_ftir_fewshot` | FTIR few-shot split initialized from the DARPA model | FTIR few-shot transfer |

Both checkpoints are released as `safetensors` in `dccc2025/TeX-UNet`.

## Benchmark

| Test split | Checkpoint | T MAE (K) | T MAPE (%) | e MSE | e SAM (rad) | X MSE | X SAM (rad) |
|---|---|---:|---:|---:|---:|---:|---:|
| DARPA IH-test | `tex_unet_v2_darpa` | 7.3284 | 2.5488 | 0.0453 | 0.2267 | 0.0311 | 0.5206 |
| FTIR-zeroshot-test | `tex_unet_v2_darpa` | 5.8309 | 1.9753 | 0.0674 | 0.0451 | 0.0219 | 0.2995 |
| FTIR-fewshot-test | `tex_unet_v2_ftir_fewshot` | 4.1004 | 1.3830 | 0.0458 | 0.1970 | 0.0220 | 0.2224 |

`e` and `X` are normalized. See the paper and Hugging Face model metadata for
the full evaluation protocol.

## Citation

```bibtex
@misc{dai2026tex1500,
  title = {TeX-1500: A Paired Real-World LWIR Hyperspectral Dataset and Benchmark for Temperature--Emissivity--Texture Decomposition},
  author = {Dai, Cheng and Lin, Jiale and Xu, Hongyi and Song, Bingxuan and Xie, Ziyang and Bao, Fanglin},
  year = {2026},
  archivePrefix = {arXiv},
  eprint = {2606.03806},
  primaryClass = {cs.CV}
}
```
