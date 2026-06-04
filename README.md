# TeX-1500

**GPU-only inference code for the TeX-1500 TeX-UNet baseline.**

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2606.03806-b31b1b)](https://arxiv.org/abs/2606.03806)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-ffcc4d)](https://huggingface.co/datasets/jialelin2007/TeX-1500)
<<<<<<< HEAD
[![Checkpoints](https://img.shields.io/badge/Checkpoints-Hugging%20Face-ffcc4d)](https://huggingface.co/dccc2025/TeX-UNet)
[![Code](https://img.shields.io/badge/Code-GitHub-111111)](https://github.com/dccc2025/TeX-1500)
[![Python](https://img.shields.io/badge/Python-3.10--3.12-3776ab)](https://www.python.org/)


TeX-1500 is a paired LWIR hyperspectral benchmark for HADAR-oriented recovery of
temperature `T`, emissivity `e`, and scalar texture `X`. The dataset contains
1,522 calibrated real-scene HSI--TeX pairs from DARPA Invisible Headlights
pushbroom imagery and FTIR acquisitions, covering multiple locations, seasons,
acquisition times, wavelength layouts, and sensor families.

This repository provides the inference-only TeX-UNet baseline: model
architecture, full-scene variable-band inference, output writers, and a compact
command line interface. Training code, raw data, and checkpoints are available in other repositories.

## Paper

**TeX-1500: A Paired Real-World LWIR Hyperspectral Dataset and Benchmark for Temperature--Emissivity--Texture Decomposition**

[Cheng Dai](https://github.com/dccc2025)\*, [Jiale Lin](https://github.com/jialelin2007)\*, Hongyi Xu, Bingxuan Song, Ziyang Xie, and Fanglin Bao

`*` Equal Contribution. Corresponding author: Fanglin Bao.

## Dataset

| Split | Location | Scenes | Images | Wavelengths | Spatial size | Bands |
|---|---:|---:|---:|---|---|---:|
| DARPA IH train | TPG, AZ / ME / FL | 74 | 1,096 | 6.8--13.2 um | 260x1200 to 480x1700 | 250/256 |
| DARPA IH valid | Sidewinder Range, TPG, AZ | 8 | 51 | 8.1--13.2 um | 260x1280 | 256 |
| DARPA IH test | Fort A. P. Hill, VA | 18 | 233 | 8.1--13.2 um | 260x1600 | 256 |
| FTIR train | Wuhan University, China | 38 | 111 | 7.9--11.5 um | 320x256 | 86 |
| FTIR test | Wuhan University, China | 16 | 31 | 7.9--11.5 um | 320x256 | 86/124/277 |

Dataset files are hosted on [Hugging Face](https://huggingface.co/datasets/jialelin2007/TeX-1500).

![Spectral coverage](assets/spectral_coverage.png)

## TeX-UNet Baseline

TeX-UNet maps calibrated HSI bands and their wavelength positions to normalized
TeX fields. During inference, it repeatedly samples 64 valid bands until each
valid band reaches the requested minimum coverage, runs the network on full
images or sliding windows, and averages the accumulated predictions.
=======
[![Model](https://img.shields.io/badge/Model-Hugging%20Face-ffcc4d)](https://huggingface.co/dccc2025/TeX-UNet)

TeX-1500 is a paired LWIR hyperspectral benchmark for temperature `T`,
emissivity `e`, and scalar texture `X` decomposition. This GitHub repository is
an inference-only code release. It contains the TeX-UNet architecture, HSI
loaders, tiled/full-scene inference, output writers, and the architecture figure
below. Dataset files and pretrained weights are hosted on Hugging Face:

- Dataset: https://huggingface.co/datasets/jialelin2007/TeX-1500
- Model weights: https://huggingface.co/dccc2025/TeX-UNet
- Paper: https://arxiv.org/abs/2606.03806
>>>>>>> d26300b (final version)

![TeX-UNet architecture](assets/tex_unet_architecture.jpg)

## Release Boundary

This repository intentionally does not store training code, raw data,
checkpoints, generated outputs, optimizer states, experiment logs, or extra
paper figures. The only tracked image asset is:

```text
assets/tex_unet_architecture.jpg
```

Large local folders such as `data/`, `checkpoints/`, and `outputs/` are ignored
by Git. Keep downloaded Hugging Face files there for local inference only.

## GPU-Only Rule

This release supports CUDA inference only. CPU inference is not a supported
runtime path. The CLI and Python API will fail if `--device cpu` is used or if
`torch.cuda.is_available()` is false.

Install a CUDA-enabled PyTorch build before installing the package. The locked
release uses torch 2.7.1; the command below uses the official CUDA 12.8 wheel
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
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT AVAILABLE")
if not torch.cuda.is_available():
    raise SystemExit("CUDA-enabled torch is required.")
PY
```

If CUDA 12.8 is not appropriate for your machine, use the official PyTorch CUDA
index that matches your driver. Do not install from the CPU wheel index.

## Download Hugging Face Files

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

Download the DARPA TeX-UNet checkpoint and its metadata:

<<<<<<< HEAD
Released TeX-UNet checkpoints are hosted on [Hugging Face](https://huggingface.co/dccc2025/TeX-UNet).
Place downloaded weights under `checkpoints/`, for example:

```text
checkpoints/tex_unet_v2_darpa.safetensors
```

The download helper is wired for the released Hugging Face layout:

```bash
uv run python scripts/download_weights.py \
  --repo-id dccc2025/TeX-UNet \
  --filename tex_unet_v2_darpa/model.safetensors \
  --output checkpoints/tex_unet_v2_darpa.safetensors
=======
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
>>>>>>> d26300b (final version)
```

Download the current public dataset preview sample:

```bash
<<<<<<< HEAD
uv run tex1500-infer \
  --input path/to/hsi.mat \
  --checkpoint checkpoints/tex_unet_v2_darpa.safetensors \
  --model-config configs/tex_unet_v2_model.json \
  --output-dir outputs/example \
  --num-bands 64 \
  --min-band-coverage 5
=======
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
>>>>>>> d26300b (final version)
```

As of this release, the Hugging Face dataset page describes `sample_0001` as a
preview sample and states that the full 1,522-sample dataset will be made
available later. The inference loader is designed to keep working as future
samples follow the documented `.mat` layout.

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

The default model samples 64 valid bands per pass. The input must therefore
contain at least 64 valid bands after `good_band_indices` filtering.

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

<<<<<<< HEAD
![DARPA IH results](assets/tex_unet_darpa_results.jpg)

![FTIR transfer results](assets/tex_unet_ftir_transfer.jpg)
=======
`e` and `X` are normalized. See the paper and Hugging Face model files for the
full evaluation protocol.
>>>>>>> d26300b (final version)

## Citation

```bibtex
<<<<<<< HEAD
@misc{dai2026tex1500pairedrealworldlwir,
      title={TeX-1500: A Paired Real-World LWIR Hyperspectral Dataset and Benchmark for Temperature-Emissivity-Texture Decomposition},
      author={Cheng Dai and Jiale Lin and Hongyi Xu and Bingxuan Song and Ziyang Xie and Fanglin Bao},
      year={2026},
      eprint={2606.03806},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2606.03806},
}
```
=======
@misc{dai2026tex1500,
  title = {TeX-1500: A Paired Real-World LWIR Hyperspectral Dataset and Benchmark for Temperature--Emissivity--Texture Decomposition},
  author = {Dai, Cheng and Lin, Jiale and Xu, Hongyi and Song, Bingxuan and Xie, Ziyang and Bao, Fanglin},
  year = {2026},
  archivePrefix = {arXiv},
  eprint = {2606.03806},
  primaryClass = {cs.CV}
}
```
>>>>>>> d26300b (final version)
