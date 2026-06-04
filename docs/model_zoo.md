# Model Zoo

<<<<<<< HEAD
Checkpoints are hosted on [Hugging Face](https://huggingface.co/dccc2025/TeX-UNet)
rather than inside this GitHub repository.

| Model | Input bands | Scope | Checkpoint |
|---|---:|---|---|
| TeX-UNet v2 DARPA IH | 64 sampled valid bands | DARPA IH test baseline | `tex_unet_v2_darpa/model.safetensors` |
| TeX-UNet v2 FTIR few-shot | 64 sampled valid bands | FTIR few-shot transfer | `tex_unet_v2_ftir_fewshot/model.safetensors` |

Released Hugging Face layout:

```text
dccc2025/TeX-UNet
  tex_unet_v2_darpa/
    model.safetensors
    config.json
    metrics.json
  tex_unet_v2_ftir_fewshot/
    model.safetensors
    config.json
    metrics.json
```
=======
Released checkpoints are hosted at:

```text
https://huggingface.co/dccc2025/TeX-UNet
```

| HF directory | File | Scope |
|---|---|---|
| `tex_unet_v2_darpa` | `model.safetensors` | DARPA IH training split; DARPA test and FTIR zero-shot baseline. |
| `tex_unet_v2_ftir_fewshot` | `model.safetensors` | FTIR few-shot fine-tuning initialized from the DARPA checkpoint. |

Each directory also includes:

```text
config.json
normalization.json
inference_config.yaml
metrics.json
```

This GitHub code supports both local `configs/tex_unet_v2_model.json` and the
HF model repo `config.json` format. When the checkpoint is stored with its HF
sidecar files, the CLI automatically reads adjacent `config.json`,
`inference_config.yaml`, and `normalization.json`; explicit CLI flags can still
override those defaults.

Download a full local copy of one model variant with:

```bash
python scripts/download_weights.py --variant tex_unet_v2_darpa
```

The local directory mirrors the Hugging Face model repository:

```text
checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/
  model.safetensors
  config.json
  inference_config.yaml
  normalization.json
  metrics.json
```

The release is GPU-only. Use a CUDA-enabled PyTorch wheel and run inference with
`--device cuda` or `--device cuda:<index>`.
>>>>>>> d26300b (final version)
