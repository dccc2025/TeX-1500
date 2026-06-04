# Model Zoo

Released TeX-UNet checkpoints are hosted on Hugging Face:

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

When the checkpoint is stored with its Hugging Face sidecar files, the CLI
automatically reads adjacent `config.json`, `inference_config.yaml`, and
`normalization.json`. Explicit CLI flags can still override those defaults.

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

This release is GPU-only. Use a CUDA-enabled PyTorch wheel and run inference
with `--device cuda` or `--device cuda:<index>`.
