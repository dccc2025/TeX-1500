# Model Zoo

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
