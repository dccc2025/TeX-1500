# Model Zoo

Checkpoints are intended to live on Hugging Face rather than inside this GitHub
repository.

| Model | Input bands | Scope | Checkpoint |
|---|---:|---|---|
| TeX-UNet v2 DARPA IH | 64 sampled valid bands | DARPA IH test baseline | Coming soon |
| TeX-UNet v2 FTIR few-shot | 64 sampled valid bands | FTIR few-shot transfer | Coming soon |

Expected Hugging Face layout:

```text
dccc2025/TeX-1500-baselines
  tex_unet_v2/
    final.pt
    config.json
    metrics.json
```

After the checkpoint repository is published, update the README and this file
with exact filenames, hashes, and release notes.
