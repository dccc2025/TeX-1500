# Benchmark

The initial benchmark reports TeX-UNet inversion results on held-out DARPA IH
pushbroom scenes and FTIR zero-/few-shot transfer scenes.

| Test split | Checkpoint | T MAE (K) | T MAPE (%) | e MSE | e SAM (rad) | X MSE | X SAM (rad) |
|---|---|---:|---:|---:|---:|---:|---:|
| DARPA IH-test | `tex_unet_v2_darpa` | 7.3284 | 2.5488 | 0.0453 | 0.2267 | 0.0311 | 0.5206 |
| FTIR-zeroshot-test | `tex_unet_v2_darpa` | 5.8309 | 1.9753 | 0.0674 | 0.0451 | 0.0219 | 0.2995 |
| FTIR-fewshot-test | `tex_unet_v2_ftir_fewshot` | 4.1004 | 1.3830 | 0.0458 | 0.1970 | 0.0220 | 0.2224 |

`e` and `X` are normalized. Full training settings and evaluation details are
described in the paper and the Hugging Face model metadata files.
