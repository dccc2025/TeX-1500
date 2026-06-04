# Release Checklist

Use this checklist before pushing the GitHub release.

## GitHub Contents

The GitHub repository should contain code, docs, configs, and one lightweight
architecture image only.

```bash
git ls-files assets
```

Expected:

```text
assets/tex_unet_architecture.jpg
```

These paths must remain ignored and untracked:

```bash
git check-ignore data checkpoints outputs
git status --ignored --short
```

## Hugging Face Model Files

Download one released model variant into the ignored local checkpoint directory:

```bash
python scripts/download_weights.py --variant tex_unet_v2_darpa
```

Expected local files:

```text
checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/model.safetensors
checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/config.json
checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/inference_config.yaml
checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/normalization.json
checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/metrics.json
```

## GPU Runtime

Install a CUDA-enabled torch wheel and verify CUDA visibility:

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
assert torch.cuda.is_available()
print(torch.cuda.get_device_name(0))
PY
```

The package must reject CPU inference:

```bash
tex1500-infer --help
python - <<'PY'
from tex1500 import require_cuda_device
try:
    require_cuda_device("cpu")
except ValueError as exc:
    print(exc)
else:
    raise SystemExit("CPU device was not rejected")
PY
```

## Single-HSI Inference

After accepting the gated dataset terms on Hugging Face, download the preview
sample and run:

```bash
hf download jialelin2007/TeX-1500 \
  data/sample_0001/hsi.mat \
  --repo-type dataset \
  --local-dir data/hf/TeX-1500

tex1500-infer \
  --input data/hf/TeX-1500/data/sample_0001/hsi.mat \
  --checkpoint checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/model.safetensors \
  --model-config checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/config.json \
  --normalization-config checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/normalization.json \
  --config checkpoints/hf/TeX-UNet/tex_unet_v2_darpa/inference_config.yaml \
  --output-dir outputs/sample_0001_darpa \
  --device cuda:0
```

Expected prediction files:

```text
prediction.mat
T.png
emissivity_midband.png
texture.png
```
