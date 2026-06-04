# Examples

This release supports CUDA inference only. Install a CUDA-enabled torch wheel
before running any example.

Download the current Hugging Face preview sample:

```bash
hf download jialelin2007/TeX-1500 \
  data/sample_0001/hsi.mat \
  --repo-type dataset \
  --local-dir data/hf/TeX-1500
```

Download the DARPA checkpoint and metadata:

```bash
python scripts/download_weights.py --variant tex_unet_v2_darpa
```

Run inference on one HSI scene:

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

For custom `.mat` files, pass explicit keys when needed:

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
