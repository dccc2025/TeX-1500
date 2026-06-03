# Examples

After downloading one TeX-1500 HSI file and a released checkpoint:

```bash
uv run tex1500-infer \
  --input data/example/hsi.mat \
  --checkpoint checkpoints/tex_unet_v2.pt \
  --model-config configs/tex_unet_v2_model.json \
  --output-dir outputs/example \
  --config configs/inference.yaml
```

Use `--hsi-key`, `--wavelength-key`, and `--good-band-key` if the file uses
non-standard MATLAB or NumPy keys.
