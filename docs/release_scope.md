# Release Scope

This GitHub repository is an inference-only code release.

Included:

- TeX-UNet model architecture.
- Full-scene and tiled variable-band inference.
- Input loaders for `.mat`, `.npy`, and `.npz`.
- Output writers for NumPy, MATLAB, and PNG previews.
- The TeX-UNet architecture figure.
- Runtime checks that enforce CUDA-only inference.

Hosted externally:

- Dataset: `jialelin2007/TeX-1500`
- Model weights: `dccc2025/TeX-UNet`

Excluded:

- Training pipeline.
- Optimizer states and experiment logs.
- Large dataset files and checkpoints.
- Extra paper figures beyond the architecture figure.

<<<<<<< HEAD
The TeX-1500 dataset and TeX-UNet checkpoints are hosted separately on Hugging
Face and downloaded into `checkpoints/` when needed.
=======
The only tracked image asset should be:

```text
assets/tex_unet_architecture.jpg
```

Local `data/`, `checkpoints/`, and `outputs/` directories are ignored by Git and
are intended only for downloaded Hugging Face files and generated predictions.
>>>>>>> d26300b (final version)
