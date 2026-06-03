# Release Scope

This repository is an inference-only release of the TeX-UNet baseline.

Included:

- TeX-UNet model architecture.
- Full-scene variable-band inference.
- HSI input loading for `.mat`, `.npy`, and `.npz`.
- Output writers for NumPy, MATLAB, and PNG previews.
- Selected paper figures and benchmark tables.

Excluded:

- Training pipeline.
- Raw/private data.
- Optimizer states and experiment logs.
- Large checkpoints.

The TeX-1500 dataset is hosted on Hugging Face. Model checkpoints should also be
hosted externally and downloaded into `checkpoints/` when needed.
