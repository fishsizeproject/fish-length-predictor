# Fish Length Predictor

#### Lightweight pipeline to predict fish lengths from images using DINOv2 visual embeddings and an AutoML XGBoost regressor.

> ✅ This repository provides an installable CLI and Python API for inference with pre-trained artifacts (which are **included**).

> ✅ A training CLI and Python API are also available. Train your own AutoML XGBoost regressor for custom images, save the artifacts, and evaluate the model (R², MAE, MSE).


## Quickstart

After creating and activating an environment with Python 3.11+:

```bash
# 1) Install the package
pip install .

# 2) Run prediction on one or more images
fish-predict path/to/image.jpg
fish-predict 'path/to/images/*.jpg' --out predictions.csv

# 3) Train new artifacts with custom images
fish-train path/to/data.csv
```

### Inference Output
For batch mode (glob pattern), a CSV with columns: `file,pred_length` is written to `--out` (default: `predictions.csv`). For single-image mode, the predicted length is printed to stdout.


## Installation Notes

- For **PyTorch** install the correct CUDA-enabled build if you have a GPU, otherwise CPU-only is fine.
- **Model artifacts** (`*.pkl`) are required for inference and **included** in this repository. You can also train your own artifacts.
- The DINOv2 backbone is loaded via `torch.hub` from `facebookresearch/dinov2` (weights will be fetched automatically on the first run).
- Training relies on **FLAML (AutoML)**, **scikit-learn** scalers, and **XGBoost** as the estimator.

---

## CLI Usage (Inference)

```bash
fish-predict <image_or_glob> \
    [--out predictions.csv] \
    [--device auto]
```
- `image_or_glob`: a single image path or a shell glob for multiple images.
- `--out`: (batch only) output CSV path. Defaults to `predictions.csv`.
- `--device`: `"auto"`, `"cpu"`, `"gpu"`, or `"cuda"`. Defaults to `"auto"`.

Examples:
```bash
fish-predict sample.jpg --device cpu
fish-predict "images/*.png" --out preds.csv
```

---

## CLI Usage (Training)

```bash
fish-train <csv> \
  [--image-col image] \
  [--target-col length] \
  [--test-size 0.2] \
  [--random-state None] \
  [--time-budget 60] \
  [--artifacts custom-artifacts] \
  [--device auto]
```
- `csv`: path to a CSV file with at least two columns (image paths and target lengths).
- `--image-col`: name of the image-path column. Defaults to `image`.
- `--target-col`: name of the length target column. Defaults to `length`.
- `--test-size`: test split fraction. Defaults to `0.2`.
- `--random-state`: random seed for reproducibility of the train/test split. Defaults to `None`.
- `--time-budget`: AutoML time budget in seconds. Defaults to `60`.
- `--artifacts`: directory to write trained artifacts. Defaults to `custom-artifacts`.
- `--device`: `"auto"`, `"cpu"`, `"gpu"`, or `"cuda"`. Defaults to `"auto"`.

Examples:
```bash
fish-train data.csv --device gpu
fish-train images/data.csv --time-budget 3600
```

---

## Disclaimer

This project is provided as-is, without warranty. Ensure you have the right to use the model artifacts.
