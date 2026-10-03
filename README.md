# Traffic Sign Recognition (GTSRB)

A multi-scale convolutional network that classifies the 43 sign classes of the [German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/gtsrb_news.html) (GTSRB), plus a command-line tool to classify new images with the trained model.

## Result

**98.90% accuracy on 2,630 held-out test images** — 3,972,139 parameters, 32×32 grayscale input.

How the data was split (see the notebook): the 39,209 GTSRB training images are split 75 / 25 into training and validation; of the 12,630 official test images, 5,000 are added to training, 5,000 to validation, and the remaining 2,630 are kept for testing. Because part of the official test set was used for training, this figure is not directly comparable with results reported on the full GTSRB test set.

## Model

- **Preprocessing:** resize to 32×32 → grayscale (Y channel) → contrast-limited adaptive histogram equalisation
- **Architecture:** three convolutional blocks (32, 64, 128 filters, 5×5) with max-pooling and dropout; features from all three blocks are pooled to a common size and concatenated (multi-scale), then classified by fully connected layers
- **Training:** Adam (learning rate 1e-3), batch size 128, up to 20 epochs with early stopping and learning-rate reduction on plateau; on-the-fly augmentation (rotation ±15°, shifts, zoom, shear)

## Files

| File | Purpose |
|---|---|
| `traffic sign.ipynb` | Full pipeline: download, preprocessing, training, evaluation, filter and feature-map visualisation, inference |
| `best_traffic_sign_model.keras` | Trained model (Keras 3 format) |
| `predict.py` | Classify an image from the command line |
| `tests/` | Tests for preprocessing and the saved model |

## Classify an image

```bash
pip install -r requirements.txt
python predict.py path/to/sign.jpg
```

```
1. No vehicles                                             100.0%
2. No passing                                                0.0%
...
```

`predict.py` applies exactly the same preprocessing as training, so the model sees images the way it was trained on them.

## Train from scratch

Open `traffic sign.ipynb` in Google Colab (or Jupyter with TensorFlow 2.x) and run all cells. The notebook downloads GTSRB itself.

## Tests

```bash
python -m pytest
```

Checks that preprocessing returns a correctly shaped and scaled tensor for several input formats, and that the saved model loads and returns a ranked 43-class probability distribution.
