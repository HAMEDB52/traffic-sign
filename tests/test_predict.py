from pathlib import Path

import numpy as np
import pytest

from predict import CLASS_NAMES, predict, preprocess

MODEL = Path(__file__).resolve().parents[1] / "best_traffic_sign_model.keras"


def test_43_classes():
    assert len(CLASS_NAMES) == 43 and len(set(CLASS_NAMES)) == 43


@pytest.mark.parametrize("shape", [(50, 60, 3), (32, 32, 3), (40, 40), (64, 64, 4)])
def test_preprocess_shape_and_range(shape):
    img = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    x = preprocess(img)
    assert x.shape == (1, 32, 32, 1) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0


@pytest.mark.skipif(not MODEL.exists(), reason="model file not present")
def test_model_predicts_probabilities():
    keras = pytest.importorskip("keras")
    model = keras.models.load_model(MODEL)
    assert model.input_shape == (None, 32, 32, 1) and model.output_shape == (None, 43)
    img = np.random.default_rng(1).integers(0, 256, (48, 48, 3), dtype=np.uint8)
    top = predict(model, img, top=5)
    assert len(top) == 5 and all(name in CLASS_NAMES for name, _ in top)
    assert all(0 <= p <= 1 for _, p in top) and top == sorted(top, key=lambda t: -t[1])
