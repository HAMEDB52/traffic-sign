"""Classify a traffic-sign image with the trained GTSRB model.

Usage: python predict.py IMAGE [--model best_traffic_sign_model.keras] [--top 5]
"""

import argparse

import numpy as np
from skimage import exposure, io, transform

CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at next intersection", "Priority road", "Yield", "Stop",
    "No vehicles", "Vehicles over 3.5 metric tons prohibited", "No entry",
    "General caution", "Dangerous curve to the left", "Dangerous curve to the right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]


def preprocess(img: np.ndarray) -> np.ndarray:
    """Apply exactly the notebook's training preprocessing; returns a (1, 32, 32, 1) batch."""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if img.shape != (32, 32, 3):
        img = transform.resize(img, (32, 32, 3), mode="constant", anti_aliasing=True)
    img = img.astype(np.float32)
    gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    gray = exposure.equalize_adapthist((gray / 255.0).astype(np.float32))
    return gray.reshape(1, 32, 32, 1).astype(np.float32)


def predict(model, img: np.ndarray, top: int = 5) -> list[tuple[str, float]]:
    probs = model.predict(preprocess(img), verbose=0)[0]
    return [(CLASS_NAMES[i], float(probs[i])) for i in np.argsort(probs)[::-1][:top]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image")
    parser.add_argument("--model", default="best_traffic_sign_model.keras")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()

    import keras  # imported here so --help works without TensorFlow

    model = keras.models.load_model(args.model)
    for rank, (name, p) in enumerate(predict(model, io.imread(args.image), args.top), 1):
        print(f"{rank}. {name:55} {p:6.1%}")


if __name__ == "__main__":
    main()
