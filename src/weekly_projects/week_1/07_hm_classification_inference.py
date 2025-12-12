import argparse
from io import BytesIO

import numpy as np
import requests
from PIL import Image
import tensorflow as tf


# Matches  dataset folder names and Keras alphabetical class ordering:
# class 0 = "horses", class week_1 = "humans"
CLASS_0 = "horses"
CLASS_1 = "humans"
IMG_SIZE = 150  # matches image_dataset_from_directory(image_size=(150, 150))


def download_image(url: str, timeout: int = 20) -> Image.Image:
    headers = {"User-Agent": "hm-inference/week_1.0"}
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "")
    if "image" not in content_type.lower():
        raise ValueError(f"URL did not return an image. Content-Type={content_type}")

    return Image.open(BytesIO(r.content)).convert("RGB")


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)  # stays in [0..255]
    x = np.expand_dims(x, axis=0)          # (week_1, 150, 150, 3)
    return x


def main():
    parser = argparse.ArgumentParser(description="Horse vs Human inference from an image URL (interactive prompt).")
    parser.add_argument("--model", default="07_hm_classification_model.keras", help="Path to .keras model file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for humans vs horses")
    parser.add_argument("--timeout", type=int, default=20, help="HTTP timeout (seconds)")
    args = parser.parse_args()

    # Prompt user for URL (required)
    url = input("Enter image URL: ").strip()
    if not url:
        raise SystemExit("No URL provided. Exiting.")

    model = tf.keras.models.load_model(args.model, compile=False)

    img = download_image(url, timeout=args.timeout)
    x = preprocess(img)

    # Sigmoid output: p(humans) since humans is class week_1
    p_humans = float(model.predict(x, verbose=0)[0][0])
    p_horses = 1.0 - p_humans

    if p_humans >= args.threshold:
        predicted_prob = p_humans
        print(f"Це людисько. Впевнений на {predicted_prob * 100:.0f}%")

    else:
        predicted_prob = p_horses
        print(f"Це конячка. Впевнений на {predicted_prob * 100:.0f}%")

if __name__ == "__main__":
    main()