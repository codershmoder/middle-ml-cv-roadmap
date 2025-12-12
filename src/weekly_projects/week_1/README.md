## Mini-project №1: End-to-end Image Classifier (Horses vs Humans)

### Overview

This mini-project implements an end-to-end image classification pipeline that distinguishes between two classes: **horses** and **humans**. It covers:

* dataset loading from a directory structure,
* training a CNN classifier,
* saving the best-performing model,
* running inference on an image from the Internet via a simple script.

The training workflow is implemented in the notebook:
`07_miniproject_01_horses_vs_humans_classification.ipynb`

---

## Project structure (recommended)

```
project/
  07_miniproject_01_horses_vs_humans_classification.ipynb
  07_hm_classification_inference.py
  07_hm_classification_model.keras
  README.md  
```

---

## Requirements

* Python 3.10+ (3.9+ should work in most cases)
* Packages:

  * tensorflow
  * numpy
  * pillow
  * requests

Install dependencies:

```bash
pip install tensorflow numpy pillow requests
```

---

## Dataset format

Dataset source:
https://www.kaggle.com/datasets/sanikamal/horses-or-humans-dataset


Important: the label mapping is based on alphabetical folder order used by Keras:

* `horses` → class **0**
* `humans` → class **1**

---

## Training pipeline (high level)

### 1) Data loading

The notebook uses `tf.keras.utils.image_dataset_from_directory(...)` to:

* read images from disk,
* resize them to a fixed shape,
* create labeled batches for training/validation.

### 2) Preprocessing and augmentation

During training, images are resized to:

* **150×150**, RGB (`(150,150,3)`)

The inference script replicates the same resizing behavior.

Important note on scaling: the trained model expects image tensors in the **same scale as training**, i.e. float32 pixel values in approximately **[0..255]** (no division by 255 and no `tf.keras.applications.*.preprocess_input`).

### 3) Model

The classifier is a binary CNN with the final layer:

* `Dense(1, activation="sigmoid")`

This means the model outputs:

* `p(humans)` as a single probability value in `[0, 1]`.

### 4) Saving the model

The best model is saved during training via `ModelCheckpoint` to:

* `07_hm_classification_model.keras`

---

## Inference (image URL → class + probability)

The `inference.py` script:

1. asks the user to paste an image URL,
2. downloads the image,
3. resizes it to **150×150**,
4. runs inference,
5. prints the predicted class and probabilities.

Run:

```bash
python 07_hm_classification_inference.py
```

Example interaction:

```text
Enter image URL: https://example.com/image.jpg
class=humans prob=0.934210 p_humans=0.934210 p_horses=0.065790
```

Interpretation:

* `p_humans` is the sigmoid output.
* `p_horses = 1 - p_humans`.
* The script predicts `humans` if `p_humans >= 0.5`, otherwise `horses`.

You can change the threshold:

```bash
python inference.py --threshold 0.6
```

---

## Notes / limitations

* The script expects the URL to return an actual image (Content-Type should contain `image/...`).
* This project is a minimal demonstration of an end-to-end pipeline; further improvements typically include:

  * more aggressive augmentation,
  * transfer learning (e.g., MobileNet/EfficientNet),
  * better validation (stratification, cross-validation, test set),
  * model calibration and threshold tuning depending on business requirements.

---

## Deliverables

* Notebook with training pipeline:

  * `07_miniproject_01_horses_vs_humans_classification.ipynb`
* Saved trained model:

  * `07_hm_classification_model.keras`
* Inference script:

  * `07_hm_classification_inference.py`
* This documentation:

  * `README.md`
