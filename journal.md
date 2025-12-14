# Learning Journal

## 2025-12-05 (Day 0)
- Created repo and project structure for CV roadmap.
- Set up Conda environment `cv-ml` with core libraries.
- Verified imports in a test notebook.

## 2025-12-06 (Day 1)
- Refresh knowledge about arrays, shapes and dimensions, as well as arrays indexing and slicing.
- Understand image formats (RGB/BGR, grayscale), dtype, image array normalization.
- Refresh knowledge about base image operation in OpenCV (imread, imshow), as well as with convertion from BGR to plot.
- The process of passing is in the file `01_numpy_scipy_basics.ipynb`. (notebooks)

## 2025-12-07 (Day 2)
Understood: 
- how to use box blur, Gaussian blur, and median blur in OpenCV and how each affects noise and edges differently.
- the idea of image gradients and how Sobel filters, Canny edge detector work (blur → gradient → thinning → thresholds) and how its thresholds influence the result.
- the difference between simple global thresholding, Otsu’s automatic threshold, and adaptive thresholding.
- how erosion, dilation, opening, and closing modify the shapes: removing small noise, thinning or thickening objects, filling small holes, and connecting nearby components.

The process of passing are in the files:
- `02_1_OpenCV_basic_image_processing.ipynb` (notebooks)
- `02_2_morphology_pipeline.ipynb` (notebooks)

## 2025-12-08 (Day 3)
Refresh knowledge about: 
- the roles of TensorFlow (low-level tensors, computation, GPU support) and Keras (high-level tf.keras API with layers, models, compile/fit/evaluate/predict).
- the difference between Sequential and Functional APIs.
- how to create a proper train/validation split (e.g., with train_test_split or validation_split), why validation data is needed, and how it’s used to monitor generalization and detect overfitting.
- how model.fit works.
- callbacks and EarlyStopping.

The process of passing are in the files:
`03_TensorFlow_Keras_quick_reminder.ipynb` (notebooks)
`03_small_sequential_model.ipynb` (src/deep_learning/models)

## 2025-12-09 (Day 4)
Understood: 
- what convolutions, kernels, stride, padding, and pooling do and how they change feature map sizes.
- what a receptive field is and how it grows as we go deeper in a CNN.
- the difference between kernel size and number of filters, and why conv layers are more efficient than dense layers for images.
- a basic picture of classic CNN architectures (LeNet, AlexNet, VGG, ResNet).

The result of this day studying is presented as a small CNN model in the next file:
`04_small_CNN.ipynb` (src/deep_learning/models)

## 2025-12-10 (Day 5)
Refresh knowledge about: 
- Overfitting vs underfitting.
- L2-regularization (weight decay).
- Dropout.
- Data augmentation for images (flips, rotations, crops/zoom and translations).

The result of this day studying is presented as a small CNN in the next files:
`05_cat_vs_dog_base_model.ipynb` (src/deep_learning/models)
`05_cat_vs_dog_model_with_augmentation.ipynb` (src/deep_learning/models)

## 2025-12-11 (Day 6)
- Understood the idea of transfer learning: using a pretrained CNN (e.g., ResNet, EfficientNet) as a backbone and adding my own classifier head instead of training a network from scratch.
- Learned the difference between feature extraction (frozen backbone + new head) and fine-tuning (unfreezing upper layers and updating some pretrained weights with a small learning rate).
- Saw how to correctly build Keras models with include_top=False, the right preprocess_input function, and appropriate loss/activation for multi-class problems.

The result of this day studying is presented as a small CNN in the next file:
`06_cat_vs_dog_model_transfer_learning.ipynb` (src/deep_learning/models)

## 2025-12-12 (Day 7)
- Built an end-to-end image classification pipeline: dataset loading/organization, train–validation split, and a consistent preprocessing flow (resize to a fixed input shape).
- Trained a binary CNN classifier (sigmoid output) on InceptionV3 backbone for horses vs humans, monitored validation metrics, and saved the best checkpoint as a reusable artifact (.keras).
- Implemented a standalone py script that downloads an image from a user-provided URL, applies the same preprocessing as training, and outputs the predicted class with probabilities.
- Documented the project in a concise README describing structure, requirements, training approach, saved artifacts, and inference usage.

The result of this day studying is presented in the next folder: src/weekly_projects/week_1)

## 2025-12-13 (Day 8)
- Learned how to choose between global thresholding (including Otsu) and adaptive thresholding based on whether lighting is uniform or uneven, and how threshold polarity (THRESH_BINARY vs THRESH_BINARY_INV) affects the mask.
- Understood how morphological operations (opening/closing, plus erosion/dilation via kernel + iterations) are used to clean binary masks by removing small noise and filling holes before segmentation.
- Learned the practical use of cv2.findContours (including why to pass mask.copy()), the difference between retrieval modes (RETR_EXTERNAL vs RETR_TREE), and how to filter contours by area to suppress noise.
- Practiced extracting shape features from contours (bounding box, polygon approximation with approxPolyDP, and circularity) to support simple “general-shape” classification (triangle/rectangle/circle-like).

The process of passing are in the file:
`08_ClassicCV_contours_segmentation_morphology.ipynb` (notebooks)