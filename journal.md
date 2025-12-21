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

## 2025-12-14 (Day 9)
Learned:
- the local feature pipeline: detect keypoints and compute descriptors (ORB in practice), then match descriptors efficiently using Hamming distance.
- how to filter noisy matches (e.g., KNN + ratio test) and why raw matches often contain many outliers.
- what a homography matrix (3×3) represents and when it is valid (primarily planar scenes like posters/books or near-pure camera rotation).
- why and how RANSAC is used with findHomography to reject outliers, produce an inlier mask, and enable reliable warpPerspective alignment/overlay.

The process of passing are in the file:
`09_Feature_detection_matching_Homography.ipynb` (notebooks)

## 2025-12-15 (Day 10)
- Learned to view optical flow as estimating 2D displacement vectors (u,v) between consecutive frames, typically for a sparse set of tracked feature points.
- Practiced the practical PyrLK pipeline in OpenCV: detect good corners (goodFeaturesToTrack) → track with calcOpticalFlowPyrLK → filter by st/err and forward–backward consistency → visualize and periodically re-detect points.
- Understood the core of Lucas–Kanade as a local least-squares solve over a window, where trackability depends on the conditioning of G=ATA (corners good; edges/flat regions problematic due to the aperture problem).
- Connected key parameters to behavior: maxLevel helps with larger motion (pyramids), winSize trades noise-robustness vs local-motion fidelity, and regularization / eigenvalue thresholds improve stability on weak features.

The process of passing are in the files:
`10_1_Lucas-Kanade_by_hand_NumPy.ipynb` (notebooks)
`10_2_Lucas-Kanade_PyrLK.ipynb` (notebooks)

## 2025-12-16 (Day 11)
- Learned Dense optical flow (Farnebäck) estimates - a full per-pixel motion field between consecutive frames (unlike LK, which tracks selected points), using a coarse-to-fine pyramid to handle larger displacements.
- Learned to interpret dense flow via HSV (hue = direction, value = magnitude) and arrow grids, and to treat “sparkly” flow—especially on low-texture regions—as a sign of low reliability.
- Practiced practical tuning for CPU real-time: downscaling first, then adjusting pyramid levels for fast motion and winsize/poly_sigma for stability vs boundary detail.
- Covered key limitations and robustness checks: brightness-constancy violations (lighting flicker), textureless surfaces, occlusions/disocclusions, depth boundaries, and the value of gradient masking and forward–backward consistency for reliability.

The process of passing are in the files:
`11_1_Dense_optical_flow_Franeback_Method.ipynb` (notebooks)
`11_2_Dense_optical_flow_Franeback_Method.ipynb` (notebooks)

## 2025-12-17 (Day 12)
- Learned the pinhole camera model basics and how a 3D world point projects to pixels via x∼K[R|T]X, separating intrinsics (𝐾) from extrinsics (R,t).
- Understood epipolar geometry basics and the Fundamental matrix basics: correspondences must satisfy x′TFx=0, and F can be estimated from pixel matches even when intrinsics are unknown.
- Understood the Essential matrix (E) basics as the calibrated version of epipolar geometry (E=K′TFK) basics, enabling recovery of R,t up to scale when K is known/approximated.
- Clarified when homography H applies (planar scene or pure rotation) and how parallax reveals true 3D structure where a single H fails.

The process of passing are in the files:
`12_1_fundamental_epipolar.ipynb` (notebooks)
`12_2_homography_warp.ipynb` (notebooks) 
`12_3_essential_pose.ipynb` (notebooks)

## 2025-12-18 (Day 13)
Learned:
- the core idea of monocular visual odometry: estimate frame-to-frame camera motion (R,t) from 2D point correspondences, with translation recoverable only up to scale.
- the high-level VO pipeline for a webcam: acquire frames → detect/track points (GFTT+KLT or ORB matching) → use RANSAC to reject outliers while estimating the Essential matrix → recoverPose to get  R and t.
- what makes VO stable or unstable: good parallax and well-distributed points help; pure rotation, planar scenes, motion blur, and moving objects degrade R,t reliability (watch inlier count/ratio).
- how to handle missing intrinsics: either calibrate once with a checkerboard to get K+distortion (best), or use an approximate K for a toy demo, expecting more drift/jitter.
- how to calibrate a camera.

The process of passing are in the files:
`13_1_VO_toy_example.ipynb` (notebooks)
`13_2_camera_calibration.ipynb` (notebooks)

## 2025-12-19 (Day 14)
- Implemented dense optical flow (Farnebäck) to estimate per-pixel motion between consecutive video frames and visualized it via HSV direction/magnitude mapping and an arrow grid overlay.
- Bbuilt a robust global motion estimate by masking unreliable vectors and aggregating flow with the median, then smoothing it over time to get a stable direction and speed signal.
- Added sparse Lucas–Kanade tracking (corners + trails) to understand feature-level motion and to compare sparse vs dense flow behavior.
- Learned to interpret flow patterns for camera motion: translation tends to produce a coherent average flow (often opposite camera direction), while rotation can create mixed directions that cancel in a global dx.

The result of this day studying is presented in the next folder: src/weekly_projects/week_2)

## 2025-12-21 (Day 15)
- Learned how RNNs model sequences by carrying a hidden state (memory) across time steps, unlike feedforward networks that process inputs independently.
- Understood why vanilla RNNs struggle with long-term dependencies (mainly vanishing/exploding gradients) and why training uses Backpropagation Through Time (BPTT).
- Learned the core intuition of LSTM and GRU gating: they learn when to keep, forget, and update information, which makes long-range learning more stable.
- Practiced key Keras sequence concepts: many-to-one vs many-to-many and how return_sequences controls whether a recurrent layer outputs one final state or the full sequence.

The process of passing are in the file:
`15_SimpleRNN_vs_LSTM_vs_GRU.ipynb` (notebooks)