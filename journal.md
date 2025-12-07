# Learning Journal

## 2025-12-05 (Day 0)
- Created repo and project structure for CV roadmap.
- Set up Conda environment `cv-ml` with core libraries.
- Verified imports in a test notebook.

## 2025-12-06 (Day 1)
- Refresh knowledge about arrays, shapes and dimensions, as well as arrays indexing and slicing.
- Understand image formats (RGB/BGR, grayscale), dtype, image array normalization.
- Refresh knowledge about base image operation in OpenCV (imread, imshow), as well as with convertion from BGR to plot.
- The process of passing is in the file `01_numpy_scipy_basics.ipynb`.

## 2025-12-07 (Day 2)
Understood: 
- how to use box blur, Gaussian blur, and median blur in OpenCV and how each affects noise and edges differently.
- the idea of image gradients and how Sobel filters, Canny edge detector work (blur → gradient → thinning → thresholds) and how its thresholds influence the result.
- the difference between simple global thresholding, Otsu’s automatic threshold, and adaptive thresholding.
- how erosion, dilation, opening, and closing modify the shapes: removing small noise, thinning or thickening objects, filling small holes, and connecting nearby components.
The process of passing are in the files `02_1_OpenCV_basic_image_processing.ipynb` and `02_2_morphology_pipeline.ipynb`.