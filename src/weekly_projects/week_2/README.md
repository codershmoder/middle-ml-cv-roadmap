**Goal**: estimate dominant camera motion direction and approximate speed from an MP4 using optical flow.

**Method**: compute dense optical flow (Farneback) between consecutive frames; robustly estimate a global motion vector via the median flow after masking tiny magnitudes/outliers; smooth over a short window.

**Visualization**:

1. HSV flow wheel (direction→hue, magnitude→value);
2. arrow grid overlay;
3. Sparse Lucas–Kanade corner tracks with trails.

**Decision rule***: if the smoothed horizontal component dominates and speed exceeds a threshold, classify as left→right vs right→left (optical flow direction), and optionally infer camera translation as the opposite direction.