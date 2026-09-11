# Computer Vision — Assignment 2

This repository contains the implementation and analysis for **Computer Vision Assignment 2**. The assignment focuses on edge detection, Hough Transform-based line and circle detection, the effect of noise on image thresholding, and Otsu's global thresholding implemented from scratch.

---

# 📁 Repository Structure

```text
Assignment2/
│
├── Q1_LoG_Hough_Line_Detection.ipynb
├── Q2_Robust_Circle_Detection_Hough_Transform.ipynb
├── Q3_Role_of_Noise_in_Image_Thresholding.ipynb
├── Q4_Otsu_Global_Thresholding_From_Scratch.ipynb
│
├── images/
│   ├── input_image1.png
│   └── input_image2.png
│
└── README.md
```

> Make sure the `images` directory is placed at the correct relative path expected by the notebooks.

---

# ⚙️ Requirements

The notebooks are implemented using **Python 3** and require the following libraries:

- **NumPy** — numerical computation and image-array manipulation
- **Matplotlib** — image visualization and plotting
- **OpenCV (`cv2`)** — image loading, grayscale conversion, Gaussian filtering, Laplacian filtering, and Canny edge detection
- **scikit-image** — Hough Transform and Canny-related functionality

---

# 📦 Installation

Install all required dependencies using:

```bash
pip install numpy matplotlib opencv-python scikit-image
```

Or, if using Conda:

```bash
conda install numpy matplotlib scikit-image
pip install opencv-python
```

---

# 📚 Imports Used

## Q1 — LoG + Hough Line Detection

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
```

## Q2 — Hough Circle Detection

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
```

## Q3 — Noise and Thresholding

```python
import numpy as np
import matplotlib.pyplot as plt
```

## Q4 — Otsu Thresholding From Scratch

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

# ▶️ Running the Notebooks

The notebooks can be executed using:

- Jupyter Notebook
- JupyterLab
- VS Code with the Jupyter extension
- Google Colab

To launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the required `.ipynb` file and execute the cells sequentially from top to bottom.

---

# Q1 — LoG + Hough Transform for Line Detection

**Notebook:**

```text
Q1_LoG_Hough_Line_Detection.ipynb
```

## Objective

The objective of this question is to detect prominent straight lines in an image using:

1. Laplacian of Gaussian (LoG)
2. Zero-crossing edge detection
3. Hough Line Transform

The effect of changing the LoG zero-crossing threshold on the detected edges and Hough lines is also investigated.

---

## Input Image

The notebook loads:

```python
IMAGE_PATH = "../images/input_image1.png"
```

If the image is stored somewhere else, update `IMAGE_PATH` accordingly.

The image is loaded using:

```python
image_bgr = cv2.imread(IMAGE_PATH)
```

and converted to RGB and grayscale representations.

---

## LoG Processing

The image is first converted to floating-point format and normalized:

```python
gray_float = gray.astype(np.float32) / 255.0
```

Gaussian smoothing is then applied:

```python
blurred = cv2.GaussianBlur(
    gray_float,
    ksize=(0, 0),
    sigmaX=GAUSSIAN_SIGMA,
    sigmaY=GAUSSIAN_SIGMA
)
```

The Laplacian is calculated using:

```python
log_response = cv2.Laplacian(
    blurred,
    cv2.CV_32F,
    ksize=3
)
```

This gives the **Laplacian of Gaussian (LoG)** response.

---

## Zero-Crossing Detection

The notebook implements a custom function:

```python
zero_crossing_edges(log_img, threshold=0.0)
```

The function detects locations where the LoG response changes sign between positive and negative values.

A magnitude threshold is additionally used to suppress weak responses.

Two thresholds are investigated:

```python
THRESHOLD_LOW = 0.0
THRESHOLD_HIGH = 0.03
```

### Interpretation

- **Low threshold:** detects more zero crossings, including weak responses.
- **High threshold:** suppresses weak responses and produces fewer, stronger edges.

---

## Hough Line Transform

The detected edge maps are passed to:

```python
hough_line()
```

and prominent lines are extracted using:

```python
hough_line_peaks()
```

The main parameters include:

```python
NUM_LINES = 20
MIN_DISTANCE = 20
MIN_ANGLE = 10
```

The Hough representation uses:

\[
x\cos(\theta) + y\sin(\theta) = \rho
\]

where:

- \(\theta\) = angle of the line
- \(\rho\) = perpendicular distance from the origin

---

## Outputs

The notebook visualizes:

- Original image
- Grayscale image
- Absolute LoG response
- LoG zero-crossing edge maps
- Hough-detected lines
- Hough accumulator spaces
- Comparison between low and high LoG thresholds

A final comparison reports:

- Number of edge pixels
- Number of detected Hough lines
- LoG threshold used

---

# Q2 — Robust Circle Detection Using Hough Transform

**Notebook:**

```text
Q2_Robust_Circle_Detection_Hough_Transform.ipynb
```

## Objective

The objective is to detect multiple circular objects using the **Hough Circle Transform**, even when the image contains:

- Different circle sizes
- Partial boundaries
- Noise

The effect of changing the radius range and detection threshold is investigated.

---

## Input Image

The notebook uses:

```python
IMAGE_PATH = "../images/input_image2.png"
```

Update the path if necessary.

---

## Preprocessing

The image is converted to grayscale and smoothed using Gaussian filtering.

```python
blurred = cv2.GaussianBlur(
    gray,
    ksize=(0, 0),
    sigmaX=GAUSSIAN_SIGMA
)
```

The Canny edge detector is then applied:

```python
edges_cv = cv2.Canny(
    blurred,
    threshold1=CANNY_LOW,
    threshold2=CANNY_HIGH
)
```

The resulting edge map is converted to a Boolean representation for use with scikit-image.

---

## Circle Detection

The Hough Circle Transform is implemented using:

```python
from skimage.transform import hough_circle, hough_circle_peaks
```

The notebook investigates multiple radius ranges:

```python
RADIUS_RANGES = [
    (10, 40),
    (20, 80),
    (40, 120)
]
```

Different peak/detection thresholds are also investigated:

```python
PEAK_THRESHOLDS = [
    0.25,
    0.40,
    0.55
]
```

---

## Important Parameters

```python
GAUSSIAN_SIGMA = 2.0

CANNY_LOW = 50
CANNY_HIGH = 150

MAX_CIRCLES = 20
MIN_CENTER_DISTANCE = 20
```

### Parameter Effects

### Radius Range

The radius range determines which circle sizes can be detected.

- Too small → large circles may be missed.
- Too large → unnecessary computation and false detections may occur.
- Appropriate range → more accurate detection.

### Detection Threshold

A lower threshold allows weaker circle candidates to be detected but may increase false positives.

A higher threshold is more selective but may cause weak or partially visible circles to be missed.

---

## Outputs

The notebook investigates:

- Original image
- Grayscale image
- Smoothed image
- Canny edge map
- Hough circle accumulator
- Detected circles
- Effects of different radius ranges
- Effects of different detection thresholds

The best parameter configuration is selected based on the accuracy of the detected circles and the number of false or missed detections.

---

# Q3 — Role of Noise in Image Thresholding

**Notebook:**

```text
Q3_Role_of_Noise_in_Image_Thresholding.ipynb
```

## Objective

This question investigates how Gaussian noise affects global thresholding and image segmentation.

A controlled synthetic grayscale image containing a shaded foreground object is used so that the segmentation result can be compared against a known ground-truth mask.

---

## Libraries Used

```python
import numpy as np
import matplotlib.pyplot as plt
```

---

## Image Construction

The notebook creates a synthetic image using NumPy.

The image consists of:

- A background
- A foreground object
- Smooth intensity variation/shading
- A known foreground mask

The ground-truth mask allows the segmentation accuracy to be measured quantitatively.

---

## Gaussian Noise

Gaussian noise is generated using a random number generator:

```python
rng = np.random.default_rng(42)
```

Noise is added using:

```python
noise = rng.normal(
    loc=0.0,
    scale=sigma,
    size=image.shape
)
```

Two noise levels are investigated:

```text
σ = 10
σ = 50
```

The noisy image is clipped to the valid 8-bit range:

```text
0 → 255
```

---

## Thresholding

A fixed global threshold is used:

```python
T = 128
```

The binary segmentation is obtained using:

```python
binary = img > T
```

Pixels above the threshold are classified as foreground, while pixels below the threshold are classified as background.

---

## Analysis

The notebook compares:

- Clean image
- Image with σ = 10 Gaussian noise
- Image with σ = 50 Gaussian noise

For each image, it visualizes:

- Grayscale image
- Foreground/background intensity histograms
- Threshold location
- Binary segmentation

---

## Quantitative Evaluation

The following measures are calculated:

### Pixel Error

\[
\text{Pixel Error}
=
\frac{\text{Number of incorrectly classified pixels}}
{\text{Total number of pixels}}
\]

### False Positive Rate

Background pixels incorrectly classified as foreground.

### False Negative Rate

Foreground pixels incorrectly classified as background.

---

## Main Observation

Increasing Gaussian noise causes greater overlap between the foreground and background intensity distributions.

Therefore, a fixed global threshold becomes less reliable as the noise level increases.

In general:

```text
More noise
    ↓
More histogram overlap
    ↓
More thresholding errors
    ↓
More false positives / false negatives
```

---

# Q4 — Otsu Global Thresholding From Scratch

**Notebook:**

```text
Q4_Otsu_Global_Thresholding_From_Scratch.ipynb
```

## Objective

This question implements **Otsu's global thresholding algorithm from scratch** using NumPy.

The objective is to automatically determine the threshold that best separates an image into foreground and background classes.

---

## Libraries Used

```python
import numpy as np
import matplotlib.pyplot as plt
```

No built-in Otsu thresholding function is used for the main implementation.

---

# Image and Histogram

A deterministic synthetic grayscale image is generated containing:

- Background
- Shaded mug-like foreground object
- Known ground-truth foreground mask

The image is represented as an 8-bit grayscale image:

```text
0 → black
255 → white
```

The histogram is calculated using:

```python
hist = np.bincount(
    image.ravel(),
    minlength=256
)
```

The histogram is normalized to obtain intensity probabilities.

---

# Otsu's Method

For every possible threshold \(T\) from 0 to 255, the image is divided into two classes:

- Class 0 → background
- Class 1 → foreground

The class probabilities are:

\[
w_0(T), \quad w_1(T)
\]

The class means are:

\[
\mu_0(T), \quad \mu_1(T)
\]

The between-class variance is:

\[
\sigma_B^2(T)
=
w_0(T)w_1(T)
[\mu_0(T)-\mu_1(T)]^2
\]

The optimal threshold is the threshold that maximizes the between-class variance:

\[
T^* = \arg\max_T \sigma_B^2(T)
\]

---

## From-Scratch Implementation

The main function is:

```python
def otsu_from_scratch(gray_image):
    ...
```

The implementation:

1. Validates the input image.
2. Computes the histogram.
3. Converts the histogram to probabilities.
4. Computes the total intensity sum.
5. Iterates through all possible thresholds.
6. Computes class weights.
7. Computes class means.
8. Computes between-class variance.
9. Selects the threshold with maximum variance.

---

# Segmentation

After obtaining the optimal threshold:

```python
binary = np.where(
    image > T_opt,
    255,
    0
).astype(np.uint8)
```

The resulting image is the Otsu segmentation.

---

# Verification and Tests

The notebook performs several checks to validate the implementation.

### Histogram Check

The normalized histogram probabilities should sum to:

\[
1
\]

### Ground-Truth Check

The clean synthetic image is expected to be segmented exactly according to the known geometric foreground mask.

### Variance Identity

The notebook verifies:

\[
\sigma_T^2
=
\sigma_W^2
+
\sigma_B^2
\]

where:

- \(\sigma_T^2\) = total variance
- \(\sigma_W^2\) = within-class variance
- \(\sigma_B^2\) = between-class variance

### Direct Criterion Verification

The calculated between-class variance is independently verified by directly splitting image pixels at every threshold.

### Constant Image Handling

The implementation also checks that a constant image is correctly rejected because it cannot produce two non-empty intensity classes.

---

# Otsu Limitations

The notebook also investigates situations where global Otsu thresholding may not work well.

Two examples are considered:

## 1. Unimodal Image

If the image contains only one dominant intensity population, there is no clear foreground/background separation.

Otsu may still select a threshold mathematically, but the resulting segmentation may not correspond to a meaningful object.

## 2. Uneven Illumination

When illumination varies spatially, pixels belonging to the same object may have substantially different intensities.

A single global threshold may therefore incorrectly classify some foreground and background regions.

This demonstrates that Otsu's method works best when the image histogram contains reasonably separable foreground and background intensity distributions.

---

# 📊 Summary of Techniques

| Question | Main Technique | Supporting Technique |
|---|---|---|
| Q1 | Hough Line Transform | LoG + Zero Crossings |
| Q2 | Hough Circle Transform | Gaussian Blur + Canny |
| Q3 | Global Thresholding | Gaussian Noise Analysis |
| Q4 | Otsu Thresholding | Histogram + Between-Class Variance |

---

# 🎯 Learning Outcomes

After completing this assignment, the following computer vision concepts are demonstrated:

- Laplacian of Gaussian
- Zero-crossing edge detection
- Hough Line Transform
- Hough Circle Transform
- Canny edge detection
- Gaussian smoothing
- Edge maps
- Hough accumulator space
- Line parameterization using \((\rho,\theta)\)
- Circle parameterization
- Radius-range selection
- Detection thresholds
- False positives and false negatives
- Gaussian noise
- Global image thresholding
- Image histograms
- Foreground/background segmentation
- Otsu's thresholding algorithm
- Between-class variance
- Within-class variance
- Limitations of global thresholding

---

# 🔧 Important Configuration Parameters

The main parameters that can be modified experimentally include:

## Q1

```python
GAUSSIAN_SIGMA = 1.4

THRESHOLD_LOW = 0.0
THRESHOLD_HIGH = 0.03

NUM_LINES = 20
MIN_DISTANCE = 20
MIN_ANGLE = 10
```

## Q2

```python
GAUSSIAN_SIGMA = 2.0

CANNY_LOW = 50
CANNY_HIGH = 150

RADIUS_RANGES = [
    (10, 40),
    (20, 80),
    (40, 120)
]

PEAK_THRESHOLDS = [
    0.25,
    0.40,
    0.55
]

MAX_CIRCLES = 20
MIN_CENTER_DISTANCE = 20
```

## Q3

```python
T = 128
```

Noise levels:

```text
σ = 10
σ = 50
```

## Q4

Otsu automatically evaluates:

```text
T = 0, 1, 2, ..., 255
```

and selects the threshold maximizing between-class variance.

---

# 📝 Notes

- Ensure that the required images are available at the paths specified in the notebooks.
- If an image cannot be found, update the corresponding `IMAGE_PATH`.
- Execute notebook cells sequentially.
- The notebooks contain both visual and quantitative analysis.
- Parameters can be modified to investigate their effects on detection and segmentation.
- Random noise experiments use fixed random seeds where reproducibility is required.
- The Q4 Otsu implementation is intentionally written from scratch rather than relying on an existing Otsu implementation.

---

# 👨‍💻 Assignment Information

**Course:** Computer Vision  
**Assignment:** Assignment 2  
**Language:** Python 3  
**Format:** Jupyter Notebooks (`.ipynb`)  
**Primary Libraries:** NumPy, Matplotlib, OpenCV, scikit-image