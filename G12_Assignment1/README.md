# Computer Vision — Assignment 1

This repository contains the implementation and analysis for **Computer Vision Assignment 1**. The assignment covers image interpolation, image smoothing, hybrid images, and edge detection.

---

## 📁 Repository Structure

```text
Assignment1/
│
├── Q1_Image_Resizing_Interpolation_Edge_Analysis.ipynb
├── Q2_Image_Smoothing.ipynb
├── Q3_Hybrid_Images.ipynb
├── Q4_Edge_Detection.ipynb
└── README.md
```

---

# ⚙️ Requirements

The notebooks are implemented in **Python 3** and use the following libraries:

- **NumPy** — numerical operations and array manipulation
- **Matplotlib** — image visualization and plotting
- **Pandas** — tabular presentation and analysis of results
- **OpenCV (cv2)** — image processing and interpolation
- **scikit-image** — sample images, colour conversion, and image transformations
- **SciPy** — 2D convolution operations
- **time** — execution-time measurement

---

## 📦 Installation

It is recommended to use a virtual environment or Conda environment.

Install all required packages using:

```bash
pip install numpy matplotlib pandas opencv-python scikit-image scipy
```

### Or, if using Conda:

```bash
conda install numpy matplotlib pandas scipy scikit-image
pip install opencv-python
```

---

# 📚 Imports Used

The following imports are used across the notebooks:

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import time

from skimage import data, color, transform
from scipy.signal import convolve2d
```

> Not every notebook uses every import. The exact imports depend on the question.

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

Then open the required `.ipynb` file.

**Run the cells sequentially from top to bottom**, since later cells may depend on variables, functions, or images created in earlier cells.

---

# Q1 — Image Resizing, Interpolation & Edge Analysis

**Notebook:**

```text
Q1_Image_Resizing_Interpolation_Edge_Analysis.ipynb
```

## Objective

This question investigates image downsampling, image reconstruction using different interpolation techniques, and the resulting reconstruction errors.

## Libraries Used

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage import data
import cv2
```

## Tasks

- Start with a `512 × 512` grayscale image.
- Downsample the image to:
  - `256 × 256`
  - `128 × 128`
- Reconstruct the images back to `512 × 512`.
- Compare:
  - Nearest-neighbour interpolation
  - Bilinear interpolation
  - Bicubic interpolation
- Implement interpolation-related operations.
- Calculate reconstruction errors.
- Calculate:
  - Mean Squared Error (MSE)
  - Peak Signal-to-Noise Ratio (PSNR)
- Visualize absolute-error and squared-error maps.
- Analyze the effect of increasing the downsampling factor.

## Main Concepts

- Image resizing
- Downsampling
- Nearest-neighbour interpolation
- Bilinear interpolation
- Bicubic interpolation
- Image reconstruction
- MSE
- PSNR
- Error visualization

---

# Q2 — Image Smoothing & Noise Analysis

**Notebook:**

```text
Q2_Image_Smoothing.ipynb
```

## Objective

This question investigates the effect of different spatial filters on images corrupted with different types of noise.

## Libraries Used

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from skimage import data, color
from scipy.signal import convolve2d

np.random.seed(42)
```

## Noise Types

The notebook investigates:

- Salt-and-pepper noise
- Gaussian noise
- Mixed noise

## Filters

The following filters are compared:

1. Box filter
2. Weighted-average filter
3. Gaussian filter
4. Median filter

Different kernel sizes are investigated, including:

```text
3 × 3
5 × 5
7 × 7
```

## Tasks

- Add salt-and-pepper noise to an image.
- Add Gaussian noise to an image.
- Apply different smoothing filters.
- Compare the filters for different kernel sizes.
- Calculate MSE and PSNR.
- Generate absolute-difference/error images.
- Create an image containing different noise types in different regions.
- Compare filter performance separately on the different regions.
- Investigate region-wise filtering.
- Determine suitable filters based on quantitative results.

## Main Concepts

- Spatial filtering
- Convolution
- Noise
- Salt-and-pepper noise
- Gaussian noise
- Box filtering
- Weighted-average filtering
- Gaussian filtering
- Median filtering
- Kernel size
- MSE
- PSNR

---

# Q3 — Hybrid Images

**Notebook:**

```text
Q3_Hybrid_Images.ipynb
```

## Objective

This question demonstrates how images can be combined using their low-frequency and high-frequency components to create **hybrid images**.

## Libraries Used

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from skimage import data, color, transform

np.random.seed(42)
```

## Basic Idea

A hybrid image combines:

- Low-frequency information from one image
- High-frequency information from another image

The high-frequency component can be obtained as:

\[
HP(I) = I - LP(I)
\]

The hybrid image is then constructed by combining the low- and high-frequency components.

## Tasks

- Load and align image pairs.
- Apply Gaussian filtering.
- Extract low-frequency components.
- Extract high-frequency components.
- Construct hybrid images.
- Experiment with:
  - Gaussian kernel size
  - Standard deviation (`sigma`)
  - Low-frequency weight
  - High-frequency weight
- Implement frequency-domain filtering using the Fourier Transform.
- Compare spatial-domain and frequency-domain approaches.
- Measure execution time.
- Construct Gaussian pyramids.
- Construct Laplacian pyramids.
- Generate hybrid images using pyramid representations.
- Investigate bilateral filtering for edge-preserving processing.
- Study the effect of viewing distance/downsampling.

## Main Concepts

- Spatial frequency
- Low-pass filtering
- High-pass filtering
- Gaussian filtering
- Fourier Transform
- Frequency-domain filtering
- Gaussian pyramid
- Laplacian pyramid
- Bilateral filtering
- Image alignment
- Hybrid images

---

# Q4 — Edge Detection

**Notebook:**

```text
Q4_Edge_Detection.ipynb
```

## Objective

This question investigates several edge-detection techniques and compares their results under different thresholding conditions.

## Libraries Used

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

from skimage import data
```

## Tasks

### 1. Manual 2D Convolution

A manual convolution operation is implemented to understand the underlying process of applying filters to an image.

### 2. First-Order Derivatives

Image derivatives are calculated in:

- X direction
- Y direction

From these, the following are obtained:

- Gradient magnitude
- Gradient direction

### 3. Thresholding

Binary edge maps are generated using different thresholds.

The effect of changing the threshold on detected edges is investigated.

### 4. Laplacian

Second-order derivatives are used to detect regions of rapid intensity change.

### 5. Laplacian of Gaussian (LoG)

Gaussian smoothing is combined with the Laplacian operator to reduce the effect of noise before detecting edges.

### 6. Canny Edge Detection

The Canny edge detector is applied and compared with the other methods.

## Main Concepts

- Image gradients
- First-order derivatives
- Second-order derivatives
- Gradient magnitude
- Gradient direction
- Thresholding
- Laplacian
- Laplacian of Gaussian
- Canny edge detection
- Noise suppression
- Edge localization

---

# 📊 Evaluation Metrics

Several experiments use **MSE** and **PSNR** to quantitatively compare image-processing results.

## Mean Squared Error — MSE

\[
MSE =
\frac{1}{MN}
\sum_{x=1}^{M}
\sum_{y=1}^{N}
[I(x,y)-I_f(x,y)]^2
\]

where:

- \(I\) is the original/reference image.
- \(I_f\) is the processed image.
- \(M,N\) are the image dimensions.

### Interpretation

- Lower MSE → smaller reconstruction/error difference.
- Higher MSE → larger difference from the reference image.

---

## Peak Signal-to-Noise Ratio — PSNR

\[
PSNR =
10\log_{10}
\left(
\frac{MAX_I^2}{MSE}
\right)
\]

For an 8-bit image:

\[
MAX_I = 255
\]

Therefore:

\[
PSNR =
10\log_{10}
\left(
\frac{255^2}{MSE}
\right)
\]

### Interpretation

- Higher PSNR → generally better similarity to the reference image.
- Lower PSNR → larger distortion/error.

---

# 🔬 Random Seed

For experiments involving random noise, the notebooks use:

```python
np.random.seed(42)
```

This ensures that the random noise generated is **reproducible**. Running the notebook multiple times with the same seed produces the same random values, making comparisons between experiments consistent.

---

# 🖼️ Image Data

Some notebooks use sample images provided by **scikit-image**, accessed through:

```python
from skimage import data
```

For example:

```python
data.camera()
```

returns a standard grayscale test image.

Other image-processing operations use NumPy arrays and OpenCV.

---

# 🧮 Important Implementation Details

## Image Representation

Images are represented as NumPy arrays.

For grayscale images, the array generally has the form:

```text
(height, width)
```

Pixel values for standard 8-bit grayscale images range from:

```text
0 → black
255 → white
```

Images are sometimes converted to floating-point representation using:

```python
.astype(np.float64)
```

to allow accurate mathematical operations without integer overflow or truncation.

---

# 🎯 Learning Outcomes

After completing this assignment, the following concepts are demonstrated:

- Image representation using NumPy
- Image resizing and reconstruction
- Interpolation techniques
- Spatial filtering
- Image convolution
- Noise generation and removal
- Quantitative image-quality evaluation
- Fourier-domain processing
- Low- and high-frequency image components
- Hybrid image construction
- Gaussian pyramids
- Laplacian pyramids
- Bilateral filtering
- Image gradients
- First- and second-order derivatives
- Laplacian edge detection
- LoG edge detection
- Canny edge detection
- Parameter and threshold analysis

---

# 📝 Notes

- Make sure all required Python packages are installed before running the notebooks.
- Execute notebook cells in order.
- Some functions and variables are defined in earlier cells and reused later.
- If running locally, ensure that the required Python environment has access to all dependencies.
- Results may depend on image-processing parameters such as kernel size, threshold, Gaussian sigma, and noise level.

---

## 👨‍💻 Assignment

**Course:** Computer Vision  
**Assignment:** Assignment 1  
**Language:** Python  
**Format:** Jupyter Notebooks (`.ipynb`)