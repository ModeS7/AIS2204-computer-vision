# Computer Vision — Course Portfolio

A collection of computer vision implementations covering image processing fundamentals, camera geometry, feature detection, stereo vision, and deep learning-based classification.

## Course Information

- **Course:** [AIS2204 - Maskinsyn](https://www.ntnu.edu/studies/courses/AIS2204)
- **Institution:** NTNU - Norwegian University of Science and Technology
- **Semester:** Fall 2022

## Projects

### Epipolar Geometry (`epipolar_geometry/`)

Implementation of the 4-point and 8-point algorithms for estimating the Essential/Fundamental matrix between two camera views. Given synthetic 3D points projected into two camera frames, the algorithm recovers the relative rotation and translation using SVD decomposition of the epipolar constraint matrix.

- `eight_point.py` — Complete 8-point algorithm: generates synthetic stereo pair, builds constraint matrix, estimates Essential matrix via SVD, extracts 4 possible [R, t] solutions
- `four_point_algorithm.ipynb` — 4-point variant notebook
- `eight_point_algorithm.ipynb` — Step-by-step 8-point walkthrough

### Feature Tracking (`feature_tracking/`)

Visual feature detection and motion estimation across image sequences using Harris corner detection and spatial gradient computation.

- `tracker.py` — Loads sequential frames, computes Harris corner response, detects and tracks features between frames

### Camera Calibration (`camera_calibration/`)

Camera intrinsic parameter estimation using checkerboard and ArUco marker patterns with OpenCV. Tested across multiple cameras (laptop webcam, Logitech C930e, Microsoft LifeCam, ODroid webcam).

- `checkerboard_calibration.py` — Live video checkerboard corner detection, iterative calibration, exports camera matrix and distortion coefficients to CSV
- `aruco_calibration.py` — ChArUco board variant using ArUco marker detection

### Image Filtering (`image_filtering/`)

Image convolution and edge detection fundamentals, progressing from manual nested-loop implementation to optimized library calls.

- `filters.py` — Averaging filters, Gaussian blur, Sobel edge detection, Laplacian, custom kernels (manual convolution → `scipy.signal.convolve2d` → `cv.filter2D`)
- `corner_detection.py` — Harris corner detection pipeline using Sobel gradients

### Neural Network (`neural_network/`)

CNN image classification on the CIFAR-10 dataset using PyTorch.

- `cifar10_classifier.ipynb` — Convolutional neural network training, evaluation, and visualization
- `ann_tutorial.ipynb` — Introductory artificial neural network concepts

### 3D Transforms (`3d_transforms/`)

Homogeneous coordinate transformations and 3D object visualization.

- `transforms.py` — Defines 3D geometry (tetrahedron), applies rotation/scaling matrices, renders with `matplotlib.Poly3DCollection`

## Project Structure

```
.
├── epipolar_geometry/
│   ├── eight_point.py
│   ├── four_point_algorithm.ipynb
│   ├── eight_point_algorithm.ipynb
│   ├── 3d_world.png                  # 3D point reconstruction result
│   └── images.png                    # Stereo image pair visualization
├── feature_tracking/
│   ├── tracker.py
│   └── test_images/                  # Sequential frames for tracking
│       ├── *.jpg                     # Test frames (5 scenes)
│       ├── bad_video/                # Low-quality sequence + eigenvalue analysis
│       └── ok_video/                 # Good-quality sequence + eigenvalue analysis
├── camera_calibration/
│   ├── checkerboard_calibration.py
│   ├── aruco_calibration.py
│   ├── 1.jpg, 2.jpg                  # Calibration board images
│   ├── calibresult.png               # Calibration result visualization
│   └── calibration_data/             # Exported calibration parameters
│       ├── camera_matrix.csv         # Default camera intrinsics
│       ├── camera_distortion.csv     # Default camera distortion
│       ├── laptop_webcam/            # Per-camera calibration CSVs
│       ├── logitech_c930e/
│       ├── microsoft_lifecam/
│       └── odroid_webcam/
├── image_filtering/
│   ├── filters.py
│   ├── corner_detection.py
│   ├── lenna.ppm                     # Classic test image
│   ├── lenna-awgn.png                # Additive white Gaussian noise
│   ├── lenna-snp.png                 # Salt-and-pepper noise
│   └── shapes.png                    # Geometric shapes for edge detection
├── neural_network/
│   ├── cifar10_classifier.ipynb
│   ├── ann_tutorial.ipynb
│   ├── output.png                    # Training results (default model)
│   ├── output2layers.png             # 2-layer architecture results
│   └── output3layers.png             # 3-layer architecture results
├── 3d_transforms/
│   └── transforms.py
├── requirements.txt
└── .gitignore
```

## Tech Stack

- **Python 3.8+**
- **OpenCV** — Image processing, camera calibration, feature detection
- **NumPy / SciPy** — Linear algebra, convolution, SVD
- **PyTorch** — CNN training and inference
- **Matplotlib** — Visualization and 3D plotting
- **Jupyter** — Interactive notebooks

## How to Run

```bash
pip install -r requirements.txt
```

Each subdirectory is self-contained. Run Python scripts directly or open notebooks with `jupyter notebook`.
