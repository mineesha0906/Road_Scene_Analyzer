# Road Scene Analyzer Using Classical Computer Vision

## 1. Project Overview

Road Scene Analyzer is an interactive Computer Vision application developed using Python, OpenCV, Scikit-Image, Scikit-Learn, NumPy, and Streamlit.

The system analyzes road images and videos using classical Computer Vision techniques. It provides multiple independent analysis modules through a simple Streamlit web interface.

The main purpose of the project is to demonstrate the practical application of fundamental Computer Vision techniques such as image preprocessing, edge detection, Hough Transform, K-Means segmentation, HOG feature extraction, and Optical Flow.

The application allows a user to upload an image or video, select an analysis technique, process the input, and visualize the resulting output.

---

## 2. Features

The application provides the following Computer Vision modules:

### 2.1 Image Preprocessing

The preprocessing module performs:

- Image resizing
- Conversion to grayscale
- Gaussian filtering for noise reduction
- CLAHE-based contrast enhancement

This prepares the image for further Computer Vision operations.

### 2.2 Canny Edge Detection

The Canny Edge Detection module identifies important boundaries and edges in the road scene.

The processing pipeline includes:

1. Conversion to grayscale
2. Gaussian smoothing
3. Canny edge detection
4. Generation of an edge map

### 2.3 Lane Detection Using Hough Transform

The lane detection module detects possible road lane markings.

The process includes:

1. Canny edge detection
2. Region of Interest selection
3. Probabilistic Hough Line Transform
4. Filtering of detected line segments
5. Drawing the detected lines on the road image

### 2.4 K-Means Image Segmentation

K-Means clustering is used to divide the image pixels into different groups based on their color characteristics.

The user can select the number of clusters, K, using the application interface.

Supported values:

- K = 2
- K = 3
- K = 4
- K = 5
- K = 6

### 2.5 HOG Feature Extraction

Histogram of Oriented Gradients (HOG) is used to extract shape and edge-based features from the input image.

The module:

- Resizes the image
- Converts it to grayscale
- Calculates HOG descriptors
- Generates a HOG visualization
- Displays the feature vector length

### 2.6 Optical Flow

The Optical Flow module analyzes motion between consecutive video frames.

The project uses the Farneback Optical Flow algorithm to estimate pixel movement between frames and generate a visual representation of motion.

The module accepts video files and processes consecutive frames to produce an optical-flow visualization.

---

## 3. Technologies Used

### Programming Language

- Python

### Libraries and Frameworks

- OpenCV
- NumPy
- Scikit-Learn
- Scikit-Image
- Pillow
- Streamlit
- Pytest

### Computer Vision Techniques

- Image preprocessing
- Gaussian filtering
- CLAHE
- Canny Edge Detection
- Region of Interest
- Hough Transform
- K-Means clustering
- Histogram of Oriented Gradients
- Farneback Optical Flow

---

## 4. System Architecture

The application follows a modular architecture.

The user interacts with the Streamlit frontend and uploads an image or video. The input is passed to the appropriate processing module depending on the analysis selected by the user.

The major layers of the system are:

1. User Interface
2. Input Handling
3. Image/Video Processing
4. Computer Vision Modules
5. Result Visualization
6. Testing

The Computer Vision functionality is separated into individual Python modules under the `src` directory.

---

## 5. Project Structure

```text
Road_Scene_Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
├── statement.md
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── edge_detection.py
│   ├── lane_detection.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   └── optical_flow.py
│
└── tests/
    └── test_modules.py