# Real-Time 2D Object Recognition

**Author:** Hyuk Jin Chung  
**Date:** February 2026  
**Language:** C++ (OpenCV)

## Overview

This project is a real-time, interactive object recognition system built in C++ using OpenCV. It serves as a comprehensive bridge between classical computer vision techniques and modern deep learning, allowing users to train and classify objects on the fly using either hand-crafted geometric features or deep semantic embeddings from a ResNet18 model.

## Features

* **Real-Time Processing Pipeline:** Captures live webcam feeds (or static images), applying HSV saturation-darkening, custom K-means thresholding, and morphological operations to segment objects cleanly.
* **Dual-Mode Classification:**
  * **Baseline Mode:** Uses classical geometric features (Oriented Bounding Boxes, Aspect Ratio, Percent Filled, and Hu Moments) and classifies via Scaled Euclidean Distance.
  * **DNN Mode:** Geometrically aligns and crops the object, passing the ROI (region of interest) through a pre-trained ResNet18 model to extract a 512-dimensional embedding. Classifies via Cosine Distance (One-Shot Learning).
* **Interactive Training:** Press a key to extract the current object's features/embeddings and instantly save them to a persistent CSV database.
* **Live Evaluation:** Built-in evaluation hooks to test the system on new objects and automatically generate a Confusion Matrix in the terminal.

## Repository Structure

* `objRecognition.cpp`: The core application loop, video I/O, UI event handling, and classification switching logic.
* `features.cpp` / `features.hpp`: Computer vision math, geometric feature extraction, bounding box generation, and DNN ROI preparation.
* `csv.cpp` / `csv.hpp`: Database management for persistently saving and loading both the geometric features and massive 512-D embeddings to and from a csv file.
* `utilities.cpp` / `utilities.hpp`: DNN embedding extraction.
* `CMakeLists.txt`: Cross-platform build configuration.

## Prerequisites

* **C++ Compiler:**
* **OpenCV** Must be compiled with `dnn` module support.
* **ResNet18 ONNX Model:** The project requires a pre-trained ResNet18 model (`resnet18-v2-7.onnx`) placed in the root directory.

## Installation & Build

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hyukjin17/Real-time-Object-Recognition.git
    cd Real-time-Object-Recognition
    ```

2.  **Create a build directory:** (use release config for faster real time video processing)
    ```bash
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
    ```

3.  **Run CMake and Compile:**
    ```bash
    cmake --build build --config Release
    make
    ```

4.  **Run the Application:**
    Ensure the `esnet18-v2-7.onnx` file is in the same directory as the executable, or that the path in the code points to it correctly.
    ```bash
    ./build/objRec
    ```

    To run the system on a static image instead of a live video feed, use the -i flag:
    ```bash
    ./build/objRec -i dir/test_img.jpg
    ```

---

## Interactive Controls

The system is fully controlled via keyboard inputs while the OpenCV video windows are actively in focus.

| Key | Action | Description |
| :---: | :--- | :--- |
| **`c`** | **Toggle Classification** | Turns the live classification text overlay ON or OFF. |
| **`m`** | **Switch Mode** | Toggles the active classifier between the Baseline (Hand-crafted Features) and DNN (ResNet18). |
| **`n`** | **New Object (Train)** | Pauses the feed and prompts the terminal for a label. Extracts features of the isolated object and saves them to `object_data.csv`. |
| **`e`** | **Evaluate** | Classifies the current object, prompts you for the *actual* ground-truth label, and records the result. |
| **`p`** | **Print Confusion Matrix** | Prints the current Evaluation Confusion Matrix to the terminal for performance analysis. |
| **`s`** | **Save Images** | Saves the current Binary, Region Map, and Bounding Box visualization frames to the disk as `.jpg` files. |
| **`q`** | **Quit** | Safely shuts down the camera and exits the program. |

---

## System Architecture

The pipeline is split into two primary phases: isolating the objects from the background, and extracting mathematical representations of those objects for classification.

### Phase 1: Pre-processing & Segmentation

Before any math or machine learning can happen, the system must cleanly separate the foreground objects from the background.
1. **Gaussian Blur:** Applies a 5x5 kernel to soften image noise and sensor grain.
2. **HSV Conversion & Darkening:** Converts the image to HSV color space. It subtracts a scaled Saturation from the Value channel to darken colorful objects, ensuring only the true white background remains bright.
3. **Custom K-Means Thresholding:** A highly optimized, 1D 2-means clustering algorithm iterates over the image histogram to dynamically find the optimal threshold boundary between foreground and background.
4. **Morphological Filtering:** Applies `cv::morphologyEx` (Open and Close operations) to strip away isolated noise pixels and fill in internal holes within the object masks (7x7 kernel).
5. **Connected Components Analysis (CCA):** Scans the binary image to assign unique IDs to distinct pixel clusters, filtering out regions that are too small or touching the image borders.

### Phase 2: Feature Extraction & Classification

The system employs a dual-mode classification architecture, allowing real-time comparison between classical methods and modern deep learning.

#### Mode A: Baseline (Hand-crafted Features)

This mode relies on geometric and statistical properties that describe the object's physical silhouette.
* **Extraction:** Calculates spatial moments (`cv::moments`) to locate the object's centroid and primary axis of orientation. It then projects the object's pixels along these axes to construct a tight Oriented Bounding Box (OBB).
* **Feature Vector:** Assembles a 4-dimensional vector comprising the Aspect Ratio, Percent Filled (object area / OBB area), and the first two log-transformed Hu Moments.
* **Classification:** Uses a **Scaled Euclidean Distance** metric. It compares the live feature vector against the database, scaling the differences by the standard deviation of the training set to prevent features with naturally larger magnitudes from dominating the math.

#### Mode B: Deep Neural Network (Semantic Embeddings)

This mode uses a pre-trained network to understand complex internal textures rather than just outer silhouettes.
* **Alignment & Extraction:** Uses the orientation angle calculated by the baseline to mathematically rotate the object so its primary axis aligns with the X-axis. It then extracts a tight, horizontally-aligned region of interest (ROI).
* **Feature Vector:** The ROI is resized to 224x224 and passed through a pre-trained ResNet18 model. The final classification layer is bypassed, and the network instead outputs the raw data from its flattened pooling layer a **512-dimensional semantic embedding**. 
* **Classification:** Uses a **Cosine Distance** metric. By measuring the angular distance between the 512-D vectors, the system achieves highly robust One-Shot classification that resists changes in lighting or scale.
