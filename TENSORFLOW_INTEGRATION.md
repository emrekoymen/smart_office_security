# TensorFlow Lite and Edge TPU Integration Guide

This document outlines the steps to integrate TensorFlow Lite and Edge TPU into the C++ implementation of the dual camera person detection system.

## Prerequisites

1.  **Install Build Tools & Dependencies:**
    ```bash
    sudo apt update
    sudo apt install -y build-essential cmake git pkg-config bazel
    ```

2.  **Install OpenCV:**
    ```bash
    sudo apt install -y libopencv-dev
    ```

3.  **Install Edge TPU Runtime & Development Libraries:**
    ```bash
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt update
    sudo apt install -y libedgetpu1-std libedgetpu-dev
    ```

4.  **Build TensorFlow Lite C++ Library from Source:**
    
    Cloning and building from source is recommended for compatibility.
    
    ```bash
    # Choose a directory to clone TensorFlow (e.g., ~/dev)
    mkdir -p ~/dev
    cd ~/dev
    
    # Clone the TensorFlow repository (adjust version tag as needed)
    # Using a specific version is recommended for stability
    # Check TensorFlow releases for suitable tags (e.g., v2.10.0)
    git clone https://github.com/tensorflow/tensorflow.git --branch v2.10.0 --depth 1
    cd tensorflow
    
    # Download dependencies required for TFLite build
    # This script downloads necessary components like flatbuffers
    ./tensorflow/lite/tools/make/download_dependencies.sh
    
    # Configure the build (optional, defaults are often fine)
    # ./configure 
    
    # Build the TensorFlow Lite shared library using Bazel
    # This command builds the core TFLite library
    bazel build --config=opt //tensorflow/lite:libtensorflowlite.so
    
    # The library will be located at:
    # ~/dev/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so
    # The required headers are within the ~/dev/tensorflow directory.
    # The flatbuffers headers are typically downloaded into:
    # ~/dev/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include
    ```

## Integration Steps

1.  **Update `CMakeLists.txt`:**
    
    Ensure your `CMakeLists.txt` correctly includes the TensorFlow Lite directories and links against the built libraries. Replace `/path/to/tensorflow/` with the actual path where you cloned the TensorFlow repository (e.g., `/home/user/dev/tensorflow`).
    
    ```cmake
    # Set minimum required CMake version
    cmake_minimum_required(VERSION 3.10)
    
    # Enable project VERSION keyword
    cmake_policy(SET CMP0048 NEW)
    
    # Set project name and version
    project(dual_camera_detector VERSION 1.0)
    
    # Set C++ standard
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED True)
    
    # Find OpenCV
    find_package(OpenCV REQUIRED)
    
    # Find Threads
    find_package(Threads REQUIRED)
    
    # Add include directories
    include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OpenCV_INCLUDE_DIRS}
        # Adjust this path to your TensorFlow clone directory
        /path/to/tensorflow 
        # Adjust this path to the downloaded flatbuffers include directory
        /path/to/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include 
    )
    
    # Add source files
    add_executable(dual_camera_detector 
        src/main.cpp 
        src/dual_camera_detector.cpp 
        src/camera_processor.cpp 
        src/model.cpp
    )
    
    # Add Edge TPU library path
    set(EDGETPU_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libedgetpu.so")
    
    # Add TensorFlow Lite library path
    # Adjust this path to your built TFLite library
    set(TFLITE_LIBRARY_PATH "/path/to/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so") 
    
    # Link libraries
    target_link_libraries(dual_camera_detector 
        ${OpenCV_LIBS} 
        ${CMAKE_THREAD_LIBS_INIT}
        ${EDGETPU_LIBRARY_PATH}
        ${TFLITE_LIBRARY_PATH}
    )
    ```

2.  **Build the Project:**
    
    Once TensorFlow Lite is built and `CMakeLists.txt` is updated:
    
    ```bash
    cd /path/to/smart_office_security # Navigate to your project root
    mkdir -p build
    cd build
    cmake ..
    make -j$(nproc) # Build the project
    ```

## Troubleshooting

*   **Linking Errors:**
    *   Double-check the `TFLITE_LIBRARY_PATH` and include paths in `CMakeLists.txt` point to the correct locations in your TensorFlow clone/build directory.
    *   Ensure Bazel built `libtensorflowlite.so` successfully.
    *   Verify that the C++ standard used for TFLite (usually C++17 by default with recent versions) matches the project's standard (`CMAKE_CXX_STANDARD 17`).
*   **Header Not Found Errors (`tensorflow/lite/...h`):**
    *   Ensure the main TensorFlow source directory (e.g., `/path/to/tensorflow`) and the flatbuffers include directory are correctly specified in `include_directories` in `CMakeLists.txt`.
*   **Runtime Errors:**
    *   Make sure the Edge TPU device is connected (if applicable).
    *   Verify the model file (`.tflite`) path provided via the command line is correct and the model is compatible with the Edge TPU (if using) or CPU.

## Resources

*   [TensorFlow Lite C++ API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c)
*   [Edge TPU C++ API](https://coral.ai/docs/edgetpu/inference-cpp/)
*   [Building TensorFlow Lite with CMake](https://www.tensorflow.org/lite/guide/build_cmake) (Alternative build method)
*   [TensorFlow Releases](https://github.com/tensorflow/tensorflow/releases) (For finding version tags) 