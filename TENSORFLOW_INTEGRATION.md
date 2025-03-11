# TensorFlow Lite and Edge TPU Integration Guide

This document outlines the steps to integrate TensorFlow Lite and Edge TPU into the C++ implementation of the dual camera person detection system.

## Prerequisites

1. Install the Edge TPU runtime:
   ```bash
   echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
   sudo apt update
   sudo apt install -y libedgetpu1-std libedgetpu-dev
   ```

2. Download TensorFlow Lite C++ library:
   ```bash
   # Option 1: Use pre-built TensorFlow Lite C++ library
   wget https://github.com/tensorflow/tensorflow/releases/download/v2.5.0/tensorflow-2.5.0-rc0-cp38-cp38-linux_x86_64.whl
   unzip tensorflow-2.5.0-rc0-cp38-cp38-linux_x86_64.whl
   
   # Option 2: Build TensorFlow Lite from source (recommended for better compatibility)
   git clone https://github.com/tensorflow/tensorflow.git --branch v2.5.0 --depth 1
   cd tensorflow
   ./configure  # Follow the prompts
   bazel build --config=opt //tensorflow/lite:libtensorflowlite.so
   ```

## Integration Steps

1. Update CMakeLists.txt:
   ```cmake
   # Remove the DISABLE_TENSORFLOW definition
   # add_definitions(-DDISABLE_TENSORFLOW)
   
   # Add TensorFlow Lite include directories
   include_directories(
       ${CMAKE_CURRENT_SOURCE_DIR}/include
       ${OpenCV_INCLUDE_DIRS}
       /path/to/tensorflow  # Adjust this path
       /path/to/tensorflow/tensorflow/lite/tools/make/downloads/flatbuffers/include  # Adjust this path
   )
   
   # Add Edge TPU library path
   set(EDGETPU_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libedgetpu.so")
   
   # Add TensorFlow Lite library path
   set(TFLITE_LIBRARY_PATH "/path/to/tensorflow/bazel-bin/tensorflow/lite/libtensorflowlite.so")  # Adjust this path
   
   # Link libraries
   target_link_libraries(dual_camera_detector
       ${OpenCV_LIBS}
       ${CMAKE_THREAD_LIBS_INIT}
       ${EDGETPU_LIBRARY_PATH}
       ${TFLITE_LIBRARY_PATH}
   )
   ```

2. Update model.cpp:
   - Remove the `#ifdef DISABLE_TENSORFLOW` blocks
   - Implement the TensorFlow Lite and Edge TPU functionality

3. Test with a simple model:
   - Download a pre-trained TensorFlow Lite model for person detection
   - Test with a single camera first
   - Then test with dual cameras

## Troubleshooting

1. If you encounter linking errors:
   - Check that the library paths are correct
   - Ensure that the TensorFlow Lite library is built with the same C++ standard (C++17)

2. If you encounter runtime errors:
   - Check that the Edge TPU is properly connected
   - Verify that the model is compatible with the Edge TPU

3. If you encounter performance issues:
   - Profile the code to identify bottlenecks
   - Consider using a smaller model or reducing the input resolution

## Resources

- [TensorFlow Lite C++ API](https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c)
- [Edge TPU C++ API](https://coral.ai/docs/edgetpu/inference-cpp/)
- [TensorFlow Lite Model Optimization](https://www.tensorflow.org/lite/performance/model_optimization) 