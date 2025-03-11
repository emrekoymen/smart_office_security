#!/bin/bash

# Script to set up TensorFlow Lite and Edge TPU for the dual camera detector

# Exit on error
set -e

echo "Setting up TensorFlow Lite and Edge TPU..."

# Create directories
mkdir -p external
cd external

# Install dependencies
echo "Installing dependencies..."
sudo apt update
sudo apt install -y cmake build-essential git python3-pip

# Install Edge TPU runtime
echo "Installing Edge TPU runtime..."
if ! dpkg -l | grep -q libedgetpu1-std; then
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt update
    sudo apt install -y libedgetpu1-std libedgetpu-dev
fi

# Clone TensorFlow repository if it doesn't exist
if [ ! -d "tensorflow" ]; then
    echo "Cloning TensorFlow repository..."
    git clone https://github.com/tensorflow/tensorflow.git --branch v2.5.0 --depth 1
fi

# Build TensorFlow Lite
echo "Building TensorFlow Lite..."
cd tensorflow

# Configure TensorFlow build
echo "Configuring TensorFlow build..."
cat > configure_tflite.sh << 'EOF'
#!/bin/bash
export PYTHON_BIN_PATH=$(which python3)
export USE_DEFAULT_PYTHON_LIB_PATH=1
export TF_NEED_OPENCL_SYCL=0
export TF_NEED_ROCM=0
export TF_NEED_CUDA=0
export TF_DOWNLOAD_CLANG=0
export TF_SET_ANDROID_WORKSPACE=0
export TF_CONFIGURE_IOS=0
./configure
EOF

chmod +x configure_tflite.sh
./configure_tflite.sh

# Build TensorFlow Lite
echo "Building TensorFlow Lite library..."
mkdir -p build_tflite
cd build_tflite

cmake ../tensorflow/lite/c \
    -DTFLITE_ENABLE_GPU=OFF \
    -DTFLITE_ENABLE_RUY=ON \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)

# Return to the project root
cd ../../..

# Update CMakeLists.txt
echo "Updating CMakeLists.txt..."

# Create backup
cp CMakeLists.txt CMakeLists.txt.bak

# Update CMakeLists.txt to use TensorFlow Lite
cat > CMakeLists.txt << 'EOF'
cmake_minimum_required(VERSION 3.10)
project(smart_office_security)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

# TensorFlow Lite paths
set(TENSORFLOW_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/external/tensorflow)
set(TFLITE_ROOT ${TENSORFLOW_ROOT}/tensorflow/lite)
set(TFLITE_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external/tensorflow/build_tflite)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORFLOW_ROOT}
    ${TFLITE_ROOT}/tools/make/downloads/flatbuffers/include
)

# Remove the DISABLE_TENSORFLOW definition
# add_definitions(-DDISABLE_TENSORFLOW)

# Edge TPU library path
set(EDGETPU_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu/libedgetpu.so")

# Source files
file(GLOB SOURCES "src/*.cpp")

# Create executable
add_executable(dual_camera_detector ${SOURCES})

# Link libraries
target_link_libraries(dual_camera_detector
    ${OpenCV_LIBS}
    ${CMAKE_THREAD_LIBS_INIT}
    ${EDGETPU_LIBRARY_PATH}
    ${TFLITE_BUILD_DIR}/libtensorflowlite_c.so
)

# Installation
install(TARGETS dual_camera_detector DESTINATION bin)
EOF

# Download model files if they don't exist
echo "Downloading model files..."
mkdir -p models

if [ ! -f "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" ]; then
    echo "Downloading Edge TPU model..."
    curl -L "https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" -o models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
fi

if [ ! -f "models/coco_labels.txt" ]; then
    echo "Downloading COCO labels..."
    curl -L "https://github.com/google-coral/test_data/raw/master/coco_labels.txt" -o models/coco_labels.txt
fi

echo "TensorFlow Lite and Edge TPU setup completed!"
echo "You can now rebuild the project with: ./build_and_run.sh" 