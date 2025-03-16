#!/bin/bash

# Build and run script for the C++ dual camera detector

# Exit on error
set -e

# Command line arguments
SETUP_TENSORFLOW=false
BUILD_ONLY=false
CLEAN_BUILD=false

# Parse command line arguments
for arg in "$@"
do
    case $arg in
        --setup-tensorflow)
        SETUP_TENSORFLOW=true
        shift
        ;;
        --build-only)
        BUILD_ONLY=true
        shift
        ;;
        --clean)
        CLEAN_BUILD=true
        shift
        ;;
        *)
        # Unknown option
        ;;
    esac
done

# Setup TensorFlow if requested
if [ "$SETUP_TENSORFLOW" = true ]; then
    echo "Setting up TensorFlow..."
    # Creating external directory if it doesn't exist
    mkdir -p external
    
    # Check if tensorflow is already downloaded
    if [ ! -d "external/tensorflow" ]; then
        echo "Downloading TensorFlow 2.4.0..."
        cd external
        wget https://github.com/tensorflow/tensorflow/archive/v2.4.0.zip
        unzip v2.4.0.zip
        mv tensorflow-2.4.0/ tensorflow
        chmod 777 tensorflow -R
        cd ..
    fi
    
    # Download dependencies and build TensorFlow Lite
    cd external/tensorflow
    ./tensorflow/lite/tools/make/download_dependencies.sh
    make -f tensorflow/lite/tools/make/Makefile
    cd ../..
    
    echo "TensorFlow setup complete!"
fi

# Clean build if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo "Cleaning build directory..."
    rm -rf build
fi

# Create build directory if it doesn't exist
mkdir -p build

# Navigate to build directory
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake ..

# Build
echo "Building..."
make -j$(nproc)

# Return to root directory
cd ..

# Run the detector if not build-only
if [ "$BUILD_ONLY" = false ]; then
    echo "Running dual camera detector..."
    ./build/dual_camera_detector "$@"
fi 