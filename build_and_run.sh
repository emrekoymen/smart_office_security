#!/bin/bash

# Build and run script for the C++ dual camera detector

# Exit on error
set -e

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

# Run the detector
echo "Running dual camera detector..."
./build/dual_camera_detector "$@" 