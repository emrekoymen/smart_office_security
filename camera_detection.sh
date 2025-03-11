#!/bin/bash

# Script to detect and test available cameras for the dual camera detector

# Exit on error
set -e

echo "Detecting available cameras..."

# Check if v4l-utils is installed
if ! command -v v4l2-ctl &> /dev/null; then
    echo "v4l-utils is not installed. Installing..."
    sudo apt update
    sudo apt install -y v4l-utils
fi

# List all video devices
echo "Available video devices:"
ls -l /dev/video*

# Get detailed information about each device
echo -e "\nDetailed camera information:"
for device in /dev/video*; do
    echo -e "\nDevice: $device"
    v4l2-ctl --device=$device --info
    echo "Supported formats:"
    v4l2-ctl --device=$device --list-formats-ext
done

# Create a simple test script to verify camera capture
cat > test_camera_capture.cpp << 'EOF'
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

void testCamera(int cameraIndex) {
    // Open camera
    cv::VideoCapture cap(cameraIndex);
    
    if (!cap.isOpened()) {
        std::cerr << "Failed to open camera " << cameraIndex << std::endl;
        return;
    }
    
    // Get camera properties
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    
    std::cout << "Camera " << cameraIndex << " opened: " 
              << width << "x" << height << " @ " << fps << " FPS" << std::endl;
    
    // Create window
    std::string windowName = "Camera " + std::to_string(cameraIndex);
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    
    // Capture and display frames
    int frameCount = 0;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    while (frameCount < 100) {
        cv::Mat frame;
        bool ret = cap.read(frame);
        
        if (!ret || frame.empty()) {
            std::cerr << "Failed to capture frame from camera " << cameraIndex << std::endl;
            break;
        }
        
        // Add frame count and camera index
        cv::putText(frame, "Camera: " + std::to_string(cameraIndex) + " Frame: " + std::to_string(frameCount), 
                   cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                   cv::Scalar(0, 255, 0), 2);
        
        // Display frame
        cv::imshow(windowName, frame);
        
        // Wait for key press (30ms)
        int key = cv::waitKey(30);
        if (key == 27) {  // ESC key
            break;
        }
        
        frameCount++;
    }
    
    // Calculate actual FPS
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime).count() / 1000.0;
    double actualFps = frameCount / elapsedTime;
    
    std::cout << "Camera " << cameraIndex << " actual FPS: " << actualFps << std::endl;
    
    // Release resources
    cap.release();
    cv::destroyWindow(windowName);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <camera_index1> [camera_index2] ..." << std::endl;
        return 1;
    }
    
    // Test each camera
    for (int i = 1; i < argc; i++) {
        int cameraIndex = std::stoi(argv[i]);
        testCamera(cameraIndex);
    }
    
    return 0;
}
EOF

# Compile the test script
echo "Compiling camera test script..."
g++ -o test_camera_capture test_camera_capture.cpp $(pkg-config --cflags --libs opencv4)

# Create a function to test a camera
test_camera() {
    local camera_index=$1
    echo "Testing camera $camera_index..."
    ./test_camera_capture $camera_index
}

# Test cameras based on the available devices
echo -e "\nTesting cameras..."
echo "Press ESC to close each camera window after testing."

# Extract camera indices from device list
camera_indices=($(ls -l /dev/video* | grep -o "video[0-9]*" | grep -o "[0-9]*"))

# Test each camera one by one
for index in "${camera_indices[@]}"; do
    read -p "Test camera $index? (y/n): " choice
    if [[ $choice == "y" || $choice == "Y" ]]; then
        test_camera $index
    fi
done

# Ask for the best cameras
echo -e "\nBased on the tests, please select the best cameras for dual camera detection:"
read -p "Enter the index for Camera 1: " camera1_index
read -p "Enter the index for Camera 2: " camera2_index

# Create a configuration file
echo "Creating camera configuration file..."
cat > camera_config.txt << EOF
# Camera configuration for dual camera detector
CAMERA1_INDEX=$camera1_index
CAMERA2_INDEX=$camera2_index
EOF

echo "Camera configuration saved to camera_config.txt"
echo "You can now run the dual camera detector with:"
echo "./build/dual_camera_detector --camera1=$camera1_index --camera2=$camera2_index"

# Offer to run the detector
read -p "Run the dual camera detector now? (y/n): " run_choice
if [[ $run_choice == "y" || $run_choice == "Y" ]]; then
    ./build/dual_camera_detector --camera1=$camera1_index --camera2=$camera2_index
fi 