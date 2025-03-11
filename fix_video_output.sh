#!/bin/bash

# Script to fix video output issues in the dual camera detector

# Exit on error
set -e

echo "Checking video codec compatibility..."

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "ffmpeg is not installed. Installing..."
    sudo apt update
    sudo apt install -y ffmpeg
fi

# Check available codecs
echo "Available video codecs:"
ffmpeg -codecs | grep -i "video encoders" -A 50

# Check OpenCV build information for codec support
echo "Checking OpenCV codec support..."
pkg-config --modversion opencv4
echo "OpenCV build information:"
opencv_version -v | grep -i "video\|codec"

# Create a test directory
mkdir -p test_video_output

# Create a simple test script to verify video writing
cat > test_video_output.cpp << 'EOF'
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

int main() {
    // Create a test video
    cv::Mat frame(720, 1280, CV_8UC3, cv::Scalar(0, 0, 255));
    
    // Test different codecs
    std::vector<std::string> codecs = {"mp4v", "avc1", "XVID", "MJPG"};
    
    for (const auto& codec : codecs) {
        // Convert codec string to fourcc code
        int fourcc = cv::VideoWriter::fourcc(
            codec[0], 
            codec.length() > 1 ? codec[1] : ' ', 
            codec.length() > 2 ? codec[2] : ' ', 
            codec.length() > 3 ? codec[3] : ' '
        );
        
        std::string filename = "test_video_output/test_" + codec + ".mp4";
        cv::VideoWriter writer(filename, fourcc, 30.0, cv::Size(1280, 720));
        
        if (!writer.isOpened()) {
            std::cerr << "Failed to open video writer with codec: " << codec << std::endl;
            continue;
        }
        
        // Write 100 frames
        for (int i = 0; i < 100; i++) {
            // Add frame number text
            cv::putText(frame, "Frame: " + std::to_string(i), 
                       cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, 
                       cv::Scalar(255, 255, 255), 2);
            
            writer.write(frame);
        }
        
        writer.release();
        std::cout << "Successfully created test video with codec: " << codec << std::endl;
    }
    
    return 0;
}
EOF

# Compile the test script
echo "Compiling test script..."
g++ -o test_video_output test_video_output.cpp $(pkg-config --cflags --libs opencv4)

# Run the test
echo "Running video codec test..."
./test_video_output

# Check the output files
echo "Checking output files..."
ls -la test_video_output/

# Analyze the output files
echo "Analyzing output files..."
for file in test_video_output/*.mp4; do
    echo "File: $file"
    ffprobe -v error -show_entries format=duration,size -show_streams -select_streams v -of default=noprint_wrappers=1 "$file"
    echo "---"
done

# Determine the best codec
echo "Determining the best codec for our system..."
best_codec=$(ls -S test_video_output/*.mp4 | head -n 1 | sed 's/.*test_\(.*\)\.mp4/\1/')
echo "Recommended codec: $best_codec"

# Update the dual_camera_detector.cpp file with the best codec
if [ -n "$best_codec" ]; then
    echo "Updating dual_camera_detector.cpp with the best codec..."
    
    # Create a backup
    cp src/dual_camera_detector.cpp src/dual_camera_detector.cpp.bak
    
    # Update the codec in the file
    sed -i "s/int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');/int fourcc = cv::VideoWriter::fourcc('$best_codec[0]', '${best_codec:1:1}', '${best_codec:2:1}', '${best_codec:3:1}');/" src/dual_camera_detector.cpp
    
    echo "Updated dual_camera_detector.cpp with codec: $best_codec"
fi

echo "Video output fix completed!"
echo "Please rebuild the project with: ./build_and_run.sh" 