#!/bin/bash
# Script to detect and test available cameras

# Exit on error
set -e

echo "Camera Detection Utility"
echo "========================"
echo

# Check for v4l-utils
if ! command -v v4l2-ctl &> /dev/null; then
    echo "v4l-utils not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y v4l-utils
fi

# List available video devices
echo "Available video devices:"
echo "------------------------"
ls -l /dev/video*
echo

# Get detailed information about each camera
echo "Camera details:"
echo "--------------"
for device in /dev/video*; do
    echo "Device: $device"
    v4l2-ctl --device=$device --info
    v4l2-ctl --device=$device --list-formats-ext
    echo
done

# Function to test a camera
test_camera() {
    local device=$1
    local index=${device#/dev/video}
    
    echo "Testing camera $index ($device)..."
    echo "Press 'q' to exit the test."
    
    # Try to open the camera with OpenCV
    python3 -c "
import cv2
import time

cap = cv2.VideoCapture($index)
if not cap.isOpened():
    print('Failed to open camera $index')
    exit(1)

# Get camera properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'Camera $index: {width}x{height} @ {fps} FPS')

# Display camera feed
cv2.namedWindow('Camera $index Test', cv2.WINDOW_NORMAL)
start_time = time.time()
frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to read frame')
        break
        
    # Calculate FPS
    frames += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        current_fps = frames / elapsed_time
        print(f'FPS: {current_fps:.2f}')
        frames = 0
        start_time = time.time()
    
    # Display FPS on frame
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display camera index
    cv2.putText(frame, f'Camera {$index}', (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Camera $index Test', frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"
}

# Ask user which cameras to test
echo "Do you want to test the cameras? (y/n)"
read -r response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    for device in /dev/video*; do
        echo "Test camera ${device#/dev/video}? (y/n)"
        read -r test_response
        if [[ "$test_response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            test_camera "$device"
        fi
    done
fi

# Create configuration
echo "Creating camera configuration..."
echo

echo "Select first camera (index number):"
read -r camera1

echo "Select second camera (index number):"
read -r camera2

echo "Configuration:"
echo "Camera 1: $camera1"
echo "Camera 2: $camera2"

# Update run_dual_camera.sh with selected cameras
sed -i "s/CAMERA1=\"0\"/CAMERA1=\"$camera1\"/" run_dual_camera.sh
sed -i "s/CAMERA2=\"1\"/CAMERA2=\"$camera2\"/" run_dual_camera.sh

echo
echo "Configuration saved to run_dual_camera.sh"
echo "You can now run the dual camera detector with:"
echo "./run_dual_camera.sh" 