# Dual Camera Person Detection System

This system uses the Google Coral USB Accelerator to perform real-time person detection on two camera feeds simultaneously. It's designed to maintain high performance (15+ FPS per camera) while providing accurate detection results.

## Features

- Processes two camera feeds simultaneously
- Uses Google Coral Edge TPU for hardware acceleration
- Displays both camera feeds side-by-side with detection overlays
- Logs performance metrics and detection events
- Triggers alerts when a person is detected in either camera feed
- Supports saving the processed video output

## Requirements

- Ubuntu PC with USB ports
- Google Coral USB Accelerator
- Two USB cameras (or one USB camera and the built-in webcam)
- Python 3.6+
- OpenCV
- PyCoral library

## Installation

1. Make sure you have the required dependencies installed:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv

# Install Python dependencies
pip3 install opencv-python numpy pillow plyer
```

2. Install the PyCoral library by following the instructions at https://coral.ai/docs/accelerator/get-started/ or by running the setup script:

```bash
./setup_coral.sh
```

## Usage

The system can be run using the provided shell script:

```bash
./run_dual_camera.sh
```

### Command-line Options

- `--camera1 SOURCE`: First camera source (default: 0 for webcam)
- `--camera2 SOURCE`: Second camera source (default: 2 for USB camera)
- `--no-display`: Disable display output
- `--save-video`: Save processed video
- `--force-cpu`: Force CPU mode (no TPU)
- `--threshold VALUE`: Detection threshold (default: 0.5)
- `--help`: Show help message

### Examples

1. Run with default settings (webcam and USB camera):

```bash
./run_dual_camera.sh
```

2. Run with specific camera sources:

```bash
./run_dual_camera.sh --camera1 0 --camera2 4
```

3. Run without display and save the output video:

```bash
./run_dual_camera.sh --no-display --save-video
```

4. Run with a custom detection threshold:

```bash
./run_dual_camera.sh --threshold 0.7
```

## Performance Optimization

To achieve the best performance:

1. Make sure the Google Coral USB Accelerator is properly connected and recognized
2. Close other applications that might be using the cameras or CPU resources
3. If FPS drops below 15, consider:
   - Reducing the camera resolution
   - Increasing the detection threshold
   - Ensuring the Edge TPU is being used (check the mode indicator)

## Troubleshooting

1. **Cameras not detected**: Check if the cameras are properly connected and recognized by the system:

```bash
ls -l /dev/video*
```

You can also test which cameras are working with:

```bash
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(f'Camera 0: {cap.isOpened()}'); cap.release()"
```

2. **Edge TPU not detected**: Check if the Coral USB Accelerator is properly connected:

```bash
lsusb | grep 1a6e:089a
```

3. **Low FPS**: Check the system load and make sure no other processes are using the cameras or CPU resources:

```bash
top
```

4. **Display issues**: If you encounter display problems, try running without display and save the video instead:

```bash
./run_dual_camera.sh --no-display --save-video
```

## Architecture

The system uses a multi-threaded approach to efficiently process two camera feeds:

1. Each camera has two dedicated threads:
   - A capture thread that continuously reads frames from the camera
   - A processing thread that runs the detection model on the frames

2. The main thread:
   - Combines the processed frames from both cameras
   - Handles display and video saving
   - Manages alerts and logging

This architecture ensures that:
- Camera capture is never blocked by processing
- Both cameras can run at their maximum possible frame rate
- The Edge TPU is efficiently utilized

## Logs and Output

- Detection logs are saved in the `logs` directory
- Processed videos are saved in the `output` directory
- Performance metrics are displayed in the terminal and saved in the logs

## License

This project is licensed under the MIT License - see the LICENSE file for details. 