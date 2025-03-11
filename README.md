# Smart Office Security - C++ Implementation

This is a high-performance C++ implementation of the dual-camera person detection system using Google's Coral Edge TPU.

## Features

- Real-time person detection using two USB cameras
- Hardware acceleration with Google Coral Edge TPU
- Multi-threaded processing for optimal performance
- Side-by-side display of both camera feeds with detection overlays
- Alert system for person detection events
- Performance metrics logging
- Video recording of detection events
- Automatic camera detection and configuration

## Requirements

- Ubuntu Linux (tested on Ubuntu 20.04/22.04)
- OpenCV 4.x
- Google Coral Edge TPU library
- C++17 compatible compiler
- CMake 3.10+

## Installation

### Install Dependencies

```bash
# Install OpenCV and development tools
sudo apt update
sudo apt install -y build-essential cmake git pkg-config
sudo apt install -y libopencv-dev

# Install Edge TPU libraries
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install -y libedgetpu1-std
sudo apt install -y python3-pycoral
```

### Build the Project

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/yourusername/smart_office_security.git
cd smart_office_security

# Checkout the C++ implementation branch
git checkout cpp_implementation

# Build the project using the build script
./build_and_run.sh
```

## Usage

### Camera Detection and Configuration

To detect and test available cameras:

```bash
./camera_detection.sh
```

This script will:
- List all available video devices
- Show detailed information about each camera
- Allow you to test each camera
- Create a configuration file with the selected cameras

### Running the Detector

Basic usage with the simplified script:

```bash
./run_detector.sh
```

With additional options:

```bash
./run_detector.sh --threshold=0.6 --save-video --output-dir=output
```

Or using the binary directly:

```bash
# Run with default settings (using webcam 0 and 1)
./build/dual_camera_detector

# Run with specific camera sources
./build/dual_camera_detector --camera1=0 --camera2=2

# Run with custom model and threshold
./build/dual_camera_detector --model=path/to/model.tflite --threshold=0.6
```

## Command Line Options

- `--model`: Path to the TFLite model (default: models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite)
- `--labels`: Path to the labels file (default: models/coco_labels.txt)
- `--threshold`: Detection threshold (default: 0.5)
- `--camera1`: First camera source (default: 0)
- `--camera2`: Second camera source (default: 1)
- `--output-dir`: Directory to save output files (default: output/)
- `--person-class-id`: Class ID for 'person' (default: 0 for COCO)
- `--no-display`: Disable display output
- `--save-video`: Save processed video
- `--force-cpu`: Force CPU mode (no TPU)

## Utility Scripts

The project includes several utility scripts to help with setup and troubleshooting:

- `build_and_run.sh`: Builds and runs the project
- `camera_detection.sh`: Detects and tests available cameras
- `fix_video_output.sh`: Fixes video output issues by testing different codecs
- `run_detector.sh`: Simplified script to run the detector with common options
- `setup_tensorflow.sh`: Sets up TensorFlow Lite and Edge TPU integration

## Performance

The C++ implementation achieves significantly better performance compared to the Python version:
- Target FPS: 15+ FPS per camera
- Lower CPU usage
- Reduced latency
- Better memory management

## Troubleshooting

### Video Output Issues

If you encounter issues with video output, run the video output fix script:

```bash
./fix_video_output.sh
```

### Camera Issues

If you have issues with cameras:

1. Check that the camera is properly connected
2. Run `./camera_detection.sh` to identify working cameras
3. Make sure no other application is using the camera

## License

[MIT License](LICENSE) 