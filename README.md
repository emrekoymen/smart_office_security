# Smart Office Security - C++ Implementation

This is a high-performance C++ implementation of the dual-camera person detection system using Google's Coral Edge TPU.

## Features

- Real-time person detection using two USB cameras
- Hardware acceleration with Google Coral Edge TPU
- Multi-threaded processing for optimal performance
- Side-by-side display of both camera feeds with detection overlays
- Alert system for person detection events
- Performance metrics logging

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

# Build the project
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Usage

```bash
# Run with default settings (using webcam 0 and 1)
./dual_camera_detector

# Run with specific camera sources
./dual_camera_detector --camera1=/dev/video0 --camera2=/dev/video1

# Run with custom model and threshold
./dual_camera_detector --model=path/to/model.tflite --threshold=0.6
```

## Command Line Options

- `--model`: Path to the TFLite model (default: models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite)
- `--labels`: Path to the labels file (default: models/coco_labels.txt)
- `--threshold`: Detection threshold (default: 0.5)
- `--camera1`: First camera source (default: 0)
- `--camera2`: Second camera source (default: 1)
- `--output-dir`: Directory to save output files (default: output/)
- `--no-display`: Disable display output
- `--save-video`: Save processed video

## Performance

The C++ implementation achieves significantly better performance compared to the Python version:
- Target FPS: 20+ FPS per camera
- Lower CPU usage
- Reduced latency
- Better memory management

## License

[MIT License](LICENSE) 