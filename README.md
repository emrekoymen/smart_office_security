# Smart Office Security System - C++ Implementation

This project implements a vision-based security system that utilizes Google Coral's pretrained person detection model to detect unauthorized persons inside an office environment.

This branch contains the C++ implementation of the system, which provides better performance and optimization compared to the Python version.

## Project Goals

- Implement a high-performance C++ version of the person detection system
- Support both single camera and dual camera setups
- Achieve 20+ FPS processing at 300x300 resolution
- Support real-time display and video saving
- Automatically use Edge TPU if available, with fallback to CPU

## Prerequisites

- Ubuntu Desktop
- CMake 3.10+
- OpenCV 4.x
- TensorFlow Lite C++ library (will be built by setup script)
- Edge TPU runtime (optional, for hardware acceleration)
- USB webcams or test video files

## Setup Instructions

### Installing Dependencies

1. Install system dependencies:
   ```bash
   sudo apt-get update
   sudo apt-get install -y \
     build-essential \
     cmake \
     libopencv-dev \
     python3-opencv \
     wget \
     unzip \
     pkg-config
   ```

2. Install Edge TPU runtime (optional, for hardware acceleration):
   ```bash
   echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
   curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
   sudo apt update
   sudo apt install -y libedgetpu1-std libedgetpu-dev
   ```

### Building the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/emrekoymen/smart_office_security.git
   cd smart_office_security
   git checkout cpp_implementation
   ```

2. Setup TensorFlow and build the project:
   ```bash
   chmod +x build_and_run.sh
   ./build_and_run.sh --setup-tensorflow --build-only
   ```

   This will:
   - Download TensorFlow v2.4.0
   - Build TensorFlow Lite
   - Configure and build the project

### Downloading Models

1. Create models directory:
   ```bash
   mkdir -p models
   ```

2. Download Edge TPU compatible models:
   ```bash
   wget -O models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
   wget -O models/coco_labels.txt https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt
   ```

## Usage

The system provides two modes of operation:

1. **Single Camera Mode**: Process a single camera feed or video file
2. **Dual Camera Mode**: Process two camera feeds simultaneously

### Running in Single Camera Mode

```bash
./run_single_camera.sh [options]
```

Options:
- `--source=SOURCE`: Camera index or video file path (default: 0)
- `--threshold=VALUE`: Detection threshold (default: 0.5)
- `--no-display`: Disable display output
- `--save-video`: Save processed video
- `--output-dir=DIR`: Output directory (default: output)
- `--force-cpu`: Force CPU mode (no TPU)

Examples:
```bash
# Run with default camera
./run_single_camera.sh

# Run with a video file
./run_single_camera.sh --source=videos/sample.mp4

# Run with camera 1, no display, and save video
./run_single_camera.sh --source=1 --no-display --save-video
```

### Running in Dual Camera Mode

```bash
./run_dual_camera.sh [options]
```

Options:
- `--camera1=INDEX`: First camera index (default: 0)
- `--camera2=INDEX`: Second camera index (default: 2)
- `--threshold=VALUE`: Detection threshold (default: 0.5)
- `--no-display`: Disable display output
- `--save-video`: Save processed video
- `--output-dir=DIR`: Output directory (default: output)
- `--force-cpu`: Force CPU mode (no TPU)

Examples:
```bash
# Run with default cameras
./run_dual_camera.sh

# Run with specific camera indexes
./run_dual_camera.sh --camera1=0 --camera2=1

# Run with no display and save video
./run_dual_camera.sh --no-display --save-video
```

### Building with Manual Options

If you want more control over the build process, you can use the following options with the build script:

```bash
# Clean build
./build_and_run.sh --clean --build-only

# Setup TensorFlow and clean build
./build_and_run.sh --setup-tensorflow --clean --build-only

# Build and run in one step (dual camera mode)
./build_and_run.sh
```

## Performance

The C++ implementation is designed for high performance:

- Target processing rate: 20+ FPS for both cameras
- Resolution: 300x300 (default, configurable)
- Automatic TPU/CPU switching
- Multi-threaded camera processing
- Optimized memory usage

## Troubleshooting

### Edge TPU Issues

If the Edge TPU is not detected:
1. Check if the Edge TPU device is connected
2. Verify that the Edge TPU runtime is installed: `dpkg -l | grep edgetpu`
3. Check the model is compiled for Edge TPU (should have "edgetpu" in the filename)

### Camera Issues

If cameras are not detected:
1. List available camera devices: `ls -l /dev/video*`
2. Try different camera indexes
3. Check camera permissions: `sudo chmod 666 /dev/video*`

### Build Issues

If you encounter build errors:
1. Make sure all dependencies are installed
2. Try a clean build: `./build_and_run.sh --clean --build-only`
3. Check CMake version: `cmake --version` (must be 3.10+)

## Future Improvements

See the [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) file for planned enhancements. 