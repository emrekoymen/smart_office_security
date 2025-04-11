# Setup Guide for Smart Office Security

This guide covers setup for both the Python and C++ implementations of the vision-based security system.

## C++ Implementation Setup (Recommended)

This section details how to set up and build the high-performance C++ version using TensorFlow Lite and the Edge TPU.

### 1. Install Prerequisites

Install necessary build tools, dependencies, OpenCV, and the Edge TPU libraries:

```bash
# Install Build Tools & Dependencies
sudo apt update
sudo apt install -y build-essential cmake git pkg-config bazel

# Install OpenCV
sudo apt install -y libopencv-dev

# Install Edge TPU Runtime & Development Libraries
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt update
sudo apt install -y libedgetpu1-std libedgetpu-dev
```

### 2. Build TensorFlow Lite C++ Library

You need to build the TensorFlow Lite C++ shared library (`libtensorflowlite.so`) from source. Follow the detailed steps in the dedicated guide: [TensorFlow Lite and Edge TPU Integration Guide](TENSORFLOW_INTEGRATION.md#build-tensorflow-lite-c-library-from-source).

This involves:
- Cloning the TensorFlow repository.
- Downloading dependencies using `download_dependencies.sh`.
- Building `libtensorflowlite.so` using Bazel.

### 3. Configure Project Build (CMake)

Before building the project, you **must** edit the `CMakeLists.txt` file in the project root directory.

Update the placeholder paths for the TensorFlow Lite include directories and the built library path (`libtensorflowlite.so`) to match the locations from Step 2.

Refer to the `CMakeLists.txt` section in the [TensorFlow Integration Guide](TENSORFLOW_INTEGRATION.md#update-cmakeliststxt) for the exact lines to modify.

### 4. Build the C++ Project

Once TensorFlow Lite is built and `CMakeLists.txt` is configured:

```bash
# Navigate to the smart_office_security project directory
cd /path/to/smart_office_security

# Create a build directory (if it doesn't exist) and navigate into it
mkdir -p build
cd build

# Configure the build using CMake
cMAKE ..

# Compile the project (use -j flag for parallel compilation)
make -j$(nproc)
```

The executable `dual_camera_detector` will be created inside the `build` directory.

### 5. Running the C++ Detector

Run the compiled executable from the `build` directory, providing necessary arguments like model path, label path, etc. Remember to use relative paths (e.g., `../models/`) if running from the `build` directory.

```bash
# Example: Run from the build directory
./dual_camera_detector \
    -m ../models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite \
    -l ../models/coco_labels.txt \
    --camera1=0 \
    --camera2=1 \
    --save-video
```

See the main [README.md](README.md#command-line-options) for all available command-line options.

---

## Python Implementation Setup (Legacy)

This section details the setup for the original Python implementation. **Note:** This version requires Python 3.8 specifically.

### Step 1: Create/Activate Conda Environment (Python 3.8)

```bash
# If needed, create the environment
# conda create -n smart_office_py python=3.8

# Activate the environment
conda activate smart_office_py
```

### Step 2: Run the Python Setup Script

```bash
./setup_coral.sh
```

This script handles Python-specific dependencies like `pycoral` and `tflite_runtime` for Python 3.8.

### Step 3: Running the Python Version

Use the various `run_*.sh` scripts designed for the Python implementation (e.g., `run_smart.sh`, `run_detection.sh`).

```bash
# Example: Run smart mode with sample video
./run_smart.sh -v videos/sample.mp4
```

### Python Troubleshooting

Refer to the troubleshooting section at the end of this guide for Python-specific issues like `ImportError` for `pycoral`.

---

## General Troubleshooting (Applies to Both)

### Model File Not Found

Ensure the path provided via the `--model` (or `-m`) argument is correct relative to where you are running the executable from. If running from the `build` directory, paths to the `models` directory should typically start with `../models/`.

### Edge TPU Not Detected/Used

- Ensure the Edge TPU device (e.g., USB Accelerator) is properly connected.
- Verify the `libedgetpu1-std` package is installed.
- For USB devices, ensure udev rules are set up correctly (often handled by `libedgetpu-dev` installation or Google's setup scripts). You can manually add them:
  ```bash
  echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a6e", MODE="0664", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-edgetpu-accelerator.rules
  echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTRS{idProduct}=="9302", MODE="0664", GROUP="plugdev"' | sudo tee -a /etc/udev/rules.d/99-edgetpu-accelerator.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger
  ```
- Check system logs (`dmesg`) for USB detection issues.

### Display Issues (C++ & Python GUI)

If you get errors related to display connection (e.g., `qt.qpa.xcb: could not connect to display`, `Gtk-WARNING **: cannot open display`):
- Ensure you are running in an environment with a graphical display server (like X11 or Wayland).
- Check `DISPLAY` environment variable (`echo $DISPLAY`).
- If running remotely (e.g., SSH), use SSH with X11 forwarding (`ssh -X user@host`).
- For the C++ version, try running without display using `--no-display`.
- For the Python version, use `./run_smart.sh` which handles display absence.

### Video Saving Issues (C++ & Python)

- Ensure `ffmpeg` and necessary codecs are installed (`sudo apt install ffmpeg`).
- Check write permissions for the output directory (`output/` by default).
- Check available disk space.


[Original Python Setup Steps - Kept for Reference]

(... Previous content of SETUP_GUIDE.md related only to Python/Conda setup ... This section can be reviewed and potentially removed or further minimized later if the Python version is fully deprecated ...) 