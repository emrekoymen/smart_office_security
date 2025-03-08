# Setup Guide for Vision-Based Security System

This guide will help you set up the vision-based security system to work with the Google Coral Edge TPU. The system **requires Python 3.8** specifically for compatibility with the Coral libraries.

## Step 1: Recreate Conda Environment with Python 3.8

First, we need to recreate your Conda environment with Python 3.8:

```bash
# Deactivate current environment
conda deactivate

# Remove existing environment 
conda remove --name smart_office --all

# Create new environment with Python 3.8
conda create -n smart_office python=3.8

# Activate the new environment
conda activate smart_office
```

## Step 2: Run the Setup Script

Once you have Python 3.8 active in your environment, run the setup script:

```bash
./setup_coral.sh
```

This script will:
- Check if you're using Python 3.8
- Install system dependencies for the Coral Edge TPU
- Download the required model files
- Install pycoral and tflite_runtime packages directly from Google's GitHub
- Install other Python dependencies

## Step 3: Choose Running Mode

The system supports three running modes:

### 1. Smart Mode (Recommended)

The smart detector automatically chooses the best output method based on your environment:
- If a display is available, it shows the video with detection boxes
- If no display is available, it saves the complete processed video as an MP4 file

```bash
./run_smart.sh -v videos/sample.mp4
```

This is the recommended mode for both desktop and headless environments.

### 2. GUI Mode with Display

If you specifically want to use the display-only mode:

```bash
./run_detection.sh -v videos/sample.mp4
```

This will display a window showing the video with bounding boxes around detected people.

### 3. Headless Mode (Frame-by-Frame Saving)

If you want to save individual frames rather than a complete video:

```bash
./run_headless.sh -v videos/sample.mp4
```

This will process the video and save individual frames with detections to the `output/` directory.

## Running with Sample Video

To run with the sample video:

```bash
# Smart mode (recommended)
./run_smart.sh -v videos/sample.mp4

# GUI mode (if display is available)
./run_detection.sh -v videos/sample.mp4

# Headless mode (saves individual frames)
./run_headless.sh -v videos/sample.mp4
```

## Viewing Results

### Smart Mode Results

In smart mode with a display, you'll see:
- Live video with bounding boxes around detected people
- Confidence scores above each detected person
- FPS counter in the top-left corner
- "Mode: CPU" or "Mode: TPU" indicating which processing mode is active

In smart mode without a display:
- A complete MP4 video will be saved to the output directory
- The video will contain all the same visual elements (bounding boxes, scores, etc.)
- Console will show detection events with confidence scores
- A summary will be shown after processing is complete

To view the saved video:
```bash
# If on a system with GUI:
xdg-open output/sample_processed_*.mp4
```

### GUI Mode Results

Same as smart mode with display.

### Headless Mode Results

In headless mode:
- Individual detection frames are saved to the `output/` directory
- Console will show detection events with confidence scores
- A summary will be shown after processing is complete

To view the saved detection frames:
```bash
# List the saved detection frames
ls -la output/

# View a specific detection frame (replace with your preferred image viewer)
# If on a system with GUI:
xdg-open output/frame_0085.jpg
```

## Troubleshooting

### ImportError for pycoral

If you encounter import errors with pycoral, here are some additional steps to try:

1. Make sure the correct packages are installed:
   ```bash
   pip list | grep coral
   pip list | grep tflite
   ```

2. If needed, manually install from Google's GitHub:
   ```bash
   pip install --no-deps https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp38-cp38-linux_x86_64.whl
   pip install --no-deps https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl
   ```

3. Test imports directly:
   ```bash
   python -c "import tflite_runtime; print('tflite_runtime imported successfully')"
   python -c "import pycoral; print('pycoral imported successfully')"
   ```

### Display Issues

If you get errors related to display, such as:
```
qt.qpa.xcb: could not connect to display
```

Use the smart mode, which will automatically fall back to saving the video:
```bash
./run_smart.sh -v videos/sample.mp4
```

### Video Saving Issues

If you have problems with the saved video:
1. Make sure you have the necessary codecs installed: `sudo apt-get install ffmpeg`
2. Try different codec options in the script (e.g., 'XVID' instead of 'mp4v')
3. Check disk space and permissions for the output directory

### Using an Actual Edge TPU

If you have an actual Edge TPU device (like a Coral USB Accelerator):

1. Connect it to your computer
2. Add the necessary udev rules:
   ```bash
   echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1a6e", MODE="0664", GROUP="plugdev"' | sudo tee /etc/udev/rules.d/99-edgetpu-accelerator.rules
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```
3. Run the system as normal, and it will automatically use the Edge TPU hardware

For further assistance, refer to the Google Coral documentation: https://coral.ai/docs/ 