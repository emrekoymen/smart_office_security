# Vision-Based Security System

This project implements a vision-based security system that utilizes Google Coral's pretrained person detection model to detect unauthorized persons inside an office environment.

## Project Goals

- Run a pretrained person detection model from Coral on an Ubuntu PC
- Process pre-recorded office security videos (low-resolution for efficiency)
- Measure the detection speed (FPS) to evaluate real-time performance
- Implement a basic alert system that triggers when a person is detected

## Setup Instructions

### Prerequisites

- Ubuntu Desktop
- Anaconda or Miniconda installed
- Python 3.8 via Conda (specific version required for Coral compatibility)
- USB webcam or test video files
- Internet connection to download models and dependencies

### Installation

1. Clone this repository

2. Set up and activate the Conda environment with Python 3.8 (required for Coral compatibility):
   ```
   conda create -n smart_office python=3.8
   conda activate smart_office
   ```

3. Run the setup script to install the Edge TPU runtime and download the model:
   ```
   ./setup_coral.sh
   ```
   This script will:
   - Install system dependencies
   - Add the Coral repository to apt sources
   - Install the Edge TPU runtime
   - Download the pretrained models
   - Install Python dependencies including the pycoral package

4. If you need sample videos for testing, you can place them in the `videos/` directory.

## Usage

First, ensure your Conda environment is activated:
```
conda activate smart_office
```

### Running the Detection System

The system provides a single unified pipeline with options for display, video saving, and hardware acceleration:

```
./run_smart.sh
```

This will run the detector with default settings:
- Uses webcam input
- Displays video feed with detection boxes
- Uses Edge TPU if available (falls back to CPU if not)
- Does not save video output

#### Command-line Options

```
Usage: ./run_smart.sh [options]
Options:
  -h, --help            Show help message
  -v, --video FILE      Use a video file instead of webcam
  -c, --camera NUM      Use a specific camera number (default: 0)
  -t, --threshold NUM   Set detection threshold (0.0-1.0, default: 0.5)
  -n, --no-display      Disable display (headless mode)
  -s, --save           Save processed video
  -o, --output DIR      Directory to save output files (default: output/)
  --cpu                Force CPU mode (no TPU)
```

### Examples

1. Run with default settings (webcam, display enabled):
   ```
   ./run_smart.sh
   ```

2. Run with a specific video file:
   ```
   ./run_smart.sh -v videos/sample.mp4
   ```

3. Run in headless mode and save the video:
   ```
   ./run_smart.sh -n -s
   ```

4. Run with a specific video file and force CPU mode:
   ```
   ./run_smart.sh -v videos/sample.mp4 --cpu
   ```

## Viewing Detection Results

The system can output results in two ways:

1. **Display Mode** (default): Shows live video feed with:
   - Green bounding boxes around detected people
   - Confidence scores above each detection
   - Current FPS and processing mode (TPU/CPU)

2. **Video Saving**: When enabled with `-s`, saves a processed video file that includes:
   - All frames with detection boxes and annotations
   - Filename includes source and timestamp (e.g., `sample_processed_1637245896.mp4`)

## Performance Metrics

The system measures and displays:
- Real-time FPS (frames per second)
- Processing mode (TPU or CPU)
- Total frames processed
- Number of frames with detections
- Average FPS for the session

## Alert System

When a person is detected with sufficient confidence (above the threshold):
1. The detection is logged to the console with confidence score
2. The frame with the detection is included in the output video (if saving enabled)
3. A desktop notification is displayed (if supported)

## Project Structure

- `smart_detector.py`: Main detection script with unified pipeline
- `utils.py`: Utility functions for video processing and alerting
- `requirements.txt`: List of Python dependencies
- `setup_coral.sh`: Script to set up the Edge TPU runtime and download models
- `run_smart.sh`: Script to run the detector with various options
- `.gitignore`: Specifies files to be ignored by git

## Troubleshooting

### Python Version Issues

This project requires Python 3.8 specifically for compatibility with the Coral Edge TPU library:
1. To check your Python version: `python --version`
2. If you have a different version, recreate the environment: `conda create -n smart_office python=3.8`

### Display Issues

If you encounter display errors:
1. Try running in headless mode with video saving:
   ```
   ./run_smart.sh -n -s
   ```
2. Check if your system has a display server running
3. Verify X11 forwarding if running remotely

### Video Saving Issues

If you have problems with the saved video:
1. Make sure you have the necessary codecs installed: `sudo apt-get install ffmpeg`
2. Check disk space and permissions for the output directory
3. Try different codec options in the script (e.g., 'XVID' instead of 'mp4v')

### Edge TPU Device

If you have an actual Edge TPU device (like a Coral USB Accelerator):
1. Connect it to your computer
2. Add the necessary udev rules (see setup script)
3. Run the system as normal, and it will automatically use the Edge TPU hardware

For further assistance, refer to the Google Coral documentation: https://coral.ai/docs/

## Progress Log

- Initial project setup completed
- Added system dependencies and model setup scripts
- Created video processing and detection utilities
- Implemented alert system and performance monitoring
- Added command-line interface and run script
- Updated for Conda environment compatibility with Python 3.8 for Coral Edge TPU support
- Unified detection pipeline with flexible output options
- Added automatic TPU/CPU fallback mechanism 