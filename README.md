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

### Smart Detection Mode (Recommended)

The smart detector automatically chooses the best output method based on your environment:
- If a display is available, it shows the video with detection boxes
- If no display is available, it saves the complete processed video as an MP4 file

```
./run_smart.sh
```

This will run the detector with default settings (webcam input).

#### Smart Mode Command-line Options

```
Usage: ./run_smart.sh [options]
Options:
  -h, --help            Show help message
  -v, --video FILE      Use a video file instead of webcam
  -c, --camera NUM      Use a specific camera number (default: 0)
  -t, --threshold NUM   Set detection threshold (0.0-1.0, default: 0.5)
  -s, --save            Force saving video even if display is available
  -o, --output DIR      Directory to save output files (default: output/)
```

### Display Mode (requires a GUI)

If you specifically want to use the display-only mode:

```
./run_detection.sh
```

This will run the detector with default settings (webcam input with display).

#### Display Mode Command-line Options

```
Usage: ./run_detection.sh [options]
Options:
  -h, --help            Show help message
  -v, --video FILE      Use a video file instead of webcam
  -c, --camera NUM      Use a specific camera number (default: 0)
  -t, --threshold NUM   Set detection threshold (0.0-1.0, default: 0.5)
  -n, --no-display      Don't display video feed
  -s, --save            Save frames with detected persons
  -o, --output DIR      Directory to save output frames (default: output/)
```

### Headless Mode (no GUI required)

For servers or environments without a display, you can also use the dedicated headless mode:

```
./run_headless.sh
```

The headless mode saves detection frames with bounding boxes to the output directory.

#### Headless Command-line Options

```
Usage: ./run_headless.sh [options]
Options:
  -h, --help            Show help message
  -v, --video FILE      Use a video file instead of webcam
  -c, --camera NUM      Use a specific camera number (default: 0)
  -t, --threshold NUM   Set detection threshold (0.0-1.0, default: 0.5)
  -a, --all             Save all frames, not just ones with detections
  -o, --output DIR      Directory to save output frames (default: output/)
```

### Examples

1. Run with the webcam (smart mode):
   ```
   ./run_smart.sh
   ```

2. Run with a specific video file (smart mode):
   ```
   ./run_smart.sh -v videos/sample.mp4
   ```

3. Run with a specific video file and force saving the video:
   ```
   ./run_smart.sh -v videos/sample.mp4 -s
   ```

4. Run the display-only mode with a video file:
   ```
   ./run_detection.sh -v videos/sample.mp4
   ```

5. Run the headless mode with a video file:
   ```
   ./run_headless.sh -v videos/sample.mp4
   ```

## Viewing Detection Results

Depending on the mode you use, the results will be available in different formats:

1. **Display Mode**: You'll see the video feed with green bounding boxes around detected people, with confidence scores displayed above each detection.

2. **Smart Mode with Display**: Same as display mode, showing live video with bounding boxes.

3. **Smart Mode without Display**: An MP4 video file will be saved to the output directory with all detections marked. The filename will include the source and timestamp, e.g., `sample_processed_1637245896.mp4`.

4. **Headless Mode**: Individual frames with detections will be saved as JPG files in the output directory.

## Performance Metrics

The system measures and displays/logs real-time FPS (frames per second) during operation. In headless mode, the FPS is printed to the console periodically.

## Alert System

When a person is detected with sufficient confidence (above the threshold):

1. The detection is logged to the console with confidence score
2. The frame with the detection is included in the output (video or images)
3. In display mode, a desktop notification is displayed (if supported)

## Project Structure

- `detector.py`: Display mode detection script
- `detector_headless.py`: Headless version for environments without display
- `smart_detector.py`: Smart detector that combines display and video saving
- `utils.py`: Utility functions for video processing and alerting
- `requirements.txt`: List of Python dependencies
- `setup_coral.sh`: Script to set up the Edge TPU runtime and download models
- `run_detection.sh`: Script to run the detector with display
- `run_headless.sh`: Script to run the detector in headless mode
- `run_smart.sh`: Script to run the smart detector
- `.gitignore`: Specifies files to be ignored by git

## Troubleshooting

### Python Version Issues

This project requires Python 3.8 specifically for compatibility with the Coral Edge TPU library:
1. To check your Python version: `python --version`
2. If you have a different version, recreate the environment: `conda create -n smart_office python=3.8`

### Display Issues

If you encounter display errors like:
```
qt.qpa.xcb: could not connect to display
```
Just use the smart detector, which will automatically fall back to saving the video:
```
./run_smart.sh -v videos/sample.mp4
```

### Video Saving Issues

If you have problems with the saved video:
1. Make sure you have the necessary codecs installed: `sudo apt-get install ffmpeg`
2. Try different codec options in the script (e.g., 'XVID' instead of 'mp4v')
3. Check disk space and permissions for the output directory

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
- Added headless mode for environments without display
- Added smart detector that can display or save video based on environment 