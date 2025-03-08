#!/bin/bash
# Script to run the person detection system

# Check for conda environment and Python version
if [[ "$CONDA_PREFIX" != *"smart_office"* ]]; then
    echo "Conda environment 'smart_office' is not active."
    echo "Please run: conda activate smart_office"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.8" ]]; then
    echo "Error: Python version 3.8 is required, but found $PYTHON_VERSION"
    echo "Please recreate your conda environment with: conda create -n smart_office python=3.8"
    echo "Then activate it with: conda activate smart_office"
    exit 1
fi

# Default values
VIDEO_SOURCE="0"  # Default to webcam
DISPLAY=true
SAVE=false
THRESHOLD="0.5"
OUTPUT_DIR="output"

# Try to create a test window to see if display is available
python -c "import cv2; cv2.namedWindow('test', cv2.WINDOW_NORMAL); cv2.destroyAllWindows()" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "No display detected. Will save frames instead of displaying."
    DISPLAY=false
    SAVE=true
fi

# Display help information
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help            Show this help message"
    echo "  -v, --video FILE      Use a video file instead of webcam"
    echo "  -c, --camera NUM      Use a specific camera number (default: 0)"
    echo "  -t, --threshold NUM   Set detection threshold (0.0-1.0, default: 0.5)"
    echo "  -n, --no-display      Don't display video feed"
    echo "  -s, --save            Save frames with detected persons"
    echo "  -o, --output DIR      Directory to save output frames (default: output/)"
    echo
    echo "Examples:"
    echo "  $0                    Run with webcam and display feed"
    echo "  $0 -v videos/sample.mp4    Run with specific video file"
    echo "  $0 -t 0.7 -s          Run with higher threshold and save detections"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--video)
            VIDEO_SOURCE="$2"
            shift 2
            ;;
        -c|--camera)
            VIDEO_SOURCE="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -n|--no-display)
            DISPLAY=false
            shift
            ;;
        -s|--save)
            SAVE=true
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Ensure models are downloaded
if [ ! -f "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" ]; then
    echo "Models not found. Running setup script..."
    ./setup_coral.sh
fi

# Create output directory if saving
if $SAVE; then
    mkdir -p "$OUTPUT_DIR"
fi

# Build command with appropriate flags
CMD="python detector.py --source \"$VIDEO_SOURCE\" --threshold $THRESHOLD --output-dir \"$OUTPUT_DIR\""
if $DISPLAY; then
    CMD="$CMD --display"
fi
if $SAVE; then
    CMD="$CMD --save"
fi

# Run the detector
echo "Starting person detection..."
echo "Video source: $VIDEO_SOURCE"
echo "Threshold: $THRESHOLD"
echo "Display: $DISPLAY"
echo "Save frames: $SAVE"
if $SAVE; then
    echo "Output directory: $OUTPUT_DIR"
fi

# Execute the command
eval $CMD

# If frames were saved, show summary
if $SAVE; then
    COUNT=$(ls -1 "$OUTPUT_DIR"/*.jpg 2>/dev/null | wc -l)
    if [ $COUNT -gt 0 ]; then
        echo "Detection complete. $COUNT frames with detections saved to $OUTPUT_DIR/"
    else
        echo "No detection frames were saved."
    fi
fi 