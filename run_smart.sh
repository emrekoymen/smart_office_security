#!/bin/bash
# Script to run the smart person detection system

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
THRESHOLD="0.5"
OUTPUT_DIR="output"
ENABLE_DISPLAY=true  # Default to display mode
SAVE_VIDEO=false  # Default to not saving video
FORCE_CPU=false  # Default to using TPU with CPU fallback

# Display help information
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help            Show this help message"
    echo "  -v, --video FILE      Use a video file instead of webcam"
    echo "  -c, --camera NUM      Use a specific camera number (default: 0)"
    echo "  -t, --threshold NUM   Set detection threshold (0.0-1.0, default: 0.5)"
    echo "  -n, --no-display      Disable display (headless mode)"
    echo "  -s, --save           Save processed video"
    echo "  -o, --output DIR      Directory to save output files (default: output/)"
    echo "  --cpu                Force CPU mode (no TPU)"
    echo
    echo "Examples:"
    echo "  $0                    Run with webcam, display enabled (default)"
    echo "  $0 -v videos/sample.mp4    Run with specific video file"
    echo "  $0 -n -s             Run in headless mode and save video"
    echo "  $0 --cpu             Force CPU mode"
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
            ENABLE_DISPLAY=false
            shift
            ;;
        -s|--save)
            SAVE_VIDEO=true
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cpu)
            FORCE_CPU=true
            shift
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

# Create output directory if saving video
if $SAVE_VIDEO; then
    mkdir -p "$OUTPUT_DIR"
fi

# Build command with appropriate flags
CMD="python smart_detector.py --source \"$VIDEO_SOURCE\" --threshold $THRESHOLD --output-dir \"$OUTPUT_DIR\""

# Add display flag
if ! $ENABLE_DISPLAY; then
    CMD="$CMD --no-display"
fi

# Add save video flag
if $SAVE_VIDEO; then
    CMD="$CMD --save-video"
fi

# Add force CPU flag
if $FORCE_CPU; then
    CMD="$CMD --force-cpu"
fi

# Run the detector
echo "Starting smart person detection..."
echo "Video source: $VIDEO_SOURCE"
echo "Threshold: $THRESHOLD"
echo "Display enabled: $ENABLE_DISPLAY"
echo "Save video: $SAVE_VIDEO"
echo "Force CPU mode: $FORCE_CPU"
if $SAVE_VIDEO; then
    echo "Output directory: $OUTPUT_DIR"
fi

# Make the script executable if not already
chmod +x smart_detector.py

# Execute the command
eval $CMD 