#!/bin/bash

# Script to run the dual camera detector with recommended settings

# Exit on error
set -e

# Check if camera_config.txt exists
if [ -f "camera_config.txt" ]; then
    echo "Loading camera configuration from camera_config.txt..."
    source camera_config.txt
else
    echo "No camera configuration found. Using default camera indices."
    CAMERA1_INDEX=0
    CAMERA2_INDEX=2
fi

# Parse command line arguments
THRESHOLD=0.5
SAVE_VIDEO=false
OUTPUT_DIR="output"
NO_DISPLAY=false
FORCE_CPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold=*)
            THRESHOLD="${1#*=}"
            shift
            ;;
        --save-video)
            SAVE_VIDEO=true
            shift
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --no-display)
            NO_DISPLAY=true
            shift
            ;;
        --force-cpu)
            FORCE_CPU=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --threshold=FLOAT     Detection threshold (default: 0.5)"
            echo "  --save-video          Save processed video"
            echo "  --output-dir=DIR      Directory to save output files (default: output/)"
            echo "  --no-display          Disable display output"
            echo "  --force-cpu           Force CPU mode (no TPU)"
            echo "  --help                Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Build command
CMD="./build/dual_camera_detector --camera1=$CAMERA1_INDEX --camera2=$CAMERA2_INDEX --threshold=$THRESHOLD"

if [ "$SAVE_VIDEO" = true ]; then
    CMD="$CMD --save-video --output-dir=$OUTPUT_DIR"
fi

if [ "$NO_DISPLAY" = true ]; then
    CMD="$CMD --no-display"
fi

if [ "$FORCE_CPU" = true ]; then
    CMD="$CMD --force-cpu"
fi

# Print command
echo "Running: $CMD"

# Run the command
eval $CMD 