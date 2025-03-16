#!/bin/bash

# Script to run the single camera detector

# Exit on error
set -e

# Default values
SOURCE="0"
THRESHOLD="0.5"
DISPLAY="true"
SAVE_VIDEO="false"
OUTPUT_DIR="output"
FORCE_CPU="false"

# Parse command line arguments
for arg in "$@"
do
    case $arg in
        --source=*)
        SOURCE="${arg#*=}"
        shift
        ;;
        --threshold=*)
        THRESHOLD="${arg#*=}"
        shift
        ;;
        --no-display)
        DISPLAY="false"
        shift
        ;;
        --save-video)
        SAVE_VIDEO="true"
        shift
        ;;
        --output-dir=*)
        OUTPUT_DIR="${arg#*=}"
        shift
        ;;
        --force-cpu)
        FORCE_CPU="true"
        shift
        ;;
        --help)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --source=SOURCE    Camera index or video file path (default: 0)"
        echo "  --threshold=VALUE  Detection threshold (default: 0.5)"
        echo "  --no-display       Disable display output"
        echo "  --save-video       Save processed video"
        echo "  --output-dir=DIR   Output directory (default: output)"
        echo "  --force-cpu        Force CPU mode (no TPU)"
        echo "  --help             Show this help message"
        exit 0
        ;;
        *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
    esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Ensure build directory exists and build is up to date
if [ ! -d "build" ]; then
    echo "Build directory not found. Running build first..."
    ./build_and_run.sh --build-only
fi

# Make sure the executable exists
if [ ! -f "build/dual_camera_detector" ]; then
    echo "Executable not found. Building..."
    ./build_and_run.sh --build-only
fi

# Build command
CMD="./build/dual_camera_detector --single-camera --source=$SOURCE --threshold=$THRESHOLD"

# Add display option
if [ "$DISPLAY" = "false" ]; then
    CMD="$CMD --no-display"
fi

# Add save video option
if [ "$SAVE_VIDEO" = "true" ]; then
    CMD="$CMD --save-video"
fi

# Add output directory
CMD="$CMD --output-dir=$OUTPUT_DIR"

# Add force CPU option
if [ "$FORCE_CPU" = "true" ]; then
    CMD="$CMD --force-cpu"
fi

# Print command
echo "Running: $CMD"

# Run the command
eval "$CMD" 