#!/bin/bash
# Script to run the dual camera detector

# Exit on error
set -e

# Default values
CAMERA1="0"
CAMERA2="2"
THRESHOLD="0.5"
DISPLAY="true"
SAVE_VIDEO="false"
OUTPUT_DIR="output"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --camera1=*)
      CAMERA1="${1#*=}"
      shift
      ;;
    --camera2=*)
      CAMERA2="${1#*=}"
      shift
      ;;
    --threshold=*)
      THRESHOLD="${1#*=}"
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
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --camera1=INDEX    First camera index or path (default: 0)"
      echo "  --camera2=INDEX    Second camera index or path (default: 2)"
      echo "  --threshold=VALUE  Detection threshold (default: 0.5)"
      echo "  --no-display       Disable display output"
      echo "  --save-video       Save processed video"
      echo "  --output-dir=DIR   Output directory (default: output)"
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

# Build command
CMD="python3 dual_camera_detector.py --camera1=$CAMERA1 --camera2=$CAMERA2 --threshold=$THRESHOLD"

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

# Print command
echo "Running: $CMD"

# Run the command
eval "$CMD" 