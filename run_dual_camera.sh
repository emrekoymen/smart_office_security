#!/bin/bash

# Run Dual Camera Person Detection System
# This script runs the dual camera detector with the Google Coral USB Accelerator

# Activate the Conda environment if available
if command -v conda &> /dev/null; then
    if conda env list | grep -q "smart_office"; then
        echo "Activating smart_office Conda environment..."
        eval "$(conda shell.bash hook)"
        conda activate smart_office
    else
        echo "Warning: smart_office Conda environment not found. Using system Python."
    fi
fi

# Check if the Coral USB Accelerator is connected
if lsusb | grep -q "1a6e:089a"; then
    echo "Google Coral USB Accelerator detected!"
else
    echo "Warning: Google Coral USB Accelerator not detected. Performance may be reduced."
fi

# Check if the model files exist
if [ ! -f "models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite" ]; then
    echo "Model files not found. Running setup script..."
    bash setup_coral.sh
fi

# Check if the output directory exists
if [ ! -d "output" ]; then
    mkdir -p output
fi

# Check if the logs directory exists
if [ ! -d "logs" ]; then
    mkdir -p logs
fi

# Default camera sources
CAMERA1="0"  # Default webcam
CAMERA2="2"  # Second camera (changed from 1 to 2 based on testing)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --camera1)
        CAMERA1="$2"
        shift
        shift
        ;;
        --camera2)
        CAMERA2="$2"
        shift
        shift
        ;;
        --no-display)
        NO_DISPLAY="--no-display"
        shift
        ;;
        --save-video)
        SAVE_VIDEO="--save-video"
        shift
        ;;
        --force-cpu)
        FORCE_CPU="--force-cpu"
        shift
        ;;
        --threshold)
        THRESHOLD="--threshold $2"
        shift
        shift
        ;;
        --help)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --camera1 SOURCE    First camera source (default: 0 for webcam)"
        echo "  --camera2 SOURCE    Second camera source (default: 2 for USB camera)"
        echo "  --no-display        Disable display output"
        echo "  --save-video        Save processed video"
        echo "  --force-cpu         Force CPU mode (no TPU)"
        echo "  --threshold VALUE   Detection threshold (default: 0.5)"
        echo "  --help              Show this help message"
        exit 0
        ;;
        *)
        echo "Unknown option: $1"
        echo "Use --help for usage information"
        exit 1
        ;;
    esac
done

# Print configuration
echo "Running dual camera detector with the following configuration:"
echo "  Camera 1: $CAMERA1"
echo "  Camera 2: $CAMERA2"
echo "  Display: ${NO_DISPLAY:+Disabled}"
echo "  Save Video: ${SAVE_VIDEO:+Enabled}"
echo "  Force CPU: ${FORCE_CPU:+Enabled}"
echo "  Threshold: ${THRESHOLD:+$THRESHOLD}"

# Run the dual camera detector
python3 dual_camera_detector.py \
    --camera1 "$CAMERA1" \
    --camera2 "$CAMERA2" \
    ${NO_DISPLAY} \
    ${SAVE_VIDEO} \
    ${FORCE_CPU} \
    ${THRESHOLD}

# Check exit status
if [ $? -ne 0 ]; then
    echo "Error: Dual camera detector exited with an error."
    exit 1
fi

echo "Dual camera detector completed successfully."
exit 0 