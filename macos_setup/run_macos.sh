#!/bin/bash
# Script to run the smart person detection system on macOS within macos_setup

# Define the conda environment name
CONDA_ENV_NAME="smart_office_macos"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: Conda is not installed or not in PATH."
    echo "Please install Miniconda or Anaconda."
    exit 1
fi

# Check if the conda environment exists
if ! conda env list | grep -q "$CONDA_ENV_NAME"; then
    echo "Conda environment '$CONDA_ENV_NAME' does not exist."
    echo "Please create it first using the environment.yml file:"
    echo "conda env create -f macos_setup/environment.yml"
    exit 1
fi

# Activate the conda environment
echo "Activating conda environment: $CONDA_ENV_NAME"
# Use conda run for better isolation and scriptability
# This avoids issues with `conda activate` in scripts
# source activate $CONDA_ENV_NAME # Avoid this in scripts

# Default values
VIDEO_SOURCE="0"  # Default to webcam
THRESHOLD="0.5"
# Use relative path from project root for output
OUTPUT_DIR="../output" # Changed from "output"
ENABLE_DISPLAY=true  # Default to display mode
SAVE_VIDEO=false  # Default to not saving video
# Use relative path from project root for models
MODEL_PATH="../models/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
LABEL_PATH="../models/coco_labels.txt"

# Display help information
show_help() {
    echo "Usage: $(basename $0) [options]" # Use basename for script name
    echo "This script should be run from the 'macos_setup' directory."
    echo "Options:"
    echo "  -h, --help            Show this help message"
    echo "  -v, --video FILE      Use a video file (relative to project root, e.g., ../videos/sample.mp4)"
    echo "  -c, --camera NUM      Use a specific camera number (default: 0)"
    echo "  -t, --threshold NUM   Set detection threshold (0.0-1.0, default: 0.5)"
    echo "  -n, --no-display      Disable display (headless mode)"
    echo "  -s, --save            Save processed video to '$OUTPUT_DIR'"
    echo "  -o, --output DIR      Directory to save output files (relative to project root, default: ../output/)"
    echo
    echo "Examples (run from macos_setup/):"
    echo "  ./run_macos.sh                    Run with webcam, display enabled"
    echo "  ./run_macos.sh -v ../videos/sample.mp4    Run with specific video file"
    echo "  ./run_macos.sh -n -s             Run in headless mode and save video"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -v|--video)
            VIDEO_SOURCE="$2" # Keep relative path from project root
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
            OUTPUT_DIR="$2" # Keep relative path from project root
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Ensure models exist relative to project root
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file not found at $MODEL_PATH."
    echo "Ensure models are present in the '../models' directory relative to the project root."
    # echo "Attempting to run setup script (if applicable)..."
    # Potentially call setup_macos.sh here if it handles model downloads
    # ./setup_macos.sh # Make sure this script handles paths correctly if called
    exit 1
fi
if [ ! -f "$LABEL_PATH" ]; then
    echo "Label file not found at $LABEL_PATH."
    echo "Ensure labels are present in the '../models' directory relative to the project root."
    exit 1
fi


# Create output directory if saving video (relative to project root)
if $SAVE_VIDEO; then
    mkdir -p "$OUTPUT_DIR"
fi

# Build command using conda run
# Note: Paths inside the command should be relative to the script's location (macos_setup)
#       or absolute paths. We use paths relative to the project root for models/output.
#       The python script itself needs to handle these paths correctly.
CMD="conda run -n $CONDA_ENV_NAME python smart_detector_macos.py \
    --model "$MODEL_PATH" \
    --labels "$LABEL_PATH" \
    --source "$VIDEO_SOURCE" \
    --threshold $THRESHOLD \
    --output-dir "$OUTPUT_DIR"" # Pass relative-to-root paths

# Add display flag
if ! $ENABLE_DISPLAY; then
    CMD="$CMD --no-display"
fi

# Add save video flag
if $SAVE_VIDEO; then
    CMD="$CMD --save-video"
fi

# Run the detector
echo "Starting smart person detection on macOS..."
echo "Running from directory: $(pwd)"
echo "Using Conda environment: $CONDA_ENV_NAME"
echo "Video source: $VIDEO_SOURCE"
echo "Model: $MODEL_PATH"
echo "Labels: $LABEL_PATH"
echo "Threshold: $THRESHOLD"
echo "Display enabled: $ENABLE_DISPLAY"
echo "Save video: $SAVE_VIDEO"
if $SAVE_VIDEO; then
    echo "Output directory: $OUTPUT_DIR"
fi

# Make the python script executable (though `python script.py` works fine)
# chmod +x smart_detector_macos.py # Already moved

# Execute the command
echo "Executing command:"
echo "$CMD"
eval $CMD
