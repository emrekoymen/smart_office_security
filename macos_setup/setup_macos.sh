#!/bin/bash
# Setup script for Smart Office Security on macOS (within macos_setup directory)
# This script primarily focuses on ensuring models are downloaded to the correct location.
# Environment setup should be done using conda and environment.yml as per macos_setup.md.

echo "Running macOS Setup Helper (within macos_setup)..."

# Define relative path to project root model directory
MODEL_DIR="../models"
MODEL_PATH="$MODEL_DIR/ssd_mobilenet_v2_coco_quant_postprocess.tflite"
LABEL_PATH="$MODEL_DIR/coco_labels.txt"

# Create the model directory in the project root if it doesn't exist
echo "Ensuring model directory exists at $MODEL_DIR..."
mkdir -p "$MODEL_DIR"

# Check and download the person detection model if missing
if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file not found at $MODEL_PATH. Downloading..."
    curl -L -o "$MODEL_PATH" https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite
    if [ $? -ne 0 ]; then
        echo "Error downloading model file. Please check your internet connection or the URL."
        exit 1
    fi
else
    echo "Model file already exists at $MODEL_PATH."
fi

# Check and download the label file if missing
if [ ! -f "$LABEL_PATH" ]; then
    echo "Label file not found at $LABEL_PATH. Downloading..."
    curl -L -o "$LABEL_PATH" https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt
    if [ $? -ne 0 ]; then
        echo "Error downloading label file. Please check your internet connection or the URL."
        exit 1
    fi
else
    echo "Label file already exists at $LABEL_PATH."
fi

# Remove legacy dependency installation (should use conda env create)
# echo "Installing Python dependencies..."
# pip install opencv-python numpy==1.24.3 Pillow==9.5.0 tqdm==4.65.0 matplotlib==3.7.1 plyer==2.1.0 tensorflow-macos

# Remove legacy script overwrite (Script should be maintained manually or via git)
# echo "Creating macOS-compatible detector..."
# cat > smart_detector_macos.py << EOF ... EOF

echo "---------------------------------------------------"
echo "macOS Setup Helper Complete."
echo "Models checked/downloaded to: $MODEL_DIR"
echo "Remember to create/activate the conda environment:"
echo "  conda env create -f macos_setup/environment.yml"
echo "  conda activate smart_office_macos"
echo "Then run the detector from within macos_setup:"
echo "  ./run_macos.sh"
echo "---------------------------------------------------"

exit 0 