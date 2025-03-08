#!/bin/bash
# Setup script for Google Coral Edge TPU Runtime

echo "Setting up Google Coral Edge TPU Runtime..."

# Check for conda environment
if [[ "$CONDA_PREFIX" == *"smart_office"* ]]; then
    # Check Python version
    PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$PYTHON_VERSION" != "3.8" ]]; then
        echo "Error: Python version 3.8 is required, but found $PYTHON_VERSION"
        echo "Please recreate your conda environment with: conda create -n smart_office python=3.8"
        echo "Then activate it with: conda activate smart_office"
        exit 1
    fi
    echo "Conda environment 'smart_office' with Python 3.8 is active."
else
    echo "Conda environment 'smart_office' is not active."
    echo "Please run: conda create -n smart_office python=3.8"
    echo "Then activate with: conda activate smart_office"
    exit 1
fi

# Install dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y libusb-1.0-0

# Add Coral repository
echo "Adding Coral repository..."
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Update package list
sudo apt-get update

# Install Edge TPU runtime
echo "Installing Edge TPU runtime..."
sudo apt-get install -y libedgetpu1-std

# Create a directory for models if it doesn't exist
mkdir -p models

# Download the person detection model
echo "Downloading person detection model..."
wget -O models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite
wget -O models/ssd_mobilenet_v2_coco_quant_postprocess.tflite https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite
wget -O models/coco_labels.txt https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt

# Install pycoral using pip directly from the github release
echo "Installing pycoral via pip..."
pip install --no-deps https://github.com/google-coral/pycoral/releases/download/v2.0.0/pycoral-2.0.0-cp38-cp38-linux_x86_64.whl

# Install tflite_runtime
echo "Installing tflite_runtime..."
pip install --no-deps https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp38-cp38-linux_x86_64.whl

# Install other Python dependencies
echo "Installing other Python dependencies..."
grep -v "pycoral" requirements.txt > requirements_without_pycoral.txt
pip install -r requirements_without_pycoral.txt

echo "Setup completed! You can now run your person detection script." 