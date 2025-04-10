# Smart Office Security - macOS Setup Guide

This guide provides instructions for setting up and running the Smart Office Security system on macOS using the dedicated scripts within this `macos_setup` directory. Since the Coral Edge TPU is not supported on macOS, this setup uses TensorFlow for CPU-based inference.

## Prerequisites

1.  **Conda:** You need `conda` (Miniconda or Anaconda) installed and working in your terminal.
2.  **(Optional) Homebrew:** Can be helpful for managing other system dependencies if needed, but not strictly required by this setup process.

## Setup Steps

1.  **Navigate to the Project Root:** Ensure your terminal is in the main `smart_office_security` directory (the parent directory of this `macos_setup` folder).

2.  **Create and Activate Conda Environment:**
    Use the provided `environment.yml` file to create the necessary conda environment.
    ```bash
    # Create the environment (this might take a few minutes)
    conda env create -f macos_setup/environment.yml

    # Activate the environment
    conda activate smart_office_macos
    ```
    *Note: If you encounter issues creating the environment (e.g., `pyjnius` error), you might need to install `plyer` separately using pip after activating the environment: `pip install plyer` (and potentially `pip install pyobjus cython` for notifications).* 

3.  **Download Model Files (if not already present):**
    The system expects the TensorFlow Lite model and labels file to be in the `models` directory in the **project root**.
    ```bash
    # Create the models directory in the project root if it doesn't exist
    mkdir -p ../models

    # Navigate to the models directory (relative to macos_setup)
    cd ../models

    # Download TFLite model and labels (if you don't have them)
    echo "Downloading person detection model..."
    curl -L -o ssd_mobilenet_v2_coco_quant_postprocess.tflite https://github.com/google-coral/test_data/raw/master/ssd_mobilenet_v2_coco_quant_postprocess.tflite
    curl -L -o coco_labels.txt https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt

    # Return to the macos_setup directory
    cd ../macos_setup
    ```

## Running the Detection System

**Important:** Always run the detector from *within* the `macos_setup` directory after activating the conda environment.

1.  **Activate Environment:**
    ```bash
    conda activate smart_office_macos
    ```

2.  **Navigate to `macos_setup` Directory:**
    ```bash
    # If you are in the project root:
    cd macos_setup
    ```

3.  **Run the Script:**
    Use the `run_macos.sh` script.
    ```bash
    # Basic usage (webcam)
    ./run_macos.sh
    ```

4.  **Common Options:**
    Pass options to the `run_macos.sh` script:
    *   Use a specific camera: `./run_macos.sh -c 1`
    *   Use a video file (path relative to project root): `./run_macos.sh -v ../videos/sample.mp4`
    *   Set detection threshold: `./run_macos.sh -t 0.6`
    *   Save video output (to `output` dir in project root): `./run_macos.sh -s`
    *   Disable display: `./run_macos.sh -n`

    **Example:** Run with a video file, save output, no display:
    ```bash
    ./run_macos.sh -v ../videos/sample.mp4 -s -n
    ```

## Troubleshooting

1.  **Environment Issues:**
    *   Ensure `conda` is installed and in your PATH.
    *   Make sure the `smart_office_macos` environment is activated (`conda activate smart_office_macos`).
    *   If environment creation fails, try removing `plyer` from `environment.yml` and installing it via `pip` after activation.

2.  **Camera Issues:**
    *   Verify camera permissions are granted to your Terminal/IDE in **System Settings > Privacy & Security > Camera**.
    *   Ensure no other app is using the camera.
    *   Try different camera indices (`-c 0`, `-c 1`, etc.).

3.  **Model Not Found:**
    *   Ensure the `.tflite` model and `.txt` labels file are present in the `models` directory **at the project root** (`smart_office_security/models/`).

4.  **Display Issues / `pyobjus` Error:**
    *   The `pyobjus` error occurs if `plyer` cannot find its backend for macOS notifications. Run `pip install pyobjus cython` in the activated environment to fix.
    *   If you don't need notifications, you can ignore the `pyobjus` error at the end.
    *   If graphical display windows fail, try running with the `--no-display` or `-n` flag.

## Notes

*   This version uses CPU-only inference via TensorFlow.
*   Performance depends heavily on your Mac's CPU.
*   Detection logs are saved in the `logs` directory **at the project root**.
*   Processed videos are saved in the `output` directory **at the project root** when using `--save-video` or `-s`.

## Support

For issues specific to the macOS version:
1. Check the logs in the `logs` directory
2. Verify all dependencies are correctly installed
3. Ensure you're using the latest version of the code
4. Submit an issue on the project's GitHub repository 