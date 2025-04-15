import tensorflow as tf
import argparse
import os

def convert_saved_model_to_tflite(saved_model_dir, output_path):
    """Converts a TensorFlow SavedModel to a TFLite model.

    Args:
        saved_model_dir: Path to the directory containing the SavedModel.
        output_path: Path where the .tflite file will be saved.
    """
    print(f"Converting SavedModel from: {saved_model_dir}")
    print(f"Output TFLite path: {output_path}")

    if not os.path.isdir(saved_model_dir):
        print(f"Error: SavedModel directory not found at {saved_model_dir}")
        return

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Enable optimizations if needed (e.g., float16 quantization)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16] # Example for float16

    tflite_model = converter.convert()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Successfully converted model and saved to {output_path}")
    print(f"TFLite model size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TensorFlow SavedModel to TFLite.")
    parser.add_argument(
        "--saved_model_dir",
        required=True,
        help="Path to the SavedModel directory (e.g., 'exported-models/my_model/saved_model/')."
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to save the output .tflite file (e.g., 'models/custom_model.tflite')."
    )

    args = parser.parse_args()

    convert_saved_model_to_tflite(args.saved_model_dir, args.output_path) 