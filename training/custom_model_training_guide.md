# Custom MobileNet SSD Model Training Guide

This guide outlines the process for training your own custom object detection model with a MobileNet SSD architecture using the TensorFlow Object Detection (TFOD) API and converting it for use, potentially with an Edge TPU.

## Training Process Overview (using TFOD API)

Training a custom model from scratch or fine-tuning a pre-trained one involves several steps using the official TFOD API.

1.  **Setup:**
    *   Install the TensorFlow Object Detection API and its dependencies.
    *   This typically involves cloning the `tensorflow/models` repository from GitHub and running provided installation scripts (`pip install .` within the `models/research` directory after compiling protos). Refer to the [official TFOD installation guide](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md) for detailed instructions.

2.  **Dataset Preparation:**
    *   Convert your custom dataset (images and annotations) into the TFRecord format. TFOD requires this specific format for efficient data loading.
    *   Annotations should include bounding box coordinates (xmin, ymin, xmax, ymax) and class labels for each object in your images.
    *   Tools and scripts are often needed to convert common annotation formats (like COCO, Pascal VOC, or labelImg XML) to TFRecord. The TFOD API repository sometimes includes helper scripts.

3.  **Label Map:**
    *   Create a `label_map.pbtxt` file.
    *   This file maps your class names (e.g., 'person', 'car') to unique integer IDs that the model will learn to predict.
    *   Example format:
        ```protobuf
        item {
          id: 1
          name: 'your_class_1'
        }
        item {
          id: 2
          name: 'your_class_2'
        }
        # ... and so on
        ```

4.  **Configuration:**
    *   Choose a pre-trained MobileNet SSD model checkpoint from the [TensorFlow 2 Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_model_zoo.md) (e.g., `ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8`). Using a pre-trained model allows for *fine-tuning*, which is usually much faster and requires less data than training from scratch.
    *   Download the chosen model's checkpoint files (`.ckpt`) and the corresponding `pipeline.config` file.
    *   **Modify the `pipeline.config`:**
        *   Update paths to point to your training/evaluation TFRecord files and your `label_map.pbtxt`.
        *   Set `num_classes` to the number of classes in your custom dataset.
        *   Specify the `fine_tune_checkpoint` path to the downloaded pre-trained checkpoint.
        *   Set `fine_tune_checkpoint_type` typically to `"detection"`.
        *   Adjust training parameters like `batch_size`, `learning_rate`, and the total number of training `num_steps`.
        *   Ensure input image size (`fixed_shape_resizer`) matches the chosen model's expectations (e.g., 640x640).

5.  **Training:**
    *   Use the `model_main_tf2.py` script provided by the TFOD API to start the training process.
    *   Command example:
        ```bash
        python model_main_tf2.py \
            --model_dir=training/my_custom_model_checkpoints \
            --pipeline_config_path=training/my_custom_model.config \
            --num_train_steps=50000 \
            --alsologtostderr
        ```
    *   Replace paths and `num_train_steps` as appropriate. Training can take a significant amount of time depending on your dataset size, hardware, and configuration. Monitor the process using TensorBoard (`tensorboard --logdir=training/my_custom_model_checkpoints`).

6.  **Export:**
    *   Once training is complete (or satisfactory), use the `exporter_main_v2.py` script from the TFOD API.
    *   This script converts the trained checkpoint into a deployable TensorFlow SavedModel format, which includes the necessary inference signatures.
    *   Command example:
        ```bash
        python exporter_main_v2.py \
            --input_type image_tensor \
            --pipeline_config_path training/my_custom_model.config \
            --trained_checkpoint_dir training/my_custom_model_checkpoints \
            --output_directory training/exported-models/my_custom_model
        ```
    *   This will create a `saved_model` subdirectory within `training/exported-models/my_custom_model`.

## Conversion to TFLite

After exporting the SavedModel, use the provided `training/convert_to_tflite.py` script.

*   **How to use:**
    ```bash
    python training/convert_to_tflite.py \
        --saved_model_dir training/exported-models/my_custom_model/saved_model/ \
        --output_path models/my_custom_model.tflite
    ```
*   This script takes the `saved_model` directory generated in the previous step and outputs a standard TensorFlow Lite model (`.tflite`).

## Next Steps for Edge TPU Compatibility

To run the model efficiently on a Google Coral Edge TPU device, further steps are required:

1.  **Quantization:**
    *   Edge TPUs require models to be quantized, typically using full integer quantization (INT8).
    *   Modify the `training/convert_to_tflite.py` script to include quantization steps. This involves:
        *   Setting `converter.optimizations = [tf.lite.Optimize.DEFAULT]`.
        *   Providing a representative dataset (a small subset of your training or validation data) to calibrate the quantization ranges. This is done via `converter.representative_dataset`.
        *   Specifying the target operations and input/output types (usually `tf.int8` or `tf.uint8`).
    *   Save the quantized model (e.g., `models/my_custom_model_quant.tflite`).

2.  **Compilation:**
    *   Use the Edge TPU Compiler tool (installable via `pip install edgetpu-compiler`) on the *quantized* `.tflite` model.
    *   Command:
        ```bash
        edgetpu_compiler models/my_custom_model_quant.tflite
        ```
    *   This produces a `.tflite` file appended with `_edgetpu` (e.g., `models/my_custom_model_quant_edgetpu.tflite`). This file is specifically compiled to run on the Edge TPU accelerator.

You can now use the `_edgetpu.tflite` model file in your application code targeting the Coral device. 