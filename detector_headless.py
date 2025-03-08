#!/usr/bin/env python3
"""
Headless Person Detection with Google Coral Edge TPU

This script uses a pre-trained MobileNet SSD model to detect persons in a video stream or file.
It runs in completely headless mode without requiring a display.
"""

import argparse
import time
import os
import cv2
import numpy as np
from PIL import Image
import sys

# Set environment variable to force OpenCV to run without display
os.environ["OPENCV_VIDEOIO_MUTE_WARNINGS"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Try to import Coral libraries
try:
    from pycoral.adapters import common
    from pycoral.adapters import detect
    from pycoral.utils.dataset import read_label_file
    from pycoral.utils.edgetpu import make_interpreter
    CORAL_AVAILABLE = True
except ImportError:
    print("Warning: pycoral not found, using tflite_runtime only")
    from tflite_runtime.interpreter import Interpreter
    CORAL_AVAILABLE = False

# Flag to track whether we're using the Edge TPU or CPU
USING_TPU = False

def load_labels(path):
    """Loads labels from a file."""
    with open(path, 'r') as f:
        lines = f.readlines()
    return {i: line.strip() for i, line in enumerate(lines)}


def load_model(model_path, labels_path):
    """Load the detection model and labels."""
    print(f"Loading model from {model_path}")
    global USING_TPU
    
    if CORAL_AVAILABLE:
        try:
            # Try to load the Edge TPU model
            interpreter = make_interpreter(model_path)
            interpreter.allocate_tensors()
            print("Successfully loaded Edge TPU model")
            USING_TPU = True
        except ValueError as e:
            # If Edge TPU is not available, fall back to regular TFLite model
            if "edgetpu" in model_path:
                cpu_model_path = model_path.replace("_edgetpu", "")
                print(f"Edge TPU not available: {e}")
                print(f"Falling back to CPU model: {cpu_model_path}")
                from tflite_runtime.interpreter import Interpreter
                interpreter = Interpreter(model_path=cpu_model_path)
                interpreter.allocate_tensors()
                USING_TPU = False
    else:
        # Use TFLite runtime directly
        if "edgetpu" in model_path:
            cpu_model_path = model_path.replace("_edgetpu", "")
            print(f"Using CPU model: {cpu_model_path}")
            model_path = cpu_model_path
        
        interpreter = Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        USING_TPU = False
    
    print(f"Loading labels from {labels_path}")
    try:
        if CORAL_AVAILABLE:
            labels = read_label_file(labels_path)
        else:
            labels = load_labels(labels_path)
    except:
        # Fallback for loading labels
        labels = load_labels(labels_path)
    
    return interpreter, labels


def process_image(interpreter, frame, threshold=0.5):
    """Process a single frame with the interpreter."""
    # Get input details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    height, width = input_shape[1], input_shape[2]
    
    # Resize and preprocess the image
    resized_frame = cv2.resize(frame, (width, height))
    
    # Convert to RGB if needed and normalize
    if resized_frame.shape[2] == 3:
        # Convert BGR to RGB
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Add batch dimension
    input_data = np.expand_dims(resized_frame, axis=0)
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    if CORAL_AVAILABLE and USING_TPU:
        # Use pycoral for Edge TPU
        return detect.get_objects(interpreter, threshold)
    else:
        # Process TFLite output manually
        # This is a simplified implementation - adjust for your model's specific output format
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        
        detections = []
        for i in range(len(scores)):
            if scores[i] >= threshold:
                # Create detection object (compatible with pycoral format)
                class Detection:
                    def __init__(self, bbox, id, score):
                        self.bbox = bbox
                        self.id = id
                        self.score = score
                
                # Normalize coordinates
                ymin, xmin, ymax, xmax = boxes[i]
                
                # Add detection
                detections.append(Detection([ymin, xmin, ymax, xmax], int(classes[i]), scores[i]))
        
        return detections


def draw_detection_box(frame, bbox, label, score):
    """Draw bounding box and label on the frame."""
    ymin, xmin, ymax, xmax = bbox
    
    # Convert normalized coordinates to pixel values
    height, width, _ = frame.shape
    xmin = int(xmin * width)
    xmax = int(xmax * width)
    ymin = int(ymin * height)
    ymax = int(ymax * height)
    
    # Calculate box dimensions
    box_width = xmax - xmin
    box_height = ymax - ymin
    
    # Draw bounding box with thicker lines
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    
    # Prepare label text with confidence score
    label_text = f"{label}: {score:.2f}"
    
    # Get text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
    
    # Draw background rectangle for text
    cv2.rectangle(frame, 
                 (xmin, ymin - text_height - 10), 
                 (xmin + text_width + 10, ymin), 
                 (0, 0, 0), 
                 -1)  # -1 means filled rectangle
    
    # Draw label text on the background rectangle
    cv2.putText(frame, 
                label_text, 
                (xmin + 5, ymin - 5), 
                font, 
                font_scale, 
                (0, 255, 0), 
                font_thickness)
    
    return frame


def main():
    parser = argparse.ArgumentParser(description='Person detection (headless)')
    parser.add_argument('--model', default='models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
                        help='Path to the detection model')
    parser.add_argument('--labels', default='models/coco_labels.txt',
                        help='Path to the labels file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--source', default='0',
                        help='Source video file or camera index (default: webcam)')
    parser.add_argument('--output-dir', default='output',
                        help='Directory to save output frames (default: output/)')
    parser.add_argument('--save-all', action='store_true',
                        help='Save all frames, not just ones with detections')
    parser.add_argument('--person-class-id', type=int, default=0,
                        help='Class ID for "person" (default: 0 for COCO)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        print("Please run setup_coral.sh to download the model files")
        return
    
    # Load model and labels
    interpreter, labels = load_model(args.model, args.labels)
    
    # Open video source
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
    
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {args.source}")
    except Exception as e:
        print(f"Error opening video source: {e}")
        return
    
    print(f"Starting detection with threshold: {args.threshold}")
    
    # Get person class label
    person_label = labels.get(args.person_class_id, "Person")
    print(f"Looking for objects of class: {person_label} (ID: {args.person_class_id})")
    
    # Frame counter for saving images
    frame_count = 0
    total_detections = 0
    
    # Track processing stats
    start_time = time.time()
    fps_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Increment frame counter
            frame_count += 1
            fps_count += 1
            
            # Process the frame
            frame_start = time.time()
            detections = process_image(interpreter, frame, args.threshold)
            
            # Filter for person detections
            person_detections = [d for d in detections if d.id == args.person_class_id]
            
            # Process detections
            for detection in person_detections:
                # Draw detection on frame
                frame = draw_detection_box(
                    frame, 
                    detection.bbox, 
                    person_label, 
                    detection.score
                )
                total_detections += 1
                
                # Print detection details
                print(f"Frame {frame_count}: Detected {person_label} with confidence {detection.score:.2f}")
            
            # Calculate current FPS
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                if elapsed > 0:
                    fps = fps_count / elapsed
                    print(f"Processing at {fps:.2f} FPS")
                    fps_count = 0
                    start_time = time.time()
            
            # Draw mode text on frame
            mode_text = "TPU" if USING_TPU else "CPU"
            cv2.putText(frame, f"Mode: {mode_text}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frames
            if args.save_all or len(person_detections) > 0:
                output_path = os.path.join(args.output_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(output_path, frame)
                if len(person_detections) > 0:
                    print(f"Saved detection to {output_path}")
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        # Clean up
        cap.release()
        
        # Print summary
        print(f"\nProcessing complete:")
        print(f"Total frames processed: {frame_count}")
        print(f"Total person detections: {total_detections}")
        print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main() 