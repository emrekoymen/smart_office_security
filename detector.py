#!/usr/bin/env python3
"""
Person Detection with Google Coral Edge TPU

This script uses a pre-trained MobileNet SSD model on the Coral Edge TPU
to detect persons in a video stream or file.
If no Edge TPU is available, it falls back to using the CPU with the regular TFLite model.
"""

import argparse
import time
import os
import cv2
import numpy as np
from PIL import Image
import sys

# Coral imports
try:
    from pycoral.adapters import common
    from pycoral.adapters import detect
    from pycoral.utils.dataset import read_label_file
    from pycoral.utils.edgetpu import make_interpreter
except ImportError:
    print("Warning: pycoral not found, using tflite_runtime only")
    from tflite_runtime.interpreter import Interpreter

# Local imports
from utils import (
    FPSCounter, 
    Logger, 
    AlertSystem,
    get_video_source,
    save_detection_image,
    draw_detection_box
)

# Flag to track whether we're using the Edge TPU or CPU
USING_TPU = True

# Check if display is available
DISPLAY_AVAILABLE = True
try:
    test_window = cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
    cv2.destroyWindow("Test")
except:
    DISPLAY_AVAILABLE = False
    print("Warning: Display not available. Running in headless mode.")

def load_labels(path):
    """Loads labels from a file."""
    with open(path, 'r') as f:
        lines = f.readlines()
    return {i: line.strip() for i, line in enumerate(lines)}


def load_model(model_path, labels_path):
    """Load the detection model and labels."""
    print(f"Loading model from {model_path}")
    global USING_TPU
    
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
    
    print(f"Loading labels from {labels_path}")
    try:
        labels = read_label_file(labels_path)
    except:
        # Fallback if read_label_file fails
        labels = load_labels(labels_path)
    
    return interpreter, labels


def process_image(interpreter, frame, threshold=0.5):
    """Process a single frame with the interpreter."""
    # Resize and convert the frame
    input_tensor_shape = interpreter.get_input_details()[0]['shape']
    _, input_height, input_width, _ = input_tensor_shape
    
    # Convert from BGR (OpenCV) to RGB (what the model expects)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to expected dimensions
    resized_frame = cv2.resize(rgb_frame, (input_width, input_height))
    
    # Run inference
    common.set_input(interpreter, resized_frame)
    interpreter.invoke()
    
    # Get detection results
    detections = detect.get_objects(interpreter, threshold)
    
    return detections


def main():
    parser = argparse.ArgumentParser(description='Person detection using Google Coral')
    parser.add_argument('--model', default='models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
                        help='Path to the detection model')
    parser.add_argument('--labels', default='models/coco_labels.txt',
                        help='Path to the labels file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--source', default='0',
                        help='Source video file or camera index (default: webcam)')
    parser.add_argument('--display', action='store_true',
                        help='Display the video feed with detections')
    parser.add_argument('--save', action='store_true',
                        help='Save frames with detections')
    parser.add_argument('--output-dir', default='output',
                        help='Directory to save output frames (default: output/)')
    parser.add_argument('--person-class-id', type=int, default=0,
                        help='Class ID for "person" (default: 0 for COCO)')
    
    args = parser.parse_args()
    
    # Override display flag if display is not available
    if args.display and not DISPLAY_AVAILABLE:
        print("Display requested but not available. Running without display.")
        args.display = False
    
    # Create output directory if saving frames
    if args.save:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        print("Please run setup_coral.sh to download the model files")
        return
    
    # Load model and labels
    interpreter, labels = load_model(args.model, args.labels)
    
    # Initialize utility classes
    fps_counter = FPSCounter()
    logger = Logger()
    alert_system = AlertSystem(min_confidence=args.threshold)
    
    # Open video source
    try:
        cap = get_video_source(args.source)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print(f"Starting detection with threshold: {args.threshold}")
    
    # Get person class label
    person_label = labels.get(args.person_class_id, "Person")
    print(f"Looking for objects of class: {person_label} (ID: {args.person_class_id})")
    
    # Frame counter for saving images
    frame_count = 0
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Start timing for this frame
            start_time = time.time()
            
            # Process the frame
            detections = process_image(interpreter, frame, args.threshold)
            
            # Filter for person detections
            person_detections = [d for d in detections if d.id == args.person_class_id]
            
            # Update FPS counter
            fps_counter.update()
            
            # Process detections
            for detection in person_detections:
                # Draw detection on frame
                frame = draw_detection_box(
                    frame, 
                    detection.bbox, 
                    f"{labels.get(detection.id, 'Unknown')}", 
                    detection.score
                )
                
                # Log detection
                logger.log_detection(
                    labels.get(detection.id, 'Unknown'), 
                    detection.score
                )
                
                # Trigger alert
                alert_system.trigger_alert(
                    detection.score, 
                    label=labels.get(detection.id, 'Unknown')
                )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log performance occasionally
            if int(time.time()) % 10 == 0:
                logger.log_performance(fps_counter.get_fps(), processing_time)
            
            # Draw FPS on frame
            frame = fps_counter.draw_fps(frame)
            
            # Determine if using Edge TPU or CPU based on the global flag
            mode_text = "TPU" if USING_TPU else "CPU"
            cv2.putText(frame, f"Mode: {mode_text}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Save frames with detections
            if args.save and len(person_detections) > 0:
                frame_filename = os.path.join(args.output_dir, f"detection_{frame_count:04d}.jpg")
                cv2.imwrite(frame_filename, frame)
                print(f"Saved detection to {frame_filename}")
                frame_count += 1
            
            # Display the frame if requested and available
            if args.display:
                try:
                    cv2.imshow('Person Detection', frame)
                    
                    # Exit on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"Error displaying frame: {e}")
                    args.display = False  # Disable display for future frames
    
    finally:
        # Clean up
        cap.release()
        if args.display:
            try:
                cv2.destroyAllWindows()
            except:
                pass
        
        print("Detection stopped")


if __name__ == '__main__':
    main() 