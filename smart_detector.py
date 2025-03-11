#!/usr/bin/env python3
"""
Smart Person Detection with Google Coral Edge TPU

This script uses a pre-trained MobileNet SSD model on the Coral Edge TPU
to detect persons in a video stream or file.
It provides a single pipeline with options for display and video saving.
By default, it uses the Edge TPU if available, with automatic fallback to CPU.
"""

import argparse
import time
import os
import cv2
import numpy as np
from PIL import Image
import sys

# Set environment variable to force OpenCV to run without errors
os.environ["OPENCV_VIDEOIO_MUTE_WARNINGS"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Coral imports
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
USING_TPU = False

def check_display_availability():
    """Check if display is available and configure OpenCV backend."""
    # First try with default backend (Qt)
    try:
        test_window = cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Test")
        print("Display is available with default backend!")
        return True
    except Exception as e:
        print(f"Default backend failed: {e}")
        
        # Try with GTK backend
        try:
            os.environ['OPENCV_VIDEOIO_PRIORITY_BACKEND'] = '0'  # Force GTK
            test_window = cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("Test")
            print("Display is available with GTK backend!")
            return True
        except Exception as e:
            print(f"GTK backend also failed: {e}")
            return False

def load_labels(path):
    """Loads labels from a file."""
    with open(path, 'r') as f:
        lines = f.readlines()
    return {i: line.strip() for i, line in enumerate(lines)}

def load_model(model_path, labels_path, force_cpu=False):
    """Load the detection model and labels."""
    print(f"Loading model from {model_path}")
    global USING_TPU
    
    if CORAL_AVAILABLE and not force_cpu:
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
    input_shape = input_details[0]['shape']
    
    # Store original frame dimensions
    orig_height, orig_width = frame.shape[:2]
    model_height, model_width = input_shape[1], input_shape[2]
    
    # Calculate scale factors
    width_scale = orig_width / model_width
    height_scale = orig_height / model_height
    
    # Resize and preprocess the image
    resized_frame = cv2.resize(frame, (model_width, model_height))
    
    # Convert to RGB if needed and normalize
    if resized_frame.shape[2] == 3:
        # Convert BGR to RGB
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Define detection classes here so they're available for both paths
    class BBox:
        def __init__(self, ymin, xmin, ymax, xmax):
            self.ymin = ymin
            self.xmin = xmin
            self.ymax = ymax
            self.xmax = xmax
        
        def __repr__(self):
            return f"BBox(xmin={self.xmin}, ymin={self.ymin}, xmax={self.xmax}, ymax={self.ymax})"

    class Detection:
        def __init__(self, bbox, id, score):
            if isinstance(bbox, list):
                self.bbox = BBox(bbox[0], bbox[1], bbox[2], bbox[3])
            else:
                self.bbox = bbox
            self.id = id
            self.score = score
    
    if CORAL_AVAILABLE and USING_TPU:
        # Using Coral API
        common.set_input(interpreter, resized_frame)
        interpreter.invoke()
        coral_detections = detect.get_objects(interpreter, threshold)
        
        # Create new detections with scaled coordinates
        detections = []
        for det in coral_detections:
            # Scale the coordinates
            xmin = det.bbox.xmin * width_scale
            xmax = det.bbox.xmax * width_scale
            ymin = det.bbox.ymin * height_scale
            ymax = det.bbox.ymax * height_scale
            
            # Create new detection with scaled coordinates
            detections.append(Detection(
                [ymin, xmin, ymax, xmax],
                det.id,
                det.score
            ))
        
        return detections
    else:
        # Manual TFLite processing
        # Add batch dimension
        input_data = np.expand_dims(resized_frame, axis=0)
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        output_details = interpreter.get_output_details()
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
        
        detections = []
        for i in range(len(scores)):
            if scores[i] >= threshold:
                # Scale bounding box coordinates to original frame size
                ymin = boxes[i][0] * height_scale
                xmin = boxes[i][1] * width_scale
                ymax = boxes[i][2] * height_scale
                xmax = boxes[i][3] * width_scale
                
                # Add detection with scaled coordinates
                detections.append(Detection(
                    [ymin, xmin, ymax, xmax],
                    int(classes[i]),
                    scores[i]
                ))
        
        return detections

def main():
    parser = argparse.ArgumentParser(description='Smart person detection system')
    parser.add_argument('--model', default='models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
                        help='Path to the detection model')
    parser.add_argument('--labels', default='models/coco_labels.txt',
                        help='Path to the labels file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--source', default='0',
                        help='Source video file or camera index (default: webcam)')
    parser.add_argument('--output-dir', default='output',
                        help='Directory to save output files (default: output/)')
    parser.add_argument('--person-class-id', type=int, default=0,
                        help='Class ID for "person" (default: 0 for COCO)')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display output')
    parser.add_argument('--save-video', action='store_true',
                        help='Save processed video')
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU mode (no TPU)')
    
    args = parser.parse_args()
    
    # Check display availability after parsing arguments
    DISPLAY_AVAILABLE = False if args.no_display else check_display_availability()
    
    # If display is not available but was requested, adjust settings
    if not args.no_display and not DISPLAY_AVAILABLE:
        print("Display requested but not available. Running without display.")
        args.no_display = True
        # Enable video saving if display is not available
        if not args.save_video:
            print("Enabling video saving since display is not available.")
            args.save_video = True
    
    # Create output directory if saving video
    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        print("Please run setup_coral.sh to download the model files")
        return
    
    # Load model and labels
    interpreter, labels = load_model(args.model, args.labels, args.force_cpu)
    
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
    
    # Initialize utility classes
    fps_counter = FPSCounter()
    logger = Logger()
    alert_system = AlertSystem(min_confidence=args.threshold)
    
    # Get video properties for potential saving
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:  # If FPS is not available, default to 30
        fps = 30.0
    
    # Initialize video writer if saving
    video_writer = None
    if args.save_video:
        # Generate output filename based on source
        if args.source.isdigit():
            source_name = f"camera_{args.source}"
        else:
            source_name = os.path.splitext(os.path.basename(args.source))[0]
        
        output_path = os.path.join(
            args.output_dir, 
            f"{source_name}_processed_{int(time.time())}.mp4"
        )
        
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            fps, 
            (frame_width, frame_height)
        )
        print(f"Saving processed video to: {output_path}")
    
    # Processing loop
    frame_count = 0
    total_detections = 0
    start_time = time.time()
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("End of video stream")
                break
            
            # Start timing for this frame
            frame_start_time = time.time()
            
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
            processing_time = time.time() - frame_start_time
            
            # Draw FPS on frame
            frame = fps_counter.draw_fps(frame)
            
            # Draw mode indicator (TPU or CPU)
            mode_text = "TPU" if USING_TPU else "CPU"
            cv2.putText(frame, f"Mode: {mode_text}", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display if enabled and available
            if not args.no_display:
                try:
                    cv2.imshow('Person Detection', frame)
                    
                    # Exit on 'q' press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception as e:
                    print(f"Display error: {e}")
                    if not args.save_video:
                        print("No display available and video saving not enabled. Exiting.")
                        break
            
            # Save frame to video if enabled
            if args.save_video and video_writer is not None:
                video_writer.write(frame)
            
            # Log performance occasionally
            if int(time.time()) % 10 == 0:
                logger.log_performance(fps_counter.get_fps(), processing_time)
            
            # Update counters
            frame_count += 1
            if len(person_detections) > 0:
                total_detections += 1
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Clean up
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
        print(f"Average FPS: {frame_count / elapsed_time:.2f}")
        print(f"Detected persons in {total_detections} frames")
        
        if args.save_video and video_writer is not None:
            print(f"Processed video saved to: {output_path}")

if __name__ == "__main__":
    main() 