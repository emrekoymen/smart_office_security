#!/usr/bin/env python3
"""
Smart Person Detection for macOS

This is a modified version of the original smart_detector.py
that works without the Coral TPU and uses TensorFlow instead.
"""

import argparse
import time
import os
import cv2
import numpy as np
from PIL import Image
import sys
import tensorflow as tf

# Add parent directory to sys.path to find the root utils.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Set environment variable to force OpenCV to run without errors
os.environ["OPENCV_VIDEOIO_MUTE_WARNINGS"] = "1"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Local imports
from utils import (
    FPSCounter, 
    Logger, 
    AlertSystem,
    get_video_source,
    save_detection_image,
    draw_detection_box
)

# Flag to track that we're using CPU only
USING_TPU = False

def check_display_availability():
    """Check if display is available and configure OpenCV backend."""
    try:
        test_window = cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("Test")
        print("Display is available!")
        return True
    except Exception as e:
        print(f"Display not available: {e}")
        return False

def load_labels(path):
    """Loads labels from a file."""
    with open(path, 'r') as f:
        lines = f.readlines()
    return {i: line.strip() for i, line in enumerate(lines)}

def load_model(model_path, labels_path, force_cpu=False):
    """Load the detection model and labels."""
    print(f"Loading model from {model_path}")
    
    # Use TensorFlow to load the model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    print(f"Loading labels from {labels_path}")
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
    
    # Resize and preprocess the image (needed for model input)
    resized_frame = cv2.resize(frame, (model_width, model_height))
    
    # Convert to RGB if needed
    if resized_frame.shape[2] == 3:
        # Convert BGR to RGB
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Define detection classes
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
    
    # Manual TFLite processing
    # Add batch dimension
    input_data = np.expand_dims(resized_frame, axis=0).astype(np.uint8)
    
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
            # Bounding box coordinates are normalized (0.0 to 1.0)
            # Scale them to the ORIGINAL frame size
            ymin = boxes[i][0] * orig_height
            xmin = boxes[i][1] * orig_width
            ymax = boxes[i][2] * orig_height
            xmax = boxes[i][3] * orig_width
            
            # Add detection with scaled coordinates
            detections.append(Detection(
                [ymin, xmin, ymax, xmax],
                int(classes[i]),
                scores[i]
            ))
    
    return detections

def main():
    parser = argparse.ArgumentParser(description='Smart person detection system')
    parser.add_argument('--model', default='models/ssd_mobilenet_v2_coco_quant_postprocess.tflite',
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
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load model and labels
    interpreter, labels = load_model(args.model, args.labels, args.force_cpu)
    
    # Initialize video source
    cap = get_video_source(args.source)
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original video FPS reported: {fps if fps > 0 else 'N/A'}")

    # Target FPS for processing
    TARGET_FPS = 10
    TARGET_FRAME_TIME = 1.0 / TARGET_FPS if TARGET_FPS > 0 else 0
    print(f"Targeting {TARGET_FPS} FPS (frame time: {TARGET_FRAME_TIME*1000:.2f} ms)")

    # Create video writer if saving is enabled
    if args.save_video:
        # Use TARGET_FPS for the output video if throttling
        output_fps = TARGET_FPS if TARGET_FPS > 0 else fps
        # Create output filename based on source
        if args.source.isdigit():
            source_name = f"camera{args.source}"
        else:
            source_name = os.path.basename(args.source).split('.')[0]
        
        output_filename = f"{source_name}_processed_{int(time.time())}.mp4"
        output_path = os.path.join(args.output_dir, output_filename)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_width, frame_height))
        print(f"Saving processed video to: {output_path} at {output_fps:.2f} FPS")
    
    # Initialize FPS counter, logger, and alert system
    fps_counter = FPSCounter()
    logger = Logger()
    alert_system = AlertSystem(min_confidence=args.threshold)
    
    # Stats tracking
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    
    print(f"Starting detection with threshold: {args.threshold}")
    print(f"Processing mode: CPU (TensorFlow)")
    
    try:
        while True:
            loop_start_time = time.time() # Time the start of the loop iteration

            # Read frame
            ret, frame = cap.read()
            if not ret:
                # End of video or camera disconnected
                print("End of video stream")
                break
            
            frame_count += 1

            # Update FPS counter (based on processed frames)
            fps_counter.update()

            # Process frame
            process_start_time = time.time() # Time the processing
            detections = process_image(interpreter, frame, args.threshold)
            process_end_time = time.time()
            
            # Track detections
            person_detected = False
            
            # Process detections
            for det in detections:
                # Only interested in the person class (typically ID 0)
                if det.id == args.person_class_id:
                    person_detected = True
                    detection_count += 1
                    
                    # Log detection
                    label = labels.get(det.id, f"Class {det.id}")
                    logger.log_detection(label, det.score)
                    
                    # Draw detection on frame
                    frame = draw_detection_box(frame, det.bbox, label, det.score)
                    
                    # Trigger alert system if confidence is high enough
                    alert_system.trigger_alert(det.score, label)
            
            # Add additional info to the frame
            current_fps = fps_counter.get_fps() # FPS of processed frames
            processing_time_ms = (process_end_time - process_start_time) * 1000
            cv2.putText(frame, f"Proc. FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: CPU", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Proc. Time: {processing_time_ms:.1f} ms", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame if enabled
            if not args.no_display:
                cv2.imshow('Smart Person Detection', frame)
                
                # Check for 'q' key to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Save frame if video saving is enabled
            if args.save_video:
                # We write the potentially modified frame (with boxes)
                writer.write(frame)
            
            # --- Time-based Throttling --- 
            loop_end_time = time.time()
            elapsed_time = loop_end_time - loop_start_time
            wait_time = TARGET_FRAME_TIME - elapsed_time
            
            if wait_time > 0:
                time.sleep(wait_time)
            # -----------------------------

    except KeyboardInterrupt:
        print("Detection stopped by user")
    
    except Exception as e:
        print(f"Error during detection: {e}")
    
    finally:
        # Clean up
        cap.release()
        if args.save_video:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Print summary
        duration = time.time() - start_time
        avg_proc_fps = frame_count / duration if duration > 0 else 0
        print("\nDetection Summary:")
        print(f"Total frames processed: {frame_count}")
        print(f"Person detections: {detection_count}")
        print(f"Average Processing FPS: {avg_proc_fps:.2f}")
        print(f"Total duration: {duration:.2f} seconds")
        
        logger.log_performance(avg_proc_fps, duration)

if __name__ == "__main__":
    main()
