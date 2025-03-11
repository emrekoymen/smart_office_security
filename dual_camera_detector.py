#!/usr/bin/env python3
"""
Dual Camera Person Detection with Google Coral Edge TPU

This script uses a pre-trained MobileNet SSD model on the Coral Edge TPU
to detect persons in two simultaneous video streams.
It processes both camera feeds in parallel and displays them side-by-side.
"""

import argparse
import time
import os
import cv2
import numpy as np
import threading
from queue import Queue
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
    
    # Store original frame dimensions
    orig_height, orig_width = frame.shape[:2]
    
    # Force model input size to 300x300 for better performance
    model_height, model_width = 300, 300
    
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

class CameraProcessor:
    """Class to handle processing of a single camera feed."""
    
    def __init__(self, camera_id, interpreter, labels, person_class_id=0, threshold=0.5):
        self.camera_id = camera_id
        self.interpreter = interpreter
        self.labels = labels
        self.person_class_id = person_class_id
        self.threshold = threshold
        self.fps_counter = FPSCounter()
        self.cap = None
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=10)
        self.running = False
        self.thread = None
        self.processing_thread = None
        self.last_frame = None
        self.last_result = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        self.total_frames = 0
        self.total_detections = 0
    
    def open_camera(self, source):
        """Open the camera source."""
        try:
            self.cap = get_video_source(source)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0
            return True
        except ValueError as e:
            print(f"Error opening camera {self.camera_id}: {e}")
            return False
    
    def capture_frames(self):
        """Thread function to capture frames from the camera."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print(f"Camera {self.camera_id}: End of video stream")
                self.running = False
                break
            
            # If queue is full, remove oldest frame
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except:
                    pass
            
            # Add new frame to queue
            try:
                self.frame_queue.put(frame, block=False)
            except:
                pass
            
            # Store last frame for direct access
            self.last_frame = frame
            self.total_frames += 1
    
    def process_frames(self):
        """Thread function to process frames from the queue."""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1)
                
                # Process the frame
                frame_start_time = time.time()
                detections = process_image(self.interpreter, frame, self.threshold)
                
                # Filter for person detections
                person_detections = [d for d in detections if d.id == self.person_class_id]
                
                # Update FPS counter
                self.fps_counter.update()
                
                # Process detections
                for detection in person_detections:
                    # Draw detection on frame
                    frame = draw_detection_box(
                        frame, 
                        detection.bbox, 
                        f"{self.labels.get(detection.id, 'Unknown')}", 
                        detection.score
                    )
                
                # Calculate processing time
                processing_time = time.time() - frame_start_time
                
                # Draw FPS on frame
                frame = self.fps_counter.draw_fps(frame)
                
                # Draw camera ID on frame
                cv2.putText(frame, f"Camera {self.camera_id}", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Draw mode indicator (TPU or CPU)
                mode_text = "TPU" if USING_TPU else "CPU"
                cv2.putText(frame, f"Mode: {mode_text}", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Create result object
                result = {
                    'frame': frame,
                    'detections': person_detections,
                    'processing_time': processing_time,
                    'fps': self.fps_counter.get_fps()
                }
                
                # If queue is full, remove oldest result
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except:
                        pass
                
                # Add result to queue
                try:
                    self.result_queue.put(result, block=False)
                except:
                    pass
                
                # Store last result for direct access
                self.last_result = result
                
                # Update detection counter
                if len(person_detections) > 0:
                    self.total_detections += 1
                
            except Exception as e:
                if "queue.Empty" not in str(e):
                    print(f"Camera {self.camera_id} processing error: {e}")
    
    def start(self):
        """Start processing the camera feed."""
        if self.cap is None:
            print(f"Camera {self.camera_id} not opened")
            return False
        
        self.running = True
        
        # Start capture thread
        self.thread = threading.Thread(target=self.capture_frames)
        self.thread.daemon = True
        self.thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return True
    
    def stop(self):
        """Stop processing the camera feed."""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        
        if self.cap:
            self.cap.release()
    
    def get_latest_result(self):
        """Get the latest processed frame and detections."""
        try:
            return self.result_queue.get_nowait()
        except:
            return self.last_result

def main():
    parser = argparse.ArgumentParser(description='Dual camera person detection system')
    parser.add_argument('--model', default='models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
                        help='Path to the detection model')
    parser.add_argument('--labels', default='models/coco_labels.txt',
                        help='Path to the labels file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    parser.add_argument('--camera1', default='0',
                        help='First camera source (default: webcam)')
    parser.add_argument('--camera2', default='1',
                        help='Second camera source (default: USB camera)')
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
    
    # Get person class label
    person_label = labels.get(args.person_class_id, "Person")
    print(f"Looking for objects of class: {person_label} (ID: {args.person_class_id})")
    
    # Initialize utility classes
    logger = Logger()
    alert_system = AlertSystem(min_confidence=args.threshold)
    
    # Initialize camera processors
    camera1 = CameraProcessor(1, interpreter, labels, args.person_class_id, args.threshold)
    camera2 = CameraProcessor(2, interpreter, labels, args.person_class_id, args.threshold)
    
    # Open cameras
    print(f"Opening camera 1: {args.camera1}")
    if not camera1.open_camera(args.camera1):
        print("Failed to open camera 1")
        return
    
    print(f"Opening camera 2: {args.camera2}")
    if not camera2.open_camera(args.camera2):
        print("Failed to open camera 2")
        camera1.stop()
        return
    
    # Start camera processors
    camera1.start()
    camera2.start()
    
    # Initialize video writer if saving
    video_writer = None
    if args.save_video:
        # Use the first camera's dimensions for the combined output
        frame_width = camera1.frame_width * 2  # Side by side
        frame_height = max(camera1.frame_height, camera2.frame_height)
        
        # Generate output filename
        output_path = os.path.join(
            args.output_dir, 
            f"dual_camera_processed_{int(time.time())}.mp4"
        )
        
        # Define codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            30.0,  # Fixed FPS for output
            (frame_width, frame_height)
        )
        print(f"Saving processed video to: {output_path}")
    
    # Processing loop
    start_time = time.time()
    last_log_time = start_time
    
    try:
        while camera1.running and camera2.running:
            # Get latest results from both cameras
            result1 = camera1.get_latest_result()
            result2 = camera2.get_latest_result()
            
            # Skip if either result is None
            if result1 is None or result2 is None:
                time.sleep(0.01)  # Short sleep to prevent CPU hogging
                continue
            
            # Get frames from results
            frame1 = result1['frame']
            frame2 = result2['frame']
            
            # Get detections from results
            detections1 = result1['detections']
            detections2 = result2['detections']
            
            # Process alerts for both cameras
            for detection in detections1 + detections2:
                # Trigger alert
                if alert_system.trigger_alert(
                    detection.score, 
                    label=labels.get(detection.id, 'Unknown')
                ):
                    # Log detection
                    logger.log_detection(
                        labels.get(detection.id, 'Unknown'), 
                        detection.score
                    )
            
            # Resize frames to same height if different
            if frame1.shape[0] != frame2.shape[0]:
                # Resize the smaller frame to match the height of the larger one
                if frame1.shape[0] < frame2.shape[0]:
                    scale = frame2.shape[0] / frame1.shape[0]
                    new_width = int(frame1.shape[1] * scale)
                    frame1 = cv2.resize(frame1, (new_width, frame2.shape[0]))
                else:
                    scale = frame1.shape[0] / frame2.shape[0]
                    new_width = int(frame2.shape[1] * scale)
                    frame2 = cv2.resize(frame2, (new_width, frame1.shape[0]))
            
            # Combine frames side by side
            combined_frame = np.hstack((frame1, frame2))
            
            # Display if enabled and available
            if not args.no_display:
                try:
                    cv2.imshow('Dual Camera Person Detection', combined_frame)
                    
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
                video_writer.write(combined_frame)
            
            # Log performance occasionally
            current_time = time.time()
            if current_time - last_log_time >= 10:
                # Log performance for both cameras
                logger.log_performance(
                    camera1.fps_counter.get_fps(), 
                    result1['processing_time']
                )
                logger.log_performance(
                    camera2.fps_counter.get_fps(), 
                    result2['processing_time']
                )
                
                # Print TPU utilization (if available)
                if USING_TPU:
                    print(f"TPU processing two camera feeds at {camera1.fps_counter.get_fps():.1f} FPS and {camera2.fps_counter.get_fps():.1f} FPS")
                
                last_log_time = current_time
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        # Stop camera processors
        camera1.stop()
        camera2.stop()
        
        # Clean up
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        
        # Print summary
        elapsed_time = time.time() - start_time
        print(f"Camera 1: Processed {camera1.total_frames} frames, detected persons in {camera1.total_detections} frames")
        print(f"Camera 2: Processed {camera2.total_frames} frames, detected persons in {camera2.total_detections} frames")
        print(f"Average FPS: Camera 1: {camera1.total_frames / elapsed_time:.2f}, Camera 2: {camera2.total_frames / elapsed_time:.2f}")
        
        if args.save_video and video_writer is not None:
            print(f"Processed video saved to: {output_path}")

if __name__ == "__main__":
    main() 