#!/usr/bin/env python3
"""
Dual Camera Person Detection with Google Coral Edge TPU

This script uses a pre-trained MobileNet SSD model on the Coral Edge TPU
to detect persons in two simultaneous video streams.
It processes both camera feeds in parallel and displays them side-by-side.
"""

import argparse
import cv2
import numpy as np
import os
import time
import threading
import queue
import logging
from datetime import datetime
from utils import FPSCounter, Logger, AlertSystem, draw_detection_box, save_detection_image

# Conditional import for Edge TPU
try:
    from pycoral.adapters import common
    from pycoral.adapters import detect
    from pycoral.utils.dataset import read_label_file
    from pycoral.utils.edgetpu import make_interpreter
    HAVE_CORAL = True
except ImportError:
    HAVE_CORAL = False
    print("Warning: PyCoral not found. Running in mock detection mode.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/dual_camera.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

class CameraProcessor:
    """
    Class to handle processing of a single camera feed
    """
    def __init__(self, camera_id, source, model_path, labels_path, threshold=0.5, person_class_id=0):
        self.camera_id = camera_id
        self.source = source
        self.model_path = model_path
        self.labels_path = labels_path
        self.threshold = threshold
        self.person_class_id = person_class_id
        
        self.cap = None
        self.frame_width = 0
        self.frame_height = 0
        self.fps = 0
        
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        self.capture_thread = None
        self.process_thread = None
        
        self.total_frames = 0
        self.total_detections = 0
        self.fps_counter = FPSCounter()
        
        # Initialize model
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize the detection model"""
        if HAVE_CORAL:
            try:
                self.interpreter = make_interpreter(self.model_path)
                self.interpreter.allocate_tensors()
                self.input_size = common.input_size(self.interpreter)
                self.labels = read_label_file(self.labels_path)
                logger.info(f"Camera {self.camera_id}: Model loaded successfully")
            except Exception as e:
                logger.error(f"Camera {self.camera_id}: Error loading model: {e}")
                self.interpreter = None
        else:
            # Mock interpreter for testing without Coral
            self.interpreter = None
            self.input_size = (300, 300)
            self.labels = {0: 'person'}
            logger.info(f"Camera {self.camera_id}: Using mock detection")
    
    def open_camera(self):
        """Open the camera source"""
        try:
            # If source is a digit string, convert to int for webcam
            if isinstance(self.source, str) and self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source))
            else:
                self.cap = cv2.VideoCapture(self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Camera {self.camera_id}: Could not open video source: {self.source}")
                return False
            
            # Get camera properties
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # If FPS is not available, use default
            if self.fps <= 0:
                self.fps = 30.0
            
            # Set resolution to 300x300 for better performance and consistency with model input
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
            
            # Update frame dimensions after setting
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"Camera {self.camera_id}: Opened {self.frame_width}x{self.frame_height} @ {self.fps} FPS")
            return True
        except Exception as e:
            logger.error(f"Camera {self.camera_id}: Error opening camera: {e}")
            return False
    
    def start(self):
        """Start processing the camera feed"""
        if not self.cap or not self.cap.isOpened():
            if not self.open_camera():
                return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.process_thread = threading.Thread(target=self.process_frames)
        
        self.capture_thread.daemon = True
        self.process_thread.daemon = True
        
        self.capture_thread.start()
        self.process_thread.start()
        
        logger.info(f"Camera {self.camera_id}: Processing started")
        return True
    
    def stop(self):
        """Stop processing the camera feed"""
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
        
        if self.process_thread:
            self.process_thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        
        logger.info(f"Camera {self.camera_id}: Processing stopped")
    
    def capture_frames(self):
        """Thread function to capture frames from the camera"""
        while self.running:
            if not self.cap or not self.cap.isOpened():
                logger.error(f"Camera {self.camera_id}: Camera disconnected")
                self.running = False
                break
            
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"Camera {self.camera_id}: Failed to read frame")
                time.sleep(0.1)
                continue
            
            # If frame queue is full, remove oldest frame
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass
    
    def process_frames(self):
        """Thread function to process frames from the queue"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            
            start_time = time.time()
            
            # Resize frame for detection
            detection_frame = cv2.resize(frame, self.input_size)
            
            # Run detection
            detections = self.detect_objects(detection_frame)
            
            # Filter for persons
            person_detections = [d for d in detections if d[0] == self.person_class_id]
            
            # Draw detection boxes
            result_frame = frame.copy()
            for detection in person_detections:
                class_id, score, bbox = detection
                label = self.labels.get(class_id, f"Class {class_id}")
                result_frame = draw_detection_box(result_frame, bbox, label, score)
            
            # Update FPS counter
            self.fps_counter.update()
            fps = self.fps_counter.get_fps()
            
            # Add FPS text to frame
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add camera ID to frame
            cv2.putText(result_frame, f"Camera {self.camera_id}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add TPU/CPU mode indicator
            mode = "TPU" if HAVE_CORAL and self.interpreter else "CPU"
            cv2.putText(result_frame, f"Mode: {mode}", (10, 110), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update counters
            self.total_frames += 1
            self.total_detections += len(person_detections)
            
            # Log detections
            if person_detections:
                logger.info(f"Camera {self.camera_id}: Detected {len(person_detections)} persons")
            
            # Create result object
            result = {
                'frame': result_frame,
                'detections': person_detections,
                'processing_time': processing_time,
                'fps': fps
            }
            
            # If result queue is full, remove oldest result
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.result_queue.put(result, block=False)
            except queue.Full:
                pass
    
    def detect_objects(self, frame):
        """Detect objects in the frame"""
        if not HAVE_CORAL or not self.interpreter:
            # Mock detection for testing without Coral
            # Return random detections occasionally
            if np.random.random() < 0.3:
                # Random box coordinates
                ymin = np.random.uniform(0.2, 0.8)
                xmin = np.random.uniform(0.2, 0.8)
                ymax = min(1.0, ymin + np.random.uniform(0.1, 0.3))
                xmax = min(1.0, xmin + np.random.uniform(0.1, 0.3))
                
                # Random score
                score = np.random.uniform(0.6, 0.9)
                
                return [(self.person_class_id, score, (ymin, xmin, ymax, xmax))]
            return []
        
        # Actual detection with Coral
        common.set_input(self.interpreter, frame)
        self.interpreter.invoke()
        return detect.get_objects(self.interpreter, self.threshold)
    
    def get_latest_result(self):
        """Get the latest processed result"""
        try:
            return self.result_queue.get(block=False)
        except queue.Empty:
            return None


class DualCameraDetector:
    """Main class for dual camera person detection"""
    def __init__(self, args):
        self.args = args
        self.camera1 = None
        self.camera2 = None
        self.video_writer = None
        self.video_writer_initialized = False
        self.display_available = self.check_display_availability()
        self.running = True
        
        # Initialize logger and alert system
        self.logger = Logger(log_dir=os.path.join(args.output_dir, "logs"))
        self.alert_system = AlertSystem(min_confidence=args.threshold)
        
        # Register signal handler
        import signal
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle interrupt signal"""
        logger.info(f"Interrupt signal ({signum}) received. Exiting...")
        self.running = False
    
    def check_display_availability(self):
        """Check if display is available"""
        try:
            test_window = "Test"
            cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
            cv2.destroyWindow(test_window)
            return True
        except:
            return False
    
    def initialize(self):
        """Initialize the detector"""
        # Check display availability
        if not self.args.no_display and not self.display_available:
            logger.info("Display requested but not available. Running without display.")
            self.args.no_display = True
            
            # Enable video saving if display is not available
            if not self.args.save_video:
                logger.info("Enabling video saving since display is not available.")
                self.args.save_video = True
        
        # Create output directory if saving video
        if self.args.save_video:
            os.makedirs(self.args.output_dir, exist_ok=True)
        
        # Initialize camera processors
        self.camera1 = CameraProcessor(
            camera_id=1,
            source=self.args.camera1,
            model_path=self.args.model,
            labels_path=self.args.labels,
            threshold=self.args.threshold,
            person_class_id=self.args.person_class_id
        )
        
        # Start first camera processor
        if not self.camera1.start():
            logger.error("Failed to start camera 1")
            return False
        
        # Check if camera2 is the same as camera1
        if self.args.camera1 == self.args.camera2:
            logger.warning("Camera 1 and Camera 2 are the same. Using a copy of Camera 1 for Camera 2.")
            self.camera2 = self.camera1  # Use the same camera processor
        else:
            # Initialize second camera processor
            self.camera2 = CameraProcessor(
                camera_id=2,
                source=self.args.camera2,
                model_path=self.args.model,
                labels_path=self.args.labels,
                threshold=self.args.threshold,
                person_class_id=self.args.person_class_id
            )
            
            # Start second camera processor
            if not self.camera2.start():
                logger.error("Failed to start camera 2")
                self.camera1.stop()
                return False
        
        logger.info("Dual camera detector initialized successfully")
        return True
    
    def run(self):
        """Run the detector"""
        if not self.display_available and not self.args.no_display:
            logger.warning("Display not available. Running in headless mode.")
            self.args.no_display = True
        
        start_time = time.time()
        frame_count = 0
        
        # Create window if display is enabled
        if not self.args.no_display:
            cv2.namedWindow('Dual Camera Person Detection', cv2.WINDOW_NORMAL)
            # Set window size to 640x480 for display
            cv2.resizeWindow('Dual Camera Person Detection', 640, 480)
        
        while self.running:
            # Get results from both cameras
            result1 = self.camera1.get_latest_result()
            
            # If both cameras are the same, use the same result
            if self.camera1 == self.camera2:
                result2 = result1
            else:
                result2 = self.camera2.get_latest_result()
            
            # Skip if either result is None
            if result1 is None or result2 is None:
                time.sleep(0.01)
                continue
            
            # Combine frames side by side
            frame1 = result1['frame']
            frame2 = result2['frame']
            
            # Ensure both frames are exactly 300x300
            frame1 = cv2.resize(frame1, (300, 300))
            frame2 = cv2.resize(frame2, (300, 300))
            
            # Create combined frame
            combined_frame = np.hstack((frame1, frame2))
            
            # Add title to the combined frame
            title_bar = np.zeros((30, combined_frame.shape[1], 3), dtype=np.uint8)
            cv2.putText(title_bar, "Dual Camera Person Detection", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add camera labels
            cv2.putText(frame1, f"Camera 1", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame2, f"Camera 2", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Combine title bar with frames
            display_frame = np.vstack((title_bar, combined_frame))
            
            # Resize display frame to 640x480 for better viewing
            display_frame = cv2.resize(display_frame, (640, 480))
            
            # Check for detections and trigger alerts
            for result in [result1, result2]:
                for detection in result['detections']:
                    class_id, score, bbox = detection
                    if class_id == self.args.person_class_id and score >= self.args.threshold:
                        # Log detection
                        self.logger.log_detection("Person", score)
                        
                        # Trigger alert
                        self.alert_system.trigger_alert(score)
                        
                        # Save detection image if configured
                        if self.args.save_detections:
                            save_detection_image(display_frame, self.args.output_dir)
            
            # Display frame
            if not self.args.no_display:
                cv2.imshow('Dual Camera Person Detection', display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # Save video
            if self.args.save_video:
                if not self.video_writer_initialized:
                    self.initialize_video_writer(display_frame)
                
                if self.video_writer:
                    self.video_writer.write(display_frame)
            
            frame_count += 1
            
            # Print stats every 100 frames
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                logger.info(f"Processed {frame_count} frames in {elapsed_time:.2f}s ({fps:.2f} FPS)")
                
                # Log detection counts
                logger.info(f"Camera 1: {self.camera1.total_detections} detections in {self.camera1.total_frames} frames")
                logger.info(f"Camera 2: {self.camera2.total_detections} detections in {self.camera2.total_frames} frames")
                
                # Log performance
                self.logger.log_performance(fps, elapsed_time / frame_count)
        
        # Clean up
        self.cleanup()
        
        # Print final stats
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        logger.info(f"Final stats: Processed {frame_count} frames in {elapsed_time:.2f}s ({fps:.2f} FPS)")
        logger.info(f"Camera 1: {self.camera1.total_detections} detections in {self.camera1.total_frames} frames")
        logger.info(f"Camera 2: {self.camera2.total_detections} detections in {self.camera2.total_frames} frames")
        
        return True
    
    def initialize_video_writer(self, frame):
        """Initialize video writer"""
        try:
            h, w = frame.shape[:2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.args.output_dir, f"dual_camera_{timestamp}.mp4")
            
            # Use mp4v codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, 20.0, (w, h))
            
            if not self.video_writer.isOpened():
                logger.error(f"Failed to open video writer: {output_path}")
                self.video_writer = None
            else:
                logger.info(f"Video recording started: {output_path}")
                self.video_writer_initialized = True
        except Exception as e:
            logger.error(f"Error initializing video writer: {e}")
            self.video_writer = None
    
    def cleanup(self):
        """Clean up resources"""
        if self.camera1:
            self.camera1.stop()
        
        # Only stop camera2 if it's different from camera1
        if self.camera2 and self.camera2 != self.camera1:
            self.camera2.stop()
        
        if self.video_writer:
            self.video_writer.release()
        
        cv2.destroyAllWindows()
        logger.info("Cleanup completed")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Dual Camera Person Detection')
    
    parser.add_argument('--model', type=str, 
                        default='models/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite',
                        help='Path to the TFLite model')
    
    parser.add_argument('--labels', type=str, 
                        default='models/coco_labels.txt',
                        help='Path to the labels file')
    
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection threshold')
    
    parser.add_argument('--camera1', type=str, default='0',
                        help='First camera source (index or path)')
    
    parser.add_argument('--camera2', type=str, default='2',
                        help='Second camera source (index or path)')
    
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save output files')
    
    parser.add_argument('--person-class-id', type=int, default=0,
                        help='Class ID for person (default: 0 for COCO)')
    
    parser.add_argument('--no-display', action='store_true',
                        help='Disable display output')
    
    parser.add_argument('--save-video', action='store_true',
                        help='Save processed video')
    
    parser.add_argument('--save-detections', action='store_true',
                        help='Save individual detection images')
    
    parser.add_argument('--force-cpu', action='store_true',
                        help='Force CPU mode (no TPU)')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Check if Edge TPU is available
    if not HAVE_CORAL:
        logger.warning("PyCoral not found. Running in mock detection mode.")
    
    # Check if model file exists
    if not os.path.exists(args.model):
        logger.warning(f"Model file not found: {args.model}")
        logger.warning("Creating models directory and using mock detection mode.")
        os.makedirs("models", exist_ok=True)
        
        # Create an empty model file for mock detection
        with open(args.model, "w") as f:
            f.write("Mock model file for testing")
    
    # Check if labels file exists
    if not os.path.exists(args.labels):
        logger.warning(f"Labels file not found: {args.labels}")
        logger.warning("Creating a default labels file.")
        os.makedirs(os.path.dirname(args.labels), exist_ok=True)
        
        # Create a default labels file
        with open(args.labels, "w") as f:
            f.write("0 person\n1 bicycle\n2 car\n3 motorcycle\n4 airplane\n5 bus\n6 train\n")
    
    # Create detector
    detector = DualCameraDetector(args)
    
    # Initialize detector
    if not detector.initialize():
        logger.error("Failed to initialize detector")
        return 1
    
    # Run detector
    try:
        if not detector.run():
            logger.error("Error during detection")
            return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 