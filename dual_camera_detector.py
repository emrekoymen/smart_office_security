#!/usr/bin/env python3
"""
Dual Camera Person Detection with Google Coral Edge TPU

This script uses a pre-trained MobileNet SSD model on the Coral Edge TPU
to detect persons in two simultaneous video streams.
It processes both camera feeds in parallel and displays them side-by-side.

Performance Optimizations:
- Efficient frame queue management to minimize latency
- Optimized display pipeline with frame rate control
- Smart result retrieval to always get the most recent frame
- Reduced CPU usage with controlled frame skipping
- Display resolution of 1280x480 for dual 640x480 camera feeds
- Maintains detection input resolution at 300x300 for optimal model performance
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
    
    This class manages camera capture, frame processing, and object detection
    in a separate thread. It implements efficient frame queue management to
    minimize latency and maintain real-time processing.
    
    Key optimizations:
    - Adaptive frame skipping when queues are full
    - Efficient frame resizing with appropriate interpolation methods
    - Smart queue management to prevent backlog and maintain freshness
    - Low-latency result retrieval for real-time display
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
                # Set a default input size even if model loading fails
                self.input_size = (300, 300)
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
            
            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Try to set resolution to 300x300 for better performance
            original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Set resolution to 300x300 for better performance and consistency with model input
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)
            
            # Get actual resolution after setting (may not be exactly what we requested)
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Get FPS
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30.0
            
            # Try to set higher FPS for webcams
            if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if actual_fps > 0:
                    self.fps = actual_fps
            
            logger.info(f"Camera {self.camera_id}: Opened {self.frame_width}x{self.frame_height} @ {self.fps} FPS (original: {original_width}x{original_height})")
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
        """
        Thread function to capture frames from the camera
        
        This method continuously captures frames from the camera and adds them
        to the frame queue for processing. It implements several optimizations:
        
        1. Adaptive frame skipping when the queue is getting full
        2. Queue management to prevent backlog and maintain freshness
        3. Non-blocking queue operations to prevent thread blocking
        4. Error handling for camera disconnection and frame reading failures
        
        These optimizations ensure that the processing pipeline always has
        the most recent frames available, minimizing latency between
        capture and display.
        """
        frame_skip = 0  # For frame skipping if needed
        
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
            
            # Skip frames if queue is getting full to maintain real-time processing
            if self.frame_queue.qsize() > 1:
                frame_skip += 1
                if frame_skip % 2 != 0:  # Skip every other frame when queue is full
                    continue
            else:
                frame_skip = 0
            
            # Clear queue if it's getting too full to prevent lag
            if self.frame_queue.qsize() > 2:
                try:
                    while self.frame_queue.qsize() > 1:
                        self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # If queue is full, don't wait - just drop the frame
                pass
    
    def process_frames(self):
        """
        Thread function to process frames from the queue
        
        This method retrieves frames from the frame queue, performs object
        detection, and adds the results to the result queue. It implements
        several optimizations:
        
        1. Efficient frame resizing using INTER_AREA for best detection quality
        2. Conditional resizing to avoid unnecessary operations
        3. Smart result queue management to prevent backlog
        4. Non-blocking queue operations to maintain responsiveness
        5. Optimized detection box drawing only when detections are present
        
        These optimizations ensure efficient processing while maintaining
        detection accuracy and minimizing CPU usage.
        """
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.01)  # Shorter timeout for responsiveness
            except queue.Empty:
                continue
            
            start_time = time.time()
            
            # Resize frame for detection - only if needed
            h, w = frame.shape[:2]
            if h != self.input_size[1] or w != self.input_size[0]:
                detection_frame = cv2.resize(frame, self.input_size, interpolation=cv2.INTER_AREA)
            else:
                detection_frame = frame
            
            # Run detection
            detections = self.detect_objects(detection_frame)
            
            # Filter for persons
            person_detections = [d for d in detections if d[0] == self.person_class_id]
            
            # Draw detection boxes - only if there are detections to improve performance
            if person_detections:
                result_frame = frame.copy()
                for detection in person_detections:
                    class_id, score, bbox = detection
                    label = self.labels.get(class_id, f"Class {class_id}")
                    result_frame = draw_detection_box(result_frame, bbox, label, score)
                    
                    # Log detection coordinates for debugging
                    x, y, w, h = bbox
                    logger.info(f"Drawing box at: ({x}, {y}), ({x+w}, {y+h}), dimensions: {w}x{h}")
            else:
                result_frame = frame
            
            # Update FPS counter
            self.fps_counter.update()
            fps = self.fps_counter.get_fps()
            
            # Add FPS text to frame - use efficient text rendering
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
            
            # Clear result queue if it's getting too full
            if self.result_queue.qsize() > 1:
                try:
                    while self.result_queue.qsize() > 0:
                        self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            
            try:
                self.result_queue.put(result, block=False)
            except queue.Full:
                # If queue is full, don't wait - just drop the result
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
        """
        Get the latest processed result
        
        This method efficiently retrieves the most recent frame from the result queue
        without waiting for older frames to be processed. It implements a smart
        queue draining approach that:
        
        1. Checks if multiple results are available
        2. If so, drains all but the most recent one to minimize latency
        3. Returns the most recent frame for display
        4. Properly handles empty queues with appropriate error handling
        
        This approach ensures that the display always shows the most current
        detection results, even if processing is slower than capture.
        """
        # More efficient way to get the latest result
        # Only get the most recent frame instead of draining the queue
        try:
            # First check if there's anything in the queue
            if self.result_queue.qsize() > 1:
                # If multiple results, drain all but the last one
                while self.result_queue.qsize() > 1:
                    self.result_queue.get_nowait()
                return self.result_queue.get_nowait()
            elif self.result_queue.qsize() == 1:
                return self.result_queue.get_nowait()
            else:
                return None
        except queue.Empty:
            return None


class DualCameraDetector:
    """
    Main class for dual camera person detection
    
    This class manages two camera processors and combines their outputs
    into a single display. It implements an optimized display pipeline
    with frame rate control for smooth visualization.
    
    Key features:
    - Parallel processing of two camera feeds
    - Optimized display pipeline with controlled frame rate
    - Side-by-side display of camera feeds at 1280x480 resolution
    - Efficient frame resizing for display while maintaining detection quality
    - Real-time FPS monitoring and display
    - Adaptive frame skipping to maintain responsiveness
    """
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
        """
        Run the detector
        
        This method implements the main detection and display loop with
        several optimizations for performance:
        
        1. Frame rate control to maintain consistent display FPS
        2. Efficient frame resizing using INTER_NEAREST for display
        3. Smart result retrieval to always get the most recent frames
        4. Optimized display pipeline with minimal CPU usage
        5. Real-time FPS calculation and display
        6. Side-by-side display of both camera feeds at 1280x480 resolution
        
        The method balances detection accuracy with display performance
        by using different resolutions for detection (300x300) and
        display (640x480 per camera).
        """
        if not self.display_available and not self.args.no_display:
            logger.warning("Display not available. Running in headless mode.")
            self.args.no_display = True
        
        start_time = time.time()
        frame_count = 0
        
        # Create window if display is enabled
        if not self.args.no_display:
            cv2.namedWindow('Dual Camera Person Detection', cv2.WINDOW_NORMAL)
            # Set window size to 1280x480 for dual 640x480 display
            cv2.resizeWindow('Dual Camera Person Detection', 1280, 480)
        
        # For performance tracking
        fps_update_interval = 0.5  # Update FPS every half second
        last_fps_update = time.time()
        display_fps = 0
        
        # For frame rate control
        target_display_fps = 30  # Target display FPS
        min_frame_time = 1.0 / target_display_fps
        last_frame_time = time.time()
        
        # Pre-allocate display frame to avoid repeated memory allocations
        display_frame = None
        
        while self.running:
            # Limit frame rate for display to reduce CPU usage
            current_time = time.time()
            elapsed_since_last_frame = current_time - last_frame_time
            
            if elapsed_since_last_frame < min_frame_time:
                # Sleep to maintain target frame rate
                time.sleep(max(0, min_frame_time - elapsed_since_last_frame))
                continue
            
            last_frame_time = current_time
            
            # Get results from both cameras
            result1 = self.camera1.get_latest_result()
            
            # If both cameras are the same, use the same result
            if self.camera1 == self.camera2:
                result2 = result1
            else:
                result2 = self.camera2.get_latest_result()
            
            # Skip if either result is None
            if result1 is None or result2 is None:
                time.sleep(0.001)  # Very short sleep to prevent CPU spinning
                continue
            
            # Get frames from results
            frame1 = result1['frame']
            frame2 = result2['frame']
            
            # Check for detections and trigger alerts
            for result, camera_id in [(result1, 1), (result2, 2)]:
                for detection in result['detections']:
                    class_id, score, bbox = detection
                    if class_id == self.args.person_class_id and score >= self.args.threshold:
                        # Log detection
                        logger.info(f"Detected: Person (Confidence: {score:.2f})")
                        
                        # Trigger alert
                        if hasattr(self, 'alert_system'):
                            self.alert_system.trigger_alert(score)
                        
                        # Save detection image if configured
                        if self.args.save_detections and hasattr(self, 'save_detection_image'):
                            self.save_detection_image(frame1 if camera_id == 1 else frame2, self.args.output_dir)
            
            # Create display frame
            if not self.args.no_display or self.args.save_video:
                # Resize both frames to 640x480 for display - use INTER_NEAREST for speed
                frame1_display = cv2.resize(frame1, (640, 480), interpolation=cv2.INTER_NEAREST)
                frame2_display = cv2.resize(frame2, (640, 480), interpolation=cv2.INTER_NEAREST)
                
                # Add camera labels
                cv2.putText(frame1_display, "Camera 1", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame2_display, "Camera 2", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Combine frames side by side
                display_frame = np.hstack((frame1_display, frame2_display))
                
                # Calculate and display FPS
                current_time = time.time()
                frame_count += 1
                
                if current_time - last_fps_update >= fps_update_interval:
                    elapsed = current_time - last_fps_update
                    display_fps = 1.0 / elapsed if elapsed > 0 else 0
                    last_fps_update = current_time
                
                # Add FPS to the display frame
                cv2.putText(display_frame, f"Display FPS: {display_fps:.1f}", (10, 470), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                if not self.args.no_display:
                    cv2.imshow('Dual Camera Person Detection', display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                # Save video
                if self.args.save_video:
                    if not hasattr(self, 'video_writer_initialized') or not self.video_writer_initialized:
                        self.initialize_video_writer(display_frame)
                    
                    if hasattr(self, 'video_writer') and self.video_writer:
                        self.video_writer.write(display_frame)
            
            # Print stats every 100 frames
            if frame_count % 100 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                logger.info(f"Processed {frame_count} frames in {elapsed_time:.2f}s ({fps:.2f} FPS)")
                
                # Log detection counts
                logger.info(f"Camera 1: {self.camera1.total_detections} detections in {self.camera1.total_frames} frames")
                logger.info(f"Camera 2: {self.camera2.total_detections} detections in {self.camera2.total_frames} frames")
        
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
    
    # Set QT_QPA_PLATFORM to offscreen if --no-display is specified
    # This prevents Qt errors when running without a display server
    if args.no_display:
        os.environ['QT_QPA_PLATFORM'] = 'offscreen'

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