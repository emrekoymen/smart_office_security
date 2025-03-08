import cv2
import numpy as np
import time
from datetime import datetime
import os
from plyer import notification

class FPSCounter:
    """Class to calculate and display FPS."""
    
    def __init__(self, avg_frames=30):
        self.frame_times = []
        self.avg_frames = avg_frames
        
    def update(self):
        """Update FPS calculation."""
        self.frame_times.append(time.time())
        # Keep only the last avg_frames times
        if len(self.frame_times) > self.avg_frames:
            self.frame_times.pop(0)
    
    def get_fps(self):
        """Calculate current FPS."""
        if len(self.frame_times) <= 1:
            return 0
        
        # Calculate time difference between oldest and newest frame
        time_diff = self.frame_times[-1] - self.frame_times[0]
        if time_diff == 0:
            return 0
            
        # FPS = (number of frames - 1) / time difference
        return (len(self.frame_times) - 1) / time_diff
    
    def draw_fps(self, frame):
        """Draw FPS on frame."""
        fps = self.get_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame


class Logger:
    """Class to handle logging detection events."""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.log_file = os.path.join(
            log_dir, f"detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    
    def log_detection(self, label, confidence, image_path=None):
        """Log a detection event."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - Detected: {label} (Confidence: {confidence:.2f})"
        
        if image_path:
            log_entry += f" - Image saved: {image_path}"
            
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")
        
        print(log_entry)
    
    def log_performance(self, fps, processing_time):
        """Log performance metrics."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"{timestamp} - Performance: FPS: {fps:.1f}, Processing time: {processing_time:.3f}s"
        
        with open(self.log_file, "a") as f:
            f.write(log_entry + "\n")


class AlertSystem:
    """Class to handle alerts when a person is detected."""
    
    def __init__(self, min_confidence=0.5, cooldown_period=10):
        self.min_confidence = min_confidence
        self.cooldown_period = cooldown_period
        self.last_alert_time = 0
        self.enabled = True
        
    def can_alert(self):
        """Check if enough time has passed since the last alert."""
        current_time = time.time()
        if current_time - self.last_alert_time > self.cooldown_period:
            return True
        return False
    
    def trigger_alert(self, confidence, label="Person"):
        """Trigger an alert if confidence is high enough and cooldown has passed."""
        if not self.enabled or confidence < self.min_confidence:
            return False
            
        if self.can_alert():
            self.last_alert_time = time.time()
            
            # Display desktop notification
            try:
                notification.notify(
                    title="Security Alert",
                    message=f"{label} detected with {confidence:.2f} confidence!",
                    app_name="Office Security System",
                    timeout=5
                )
            except Exception as e:
                print(f"Could not send notification: {e}")
            
            return True
        
        return False


def get_video_source(source="0"):
    """
    Get video source - can be a file path, webcam index, or RTSP URL.
    Returns a cv2.VideoCapture object.
    """
    # If source is a digit string, convert to int for webcam
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    
    # Check if opened successfully
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")
        
    return cap


def save_detection_image(frame, output_dir="detections"):
    """Save a frame with detection to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filename = os.path.join(
        output_dir, f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    )
    
    cv2.imwrite(filename, frame)
    return filename


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