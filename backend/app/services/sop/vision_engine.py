# app/services/sop/vision_engine.py
"""
Vision Engine for SOP Assessment
Uses YOLO for object detection and MediaPipe for hand tracking
"""
import cv2
import logging as log
from typing import Generator, Tuple, List, Dict, Any, Optional

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    log.warning("MediaPipe not installed. Hand tracking will be disabled.")

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    log.warning("Ultralytics YOLO not installed. Object detection will be disabled.")

from .config import (
    CLASS_MAP, TRACKER_CONFIG, CONF_THRESHOLD, 
    TARGET_CLASSES, MP_CONFIDENCE
)

log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class VisionEngine:
    """
    Vision processing engine combining YOLO object detection 
    with MediaPipe hand tracking for SOP assessment.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the vision engine.
        
        Args:
            model_path: Path to YOLO model weights (.pt file)
        """
        self.model = None
        self.hands_detector = None
        self.mp_hands = None
        
        # Initialize YOLO
        if YOLO_AVAILABLE:
            try:
                log.info(f"Loading YOLO model: {model_path}...")
                self.model = YOLO(model_path)
                log.info("YOLO model loaded successfully")
            except Exception as e:
                log.error(f"Failed to load YOLO model: {e}")
        
        # Initialize MediaPipe Hands
        if MEDIAPIPE_AVAILABLE:
            try:
                log.info("Loading MediaPipe Hands...")
                self.mp_hands = mp.solutions.hands
                self.hands_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=MP_CONFIDENCE,
                    min_tracking_confidence=MP_CONFIDENCE
                )
                log.info("MediaPipe Hands loaded successfully")
            except Exception as e:
                log.error(f"Failed to load MediaPipe: {e}")
    
    def process_frame(self, frame) -> Tuple[List[Dict], List[Dict]]:
        """
        Process a single frame for objects and hands.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Tuple of (detected_objects, detected_hands)
        """
        detected_objects = []
        detected_hands = []
        
        # A. YOLO Object Detection with Tracking
        if self.model is not None:
            try:
                results = self.model.track(
                    source=frame,
                    tracker=TRACKER_CONFIG,
                    persist=True,
                    conf=CONF_THRESHOLD,
                    classes=TARGET_CLASSES,
                    verbose=False
                )
                
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.int().cpu().numpy()
                    class_ids = results[0].boxes.cls.int().cpu().numpy()
                    
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        label = CLASS_MAP.get(class_id, "unknown")
                        x1, y1, x2, y2 = map(int, box)
                        detected_objects.append({
                            "id": track_id,
                            "label": label,
                            "bbox": (x1, y1, x2, y2),
                            "center": (int((x1+x2)/2), int((y1+y2)/2))
                        })
            except Exception as e:
                log.error(f"YOLO detection error: {e}")
        
        # B. MediaPipe Hand Detection
        if self.hands_detector is not None:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_results = self.hands_detector.process(frame_rgb)
                
                if mp_results.multi_hand_landmarks:
                    h, w, c = frame.shape
                    
                    for idx, hand_landmarks in enumerate(mp_results.multi_hand_landmarks):
                        # Get hand label (Left/Right)
                        hand_label = mp_results.multi_handedness[idx].classification[0].label
                        
                        # Convert landmarks to pixel coordinates
                        landmarks_pixel = []
                        x_list = []
                        y_list = []
                        
                        for lm in hand_landmarks.landmark:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            landmarks_pixel.append((cx, cy))
                            x_list.append(cx)
                            y_list.append(cy)
                        
                        # Create bounding box around hand
                        x_min, x_max = min(x_list), max(x_list)
                        y_min, y_max = min(y_list), max(y_list)
                        
                        detected_hands.append({
                            "id": f"hand_{idx}",
                            "type": f"HAND_{hand_label.upper()}",
                            "landmarks": landmarks_pixel,
                            "bbox": (x_min, y_min, x_max, y_max),
                            "wrist": landmarks_pixel[0],
                            "index_tip": landmarks_pixel[8],
                            "thumb_tip": landmarks_pixel[4]
                        })
            except Exception as e:
                log.error(f"MediaPipe detection error: {e}")
        
        return detected_objects, detected_hands
    
    def process_stream(self, video_source) -> Generator[Tuple[Any, List[Dict], List[Dict]], None, None]:
        """
        Process a video stream frame by frame.
        
        Args:
            video_source: Path to video file or camera index (0 for webcam)
            
        Yields:
            Tuple of (frame, detected_objects, detected_hands)
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            log.error(f"Failed to open video source: {video_source}")
            return
        
        log.info(f"Processing video stream: {video_source}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                detected_objects, detected_hands = self.process_frame(frame)
                yield frame, detected_objects, detected_hands
                
        finally:
            cap.release()
            log.info("Video stream processing completed")
    
    def process_video_file(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Process entire video file and return all detections.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of frame results with objects and hands
        """
        results = []
        frame_count = 0
        
        for frame, objects, hands in self.process_stream(video_path):
            results.append({
                "frame_number": frame_count,
                "objects": objects,
                "hands": hands
            })
            frame_count += 1
        
        log.info(f"Processed {frame_count} frames from {video_path}")
        return results
    
    def release(self):
        """Release resources."""
        if self.hands_detector:
            self.hands_detector.close()
