# app/services/sop/config.py
"""
Configuration for SOP Video Assessment System (Stage B)
"""
import os
from pathlib import Path

# --- MODEL PATHS ---
# YOLO model for object detection (custom trained for SOP steps)
MODEL_PATH = os.getenv("SOP_MODEL_PATH", "models/best.pt")

# ByteTrack configuration for object tracking
TRACKER_CONFIG = "bytetrack.yaml"

# --- DETECTION THRESHOLDS ---
CONF_THRESHOLD = 0.5  # Minimum confidence for YOLO detection
MP_CONFIDENCE = 0.5   # MediaPipe hand detection confidence

# --- INTERACTION LOGIC ---
GRAB_DISTANCE_THRESHOLD = 200    # Pixel distance to consider "grabbing"
VERIFY_TIME_THRESHOLD = 0.4      # Time to hold for verification (seconds)

# --- SOP STATE MACHINE ---
REQUIRED_TOUCH_FRAMES = 10       # Frames of continuous touch to start step
MIN_WORK_DURATION = 1.0          # Minimum work time for step completion (seconds)
HAND_OFF_TIMEOUT = 2.0           # Time after releasing to confirm step done (seconds)

# --- WRONG ORDER DETECTION ---
MAX_STEP_IDLE_TIME = 25.0        # Max seconds waiting before timeout (fatal error)
FATAL_ERROR_WAIT_TIME = 10.0     # Time to wait after fatal error before retry
WRONG_STEP_TOLERANCE = 3         # Number of consecutive wrong detections before error

# --- ERROR TYPES ---
ERROR_TYPE_WRONG_ORDER = "WRONG_ORDER"
ERROR_TYPE_TIMEOUT = "TIMEOUT"
ERROR_TYPE_MISSED_STEP = "MISSED_STEP"

# --- TARGET CLASSES ---
# Map YOLO class IDs to step labels
# These should match your custom-trained YOLO model classes
TARGET_CLASSES = [0, 1, 2, 3]  # Class IDs to detect

CLASS_MAP = {
    0: "step1",
    1: "step2", 
    2: "step3",
    3: "step4"
}

# --- VISUALIZATION COLORS (BGR format for OpenCV) ---
COLORS = {
    "step1": (0, 165, 255),    # Orange
    "step2": (255, 255, 0),    # Cyan
    "step3": (255, 0, 0),      # Blue
    "step4": (0, 255, 255),    # Yellow
    "HAND_LEFT": (0, 255, 0),  # Green
    "HAND_RIGHT": (255, 0, 255) # Magenta
}

def get_color(label: str) -> tuple:
    """Get color for a given label."""
    return COLORS.get(label, (200, 200, 200))

# --- PATHS ---
SOP_RULES_PATH = Path(__file__).parent / "data" / "sop_rules.json"
REPORTS_FOLDER = Path(__file__).parent / "reports"
MODELS_FOLDER = Path(__file__).parent / "models"

# Create directories if they don't exist
REPORTS_FOLDER.mkdir(parents=True, exist_ok=True)
MODELS_FOLDER.mkdir(parents=True, exist_ok=True)
