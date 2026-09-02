"""
Air Canvas - Backend Module
Handles hand detection, gesture recognition, drawing logic, and canvas management.
"""

from __future__ import annotations

import os
import importlib
from collections import deque
from datetime import datetime
import logging
from threading import Lock
from typing import TYPE_CHECKING, Dict, Optional, Tuple, List, Any

# Type checking imports (extensions like Pylance can resolve these).  Runtime imports are handled below.
if TYPE_CHECKING:
    import numpy as np
    import cv2
    import numpy.typing as npt

# Attempt to import key scientific libraries; if they fail, we continue with a flag.
BACKEND_IMPORT_ERROR: Optional[str] = None
try:
    import numpy as np
    import cv2
except Exception as e:
    np = None  # type: ignore
    cv2 = None  # type: ignore
    BACKEND_IMPORT_ERROR = str(e)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============ MediaPipe Hand Tracking (Optional) ============
# Allows forcing MediaPipe mode via environment variable for debugging.
FORCE_MEDIAPIPE = str(os.getenv('FORCE_MEDIAPIPE', '0')).lower() in ('1', 'true', 'yes')

mp = None
mp_hands = None
mp_drawing = None
mp_image = None
HAND_DETECTOR = None
USE_MEDIAPIPE = False
MEDIA_PIPE_ERROR: Optional[str] = None


def init_mediapipe(force: bool = False) -> bool:
    """Attempt to initialize MediaPipe Hand Landmarker.

    Returns True if MediaPipe is available and initialized.
    """
    global mp, mp_hands, mp_drawing, HAND_DETECTOR, USE_MEDIAPIPE, MEDIA_PIPE_ERROR

    if USE_MEDIAPIPE and (HAND_DETECTOR is not None or mp_hands is not None) and not force:
        return True

    # Ensure model file exists. If not, try to download it.
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'hand_landmarker.task')

    if not os.path.exists(model_path):
        try:
            import urllib.request
            logger.info('Downloading MediaPipe hand_landmarker task model...')
            req = urllib.request.Request(
                'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task',
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as resp, open(model_path, 'wb') as out_f:
                out_f.write(resp.read())
            logger.info('Downloaded hand_landmarker.task')
        except Exception as e:
            logger.warning(f"Failed to download MediaPipe model task file: {e}")

    try:
        # 1. Try modern MediaPipe Tasks API (v0.10+)
        if os.path.exists(model_path):
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core import image as mp_img_module
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
            from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

            base_options = BaseOptions(model_asset_path=model_path)
            options = HandLandmarkerOptions(
                base_options=base_options,
                running_mode=VisionTaskRunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.45,
                min_hand_presence_confidence=0.40,
                min_tracking_confidence=0.40,
            )

            HAND_DETECTOR = HandLandmarker.create_from_options(options)
            USE_MEDIAPIPE = True
            MEDIA_PIPE_ERROR = None
            globals()['mp_image'] = mp_img_module
            logger.info('MediaPipe Tasks HandLandmarker initialized successfully')
            return True
    except Exception as e:
        logger.warning(f'MediaPipe Tasks initialization failed: {e}')

    try:
        # 2. Try legacy MediaPipe Solutions API fallback (static_image_mode=True for stateless HTTP requests)
        import mediapipe as mp_lib
        mp_solutions_hands = mp_lib.solutions.hands
        mp_hands = mp_solutions_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.40
        )
        USE_MEDIAPIPE = True
        MEDIA_PIPE_ERROR = None
        HAND_DETECTOR = None
        logger.info('MediaPipe Solutions Hands initialized successfully as fallback')
        return True
    except Exception as e:
        USE_MEDIAPIPE = False
        HAND_DETECTOR = None
        mp_hands = None
        MEDIA_PIPE_ERROR = str(e)
        logger.warning(f'MediaPipe initialization failed: {e}')
        return False


# Try to initialize MediaPipe on module import.
# Note: if MediaPipe fails to load, it is safe to continue using the fallback detector.
init_mediapipe(force=FORCE_MEDIAPIPE)

# ============ Thread-safe Canvas Operations ============
canvas_lock = Lock()


def _ensure_backend_available():
    """Raise an error if required backend libs (numpy/cv2) failed to import."""
    if BACKEND_IMPORT_ERROR:
        raise RuntimeError(f"Backend import failed: {BACKEND_IMPORT_ERROR}")

# ============ Hand Detection ============
def detect_hand(frame_rgb: np.ndarray) -> Optional[Dict]:  # type: ignore[valid-type]
    """Detect hand in frame.

    Uses MediaPipe Hands when available to provide robust and accurate hand
    tracking. If MediaPipe is not installed, falls back to a simple skin-tone
    based segmentation as a best-effort backup.

    Returns:
        Dictionary with hand data or None if no hand detected.
    """
    # Ensure required libraries loaded
    _ensure_backend_available()

    orig_h, orig_w, _ = frame_rgb.shape

    # Downsample frame for fast lightweight CPU inference (320x240) to prevent Gunicorn worker timeouts
    target_w, target_h = 320, 240
    if orig_w > target_w or orig_h > target_h:
        proc_frame = cv2.resize(frame_rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        proc_frame = frame_rgb

    # --- MediaPipe Tasks-based detection (preferred) ---
    if USE_MEDIAPIPE and HAND_DETECTOR is not None and mp_image is not None:
        try:
            image = mp_image.Image(mp_image.ImageFormat.SRGB, proc_frame)
            results = HAND_DETECTOR.detect(image)

            if results and results.hand_landmarks:
                hand_landmarks = results.hand_landmarks[0]
                # Scale normalized landmarks back to original canvas dimensions
                points = [(int(lm.x * orig_w), int(lm.y * orig_h)) for lm in hand_landmarks]
                if points:
                    cx = int(np.mean([p[0] for p in points]))
                    cy = int(np.mean([p[1] for p in points]))
                    index_tip = points[8] if len(points) > 8 else (cx, cy)
                    bbox = (
                        min(p[0] for p in points),
                        min(p[1] for p in points),
                        max(p[0] for p in points),
                        max(p[1] for p in points)
                    )
                    return {
                        'center': (cx, cy),
                        'index_tip': index_tip,
                        'landmarks': points,
                        'bbox': bbox,
                        'hand_landmarks': hand_landmarks
                    }
        except Exception as e:
            logger.warning(f"MediaPipe Tasks detection error: {e}")

    # --- MediaPipe Solutions API detection fallback ---
    if USE_MEDIAPIPE and mp_hands is not None:
        try:
            results = mp_hands.process(proc_frame)
            if results and results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                # Scale normalized landmarks back to original canvas dimensions
                points = [(int(lm.x * orig_w), int(lm.y * orig_h)) for lm in hand_landmarks.landmark]
                if points:
                    cx = int(np.mean([p[0] for p in points]))
                    cy = int(np.mean([p[1] for p in points]))
                    index_tip = points[8] if len(points) > 8 else (cx, cy)
                    bbox = (
                        min(p[0] for p in points),
                        min(p[1] for p in points),
                        max(p[0] for p in points),
                        max(p[1] for p in points)
                    )
                    return {
                        'center': (cx, cy),
                        'index_tip': index_tip,
                        'landmarks': points,
                        'bbox': bbox,
                        'hand_landmarks': hand_landmarks.landmark
                    }
        except Exception as e:
            logger.warning(f"MediaPipe Solutions detection error: {e}")

    # --- Fallback: skin-tone segmentation (less accurate) ---
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Skin tone range - optimized for robustness
    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Simple morphological operations - less aggressive
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Find largest contour (assumes this is the hand)
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    # Filter by area
    if area < 300:
        return None

    # Calculate centroid
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return None

    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Calculate convex hull
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0

    # Hand shape validation
    if not (0.3 <= solidity <= 1.0):
        return None

    # Find topmost contour point to estimate index tip
    topmost = tuple(largest[largest[:, :, 1].argmin()][0])
    index_tip = (int(topmost[0]), int(topmost[1]))
    bottommost = tuple(largest[largest[:, :, 1].argmax()][0])
    wrist = (int(bottommost[0]), int(bottommost[1]))

    # Construct synthetic 21-landmark set so classify_gesture works in skin-tone fallback mode
    synthetic_landmarks = [wrist] * 21
    mid_y = int((wrist[1] + index_tip[1]) / 2)
    synthetic_landmarks[6] = (index_tip[0], mid_y)
    synthetic_landmarks[8] = index_tip  # Index fingertip

    x, y, w_box, h_box = cv2.boundingRect(largest)
    bbox = (x, y, x + w_box, y + h_box)

    return {
        'center': (cx, cy),
        'index_tip': index_tip,
        'landmarks': synthetic_landmarks,
        'bbox': bbox,
        'contour': largest,
        'hull': hull,
        'area': area,
        'solidity': solidity,
        'mask': mask
    }


def classify_gesture(landmarks: List[Tuple[int, int]], image_shape: Tuple[int, int, int]) -> str:
    """Classify hand gestures from landmarks.

    Uses a stronger heuristic to detect finger extension and pinch gestures.
    This improves accuracy and avoids false positives when the hand is tilted.
    """
    if not landmarks or len(landmarks) < 21:
        return 'neutral'

    # Determine hand size (used for normalization)
    wrist = np.array(landmarks[0])
    middle_mcp = np.array(landmarks[9])
    hand_size = np.linalg.norm(middle_mcp - wrist)
    if hand_size < 1e-3:
        hand_size = 1.0

    def finger_extended(tip_idx: int, pip_idx: int) -> bool:
        tip = np.array(landmarks[tip_idx])
        pip = np.array(landmarks[pip_idx])
        # A finger is considered extended if the tip is further from the wrist than the pip joint
        # (this works across different orientations more robustly than just y-coordinate checks).
        return np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist)

    fingers = {
        'thumb': finger_extended(4, 2),
        'index': finger_extended(8, 6),
        'middle': finger_extended(12, 10),
        'ring': finger_extended(16, 14),
        'pinky': finger_extended(20, 18)
    }

    # Index + middle finger up -> Selection mode (must be checked before single-finger drawing).
    if fingers['index'] and fingers['middle'] and not fingers['ring'] and not fingers['pinky']:
        return 'selection'

    # Index finger up -> Drawing mode
    if fingers['index'] and not fingers['middle'] and not fingers['ring'] and not fingers['pinky']:
        return 'drawing'

    # If most fingers are extended, treat as open palm (eraser)
    extended_count = sum(1 for v in fingers.values() if v)
    if extended_count >= 4:
        return 'open_palm'

    return 'neutral'


# ============ Motion Detection ============
class MotionDetector:
    """Advanced motion detection with improved smoothing and gesture classification."""
    
    def __init__(
        self,
        window_size: int = 6,
        smoothing_factor: float = 0.85,
        outlier_threshold: float = 200.0,
    ):
        """Initialize motion detector with responsive gesture tracking.
        
        Args:
            window_size: Number of position samples to keep (6-8 for responsiveness)
            smoothing_factor: Alpha for exponential smoothing (higher = follow raw tip more closely)
            outlier_threshold: Max jump (px) between frames; raise for fast finger motion while drawing
        """
        self.positions = deque(maxlen=window_size)
        self.velocities = deque(maxlen=window_size - 1)
        self.smoothed_pos = None
        self.smoothed_vel = 0.0
        self.alpha = smoothing_factor
        self.outlier_threshold = outlier_threshold
        self.stable_frames = 0
        self.stable_threshold = 2  # Frames to consider stable
        self.motion_frames = 0  # Frames of motion detected
        self.last_valid_pos = None
    
    def add_position(self, pos: Tuple[int, int]) -> None:
        """
        Add position with simple but effective outlier handling.
        
        Args:
            pos: Current hand position (x, y)
        """
        # Simple outlier rejection
        if self.positions:
            last = self.positions[-1]
            jump = np.sqrt((pos[0] - last[0])**2 + (pos[1] - last[1])**2)
            
            # Only reject extreme jumps
            if jump > self.outlier_threshold:
                return  # Skip this frame
        
        self.positions.append(pos)
        self.last_valid_pos = pos
        
        # Single-stage exponential smoothing for responsiveness
        if self.smoothed_pos is None:
            self.smoothed_pos = np.array(pos, dtype=np.float32)
        else:
            self.smoothed_pos = (self.alpha * np.array(pos, dtype=np.float32) + 
                                (1 - self.alpha) * self.smoothed_pos)
        
        # Calculate velocity
        if len(self.positions) >= 2:
            prev = self.positions[-2]
            curr = self.positions[-1]
            raw_vel = np.sqrt((curr[0] - prev[0])**2 + (curr[1] - prev[1])**2)
            
            # Simple velocity smoothing
            self.smoothed_vel = 0.5 * raw_vel + 0.5 * self.smoothed_vel
            self.velocities.append(self.smoothed_vel)
    
    def get_smoothed_position(self) -> Optional[Tuple[int, int]]:
        """Get smoothed hand position with sub-pixel precision."""
        if self.smoothed_pos is not None:
            return tuple(np.round(self.smoothed_pos).astype(int))
        return None
    
    def get_motion_state(self) -> Tuple[str, float]:
        """
        Classify motion state based on velocity.
        
        Returns:
            Tuple of (state, velocity) where state is 'stationary', 'moving', or 'fast'
        """
        if not self.velocities or len(self.velocities) < 1:
            return 'stationary', 0.0
        
        # Use recent velocity average
        recent_vels = list(self.velocities)[-2:]
        avg_vel = np.mean(recent_vels)
        
        # Simple velocity-based classification
        if avg_vel < 0.8:
            self.stable_frames += 1
            self.motion_frames = 0
            return 'stationary', avg_vel
        elif avg_vel < 5.0:
            self.stable_frames = 0
            self.motion_frames += 1
            return 'moving', avg_vel
        else:
            self.stable_frames = 0
            self.motion_frames += 1
            return 'fast', avg_vel
    
    def is_moving(self) -> bool:
        """Check if hand is currently moving."""
        return self.motion_frames >= 1
    
    def is_stable(self) -> bool:
        """Check if hand has been stationary for threshold frames."""
        return self.stable_frames >= self.stable_threshold
    
    def reset(self) -> None:
        """Clear all tracking data."""
        self.positions.clear()
        self.velocities.clear()
        self.smoothed_pos = None
        self.smoothed_vel = 0.0
        self.stable_frames = 0
        self.motion_frames = 0
        self.last_valid_pos = None


# ============ Canvas Configuration ============
CANVAS_WIDTH = 640
CANVAS_HEIGHT = 480
TOOLBAR_HEIGHT = 70

# UI Buttons definition
UI_BUTTONS = [
    {"label": "CLEAR", "color": (200, 200, 200), "text_color": (0, 0, 0), "action": "clear"},
    {"label": "ERASER", "color": (255, 255, 255), "text_color": (0, 0, 0), "action": "eraser"},
    {"label": "BLUE", "color": (255, 0, 0), "text_color": (255, 255, 255), "action": "color", "index": 0},
    {"label": "GREEN", "color": (0, 255, 0), "text_color": (255, 255, 255), "action": "color", "index": 1},
    {"label": "RED", "color": (0, 0, 255), "text_color": (255, 255, 255), "action": "color", "index": 2},
    {"label": "YELLOW", "color": (0, 255, 255), "text_color": (0, 0, 0), "action": "color", "index": 3},
]

# Colors in BGR format
COLORS = [
    (255, 0, 0),      # Blue (Index 0)
    (0, 255, 0),      # Green (Index 1)
    (0, 0, 255),      # Red (Index 2)
    (0, 255, 255),    # Yellow (Index 3)
    (0, 0, 0),        # Eraser (Index 4: draws black on paintWindow)
]
COLOR_NAMES = ['Blue', 'Green', 'Red', 'Yellow', 'Eraser']

# Brush settings
brush_size = 4
MIN_BRUSH = 1
MAX_BRUSH = 20

# ============ Canvas and Drawing State ============
if BACKEND_IMPORT_ERROR:
    paintWindow = None
else:
    # Persistent drawing layer (same shape as resized BGR frame — equivalent to np.zeros_like(frame)).
    paintWindow = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)

# Stroke storage: list of deques — each deque is one continuous stroke (finger down → finger up).
# Points are (x, y) in frame coordinates; not cleared each frame (only on clear_canvas / lost hand).
stroke_history: List[deque] = []
# Whether we were actively drawing on the previous frame (index up in canvas).
prev_drawing_frame = False

# State tracking
colorIndex = 0
current_color = COLORS[colorIndex]
is_drawing = False
last_finger_position = None
hand_in_toolbar = False
prev_was_drawing = False  # ink laid this stroke segment (for undo batching)
current_gesture = 'neutral'
motion_detector = MotionDetector()
missing_hand_frames = 0
MAX_MISSING_HAND_FRAMES = 8
# Light debounce only — avoids 1-frame gesture noise (was hiding all ink when set too high).
STABLE_DRAWING_FRAMES_NEEDED = 1
stable_drawing_frames = 0
_debug_frame_count = 0

# Statistics
stroke_count = 0
total_distance = 0.0
colors_used = {0}
start_time = datetime.now()
session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

# Undo/Redo
undo_stack = []
redo_stack = []
MAX_UNDO_HISTORY = 30
last_canvas_hash = None



def draw_ui_overlay(display_frame: np.ndarray, current_color_idx: int) -> None:
    """Draw a clean, professional top bar with color buttons and labels."""
    button_width = CANVAS_WIDTH // len(UI_BUTTONS)
    
    for i, btn in enumerate(UI_BUTTONS):
        x1, x2 = i * button_width, (i + 1) * button_width
        
        # Determine if this button represents the currently selected tool
        is_selected = False
        if btn["action"] == "eraser" and current_color_idx == 4:
            is_selected = True
        elif btn["action"] == "color" and current_color_idx == btn["index"]:
            is_selected = True

        # Draw the button interior
        cv2.rectangle(display_frame, (x1, 0), (x2, TOOLBAR_HEIGHT), btn["color"], -1)
        
        # Highlight border if selected, otherwise subtle border
        border_color = (0, 0, 0) if is_selected else (200, 200, 200)
        thickness = 3 if is_selected else 1
        cv2.rectangle(display_frame, (x1, 0), (x2, TOOLBAR_HEIGHT), border_color, thickness)
        
        # Centered text logic
        text_size = cv2.getTextSize(btn["label"], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x1 + (button_width - text_size[0]) // 2
        text_y = TOOLBAR_HEIGHT // 2 + 5
        
        # Draw shadow first for better readability
        cv2.putText(display_frame, btn["label"], (text_x + 1, text_y + 1),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
        cv2.putText(display_frame, btn["label"], (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, btn["text_color"], 2)


def canvas_hash() -> int:
    """Generate hash of canvas state."""
    return hash(paintWindow.tobytes())


def save_to_undo_stack() -> None:
    """Save canvas state to undo stack (only if changed)."""
    global undo_stack, redo_stack, last_canvas_hash
    
    current_hash = canvas_hash()
    if current_hash == last_canvas_hash:
        return
    
    with canvas_lock:
        undo_stack.append(paintWindow.copy())
        redo_stack.clear()
        last_canvas_hash = current_hash
        
        if len(undo_stack) > MAX_UNDO_HISTORY:
            undo_stack.pop(0)


def undo() -> bool:
    """Undo last drawing action."""
    global paintWindow, undo_stack, redo_stack, last_canvas_hash
    
    if not undo_stack:
        return False
    
    with canvas_lock:
        redo_stack.append(paintWindow.copy())
        paintWindow = undo_stack.pop()
        last_canvas_hash = canvas_hash()
    
    logger.info("Undo executed")
    return True


def redo() -> bool:
    """Redo last undone action."""
    global paintWindow, undo_stack, redo_stack, last_canvas_hash
    
    if not redo_stack:
        return False
    
    with canvas_lock:
        undo_stack.append(paintWindow.copy())
        paintWindow = redo_stack.pop()
        last_canvas_hash = canvas_hash()
    
    logger.info("Redo executed")
    return True


def clear_canvas() -> None:
    """Clear entire canvas and reset state."""
    if BACKEND_IMPORT_ERROR:
        raise RuntimeError(f"Cannot clear canvas: {BACKEND_IMPORT_ERROR}")

    global paintWindow, stroke_history
    global stroke_count, total_distance, colors_used
    global is_drawing, last_finger_position, prev_was_drawing
    global last_canvas_hash, colorIndex, current_color
    global missing_hand_frames, stable_drawing_frames
    global prev_drawing_frame
    
    with canvas_lock:
        paintWindow = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
        stroke_history.clear()
        
        stroke_count = 0
        total_distance = 0.0
        colors_used = {colorIndex}
        current_color = COLORS[colorIndex]
        is_drawing = False
        last_finger_position = None
        last_canvas_hash = None
        prev_was_drawing = False
        prev_drawing_frame = False
    
    motion_detector.reset()
    missing_hand_frames = 0
    stable_drawing_frames = 0
    logger.info("Canvas cleared")


def _begin_new_stroke_deque() -> None:
    """Append a new empty deque for the current finger-down stroke (call once per stroke start)."""
    global stroke_count
    stroke_history.append(deque(maxlen=8192))
    stroke_count += 1
    logger.info("Stroke #%d started (deque count=%d)", stroke_count, len(stroke_history))


def _append_point_active_stroke(pt: Tuple[int, int]) -> None:
    """Append a point to the latest stroke deque."""
    if not stroke_history:
        _begin_new_stroke_deque()
    stroke_history[-1].append(pt)


def _clamp_xy(x: int, y: int, w: int, h: int) -> Tuple[int, int]:
    """Keep drawing coordinates inside the frame (avoids off-canvas strokes)."""
    return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))


def _index_tip_pixel(hand_data: Dict) -> Tuple[int, int]:
    """MediaPipe landmark 8 (index fingertip) in pixel space, clamped — matches what you see on camera."""
    if 'landmarks' in hand_data and len(hand_data['landmarks']) > 8:
        x, y = hand_data['landmarks'][8]
    else:
        x, y = hand_data.get('index_tip', hand_data['center'])
    return _clamp_xy(int(round(x)), int(round(y)), CANVAS_WIDTH, CANVAS_HEIGHT)


def _interpolate_points(start: Tuple[int, int], end: Tuple[int, int], max_step: float = 4.0) -> List[Tuple[int, int]]:
    """Generate intermediate points so fast hand motion does not create gaps."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    distance = float(np.hypot(dx, dy))
    if distance <= max_step:
        return [end]

    steps = int(distance // max_step)
    points: List[Tuple[int, int]] = []
    for i in range(1, steps + 1):
        t = i / (steps + 1)
        points.append((int(start[0] + dx * t), int(start[1] + dy * t)))
    points.append(end)
    return points



def process_frame(frame: np.ndarray) -> np.ndarray:  # type: ignore[valid-type]
    """Process video frame: detect hand, update drawing state, render feedback.

    Args:
        frame: Input video frame (BGR)

    Returns:
        Processed frame with visualization
    """
    # Ensure required libraries loaded
    _ensure_backend_available()
    global paintWindow, colorIndex, current_color, is_drawing, last_finger_position
    global stroke_count, total_distance, colors_used
    global prev_was_drawing, hand_in_toolbar, motion_detector, current_gesture
    global missing_hand_frames, stable_drawing_frames
    global prev_drawing_frame, stroke_history, _debug_frame_count

    _debug_frame_count += 1

    frame = cv2.resize(frame, (CANVAS_WIDTH, CANVAS_HEIGHT))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    display_frame = frame.copy()
    
    # Detect hand
    hand_data = detect_hand(frame_rgb)
    
    # Draw top toolbar overlay
    draw_ui_overlay(display_frame, colorIndex)
    
    if not hand_data:
        # Tolerate short flickers; only reset stroke linkage after hand is gone for several frames.
        missing_hand_frames += 1
        if missing_hand_frames >= MAX_MISSING_HAND_FRAMES:
            is_drawing = False
            current_gesture = 'neutral'
            last_finger_position = None
            motion_detector.reset()
            stable_drawing_frames = 0

            if prev_drawing_frame and stroke_history:
                logger.info(
                    "Drawing paused (hand lost): last stroke points=%d",
                    len(stroke_history[-1]),
                )
            prev_drawing_frame = False
            if prev_was_drawing:
                prev_was_drawing = False
    else:
        missing_hand_frames = 0
        # Use raw index tip (landmark 8) for drawing — no heavy smoothing (that lagged behind the green dot).
        finger_tip = _index_tip_pixel(hand_data)
        motion_detector.add_position(finger_tip)

        hand_x, hand_y = finger_tip
        hand_in_toolbar = hand_y < TOOLBAR_HEIGHT

        # Determine gesture (draw / select / erase / neutral)
        current_gesture = 'neutral'
        if 'landmarks' in hand_data:
            current_gesture = classify_gesture(hand_data['landmarks'], frame.shape)

        # Do not treat toolbar area as drawing canvas (color UI uses gestures here).
        in_draw_area = hand_y >= TOOLBAR_HEIGHT
        # Debounce: only ink after stable "index only" pose — avoids lines during peace sign / palm / flicker.
        if current_gesture == 'drawing' and in_draw_area:
            stable_drawing_frames += 1
        else:
            stable_drawing_frames = 0
        can_draw = (
            current_gesture == 'drawing'
            and in_draw_area
            and stable_drawing_frames >= STABLE_DRAWING_FRAMES_NEEDED
        )
        # Peace sign / toolbar idle: end active stroke linkage (finger lifted semantically).
        if current_gesture == 'selection' or (
            current_gesture == 'neutral' and not in_draw_area
        ):
            if prev_drawing_frame and stroke_history:
                logger.info("Stroke segment ended (selection/toolbar): points=%d", len(stroke_history[-1]))
            prev_drawing_frame = False
            prev_was_drawing = False
            last_finger_position = None

        # Show current gesture mode on screen
        gesture_label = {
            'drawing': 'Draw (Index Up)',
            'selection': 'Select (2 Fingers)',
            'open_palm': 'Erase (Open Palm)',
            'neutral': 'Idle'
        }.get(current_gesture, 'Idle')
        cv2.putText(display_frame, f"Mode: {gesture_label}", (10, CANVAS_HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Visualization - show detected hand landmarks
        if 'hand_landmarks' in hand_data:
            # Draw landmark points and connections for feedback
            pts = hand_data['landmarks']
            for pt in pts:
                cv2.circle(display_frame, pt, 4, (0, 255, 0), -1)

            # Draw simple connections (thumb->index, index->middle, etc.)
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),        # Index
                (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
                (0, 13), (13, 14), (14, 15), (15, 16), # Ring
                (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            ]
            for a, b in connections:
                if a < len(pts) and b < len(pts):
                    cv2.line(display_frame, pts[a], pts[b], (0, 255, 0), 2)

            # Optional bounding box for more obvious detection
            if 'bbox' in hand_data:
                x1, y1, x2, y2 = hand_data['bbox']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        else:
            # Fallback: simple circle + contour overlay
            cv2.circle(display_frame, finger_tip, 10, (0, 255, 0), 2)
            if 'hull' in hand_data:
                cv2.drawContours(display_frame, [hand_data['hull']], 0, (100, 255, 100), 1)

        motion_state, velocity = motion_detector.get_motion_state()

        # If in toolbar region AND selection gesture, allow color selection
        if current_gesture == 'selection' and hand_in_toolbar:
            is_drawing = False

            # Select tool when hand is stable on toolbar
            if motion_detector.is_stable():
                button_width = CANVAS_WIDTH // len(UI_BUTTONS)
                button_idx = hand_x // button_width
                
                if 0 <= button_idx < len(UI_BUTTONS):
                    btn = UI_BUTTONS[button_idx]
                    if btn["action"] == "clear":
                        clear_canvas()
                        # Move finger out to avoid spamming clear
                        motion_detector.reset() 
                    elif btn["action"] == "eraser":
                        colorIndex = 4
                        current_color = COLORS[colorIndex]
                        print(f"current_color changed to: {current_color}")
                        colors_used.add(colorIndex)
                    elif btn["action"] == "color":
                        colorIndex = btn["index"]
                        current_color = COLORS[colorIndex]
                        print(f"current_color changed to: {current_color}")
                        colors_used.add(colorIndex)

            if prev_drawing_frame and stroke_history:
                logger.info("Stroke segment ended (toolbar UI): points=%d", len(stroke_history[-1]))
            prev_drawing_frame = False
            prev_was_drawing = False

        else:
            # Erase only on open palm; ink only when pose is stable (can_draw), not during other gestures.
            is_erasing = (current_gesture == 'open_palm')

            if can_draw:
                is_drawing = True
                motion_detector.motion_frames = max(1, min(3, motion_detector.motion_frames + 1))

                # New stroke only when finger just started drawing (index up) — not every frame.
                if not prev_drawing_frame:
                    _begin_new_stroke_deque()
                    colors_used.add(colorIndex)
                    save_to_undo_stack()
                    if _debug_frame_count % 30 == 0:
                        logger.info(
                            "Drawing mode ON | color=%s brush=%s",
                            COLOR_NAMES[colorIndex],
                            brush_size,
                        )

                prev_drawing_frame = True
                is_explicit_eraser = (colorIndex == 4)
                line_thickness = max(1, int(max(brush_size * 2, 20) if is_explicit_eraser else brush_size))

                if last_finger_position:
                    dx = finger_tip[0] - last_finger_position[0]
                    dy = finger_tip[1] - last_finger_position[1]
                    distance = float(np.hypot(dx, dy))
                    if distance > 0.05:
                        total_distance += distance

                    points_to_draw = _interpolate_points(last_finger_position, finger_tip)
                    prev_point = last_finger_position
                    with canvas_lock:
                        for p in points_to_draw:
                            cv2.line(
                                paintWindow,
                                prev_point,
                                p,
                                current_color,
                                line_thickness,
                                cv2.LINE_8,
                            )
                            _append_point_active_stroke(p)
                            prev_point = p
                    prev_was_drawing = True
                else:
                    # First sample of this stroke: draw a visible dot so short taps register.
                    with canvas_lock:
                        cv2.circle(
                            paintWindow,
                            finger_tip,
                            max(2, line_thickness // 2),
                            current_color,
                            -1,
                            lineType=cv2.LINE_8,
                        )
                        _append_point_active_stroke(finger_tip)
                    prev_was_drawing = True

                last_finger_position = finger_tip

                if _debug_frame_count % 45 == 0 and stroke_history:
                    logger.debug(
                        "Drawing active | last stroke points=%d | total stroke deques=%d",
                        len(stroke_history[-1]),
                        len(stroke_history),
                    )

                # Visual feedback - show brush size
                disp_color = (255, 255, 255) if colorIndex == 4 else (0, 255, 100)
                feedback_brush = int(max(brush_size * 2, 20)) if colorIndex == 4 else brush_size
                cv2.circle(display_frame, finger_tip, feedback_brush, disp_color, 2)

            # Index-up pose in canvas but debounce not satisfied yet.
            elif current_gesture == 'drawing' and in_draw_area and not can_draw:
                is_drawing = False
                motion_detector.motion_frames = 0
                last_finger_position = finger_tip

            # Drawing pose in toolbar: no ink.
            elif current_gesture == 'drawing' and hand_in_toolbar:
                is_drawing = False
                motion_detector.motion_frames = 0
                last_finger_position = None
                if prev_drawing_frame and stroke_history:
                    logger.info("Stroke segment ended (toolbar): points=%d", len(stroke_history[-1]))
                prev_drawing_frame = False

            # Eraser mode when open palm detected
            elif is_erasing:
                is_drawing = False
                motion_detector.motion_frames = max(1, min(3, motion_detector.motion_frames + 1))
                if prev_drawing_frame and stroke_history:
                    logger.debug("Stroke paused for erase gesture; last stroke points=%d", len(stroke_history[-1]))
                prev_drawing_frame = False

                erase_radius = int(max(brush_size * 2, 20))
                cv2.circle(display_frame, finger_tip, erase_radius, (255, 255, 255), 2)

                if last_finger_position:
                    with canvas_lock:
                        cv2.circle(paintWindow, finger_tip, erase_radius, (0, 0, 0), -1)
                        # Also draw line to prevent dotty erasing
                        cv2.line(paintWindow, last_finger_position, finger_tip, (0, 0, 0), erase_radius * 2, cv2.LINE_8)

                if not prev_was_drawing:
                    save_to_undo_stack()
                prev_was_drawing = True
                last_finger_position = finger_tip

            else:
                # Idle / not drawing (index not up): finger lifted — stroke ends for next down stroke.
                is_drawing = False
                motion_detector.motion_frames = 0
                last_finger_position = finger_tip # Keep track to prevent jump lines next time

                if prev_drawing_frame and stroke_history:
                    logger.info("Stroke segment ended (finger up / idle): points=%d", len(stroke_history[-1]))
                prev_drawing_frame = False
                prev_was_drawing = False
    
    # Display status info
    status = f"Brush: {brush_size}px | Mode: {current_gesture.capitalize()} | Color: {COLOR_NAMES[colorIndex]} | Strokes: {stroke_count}"
    cv2.putText(display_frame, status, (10, CANVAS_HEIGHT - 15),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[colorIndex], 1)
    
    if is_drawing:
        cv2.putText(display_frame, "✓ DRAWING", (CANVAS_WIDTH - 180, CANVAS_HEIGHT - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Overlay persistent canvas (same shape as frame). cv2.add() saturates on bright skin/video and
    # hides strokes; np.maximum keeps ink visible on light backgrounds.
    with canvas_lock:
        output_frame = np.maximum(display_frame.astype(np.uint8), paintWindow.astype(np.uint8))

    return output_frame



def get_statistics() -> Dict:
    """Get comprehensive drawing statistics."""
    try:
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if stroke_count > 0 and elapsed > 0:
            strokes_per_min = int((stroke_count / elapsed) * 60)
            avg_distance = int(total_distance / stroke_count)
        else:
            strokes_per_min = 0
            avg_distance = 0
        
        color_list = sorted(list(colors_used))
        color_name_list = [COLOR_NAMES[i] for i in color_list if 0 <= i < len(COLOR_NAMES)]
        
        return {
            'strokes': int(stroke_count),
            'time': int(elapsed),
            'distance': int(total_distance),
            'colors_used': color_list,
            'color_names': color_name_list,
            'avg_distance': avg_distance,
            'strokes_per_minute': strokes_per_min,
            'brush_size': int(brush_size),
            'is_drawing': bool(is_drawing),
            'current_color': COLOR_NAMES[colorIndex],
            'session_id': str(session_id)
        }
    except Exception as e:
        logger.error(f"Error in get_statistics: {e}")
        return {
            'strokes': 0,
            'time': 0,
            'distance': 0,
            'colors_used': [0],
            'color_names': ['Red'],
            'avg_distance': 0,
            'strokes_per_minute': 0,
            'brush_size': brush_size,
            'is_drawing': False,
            'current_color': 'Red',
            'session_id': session_id,
            'error': str(e)
        }


def set_brush_size(size: int) -> None:
    """Set brush size (constrained between MIN and MAX)."""
    global brush_size
    brush_size = max(MIN_BRUSH, min(MAX_BRUSH, int(size)))
    logger.info(f"Brush size set to {brush_size}px")


def set_color(color_idx: int) -> bool:
    """Set current color by index."""
    global colorIndex, colors_used, current_color
    if 0 <= color_idx < len(COLORS):
        colorIndex = color_idx
        current_color = COLORS[colorIndex]
        print(f"current_color changed to: {current_color}")
        colors_used.add(color_idx)
        logger.info(f"Color changed to {COLOR_NAMES[colorIndex]}")
        return True
    return False


def get_current_color() -> Dict:
    """Get current color information."""
    bgr = COLORS[colorIndex]
    return {
        'name': COLOR_NAMES[colorIndex],
        'bgr': bgr,
        'index': colorIndex
    }


def save_drawing(filename: Optional[str] = None) -> Dict:
    """Save drawing to file.

    Args:
        filename: Custom filename (with extension). If None, generates timestamp-based name.

    Returns:
        Dictionary with success status and details
    """
    if BACKEND_IMPORT_ERROR:
        return {'success': False, 'error': BACKEND_IMPORT_ERROR}

    try:
        if filename is None:
            filename = f"saved_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        filepath = f"static/{filename}"

        with canvas_lock:
            cv2.imwrite(filepath, paintWindow)

        logger.info(f"Drawing saved to {filepath}")
        return {'success': True, 'filename': filename, 'path': filepath}
    except Exception as e:
        logger.error(f"Error saving drawing: {e}")
        return {'success': False, 'error': str(e)}


def get_canvas() -> np.ndarray:  # type: ignore[valid-type]
    """Get copy of current canvas."""
    with canvas_lock:
        return paintWindow.copy()


def reset_session() -> None:
    """Reset entire session (clear canvas and reset statistics)."""
    global start_time, session_id
    clear_canvas()
    start_time = datetime.now()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("Session reset")


def get_gesture_info() -> List[Dict]:
    """Get hand gesture instructions."""
    return [
        {'gesture': 'Move your hand inside the frame', 'action': 'Draw on the canvas'},
        {'gesture': 'Hold still over the top toolbar', 'action': 'Change color'},
        {'gesture': 'Use UI buttons for save/clear/undo/redo', 'action': 'Control the canvas'}
    ]

