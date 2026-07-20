"""
Air Canvas - Flask Backend Server
Handles video streaming, drawing state management, and API endpoints.
"""

from flask import Flask, render_template, Response, jsonify, request
import os
import atexit
import time
from datetime import datetime
import logging
import threading
import sys

import numpy as np

def is_python_stable():
    """Check if the current Python version is a stable (final) release."""
    # sys.version_info.releaselevel can be 'alpha', 'beta', 'candidate', or 'final'.
    return sys.version_info.releaselevel == 'final'

# Attempt to import the drawing backend; if it fails, we will run a minimal error server.
AIR_CANVAS_ERROR = None
try:
    import air_canvas

    # Detect if the backend failed to load key dependencies (numpy/cv2)
    backend_error = getattr(air_canvas, 'BACKEND_IMPORT_ERROR', None)
    if backend_error:
        AIR_CANVAS_ERROR = backend_error
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.getLogger(__name__).error(f"air_canvas backend error: {backend_error}")
except Exception as e:
    air_canvas = None  # type: ignore
    AIR_CANVAS_ERROR = str(e)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger(__name__).error(f"Failed to import air_canvas: {e}")

# Attempt to import OpenCV (cv2). If it fails, we continue running but will return errors from the API.
CV2_ERROR = None
try:
    import cv2
except Exception as e:
    cv2 = None  # type: ignore
    CV2_ERROR = str(e)
    if AIR_CANVAS_ERROR is None:
        AIR_CANVAS_ERROR = CV2_ERROR
    logging.getLogger(__name__).error(f"Failed to import cv2: {e}")

# ============ Logging Configuration ============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check Python stability on startup
if not is_python_stable():
    logger.warning(f"Running on pre-release Python: {sys.version}. This may cause issues with NumPy/OpenCV. Use a stable release like 3.11.x.")

# ============ Flask Setup ============
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# ============ Camera State ============
camera = None
camera_lock = threading.Lock()

APP_STATUS = {
    'status': 'Initializing...',
    'frames_processed': 0,
    'last_action': 'None',
    'fps': 0,
    'last_error': None,
    'hand_detected': False
}


# ============ Camera Management ============
def init_camera() -> bool:
    """Initialize camera with error handling."""
    global camera

    if CV2_ERROR:
        logger.error(f"Cannot initialize camera because cv2 failed to import: {CV2_ERROR}")
        APP_STATUS['last_error'] = f"cv2 import error: {CV2_ERROR}"
        return False

    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("Failed to open camera - device not available")
            APP_STATUS['last_error'] = 'Camera initialization failed'
            return False

        # Set camera properties
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)

        logger.info("Camera initialized successfully")
        APP_STATUS['status'] = 'Ready'
        return True
    except Exception as e:
        logger.error(f"Camera initialization error: {e}")
        APP_STATUS['last_error'] = str(e)
        return False


def release_camera() -> None:
    """Release camera resources (process exit only — not called during streaming)."""
    global camera
    try:
        if camera and camera.isOpened():
            camera.release()
            logger.info("Camera released")
    except Exception as e:
        logger.error(f"Error releasing camera: {e}")


def _server_camera_available() -> bool:
    """Check whether a local/server webcam can be opened."""
    if CV2_ERROR or cv2 is None:
        return False

    with camera_lock:
        if camera is not None and camera.isOpened():
            return True

    test_camera = None
    try:
        test_camera = cv2.VideoCapture(0)
        return test_camera.isOpened()
    except Exception:
        return False
    finally:
        if test_camera is not None:
            test_camera.release()


# Release hardware on interpreter exit; do not tie this to app.run() return order.
atexit.register(release_camera)


def _placeholder_frame(message: str = "Camera unavailable — check device / permissions"):
    """Single-color frame so MJPEG stream never dies when the camera is busy."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (32, 32, 32)
    cv2.putText(img, message, (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    return img


def generate_frames():
    """
    MJPEG generator: runs forever, yields multipart JPEG chunks.
    Camera is read under lock; never release inside this loop.
    """
    global camera, APP_STATUS

    frame_count = 0
    APP_STATUS['status'] = 'Running'
    logger.info("Video stream generator started")

    while True:
        frame = None
        try:
            with camera_lock:
                if CV2_ERROR or cv2 is None:
                    pass
                elif camera is None or not camera.isOpened():
                    init_camera()
                if camera is not None and camera.isOpened():
                    ok, frame = camera.read()
                    if not ok:
                        frame = None
        except Exception as e:
            logger.error(f"Camera read error: {e}")
            frame = None

        if frame is None:
            APP_STATUS['status'] = 'Camera Error'
            time.sleep(0.05)
            try:
                _, buffer = cv2.imencode('.jpg', _placeholder_frame(), [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception:
                pass
            continue

        try:
            frame = cv2.flip(frame, 1)
            if AIR_CANVAS_ERROR or air_canvas is None:
                pass
            else:
                frame = air_canvas.process_frame(frame)

            ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            frame_count += 1
            APP_STATUS['frames_processed'] = frame_count
            APP_STATUS['status'] = 'Running'

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            APP_STATUS['last_error'] = str(e)
            try:
                _, buffer = cv2.imencode('.jpg', _placeholder_frame("Processing error"), [cv2.IMWRITE_JPEG_QUALITY, 75])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            except Exception:
                time.sleep(0.05)

# ============ API Routes ============

@app.route('/')
def home():
    """Serve main application page."""
    if AIR_CANVAS_ERROR:
        # Provide a friendly message when dependencies are broken
        version_info = f"You are running Python {sys.version.split()[0]}."
        stability_note = "" if is_python_stable() else " <strong>This appears to be a pre-release version, which may cause issues with binary packages.</strong>"
        return (
            f"<h2>Air Canvas cannot start</h2>"
            f"<p><strong>Error:</strong> {AIR_CANVAS_ERROR}</p>"
            f"<p>{version_info}{stability_note}</p>"
            f"<p>Please ensure you are using a stable Python version (e.g., 3.10.15 or 3.11.x) "
            f"and that all requirements are installed (see requirements.txt).</p>"
            f"<p>On Windows, try: <code>py -3.11 -m venv .venv && .\\.venv\\Scripts\\Activate.ps1 && pip install -r requirements.txt</code></p>",
            500
        )

    try:
        logger.info("Home page accessed")
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error loading home page: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/video')
def video():
    """Stream video feed with hand gesture detection."""
    if AIR_CANVAS_ERROR:
        return jsonify({'error': 'Application dependencies missing', 'details': AIR_CANVAS_ERROR}), 500
    if CV2_ERROR or cv2 is None:
        return jsonify({'error': 'OpenCV (cv2) not available', 'details': CV2_ERROR}), 503

    try:
        with camera_lock:
            if not camera or not camera.isOpened():
                if not init_camera():
                    return jsonify({'error': 'Camera unavailable', 'details': APP_STATUS.get('last_error')}), 503
        logger.debug("Video stream requested")
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    except Exception as e:
        logger.error(f"Video stream error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save', methods=['GET', 'POST'])
def save():
    """Save current drawing to file"""
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"saved_{timestamp}.png"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save drawing
        result = air_canvas.save_drawing(filename)
        
        if result['success']:
            logger.info(f"Drawing saved: {filename}")
            APP_STATUS['last_action'] = f'Saved: {filename}'
            return jsonify({
                "message": "✅ Saved Successfully!",
                "filename": filename,
                "filepath": filepath,
                "status": "success",
                "timestamp": timestamp
            }), 200
        else:
            raise Exception(result.get('error', 'Unknown error'))
    
    except Exception as e:
        logger.error(f"Error saving drawing: {e}")
        APP_STATUS['last_action'] = f'Error: {str(e)}'
        APP_STATUS['last_error'] = str(e)
        return jsonify({
            "message": f"❌ Error saving: {str(e)}",
            "status": "error"
        }), 500

@app.route('/clear', methods=['GET', 'POST'])
def clear():
    """Clear the drawing canvas."""
    try:
        air_canvas.clear_canvas()
        logger.info("Canvas cleared")
        APP_STATUS['last_action'] = 'Canvas cleared'
        return jsonify({
            "message": "✅ Canvas Cleared!",
            "status": "success"
        }), 200
    except Exception as e:
        logger.error(f"Error clearing canvas: {e}")
        APP_STATUS['last_error'] = str(e)
        return jsonify({
            "message": f"❌ Error clearing: {str(e)}",
            "status": "error"
        }), 500

@app.route('/undo', methods=['GET', 'POST'])
def undo():
    """Undo last drawing action."""
    try:
        success = air_canvas.undo()
        if success:
            logger.info("Undo executed")
            APP_STATUS['last_action'] = 'Undo'
            return jsonify({
                "message": "↶ Undone",
                "status": "success"
            }), 200
        else:
            return jsonify({
                "message": "⚠️ Nothing to undo",
                "status": "warning"
            }), 200
    except Exception as e:
        logger.error(f"Error undoing: {e}")
        APP_STATUS['last_error'] = str(e)
        return jsonify({
            "message": f"❌ Error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/redo', methods=['GET', 'POST'])
def redo():
    """Redo last undone action."""
    try:
        success = air_canvas.redo()
        if success:
            logger.info("Redo executed")
            APP_STATUS['last_action'] = 'Redo'
            return jsonify({
                "message": "↷ Redone",
                "status": "success"
            }), 200
        else:
            return jsonify({
                "message": "⚠️ Nothing to redo",
                "status": "warning"
            }), 200
    except Exception as e:
        logger.error(f"Error redoing: {e}")
        APP_STATUS['last_error'] = str(e)
        return jsonify({
            "message": f"❌ Error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/status')
def status():
    """Get current application and drawing status."""
    try:
        status_data = APP_STATUS.copy()
        stats = air_canvas.get_statistics()
        status_data['drawing_stats'] = stats

        status_data['mediapipe'] = {
            'available': getattr(air_canvas, 'USE_MEDIAPIPE', False),
            'forced': getattr(air_canvas, 'FORCE_MEDIAPIPE', False),
            'error': getattr(air_canvas, 'MEDIA_PIPE_ERROR', None)
        }

        return jsonify(status_data), 200
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/toggle-mediapipe', methods=['POST'])
def toggle_mediapipe():
    """Force attempt to enable (or disable) MediaPipe hand tracking."""
    try:
        data = request.json or {}
        enable = data.get('enable', True)

        # Update the force flag and re-init
        air_canvas.FORCE_MEDIAPIPE = bool(enable)
        initialized = air_canvas.init_mediapipe(force=True)

        return jsonify({
            'mediapipe_available': initialized,
            'force': air_canvas.FORCE_MEDIAPIPE,
            'error': getattr(air_canvas, 'MEDIA_PIPE_ERROR', None)
        }), 200
    except Exception as e:
        logger.error(f"Error toggling MediaPipe: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/stats')
def stats():
    """Get detailed drawing statistics."""
    try:
        stats = air_canvas.get_statistics()
        return jsonify(stats), 200
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to get statistics'
        }), 500

@app.route('/set-brush-size', methods=['POST'])
def set_brush_size():
    """Set the brush size for drawing."""
    try:
        data = request.get_json(silent=True) or {}
        size = int(data.get('size', 4))
        air_canvas.set_brush_size(size)
        logger.info(f"Brush size set to {size}")
        return jsonify({
            "message": f"Brush size set to {size}px",
            "size": size,
            "status": "success"
        }), 200
    except Exception as e:
        logger.error(f"Error setting brush size: {e}")
        return jsonify({
            "message": f"❌ Error: {str(e)}",
            "status": "error"
        }), 400

@app.route('/set-color', methods=['POST'])
def set_color():
    """Set the drawing color (0-3 index)."""
    try:
        data = request.get_json(silent=True) or {}
        color_idx = int(data.get('color', 0))
        
        success = air_canvas.set_color(color_idx)
        if success:
            color_info = air_canvas.get_current_color()
            logger.info(f"Color changed to {color_info['name']}")
            return jsonify({
                "message": f"Color changed to {color_info['name']}",
                "color": color_info,
                "status": "success"
            }), 200
        else:
            return jsonify({
                "message": "Invalid color index",
                "status": "error"
            }), 400
    except Exception as e:
        logger.error(f"Error setting color: {e}")
        return jsonify({
            "message": f"❌ Error: {str(e)}",
            "status": "error"
        }), 400

@app.route('/gesture-help')
def gesture_help():
    """Get gesture instructions."""
    try:
        instructions = air_canvas.get_gesture_info()
        return jsonify({
            "gestures": instructions,
            "status": "success"
        }), 200
    except Exception as e:
        logger.error(f"Error getting gesture help: {e}")
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

@app.route('/reset-session', methods=['POST'])
def reset_session():
    """Reset the entire drawing session."""
    try:
        air_canvas.reset_session()
        logger.info("Session reset")
        APP_STATUS['last_action'] = 'Session reset'
        return jsonify({
            "message": "✅ Session Reset!",
            "status": "success"
        }), 200
    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        return jsonify({
            "message": f"❌ Error: {str(e)}",
            "status": "error"
        }), 500

@app.route('/camera-status')
def camera_status():
    """Report whether the server has a usable webcam."""
    force_browser = str(os.getenv('USE_BROWSER_CAMERA', '0')).lower() in ('1', 'true', 'yes')
    available = False
    if not force_browser and not CV2_ERROR and cv2 is not None:
        available = _server_camera_available()

    return jsonify({
        'camera_available': available,
        'browser_camera_required': force_browser or not available,
        'mode': 'browser' if (force_browser or not available) else 'server',
    }), 200


@app.route('/process-frame', methods=['POST'])
def process_frame_route():
    """Process a browser webcam frame and return the annotated JPEG."""
    if AIR_CANVAS_ERROR:
        return jsonify({'error': 'Application dependencies missing', 'details': AIR_CANVAS_ERROR}), 500
    if CV2_ERROR or cv2 is None:
        return jsonify({'error': 'OpenCV (cv2) not available', 'details': CV2_ERROR}), 503

    try:
        raw = None
        if request.files.get('frame'):
            raw = request.files['frame'].read()
        elif request.is_json:
            import base64
            payload = request.get_json(silent=True) or {}
            image_data = payload.get('image', '')
            if ',' in image_data:
                image_data = image_data.split(',', 1)[1]
            raw = base64.b64decode(image_data)

        if not raw:
            return jsonify({'error': 'No frame data provided'}), 400

        frame_array = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'error': 'Invalid image data'}), 400

        frame = cv2.flip(frame, 1)
        if air_canvas is not None:
            frame = air_canvas.process_frame(frame)

        APP_STATUS['frames_processed'] = APP_STATUS.get('frames_processed', 0) + 1
        APP_STATUS['status'] = 'Running'

        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return jsonify({'error': 'Failed to encode frame'}), 500

        return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        logger.error(f"Process frame error: {e}")
        APP_STATUS['last_error'] = str(e)
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    if AIR_CANVAS_ERROR:
        return jsonify({
            "status": "unhealthy",
            "error": AIR_CANVAS_ERROR,
            "timestamp": datetime.now().isoformat()
        }), 500

    return jsonify({
        "status": "healthy",
        "app_status": APP_STATUS['status'],
        "camera_available": _server_camera_available(),
        "browser_camera_required": not _server_camera_available(),
        "timestamp": datetime.now().isoformat()
    }), 200

# ============ Error Handlers ============
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    logger.warning(f"404 error: {error}")
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"500 error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


# ============ Application Startup ============
if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Open camera once at startup; streaming loop also retries if needed.
    if not init_camera():
        logger.warning("Camera not ready at startup — will retry when /video is opened.")

    port = int(os.getenv('PORT', 5000))
    host = os.getenv('FLASK_HOST', '0.0.0.0' if os.getenv('PORT') else '127.0.0.1')
    logger.info("Starting Flask — open http://%s:%s/ (Ctrl+C to stop)", host, port)
    # use_reloader=False: avoids double process on Windows and keeps a single VideoCapture.
    app.run(
        debug=False,
        host=host,
        port=port,
        threaded=True,
        use_reloader=False,
    )