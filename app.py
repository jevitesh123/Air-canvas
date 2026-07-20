"""Gunicorn entry point when Render runs `gunicorn app:app` from repo root."""
import importlib.util
import os
import sys
from pathlib import Path

AIR_CANVAS_DIR = Path(__file__).resolve().parent / "Air_Canvas"
sys.path.insert(0, str(AIR_CANVAS_DIR))
os.chdir(AIR_CANVAS_DIR)

_spec = importlib.util.spec_from_file_location(
    "air_canvas_flask_app",
    AIR_CANVAS_DIR / "app.py",
)
_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_module)

app = _module.app
