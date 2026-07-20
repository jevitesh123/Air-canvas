"""WSGI entry point for deployment from the repository root."""
import os
import sys
from pathlib import Path

AIR_CANVAS_DIR = Path(__file__).resolve().parent / "Air_Canvas"
sys.path.insert(0, str(AIR_CANVAS_DIR))
os.chdir(AIR_CANVAS_DIR)

from app import app  # noqa: E402
