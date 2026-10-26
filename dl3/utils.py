"""Shared project paths, labels, and upload helpers."""

from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = BASE_DIR / "dataset"
RESULTS_DIR = BASE_DIR / "results"
DATABASE_PATH = BASE_DIR / "results" / "classroom_sessions.sqlite3"

CLASS_NAMES = ("facing_forward", "looking_away", "head_down", "unclear")
DISPLAY_NAMES = {
    "facing_forward": "Facing forward",
    "looking_away": "Looking away",
    "head_down": "Head down",
    "unclear": "Unclear",
}
IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}
VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}


def ensure_project_dirs() -> None:
    for directory in (UPLOAD_DIR, MODEL_DIR, DATASET_DIR, RESULTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def allowed_extension(filename: str, extensions: set[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in extensions


def decode_image(file_bytes: bytes) -> np.ndarray | None:
    encoded = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def image_data_url(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not success:
        raise ValueError("Could not encode the processed image.")
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")