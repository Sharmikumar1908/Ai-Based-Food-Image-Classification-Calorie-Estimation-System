"""Input preparation shared by training and inference."""

import cv2
import numpy as np

IMAGE_SIZE = (96, 96)


def prepare_crop(face_crop: np.ndarray) -> np.ndarray:
    """Resize a BGR face crop and return a normalized RGB batch of size one."""
    if face_crop is None or face_crop.size == 0:
        raise ValueError("The detected face crop is empty.")
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[np.newaxis, ...]