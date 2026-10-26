"""Lightweight face-region detection using OpenCV's bundled Haar cascades."""

from pathlib import Path

import cv2
import numpy as np


class FaceDetector:
    def __init__(self) -> None:
        cascade_dir = Path(cv2.data.haarcascades)
        self._cascades = [
            cv2.CascadeClassifier(str(cascade_dir / "haarcascade_frontalface_default.xml")),
            cv2.CascadeClassifier(str(cascade_dir / "haarcascade_profileface.xml")),
        ]
        if any(cascade.empty() for cascade in self._cascades):
            raise RuntimeError("OpenCV face detector files could not be loaded.")

    def detect(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        candidates: list[tuple[int, int, int, int]] = []
        for cascade in self._cascades:
            boxes = cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28)
            )
            candidates.extend(tuple(int(value) for value in box) for box in boxes)

        accepted: list[tuple[int, int, int, int]] = []
        for box in sorted(candidates, key=lambda item: item[2] * item[3], reverse=True):
            x, y, width, height = box
            overlaps = False
            for other_x, other_y, other_width, other_height in accepted:
                intersection_width = max(0, min(x + width, other_x + other_width) - max(x, other_x))
                intersection_height = max(0, min(y + height, other_y + other_height) - max(y, other_y))
                intersection = intersection_width * intersection_height
                union = width * height + other_width * other_height - intersection
                if union and intersection / union > 0.35:
                    overlaps = True
                    break
            if not overlaps:
                accepted.append(box)
        return accepted[:40]