"""Frame and video analysis; outputs visual-cue summaries, not attention scores."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detection import FaceDetector
from model import get_classifier
from utils import CLASS_NAMES, DISPLAY_NAMES


def empty_counts() -> dict[str, int]:
    return {name: 0 for name in CLASS_NAMES}


def analyze_frame(
    frame: np.ndarray,
    detector: FaceDetector | None = None,
    classifier: Any | None = None,
) -> dict[str, Any]:
    detector = detector or FaceDetector()
    classifier = classifier or get_classifier()
    annotated = frame.copy()
    counts = empty_counts()
    sources: set[str] = set()
    boxes = detector.detect(frame)
    if not boxes:
        sources.add("trained CNN" if getattr(classifier, "is_trained", False) else "demo")

    for index, (x, y, width, height) in enumerate(boxes, start=1):
        crop = frame[y : y + height, x : x + width]
        label, confidence, source = classifier.predict(crop)
        if label not in counts:
            label = "unclear"
        counts[label] += 1
        sources.add(source)
        display = DISPLAY_NAMES[label]
        caption = f"subject-{index:02d} | {display} | {confidence:.0%}"
        color = (56, 151, 95) if label != "unclear" else (40, 147, 190)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            annotated,
            caption,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    return {
        "annotated": annotated,
        "counts": counts,
        "detected_subjects": len(boxes),
        "observations": len(boxes),
        "model_mode": _mode_label(sources),
    }


def analyze_image(frame: np.ndarray) -> dict[str, Any]:
    return analyze_frame(frame)


def analyze_video(
    video_path: str | Path,
    sample_seconds: float = 1.0,
    max_samples: int = 300,
) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded video could not be opened.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, int(round((fps if fps > 0 else 1.0) * sample_seconds)))
    counts = empty_counts()
    total_observations = 0
    peak_subjects = 0
    sample_count = 0
    frame_index = 0
    final_image: np.ndarray | None = None
    modes: set[str] = set()
    detector = FaceDetector()
    classifier = get_classifier()

    try:
        while sample_count < max_samples:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_step == 0:
                result = analyze_frame(frame, detector=detector, classifier=classifier)
                final_image = result["annotated"]
                total_observations += result["observations"]
                peak_subjects = max(peak_subjects, result["detected_subjects"])
                for name in CLASS_NAMES:
                    counts[name] += result["counts"][name]
                modes.add(result["model_mode"])
                sample_count += 1
            frame_index += 1
    finally:
        capture.release()

    if final_image is None:
        raise ValueError("No readable frames were found in the uploaded video.")
    return {
        "annotated": final_image,
        "counts": counts,
        "detected_subjects": peak_subjects,
        "observations": total_observations,
        "sampled_frames": sample_count,
        "model_mode": _mode_label(modes),
    }


def _mode_label(sources: set[str]) -> str:
    if not sources:
        return "demo (no usable trained model)"
    if all(source == "trained CNN" for source in sources):
        return "trained CNN"
    return "demo (no usable trained model)"