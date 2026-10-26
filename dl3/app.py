"""Flask entry point for the local classroom visual-cue prototype."""

import base64
import csv
import io
import secrets
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask_bootstrap import Bootstrap5
from flask import Flask, Response, flash, redirect, render_template, request, url_for

import database
from analysis import analyze_frame, analyze_video
from utils import (
    CLASS_NAMES,
    DATABASE_PATH,
    DISPLAY_NAMES,
    IMAGE_EXTENSIONS,
    UPLOAD_DIR,
    VIDEO_EXTENSIONS,
    allowed_extension,
    decode_image,
    ensure_project_dirs,
    image_data_url,
)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def create_app(test_config: dict[str, Any] | None = None) -> Flask:
    ensure_project_dirs()
    application = Flask(__name__)
    application.config.update(
        SECRET_KEY=secrets.token_hex(24),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES,
        DATABASE=str(DATABASE_PATH),
        UPLOAD_FOLDER=str(UPLOAD_DIR),
    )
    if test_config:
        application.config.update(test_config)
    Bootstrap5(application)
    database.init_db(application.config["DATABASE"])

    @application.get("/")
    def home():
        return render_template("home.html")

    @application.route("/analyze", methods=["GET", "POST"])
    def analyze():
        if request.method == "GET":
            return render_template("analyze.html")

        captured_image = request.form.get("webcam_image", "")
        if captured_image:
            result = _analyze_webcam_capture(captured_image)
            if isinstance(result, str):
                flash(result, "danger")
                return redirect(url_for("analyze"))
            return _save_and_render(result, "webcam")

        uploaded = request.files.get("media")
        if uploaded is None or not uploaded.filename:
            flash("Choose an image or video, or capture a webcam frame.", "warning")
            return redirect(url_for("analyze"))

        filename = uploaded.filename
        if allowed_extension(filename, IMAGE_EXTENSIONS):
            frame = decode_image(uploaded.read())
            if frame is None:
                flash("That image could not be decoded. Try a different image file.", "danger")
                return redirect(url_for("analyze"))
            return _save_and_render(analyze_frame(frame), "image")

        if not allowed_extension(filename, VIDEO_EXTENSIONS):
            flash("Unsupported format. Use JPG, PNG, WEBP, MP4, AVI, MOV, or MKV.", "danger")
            return redirect(url_for("analyze"))

        suffix = Path(filename).suffix.lower()
        temporary_path = Path(application.config["UPLOAD_FOLDER"]) / f"{secrets.token_hex(12)}{suffix}"
        temporary_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            uploaded.save(temporary_path)
            result = analyze_video(temporary_path)
            return _save_and_render(result, "video")
        except (ValueError, OSError) as error:
            flash(str(error), "danger")
            return redirect(url_for("analyze"))
        finally:
            temporary_path.unlink(missing_ok=True)

    @application.get("/history")
    def history():
        return render_template("history.html", sessions=database.list_sessions(application.config["DATABASE"]))

    @application.get("/history.csv")
    def export_history():
        output = io.StringIO(newline="")
        columns = ["id", "created_at", "media_type", "detected_subjects", "observations", "model_mode", *CLASS_NAMES]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(database.list_sessions(application.config["DATABASE"]))
        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=classroom_session_summaries.csv"},
        )

    @application.post("/history/<int:session_id>/delete")
    def remove_session(session_id: int):
        database.delete_session(session_id, application.config["DATABASE"])
        flash("Session summary deleted.", "success")
        return redirect(url_for("history"))

    @application.get("/sample")
    def sample():
        counts = {"facing_forward": 2, "looking_away": 1, "head_down": 1, "unclear": 2}
        result = {
            "annotated": _create_sample_preview(),
            "counts": counts,
            "detected_subjects": sum(counts.values()),
            "observations": sum(counts.values()),
            "model_mode": "simulated preview (not an inference)",
        }
        return render_template(
            "results.html",
            result=result,
            processed_image=image_data_url(result["annotated"]),
            display_names=DISPLAY_NAMES,
            simulated=True,
            session_id=None,
        )

    @application.get("/about")
    def about():
        return render_template("about.html")

    @application.errorhandler(413)
    def request_too_large(_error):
        flash("Upload is larger than the 100 MB limit.", "danger")
        return redirect(url_for("analyze"))

    def _save_and_render(result: dict[str, Any], media_type: str):
        summary = {**result, "media_type": media_type}
        session_id = database.add_session(summary, application.config["DATABASE"])
        return render_template(
            "results.html",
            result=result,
            processed_image=image_data_url(result["annotated"]),
            display_names=DISPLAY_NAMES,
            simulated=False,
            session_id=session_id,
        )

    return application


def _analyze_webcam_capture(data_url: str) -> dict[str, Any] | str:
    try:
        encoded = data_url.split(",", 1)[1]
        image_bytes = base64.b64decode(encoded, validate=True)
    except (IndexError, ValueError):
        return "The webcam image was not valid. Capture a new frame and try again."
    frame = decode_image(image_bytes)
    if frame is None:
        return "The webcam image could not be decoded. Capture a new frame and try again."
    return analyze_frame(frame)


def _create_sample_preview() -> np.ndarray:
    """Draw a synthetic classroom preview; these boxes and labels are illustrative only."""
    canvas = np.full((440, 960, 3), (232, 237, 232), dtype=np.uint8)
    cv2.rectangle(canvas, (60, 35), (900, 205), (52, 74, 72), -1)
    cv2.putText(canvas, "SIMULATED PREVIEW - NOT AN INFERENCE", (82, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 247, 242), 2)
    labels = ["Facing forward", "Facing forward", "Looking away", "Head down", "Unclear", "Unclear"]
    colors = [(56, 151, 95), (56, 151, 95), (54, 120, 185), (54, 120, 185), (40, 147, 190), (40, 147, 190)]
    for index, (label, color) in enumerate(zip(labels, colors)):
        column = index % 3
        row = index // 3
        center_x = 210 + column * 270
        center_y = 285 + row * 115
        cv2.circle(canvas, (center_x, center_y), 28, (174, 132, 101), -1)
        cv2.rectangle(canvas, (center_x - 48, center_y - 44), (center_x + 48, center_y + 48), color, 2)
        cv2.putText(canvas, f"sample-{index + 1:02d} | {label}", (center_x - 75, center_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (35, 48, 46), 1, cv2.LINE_AA)
    return canvas


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)