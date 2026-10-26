import io

import cv2
import numpy as np
import pytest

import app as app_module
import database
from model import AttentionClassifier
from utils import CLASS_NAMES


@pytest.fixture
def client(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    application = app_module.create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "sessions.sqlite3"),
            "UPLOAD_FOLDER": str(upload_dir),
        }
    )
    return application.test_client(), application.config["DATABASE"], upload_dir


def sample_analysis(frame):
    counts = {name: 0 for name in CLASS_NAMES}
    counts["unclear"] = 1
    return {
        "annotated": frame,
        "counts": counts,
        "detected_subjects": 1,
        "observations": 1,
        "model_mode": "demo (no usable trained model)",
    }


def encoded_image():
    image = np.full((80, 100, 3), 220, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def test_image_upload_creates_aggregate_summary(client, monkeypatch):
    test_client, db_path, _upload_dir = client
    monkeypatch.setattr(app_module, "analyze_frame", sample_analysis)
    response = test_client.post(
        "/analyze",
        data={"media": (io.BytesIO(encoded_image()), "classroom.jpg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Visual cue summary" in response.data
    sessions = database.list_sessions(db_path)
    assert len(sessions) == 1
    assert sessions[0]["media_type"] == "image"
    assert sessions[0]["unclear"] == 1


def test_video_upload_is_processed_and_temporary_file_removed(client, monkeypatch):
    test_client, db_path, upload_dir = client

    def fake_video(path):
        assert path.is_file()
        return sample_analysis(np.zeros((20, 20, 3), dtype=np.uint8))

    monkeypatch.setattr(app_module, "analyze_video", fake_video)
    response = test_client.post(
        "/analyze",
        data={"media": (io.BytesIO(b"test video bytes"), "classroom.mp4")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert database.list_sessions(db_path)[0]["media_type"] == "video"
    assert list(upload_dir.iterdir()) == []


def test_classifier_uses_unclear_without_a_trained_model(tmp_path):
    classifier = AttentionClassifier(tmp_path / "missing.keras")
    label, confidence, source = classifier.predict(np.zeros((32, 32, 3), dtype=np.uint8))
    assert label == "unclear"
    assert confidence == 0.0
    assert source.startswith("demo")


def test_history_dashboard_shows_session_category_statistics(client):
    test_client, db_path, _upload_dir = client
    counts = {name: 0 for name in CLASS_NAMES}
    counts.update({"facing_forward": 3, "looking_away": 2, "head_down": 1, "unclear": 4})
    database.add_session(
        {
            "media_type": "video",
            "detected_subjects": 4,
            "observations": 10,
            "model_mode": "demo",
            "counts": counts,
        },
        db_path,
    )
    response = test_client.get("/history")
    assert response.status_code == 200
    assert b"F 3" in response.data
    assert b"A 2" in response.data
    assert b"D 1" in response.data
    assert b"U 4" in response.data


def test_delete_session_and_export_csv(client):
    test_client, db_path, _upload_dir = client
    session_id = database.add_session({"media_type": "image", "counts": {}}, db_path)
    csv_response = test_client.get("/history.csv")
    assert csv_response.status_code == 200
    assert b"facing_forward" in csv_response.data
    delete_response = test_client.post(f"/history/{session_id}/delete")
    assert delete_response.status_code == 302
    assert database.list_sessions(db_path) == []