"""Small CNN inference wrapper; predictions are disabled until a model is trained."""

from pathlib import Path

import numpy as np

from preprocessing import prepare_crop
from utils import CLASS_NAMES, MODEL_DIR

CONFIDENCE_THRESHOLD = 0.55
DEFAULT_MODEL_PATH = MODEL_DIR / "attention_cue_cnn.keras"


class AttentionClassifier:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self._model = None
        self._load_error = False

    @property
    def is_trained(self) -> bool:
        return self.model_path.is_file() and not self._load_error

    def _load(self) -> None:
        if self._model is not None or self._load_error:
            return
        if not self.model_path.is_file():
            return
        try:
            from tensorflow import keras

            self._model = keras.models.load_model(self.model_path, compile=False)
        except (ImportError, OSError, ValueError):
            self._load_error = True

    def predict(self, face_crop: np.ndarray) -> tuple[str, float, str]:
        self._load()
        if self._model is None:
            return "unclear", 0.0, "demo (no usable trained model)"

        probabilities = np.asarray(self._model.predict(prepare_crop(face_crop), verbose=0))[0]
        if probabilities.size != len(CLASS_NAMES):
            self._load_error = True
            self._model = None
            return "unclear", 0.0, "demo (model output mismatch)"
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])
        label = CLASS_NAMES[best_index]
        if confidence < CONFIDENCE_THRESHOLD:
            label = "unclear"
        return label, confidence, "trained CNN"


_classifier: AttentionClassifier | None = None


def get_classifier() -> AttentionClassifier:
    global _classifier
    if _classifier is None:
        _classifier = AttentionClassifier()
    return _classifier


def build_cnn(input_shape: tuple[int, int, int] = (96, 96, 3), class_count: int = 4):
    """Create a compact CPU-friendly CNN with training-only augmentation."""
    from tensorflow import keras

    inputs = keras.Input(shape=input_shape)
    x = keras.layers.RandomRotation(0.06)(inputs)
    x = keras.layers.RandomZoom(0.08)(x)
    for filters in (16, 32, 64):
        x = keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)
    outputs = keras.layers.Dense(class_count, activation="softmax")(x)
    model = keras.Model(inputs, outputs, name="attention_cue_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model