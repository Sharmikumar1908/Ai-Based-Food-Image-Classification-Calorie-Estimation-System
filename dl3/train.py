"""Train the compact visual-cue CNN and evaluate it on labeled test images."""

import json
import os
from pathlib import Path
from typing import Any

from model import build_cnn
from utils import CLASS_NAMES, DATASET_DIR, MODEL_DIR, RESULTS_DIR


def _load_split(directory: Path, batch_size: int, shuffle: bool, seed: int):
    import tensorflow as tf

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        class_names=list(CLASS_NAMES),
        image_size=(96, 96),
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    return dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32) / 255.0, labels),
        num_parallel_calls=1,
    ).prefetch(1)


def _validate_classes(directory: Path) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset folder does not exist: {directory}")
    available = {path.name for path in directory.iterdir() if path.is_dir()}
    missing = set(CLASS_NAMES) - available
    if missing:
        raise ValueError(f"Missing class folders in {directory}: {', '.join(sorted(missing))}")


def run_training(
    dataset_dir: Path = DATASET_DIR,
    model_dir: Path = MODEL_DIR,
    results_dir: Path = RESULTS_DIR,
    epochs: int = 20,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    train_dir = dataset_dir / "train"
    validation_dir = dataset_dir / "validation"
    test_dir = dataset_dir / "test"
    _validate_classes(train_dir)
    _validate_classes(validation_dir)
    train_data = _load_split(train_dir, batch_size, True, seed)
    validation_data = _load_split(validation_dir, batch_size, False, seed)

    model = build_cnn()
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "attention_cue_cnn.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )

    report: dict[str, Any] = {
        "classes": list(CLASS_NAMES),
        "epochs_completed": len(history.epoch),
        "final_training_metrics": {key: float(values[-1]) for key, values in history.history.items()},
        "test_evaluation": None,
        "model_file": str(model_path.name),
    }
    if test_dir.is_dir() and all((test_dir / name).is_dir() for name in CLASS_NAMES):
        test_data = _load_split(test_dir, batch_size, False, seed)
        loss, accuracy = model.evaluate(test_data, verbose=0)
        expected: list[int] = []
        predicted: list[int] = []
        for images, labels in test_data:
            expected.extend(labels.numpy().astype(int).tolist())
            predicted.extend(model.predict(images, verbose=0).argmax(axis=1).astype(int).tolist())
        from sklearn.metrics import classification_report, confusion_matrix

        report["test_evaluation"] = {
            "loss": float(loss),
            "accuracy": float(accuracy),
            "classification_report": classification_report(
                expected,
                predicted,
                labels=list(range(len(CLASS_NAMES))),
                target_names=list(CLASS_NAMES),
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                expected, predicted, labels=list(range(len(CLASS_NAMES)))
            ).tolist(),
        }
        _save_confusion_matrix(report["test_evaluation"]["confusion_matrix"], results_dir)

    report_path = results_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _save_confusion_matrix(matrix: list[list[int]], results_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES).plot(
        ax=axis, values_format="d", colorbar=False
    )
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png", dpi=140)
    plt.close(figure)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    report = run_training(args.dataset_dir, epochs=args.epochs, batch_size=args.batch_size)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()