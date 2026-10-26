"""Split manually labeled images from dataset/raw into train/validation/test."""

import argparse
import random
import shutil
from pathlib import Path

import cv2

from utils import CLASS_NAMES, DATASET_DIR, IMAGE_EXTENSIONS


def prepare_dataset(
    raw_dir: Path = DATASET_DIR / "raw",
    output_dir: Path = DATASET_DIR,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, int]:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and leave data for test.")
    sources = {name: raw_dir / name for name in CLASS_NAMES}
    missing = [name for name, path in sources.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Missing labeled folders under {raw_dir}: {', '.join(missing)}")

    destinations = [output_dir / split for split in ("train", "validation", "test")]
    if any(path.exists() and any(path.iterdir()) for path in destinations):
        raise FileExistsError("Dataset split folders are not empty; move them before preparing a new split.")

    generator = random.Random(seed)
    totals = {"train": 0, "validation": 0, "test": 0}
    for class_name, source in sources.items():
        paths = [
            path
            for path in source.iterdir()
            if path.is_file()
            and path.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS
            and cv2.imread(str(path)) is not None
        ]
        if len(paths) < 3:
            raise ValueError(f"Class '{class_name}' needs at least 3 readable images; found {len(paths)}.")
        generator.shuffle(paths)
        train_end = min(max(1, int(len(paths) * train_fraction)), len(paths) - 2)
        validation_end = max(train_end + 1, int(len(paths) * (train_fraction + validation_fraction)))
        validation_end = min(validation_end, len(paths) - 1)
        groups = {
            "train": paths[:train_end],
            "validation": paths[train_end:validation_end],
            "test": paths[validation_end:],
        }
        for split, group in groups.items():
            target = output_dir / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            for image_path in group:
                shutil.copy2(image_path, target / image_path.name)
                totals[split] += 1
    return totals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DATASET_DIR / "raw")
    parser.add_argument("--output-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    totals = prepare_dataset(raw_dir=args.raw_dir, output_dir=args.output_dir, seed=args.seed)
    print("Prepared image totals:", totals)


if __name__ == "__main__":
    main()