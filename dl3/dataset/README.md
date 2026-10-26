# Labeled training images

No classroom or student images are bundled. Supply images only when you have appropriate permission and a labeling process you can explain. Avoid collecting identifying details, and do not use this prototype to make decisions about individual students.

Place one image per manually reviewed label in these folders:

```text
dataset/raw/
├── facing_forward/
├── looking_away/
├── head_down/
└── unclear/
```

Use those exact lowercase folder names. Use varied, permission-cleared crops and conditions; balance classes where possible. At least three readable images per class are required to create non-empty train, validation, and test splits, but that is only a script minimum and is nowhere near enough to establish model quality. More representative, independent examples are needed for meaningful evaluation.

Create the deterministic 70/15/15 split with:

```powershell
python prepare_dataset.py
```

The script copies files into `dataset/train/`, `dataset/validation/`, and `dataset/test/`. It refuses to overwrite non-empty split folders. Keep subjects or recording sessions separated across splits where possible to reduce data leakage. Remove or move existing split folders before preparing a new split.

The CNN expects a single face crop per image. This project does not automatically crop an image collection or supply pretrained weights. The Haar face detector is only used by the live app. A trained checkpoint is created by `python train.py`; its reported metrics are only meaningful for the dataset and split that were actually evaluated.