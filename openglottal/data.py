"""YOLO dataset construction from GIRAFE-format segmentation masks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_W = IMG_H = 256   # all GIRAFE frames are 256×256
DILATE = 10           # pixels added on each side of the tight mask bbox


def mask_to_yolo(mask_path: str | Path, dilate: int = DILATE) -> str:
    """
    Convert a binary segmentation mask to a YOLO bbox label string.

    Parameters
    ----------
    mask_path:
        Path to a grayscale PNG mask where glottis pixels are > 0.
    dilate:
        Number of pixels to expand the tight bounding box on each side.

    Returns
    -------
    A YOLO-format label string ``"0 cx cy w h"`` (normalised), or ``""``
    if the mask is empty.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.max() == 0:
        return ""
    ys, xs = np.where(mask > 0)
    x1 = max(0, xs.min() - dilate)
    x2 = min(IMG_W, xs.max() + dilate)
    y1 = max(0, ys.min() - dilate)
    y2 = min(IMG_H, ys.max() + dilate)
    cx = (x1 + x2) / 2 / IMG_W
    cy = (y1 + y2) / 2 / IMG_H
    w = (x2 - x1) / IMG_W
    h = (y2 - y1) / IMG_H
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_yolo_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    training_json: str | Path,
    output_dir: str | Path,
    dilate: int = DILATE,
    force: bool = False,
) -> Path:
    """
    Build a YOLO-format dataset from GIRAFE images and masks.

    Directory layout created under ``output_dir``::

        output_dir/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        └── labels/
            ├── train/
            ├── val/
            └── test/

    Parameters
    ----------
    images_dir:
        Directory containing PNG frames (e.g. ``GIRAFE/Training/imagesTr``).
    labels_dir:
        Directory containing PNG masks (e.g. ``GIRAFE/Training/labelsTr``).
    training_json:
        Path to the split JSON with keys ``training``, ``Val``, ``test``.
    output_dir:
        Root directory for the generated YOLO dataset.
    dilate:
        Pixel dilation applied to each mask bbox.
    force:
        Rebuild even if the dataset already exists.

    Returns
    -------
    Path to the generated ``dataset.yaml`` file.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    output_dir = Path(output_dir)

    splits = json.load(open(training_json))
    split_map = {
        "train": splits["training"],
        "val": splits["Val"],
        "test": splits["test"],
    }

    def _complete() -> bool:
        for split in split_map:
            if not (output_dir / "images" / split).exists():
                return False
            if not (output_dir / "labels" / split).exists():
                return False
        return True

    if _complete() and not force:
        print("Dataset already exists — skipping build. Pass force=True to rebuild.")
    else:
        for split, fnames in split_map.items():
            img_out = output_dir / "images" / split
            lbl_out = output_dir / "labels" / split
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)
            for fname in fnames:
                shutil.copy(images_dir / fname, img_out / fname)
                label = mask_to_yolo(labels_dir / fname, dilate=dilate)
                stem = Path(fname).stem
                (lbl_out / f"{stem}.txt").write_text(label)
        print(f"Dataset built at {output_dir}")

    yaml_path = output_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"nc: 1\n"
        f"names: ['glottis']\n"
    )
    return yaml_path
