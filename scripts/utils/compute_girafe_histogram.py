"""Compute and save aggregate grayscale histogram from GIRAFE training images.

The histogram is used as a reference CDF for histogram-matching BAGLS frames
into the GIRAFE pixel distribution at inference time.

Usage
-----
python scripts/compute_girafe_histogram.py \
    --images-dir GIRAFE/Training/imagesTr \
    --output     outputs/girafe_histogram.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def compute_aggregate_histogram(images_dir: Path) -> np.ndarray:
    """Return a 256-bin aggregate histogram (sum over all grayscale training images)."""
    agg = np.zeros(256, dtype=np.int64)
    img_paths = sorted(images_dir.glob("*.png"))
    if not img_paths:
        raise FileNotFoundError(f"No PNG files found in {images_dir}")

    for i, p in enumerate(img_paths):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
        agg += hist
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(img_paths)}] ...")

    return agg


def main() -> None:
    p = argparse.ArgumentParser(
        description="Compute GIRAFE aggregate grayscale histogram for domain adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir", required=True,
                   help="GIRAFE training images directory (imagesTr).")
    p.add_argument("--output",     default="outputs/girafe_histogram.npy",
                   help="Output .npy path for the 256-bin histogram.")
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    out_path   = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Computing aggregate histogram from {images_dir} ...")
    hist = compute_aggregate_histogram(images_dir)
    np.save(out_path, hist)

    total_px = hist.sum()
    mean_val = (np.arange(256) * hist).sum() / total_px
    print(f"\nDone. Total pixels : {total_px:,}")
    print(f"Mean grayscale val : {mean_val:.1f}")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
