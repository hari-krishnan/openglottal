"""Estimate z-normalization parameters from BAGLS training images.

Usage
-----
python scripts/estimate_normalization_bagls.py \\
  --images-dir /path/to/BAGLS/training \\
  --output outputs/normalization_bagls.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def estimate_normalization(images_dir: Path, max_images: int = 0) -> dict:
    """Compute mean and std over all grayscale images in images_dir.
    
    Uses Welford's online algorithm to avoid memory issues with large datasets.
    """
    mean_val = 0.0
    m2 = 0.0  # For variance
    count = 0
    
    img_files = sorted([
        f for f in images_dir.iterdir()
        if f.suffix.lower() == ".png" and not f.name.endswith("_seg.png")
    ])
    
    # optionally sample a subset at random rather than just taking first N
    if max_images and max_images < len(img_files):
        import random
        img_files = random.sample(img_files, max_images)
    
    print(f"Estimating normalization from {len(img_files)} images in {images_dir}")
    
    for i, img_path in enumerate(img_files):
        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(img_files)}] ...")
        
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Normalize to [0, 1]
        img_norm = img_gray.astype(np.float32) / 255.0
        
        # Update using Welford's online algorithm for each pixel
        for val in img_norm.ravel():
            count += 1
            delta = float(val) - mean_val
            mean_val += delta / count
            delta2 = float(val) - mean_val
            m2 += delta * delta2
    
    if count == 0:
        raise ValueError("No images found")
    
    variance = m2 / count
    std_val = np.sqrt(variance)
    
    print(f"\nStatistics over {len(img_files)} training images ({count} total pixels):")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std:  {std_val:.6f}")
    
    return {
        "mean": mean_val,
        "std": std_val,
        "n_images": len(img_files),
    }


def main():
    p = argparse.ArgumentParser(
        description="Estimate z-normalization parameters from BAGLS training images."
    )
    p.add_argument("--images-dir", required=True,
                   help="BAGLS training directory (contains N.png files).")
    p.add_argument("--output", required=True,
                   help="Output JSON file path.")
    p.add_argument("--max-images", type=int, default=0,
                   help="Limit to N randomly chosen images (0 = all).")
    args = p.parse_args()
    
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise ValueError(f"Directory not found: {images_dir}")
    
    stats = estimate_normalization(images_dir, args.max_images)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Normalization parameters saved to: {out_path}\n")


if __name__ == "__main__":
    main()
