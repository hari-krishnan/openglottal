"""Estimate z-normalization parameters (mean, std) from training images.

Computes the global mean and standard deviation across all training images
to enable z-score normalization during training and inference.

Usage
-----
python scripts/estimate_normalization.py \\
    --images-dir /path/to/GIRAFE/Training/imagesTr \\
    --training-json /path/to/GIRAFE/Training/training.json \\
    --output outputs/normalization.json

Then use these parameters in training and inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def estimate_normalization(
    images_dir: str | Path,
    training_json: str | Path,
) -> dict[str, float]:
    """
    Compute mean and std of grayscale training images.
    
    Parameters
    ----------
    images_dir : path
        Directory containing training images (PNG).
    training_json : path
        JSON file with "training" key containing list of image filenames.
    
    Returns
    -------
    Dict with keys "mean" and "std" (values are in [0, 1] range after /255).
    """
    images_dir = Path(images_dir)
    
    with open(training_json) as f:
        splits = json.load(f)
    
    training_fnames = splits.get("training", [])
    if not training_fnames:
        raise ValueError("No training images found in JSON")
    
    # Collect all pixel values
    all_pixels = []
    
    for i, fname in enumerate(training_fnames):
        img_path = images_dir / fname
        if not img_path.exists():
            print(f"  Warning: {fname} not found")
            continue
        
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  Warning: Failed to read {fname}")
            continue
        
        # Normalize to [0, 1]
        img_normalized = img.astype(np.float32) / 255.0
        all_pixels.append(img_normalized.ravel())
        
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(training_fnames)}] ...")
    
    # Concatenate all pixels
    all_pixels = np.concatenate(all_pixels)
    
    # Compute statistics
    mean = float(np.mean(all_pixels))
    std = float(np.std(all_pixels))
    
    print(f"\nStatistics over {len(training_fnames)} training images:")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std:  {std:.6f}")
    print(f"  Range: [{all_pixels.min():.6f}, {all_pixels.max():.6f}]")
    
    return {"mean": mean, "std": std, "n_images": len(training_fnames)}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Estimate z-normalization parameters from training images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir", required=True,
                   help="Training images directory.")
    p.add_argument("--training-json", required=True,
                   help="JSON file with training split.")
    p.add_argument("--output", default="outputs/normalization.json",
                   help="Output JSON file for normalization parameters.")
    args = p.parse_args()
    
    print(f"Estimating normalization from {args.images_dir} ...\n")
    
    norm_params = estimate_normalization(args.images_dir, args.training_json)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(norm_params, f, indent=2)
    
    print(f"\nNormalization parameters saved to: {output_path}")


if __name__ == "__main__":
    main()
