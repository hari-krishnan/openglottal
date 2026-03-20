"""Create z-normalized versions of GIRAFE training images without clipping loss.

This preprocesses GIRAFE images to z-normalized form using full-range mapping
to preserve the complete z-score distribution without destructive clipping.

Key difference from preprocess_znorm_girafe.py:
    OLD (lossy):  clip z-scores to [0,1], then multiply by 255
    NEW (lossless): map z-scores from [-N, +N] linearly to [0, 255]

For GIRAFE: z-scores are roughly in [-2, +2], so we map:
    z ∈ [-2, +2] → uint8 ∈ [0, 255]

Usage
-----
python scripts/preprocess_znorm_girafe_fullrange.py \\
    --images-dir GIRAFE/Training/imagesTr \\
    --training-json GIRAFE/Training/training.json \\
    --znorm-json outputs/normalization_girafe.json \\
    --output-dir yolo_data_znorm_fullrange/images/train \\
    --val-output-dir yolo_data_znorm_fullrange/images/val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def preprocess_images(
    image_files: list[Path],
    znorm_stats: dict,
    output_dir: Path,
) -> None:
    """
    Apply z-normalization to images and save without clipping loss.
    
    Process:
      1. Load image as grayscale
      2. Convert to [0,1]
      3. Apply z-norm: (img - mean) / std
      4. Map [-N, +N] to [0, 255] linearly (no clipping)
      5. Convert to uint8 and save
    
    Parameters
    ----------
    image_files : list[Path]
        List of image paths to preprocess
    znorm_stats : dict
        Dict with 'mean' and 'std' keys
    output_dir : Path
        Directory to save normalized images
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mean = znorm_stats["mean"]
    std = znorm_stats["std"]
    
    # Determine mapping range: assume z-scores fit in [-3*std_dev, +3*std_dev]
    # which gives roughly [-2, +2] for GIRAFE
    z_range = 3.0
    z_min = -z_range
    z_max = z_range
    
    print(f"  Mapping z-scores [{z_min:.1f}, {z_max:.1f}] → [0, 255]")
    
    for i, img_path in enumerate(image_files):
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(image_files)}] ...")
        
        # Load and convert to grayscale
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Warning: Failed to load {img_path}")
            continue
        
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Apply z-normalization
        img_znorm = (img_gray - mean) / std
        
        # Map z-scores linearly to [0, 255] without clipping
        # z_min maps to 0, z_max maps to 255
        img_uint8 = ((img_znorm - z_min) / (z_max - z_min) * 255).astype(np.uint8)
        
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), img_uint8)


def main():
    p = argparse.ArgumentParser(
        description="Preprocess GIRAFE images to z-normalized form (full range, no clipping)."
    )
    p.add_argument("--images-dir", required=True,
                   help="GIRAFE training images directory.")
    p.add_argument("--training-json", required=True,
                   help="Training split JSON with 'training' and 'Val' keys.")
    p.add_argument("--znorm-json", required=True,
                   help="Z-normalization parameters (mean, std).")
    p.add_argument("--output-dir", required=True,
                   help="Output directory for z-normalized training images.")
    p.add_argument("--val-output-dir", required=True,
                   help="Output directory for z-normalized validation images.")
    args = p.parse_args()
    
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    val_output_dir = Path(args.val_output_dir)
    
    # Load normalization parameters
    with open(args.znorm_json) as f:
        znorm_stats = json.load(f)
    
    print(f"Z-norm parameters: mean={znorm_stats['mean']:.6f}, std={znorm_stats['std']:.6f}\n")
    
    # Load training split
    with open(args.training_json) as f:
        splits = json.load(f)
    
    train_fnames = splits["training"]
    val_fnames = splits["Val"]
    
    # Get full paths
    train_paths = [images_dir / fn for fn in train_fnames]
    val_paths = [images_dir / fn for fn in val_fnames]
    
    # Preprocess training images
    print(f"Preprocessing {len(train_paths)} training images (full-range mapping)...")
    preprocess_images(train_paths, znorm_stats, output_dir)
    
    # Preprocess validation images
    print(f"Preprocessing {len(val_paths)} validation images (full-range mapping)...")
    preprocess_images(val_paths, znorm_stats, val_output_dir)
    
    print(f"\n✓ Training images saved to: {output_dir}")
    print(f"✓ Validation images saved to: {val_output_dir}")
    print(f"  Total: {len(train_paths)} train + {len(val_paths)} val")


if __name__ == "__main__":
    main()
