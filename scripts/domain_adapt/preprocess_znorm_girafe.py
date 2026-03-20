"""Create z-normalized versions of GIRAFE training images for YOLO training.

This preprocesses GIRAFE images to z-normalized form, allowing YOLO to train
on normalized data without modifying the YOLO pipeline itself.

Usage
-----
python scripts/preprocess_znorm_girafe.py \\
    --images-dir GIRAFE/Training/imagesTr \\
    --training-json GIRAFE/Training/training.json \\
    --znorm-json outputs/normalization_girafe.json \\
    --output-dir yolo_data_znorm/images/train \\
    --val-output-dir yolo_data_znorm/images/val
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
    Apply z-normalization to images and save as uint8.
    
    Process:
      1. Load image as grayscale
      2. Convert to [0,1]
      3. Apply z-norm: (img - mean) / std
      4. Clip to valid range [0,1]
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
        
        # Clip to [0,1] to keep valid uint8 range
        img_znorm = np.clip(img_znorm, 0.0, 1.0)
        
        # Convert to uint8 and save
        img_uint8 = (img_znorm * 255).astype(np.uint8)
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), img_uint8)


def main():
    p = argparse.ArgumentParser(
        description="Preprocess GIRAFE images to z-normalized form for YOLO training."
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
    print(f"Preprocessing {len(train_paths)} training images...")
    preprocess_images(train_paths, znorm_stats, output_dir)
    
    # Preprocess validation images
    print(f"Preprocessing {len(val_paths)} validation images...")
    preprocess_images(val_paths, znorm_stats, val_output_dir)
    
    print(f"\n✓ Training images saved to: {output_dir}")
    print(f"✓ Validation images saved to: {val_output_dir}")
    print(f"  Total: {len(train_paths)} train + {len(val_paths)} val")


if __name__ == "__main__":
    main()
