"""Build YOLO dataset structure from z-normalized GIRAFE images.

Creates a YOLO-compatible dataset.yaml and directory structure pointing to
the z-normalized preprocessed images.

Usage
-----
python scripts/build_yolo_znorm_dataset.py \\
    --labels-dir GIRAFE/Training/labelsTr \\
    --training-json GIRAFE/Training/training.json \\
    --images-train yolo_data_znorm/images/train \\
    --images-val yolo_data_znorm/images/val \\
    --output-dir yolo_data_znorm_dataset
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_yolo_dataset(
    labels_dir: Path,
    training_json: Path,
    images_train: Path,
    images_val: Path,
    output_dir: Path,
) -> Path:
    """
    Create YOLO dataset structure with z-normalized images.
    
    Creates:
      - output_dir/images/train/ (symlinks to z-normalized training images)
      - output_dir/images/val/   (symlinks to z-normalized validation images)
      - output_dir/labels/train/ (YOLO-format labels)
      - output_dir/labels/val/   (YOLO-format labels)
      - output_dir/dataset.yaml  (YOLO config)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training split
    with open(training_json) as f:
        splits = json.load(f)
    
    train_fnames = splits["training"]
    val_fnames = splits["Val"]
    
    # Create directory structure
    output_images_train = output_dir / "images" / "train"
    output_images_val = output_dir / "images" / "val"
    output_labels_train = output_dir / "labels" / "train"
    output_labels_val = output_dir / "labels" / "val"
    
    output_images_train.mkdir(parents=True, exist_ok=True)
    output_images_val.mkdir(parents=True, exist_ok=True)
    output_labels_train.mkdir(parents=True, exist_ok=True)
    output_labels_val.mkdir(parents=True, exist_ok=True)
    
    # Link normalized images to YOLO dataset
    print(f"Linking {len(train_fnames)} training images...")
    for fname in train_fnames:
        src = images_train / fname
        dst = output_images_train / fname
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
    
    print(f"Linking {len(val_fnames)} validation images...")
    for fname in val_fnames:
        src = images_val / fname
        dst = output_images_val / fname
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
    
    # Copy YOLO-format labels
    print(f"Copying {len(train_fnames)} training labels...")
    for fname in train_fnames:
        label_name = Path(fname).stem + ".txt"
        src = labels_dir / label_name
        dst = output_labels_train / label_name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
    
    print(f"Copying {len(val_fnames)} validation labels...")
    for fname in val_fnames:
        label_name = Path(fname).stem + ".txt"
        src = labels_dir / label_name
        dst = output_labels_val / label_name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
    
    # Create dataset.yaml
    dataset_yaml = output_dir / "dataset.yaml"
    yaml_content = f"""path: {output_dir.resolve()}
train: images/train
val: images/val

nc: 1  # Single class: glottis
names: ['glottis']
"""
    with open(dataset_yaml, "w") as f:
        f.write(yaml_content)
    
    print(f"\nDataset structure created at: {output_dir}")
    print(f"  - {len(train_fnames)} training images")
    print(f"  - {len(val_fnames)} validation images")
    print(f"  - Config: {dataset_yaml}")
    
    return dataset_yaml


def main():
    p = argparse.ArgumentParser(
        description="Build YOLO dataset structure from z-normalized GIRAFE images."
    )
    p.add_argument("--labels-dir", required=True,
                   help="GIRAFE training labels directory (labelsTr).")
    p.add_argument("--training-json", required=True,
                   help="Training split JSON.")
    p.add_argument("--images-train", required=True,
                   help="Z-normalized training images directory.")
    p.add_argument("--images-val", required=True,
                   help="Z-normalized validation images directory.")
    p.add_argument("--output-dir", required=True,
                   help="Output YOLO dataset directory.")
    args = p.parse_args()
    
    build_yolo_dataset(
        labels_dir=Path(args.labels_dir),
        training_json=Path(args.training_json),
        images_train=Path(args.images_train),
        images_val=Path(args.images_val),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
