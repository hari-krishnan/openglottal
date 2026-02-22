"""Train YOLOv8n for glottis detection on the GIRAFE dataset.

Example
-------
python scripts/train_yolo.py \\
    --images-dir  /path/to/GIRAFE/Training/imagesTr \\
    --labels-dir  /path/to/GIRAFE/Training/labelsTr \\
    --training-json /path/to/GIRAFE/Training/training.json \\
    --output-dir  yolo_data \\
    --epochs 100 \\
    --project runs/detect/glottis
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from openglottal.data import build_yolo_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8n glottis detector.")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--training-json", required=True)
    p.add_argument("--output-dir", default="yolo_data", help="YOLO dataset root.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", default="runs/detect/glottis")
    p.add_argument("--name", default="exp")
    p.add_argument("--force-rebuild", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    yaml_path = build_yolo_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        training_json=args.training_json,
        output_dir=args.output_dir,
        force=args.force_rebuild,
    )

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")


if __name__ == "__main__":
    main()
