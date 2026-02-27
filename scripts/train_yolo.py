"""Train YOLOv8n for glottis detection on the GIRAFE dataset.

Example
-------
python scripts/train_yolo.py \\
    --images-dir  /path/to/GIRAFE/Training/imagesTr \\
    --labels-dir  /path/to/GIRAFE/Training/labelsTr \\
    --training-json /path/to/GIRAFE/Training/training.json \\
    --output-dir  yolo_data \\
    --epochs 100 \\
    --project outputs/yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Workaround: matplotlib font_manager can raise KeyError('_items') on macOS with Python 3.14
# when Ultralytics checks fonts for plotting. Patch to avoid crash.
try:
    import matplotlib.font_manager as _fm
    _orig_get_macos_fonts = getattr(_fm, "_get_macos_fonts", None)
    if _orig_get_macos_fonts is not None:
        def _safe_get_macos_fonts():
            try:
                return _orig_get_macos_fonts()
            except KeyError:
                return []
        _fm._get_macos_fonts = _safe_get_macos_fonts
except Exception:
    pass

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
    p.add_argument("--project", default="outputs/yolo",
                   help="Ultralytics project path (run save dir; outputs/ is gitignored).")
    p.add_argument("--name", default="exp",
                   help="Run name (subdir under project). Weights at <project>/<name>/weights/best.pt.")
    p.add_argument("--run-dir", default=None,
                   help="Custom save dir for this run (overrides project/name). e.g. outputs/yolo/bagls.")
    p.add_argument("--label-suffix", default="",
                   help="Mask filename is stem+label_suffix+.png (e.g. _seg for BAGLS).")
    p.add_argument("--subset-frac", type=float, default=None,
                   help="Use only this fraction of train/val (e.g. 0.1 for 1/10). Default: use all.")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--resume", default=None,
                   help="Resume: load YOLO from this checkpoint (e.g. outputs/yolo/exp/weights/last.pt) and continue training.")
    p.add_argument("--device", default=None,
                   help="Device for training (e.g. mps, cuda, cpu). Default: auto.")
    return p.parse_args()


def main() -> None:
    import json
    import random

    args = parse_args()

    if args.run_dir:
        run_path = Path(args.run_dir)
        project = str(run_path.parent) if run_path.parent != Path(".") else "runs"
        name = run_path.name
    else:
        project = args.project
        name = args.name

    training_json = args.training_json
    if args.subset_frac is not None and 0 < args.subset_frac < 1:
        splits = json.load(open(args.training_json))
        random.seed(42)
        n_train = max(1, int(len(splits["training"]) * args.subset_frac))
        n_val = max(1, int(len(splits["Val"]) * args.subset_frac))
        sub = {
            "training": random.sample(splits["training"], n_train),
            "Val": random.sample(splits["Val"], n_val),
            "test": splits.get("test", []),
        }
        import tempfile
        fd, path = tempfile.mkstemp(suffix=".json", prefix="yolo_subset_")
        with open(fd, "w") as f:
            json.dump(sub, f, indent=2)
        training_json = path
        print(f"Subset {args.subset_frac:.0%}: {n_train} train, {n_val} val")

    yaml_path = build_yolo_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        training_json=training_json,
        output_dir=args.output_dir,
        force=args.force_rebuild,
        mask_suffix=args.label_suffix,
    )

    model = YOLO(args.resume if args.resume else "yolov8n.pt")
    if args.resume:
        print(f"Resuming from {args.resume}", flush=True)
    train_kw = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=project,
        name=name,
        exist_ok=True,
    )
    if args.device is not None:
        train_kw["device"] = args.device
    results = model.train(**train_kw)

    best = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")


if __name__ == "__main__":
    main()
