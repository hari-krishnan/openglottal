"""Build training.json from GIRAFE Training directory (imagesTr + labelsTr).

GIRAFE has separate image and label dirs; image and label share the same filename.
Test split is fixed to 4 held-out patients (same as eval_girafe.py). The rest
are split into training and Val.

Usage
-----
python scripts/prepare_girafe_splits.py \\
    --images-dir GIRAFE/Training/imagesTr \\
    --labels-dir GIRAFE/Training/labelsTr \\
    --output GIRAFE/Training/training.json \\
    --val-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# Must match scripts/eval_girafe.py so test set is consistent
TEST_PATIENT_PREFIXES = ("patient57A3", "patient61", "patient63", "patient64")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build training.json from GIRAFE imagesTr + labelsTr.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir", required=True,
                   help="Directory containing PNG frames (e.g. GIRAFE/Training/imagesTr).")
    p.add_argument("--labels-dir", required=True,
                   help="Directory containing PNG masks (same filenames as images).")
    p.add_argument("--output", required=True,
                   help="Path to write training.json (keys: training, Val, test).")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of non-test frames for validation (rest for training).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/val split.")
    args = p.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {labels_dir}")

    # All image filenames that have a matching label (same name)
    valid = []
    for f in sorted(images_dir.iterdir()):
        if f.suffix.lower() != ".png":
            continue
        if (labels_dir / f.name).exists():
            valid.append(f.name)

    if not valid:
        raise FileNotFoundError(
            f"No image+label pairs found in {images_dir} / {labels_dir}. "
            "Check --images-dir and --labels-dir."
        )

    # Fixed test set: frames belonging to the 4 test patients
    test = [f for f in valid if any(f.startswith(p) for p in TEST_PATIENT_PREFIXES)]
    train_val_pool = [f for f in valid if f not in test]

    random.seed(args.seed)
    random.shuffle(train_val_pool)
    n_val = max(0, int(len(train_val_pool) * args.val_frac))
    n_train = len(train_val_pool) - n_val
    training = train_val_pool[:n_train]
    Val = train_val_pool[n_train:]

    out = {
        "training": training,
        "Val": Val,
        "test": test,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"GIRAFE split: {len(training)} train, {len(Val)} val, {len(test)} test → {out_path}")


if __name__ == "__main__":
    main()
