"""Build training.json from a BAGLS training directory (N.png + N_seg.png pairs).

BAGLS uses one directory per split; images are N.png with masks N_seg.png.
This script scans a single directory, shuffles, and splits into train/val
so you can train on BAGLS with the same scripts as GIRAFE.

Usage
-----
python scripts/prepare_bagls_splits.py \\
    --bagls-dir BAGLS/training \\
    --output BAGLS/training.json \\
    --val-frac 0.1 \\
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build training.json from BAGLS directory (N.png + N_seg.png).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bagls-dir", required=True,
                   help="Directory containing N.png and N_seg.png pairs.")
    p.add_argument("--output", required=True,
                   help="Path to write training.json (keys: training, Val, test).")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of frames for validation (rest for training).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for train/val split.")
    args = p.parse_args()

    root = Path(args.bagls_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")

    # List image filenames (exclude _seg.png)
    fnames = sorted(
        f.name for f in root.iterdir()
        if f.suffix == ".png" and not f.name.endswith("_seg.png")
    )
    # Require matching mask for each
    valid = []
    for name in fnames:
        stem = Path(name).stem
        seg = root / f"{stem}_seg.png"
        if seg.exists():
            valid.append(name)
        else:
            pass  # skip frames without mask

    if not valid:
        raise FileNotFoundError(
            f"No N.png + N_seg.png pairs found in {root}. "
            "Check --bagls-dir (e.g. BAGLS/training after extracting training.zip)."
        )

    random.seed(args.seed)
    random.shuffle(valid)
    n_val = max(1, int(len(valid) * args.val_frac))
    n_train = len(valid) - n_val
    training = valid[:n_train]
    Val = valid[n_train:]

    out = {
        "training": training,
        "Val": Val,
        "test": [],  # BAGLS test is a separate directory; leave empty for training
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"BAGLS split: {len(training)} train, {len(Val)} val → {out_path}")


if __name__ == "__main__":
    main()
