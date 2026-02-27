"""Train the U-Net segmentation model on the GIRAFE or BAGLS dataset.

Examples
--------
GIRAFE (local):
  python scripts/train_unet.py \\
    --images-dir  /path/to/GIRAFE/Training/imagesTr \\
    --labels-dir  /path/to/GIRAFE/Training/labelsTr \\
    --training-json /path/to/GIRAFE/Training/training.json \\
    --output outputs/openglottal_unet.pt --epochs 50

BAGLS on Kaggle (add dataset gomezp/benchmark-for-automatic-glottis-segmentation):
  python scripts/train_unet.py --use-kaggle-bagls \\
    --training-json /kaggle/input/openglottal/BAGLS/training.json \\
    --output /kaggle/working/openglottal_unet.pt --epochs 50

Efficient I/O with HDF5 cache (build once, then train):
  python scripts/train_unet.py --images-dir ... --labels-dir ... --training-json ... \\
    --hdf5 outputs/bagls --build-hdf5 --epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from openglottal.data import (
    build_glottis_hdf5,
    resolve_kaggle_data_paths,
    GlottisDatasetHDF5,
)
from openglottal.models import UNet, GlottisDataset
from openglottal.utils import dice_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net glottis segmenter.")
    p.add_argument("--images-dir", default=None,
                   help="Images directory. Ignored if --use-kaggle-bagls.")
    p.add_argument("--labels-dir", default=None,
                   help="Labels directory. Ignored if --use-kaggle-bagls.")
    p.add_argument("--training-json", required=True,
                   help="Path to split JSON (keys: training, Val).")
    p.add_argument("--use-kaggle-bagls", action="store_true",
                   help="Use BAGLS from Kaggle input (when running on Kaggle). Sets images/labels dirs.")
    p.add_argument("--hdf5", metavar="PATH", default=None,
                   help="Use HDF5 cache for I/O. PATH is prefix: use PATH_train.h5 and PATH_val.h5.")
    p.add_argument("--build-hdf5", action="store_true",
                   help="Build HDF5 caches from current images/labels before training (requires --hdf5).")
    p.add_argument("--output", default="outputs/openglottal_unet.pt",
                   help="Path to save best checkpoint (outputs/ is gitignored).")
    p.add_argument("--output-suffix", default=None,
                   help="If set, save as <output_stem>_<suffix>.pt (e.g. letterbox, bagls).")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--features", nargs="+", type=int, default=[32, 64, 128, 256])
    p.add_argument("--device", default=None,
                   help="Device: mps, cuda, or cpu. Auto-detected if omitted.")
    p.add_argument("--label-suffix", default="",
                   help="Mask filename is stem+label_suffix+.png (e.g. _seg for BAGLS).")
    p.add_argument("--log-dir", default="runs",
                   help="TensorBoard log directory. Logs go to <log-dir>/<run_name>.")
    p.add_argument("--patience", type=int, default=0,
                   help="Early stopping: stop if val loss does not improve for this many epochs (0 = disabled). e.g. 5")
    p.add_argument("--resume", default=None,
                   help="Resume: load model weights from this checkpoint and continue training (epoch 1, fresh optimizer).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = (
            torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cuda") if torch.cuda.is_available()
            else torch.device("cpu")
        )
    print(f"Device: {device}", flush=True)

    # Resolve images/labels dirs (Kaggle BAGLS or explicit paths)
    if args.use_kaggle_bagls:
        resolved = resolve_kaggle_data_paths("bagls", "training")
        if resolved is None:
            raise SystemExit(
                " --use-kaggle-bagls set but not on Kaggle or BAGLS dataset not added. "
                "Add dataset 'gomezp/benchmark-for-automatic-glottis-segmentation' to your notebook."
            )
        images_dir = resolved[0]
        labels_dir = resolved[1]
        if not args.label_suffix:
            args.label_suffix = "_seg"
        print(f"Using Kaggle BAGLS: {images_dir}")
    else:
        if not args.images_dir or not args.labels_dir:
            raise SystemExit("Provide --images-dir and --labels-dir, or use --use-kaggle-bagls.")
        images_dir = Path(args.images_dir)
        labels_dir = Path(args.labels_dir)

    splits = json.load(open(args.training_json))
    train_fnames = splits["training"]
    val_fnames = splits["Val"]

    if args.build_hdf5 and not args.hdf5:
        raise SystemExit("--build-hdf5 requires --hdf5 PATH.")

    if args.hdf5:
        h5_prefix = Path(args.hdf5)
        h5_train = h5_prefix.parent / f"{h5_prefix.name}_train.h5"
        h5_val = h5_prefix.parent / f"{h5_prefix.name}_val.h5"
        if args.build_hdf5 or not h5_train.exists() or not h5_val.exists():
            print("Building HDF5 caches ...", flush=True)
            build_glottis_hdf5(
                train_fnames, images_dir, labels_dir,
                h5_train, label_suffix=args.label_suffix,
            )
            build_glottis_hdf5(
                val_fnames, images_dir, labels_dir,
                h5_val, label_suffix=args.label_suffix,
            )
            print(f"  → {h5_train}, {h5_val}")
        train_ds = GlottisDatasetHDF5(h5_train, augment=True)
        val_ds = GlottisDatasetHDF5(h5_val, augment=False)
    else:
        train_ds = GlottisDataset(
            train_fnames, images_dir, labels_dir,
            augment=True, label_suffix=args.label_suffix,
        )
        val_ds = GlottisDataset(
            val_fnames, images_dir, labels_dir,
            augment=False, label_suffix=args.label_suffix,
        )

    train_ldr = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_ldr = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet(1, 1, tuple(args.features)).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        model.load_state_dict(state, strict=True)
        print(f"Resumed from {args.resume}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net — {n_params / 1e6:.2f}M parameters", flush=True)

    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    out_path = Path(args.output)
    if args.output_suffix:
        out_path = out_path.parent / f"{out_path.stem}_{args.output_suffix}{out_path.suffix}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = out_path.stem
    log_dir = Path(args.log_dir) / run_name
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard: tensorboard --logdir {log_dir}\n")

    best_val = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for imgs, msks in train_ldr:
            imgs, msks = imgs.to(device), msks.to(device)
            logits = model(imgs)
            loss = 0.5 * bce(logits, msks) + 0.5 * dice_loss(logits, msks)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_loss += loss.item() * len(imgs)
        train_loss /= len(train_ds)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, msks in val_ldr:
                imgs, msks = imgs.to(device), msks.to(device)
                logits = model(imgs)
                loss = 0.5 * bce(logits, msks) + 0.5 * dice_loss(logits, msks)
                val_loss += loss.item() * len(imgs)
        val_loss /= len(val_ds)
        scheduler.step()

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)

        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), out_path)
            print(f"  → saved best checkpoint (val={best_val:.4f})", flush=True)
        else:
            epochs_no_improve += 1
            if args.patience > 0 and epochs_no_improve >= args.patience:
                print(f"\nEarly stopping: no val improvement for {args.patience} epochs.", flush=True)
                break

    writer.close()
    print(f"\nDone. Best checkpoint: {out_path}", flush=True)


if __name__ == "__main__":
    main()
