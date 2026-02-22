"""Train the U-Net segmentation model on the GIRAFE dataset.

Example
-------
python scripts/train_unet.py \\
    --images-dir  /path/to/GIRAFE/Training/imagesTr \\
    --labels-dir  /path/to/GIRAFE/Training/labelsTr \\
    --training-json /path/to/GIRAFE/Training/training.json \\
    --output glottal_detector/unet_glottis_v2.pt \\
    --epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from openglottal.models import UNet, GlottisDataset
from openglottal.utils import dice_loss


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net glottis segmenter.")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--training-json", required=True)
    p.add_argument("--output", default="glottal_detector/unet_glottis_v2.pt",
                   help="Path to save best checkpoint.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--features", nargs="+", type=int, default=[32, 64, 128, 256])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = (
        torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device}")

    splits = json.load(open(args.training_json))
    train_ds = GlottisDataset(splits["training"], args.images_dir, args.labels_dir, augment=True)
    val_ds = GlottisDataset(splits["Val"], args.images_dir, args.labels_dir, augment=False)
    train_ldr = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_ldr = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet(1, 1, tuple(args.features)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net — {n_params / 1e6:.2f}M parameters")

    bce = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
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

        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_path)
            print(f"  → saved best checkpoint (val={best_val:.4f})")

    print(f"\nDone. Best checkpoint: {out_path}")


if __name__ == "__main__":
    main()
