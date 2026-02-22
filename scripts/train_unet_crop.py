"""Train U-Net on YOLO-cropped glottis patches.

Instead of training on the full 256×256 GIRAFE frames, this script:
  1. Runs YOLO on each training/validation image to detect the glottis bbox
  2. Crops image + GT mask to the padded bbox
  3. Resizes each crop to ``--crop-size`` × ``--crop-size`` (default 256)
  4. Trains U-Net on these patches

At inference time the matching pipeline is:
  YOLO detect → crop → resize to crop_size → U-Net → resize back → paste into full frame

This gives U-Net higher effective resolution on the glottis region and removes
background clutter, typically improving Dice on well-detected patients.
Frames where YOLO finds no detection are excluded from training/validation.

Example
-------
python scripts/train_unet_crop.py \\
    --images-dir    GIRAFE/Training/imagesTr \\
    --labels-dir    GIRAFE/Training/labelsTr \\
    --training-json GIRAFE/Training/training.json \\
    --yolo-weights  runs/detect/glottal_detector/yolov8n_girafe/weights/best.pt \\
    --output        glottal_detector/unet_glottis_crop.pt \\
    --crop-size     256 \\
    --epochs        50 \\
    --device        mps
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openglottal.models import UNet, TemporalDetector
from openglottal.utils import dice_loss


# ── Dataset ───────────────────────────────────────────────────────────────────

class CroppedGlottisDataset(Dataset):
    """
    Pre-computes YOLO crops for all split images and serves them as tensors.

    Images where YOLO finds no detection are silently skipped.  The crop is
    padded by ``padding`` pixels (same as TemporalDetector default) before
    being resized to ``crop_size`` × ``crop_size``.
    """

    def __init__(
        self,
        fnames: list[str],
        img_dir: Path,
        lbl_dir: Path,
        yolo_model,
        crop_size: int = 256,
        padding: int = 8,
        conf: float = 0.25,
        augment: bool = False,
    ) -> None:
        self.crop_size = crop_size
        self.augment = augment
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        for fname in fnames:
            img_bgr = cv2.imread(str(img_dir / fname))
            lbl_gray = cv2.imread(str(lbl_dir / fname), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or lbl_gray is None:
                continue

            results = yolo_model(img_bgr, conf=conf, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                continue

            idx = int(boxes.conf.argmax())
            x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
            H, W = img_bgr.shape[:2]
            x1 = max(0, int(x1) - padding)
            y1 = max(0, int(y1) - padding)
            x2 = min(W, int(x2) + padding)
            y2 = min(H, int(y2) + padding)

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            crop_img = cv2.resize(gray[y1:y2, x1:x2],
                                  (crop_size, crop_size),
                                  interpolation=cv2.INTER_LINEAR)
            crop_lbl = cv2.resize(lbl_gray[y1:y2, x1:x2],
                                  (crop_size, crop_size),
                                  interpolation=cv2.INTER_NEAREST)
            self.samples.append((crop_img, crop_lbl))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img, msk = self.samples[idx]
        t_img = torch.from_numpy(img.astype("float32") / 255.0).unsqueeze(0)
        t_msk = torch.from_numpy((msk > 0).astype("float32")).unsqueeze(0)

        if self.augment:
            if random.random() > 0.5:
                t_img, t_msk = TF.hflip(t_img), TF.hflip(t_msk)
            if random.random() > 0.5:
                t_img, t_msk = TF.vflip(t_img), TF.vflip(t_msk)

            angle = random.uniform(-30, 30)
            t_img = TF.rotate(t_img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            t_msk = TF.rotate(t_msk, angle, interpolation=TF.InterpolationMode.NEAREST)

            if random.random() > 0.5:
                scale = random.uniform(0.85, 1.15)
                new_size = int(self.crop_size * scale)
                t_img = TF.resize(t_img, [new_size, new_size],
                                  interpolation=TF.InterpolationMode.BILINEAR)
                t_msk = TF.resize(t_msk, [new_size, new_size],
                                  interpolation=TF.InterpolationMode.NEAREST)
                if new_size > self.crop_size:
                    off = (new_size - self.crop_size) // 2
                    t_img = TF.crop(t_img, off, off, self.crop_size, self.crop_size)
                    t_msk = TF.crop(t_msk, off, off, self.crop_size, self.crop_size)
                else:
                    pad = self.crop_size - new_size
                    pl, pr = pad // 2, pad - pad // 2
                    t_img = TF.pad(t_img, [pl, pl, pr, pr])
                    t_msk = TF.pad(t_msk, [pl, pl, pr, pr])

            if random.random() > 0.5:
                sigma = random.uniform(0.01, 0.05)
                t_img = torch.clamp(t_img + torch.randn_like(t_img) * sigma, 0.0, 1.0)

            if random.random() > 0.5:
                ks = random.choice([3, 5])
                t_img = TF.gaussian_blur(t_img, kernel_size=ks, sigma=random.uniform(0.5, 1.5))

            if random.random() > 0.5:
                t_img = torch.clamp(t_img * random.uniform(0.7, 1.3), 0.0, 1.0)

            if random.random() > 0.5:
                t_img = TF.adjust_contrast(t_img, random.uniform(0.7, 1.3))

        return t_img, t_msk


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train U-Net on YOLO-cropped glottis patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir",    required=True)
    p.add_argument("--labels-dir",    required=True)
    p.add_argument("--training-json", required=True)
    p.add_argument("--yolo-weights",  required=True,
                   help="Trained YOLO weights used to generate crops.")
    p.add_argument("--output",        default="glottal_detector/unet_glottis_crop.pt",
                   help="Path to save best checkpoint.")
    p.add_argument("--crop-size",     type=int,   default=256,
                   help="Side length (px) to which every YOLO crop is resized.")
    p.add_argument("--padding",       type=int,   default=8,
                   help="Extra pixels added to each side of the YOLO bbox.")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch",         type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--features", nargs="+", type=int, default=[32, 64, 128, 256])
    p.add_argument("--device",        default=None,
                   help="torch device (mps / cuda / cpu). Auto-detected if omitted.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = (
            torch.device("mps")  if torch.backends.mps.is_available()  else
            torch.device("cuda") if torch.cuda.is_available()           else
            torch.device("cpu")
        )
    print(f"Device : {device}")

    # Load YOLO (CPU is fine for offline pre-processing)
    print(f"Loading YOLO from {args.yolo_weights} ...")
    from ultralytics import YOLO as _YOLO
    yolo = _YOLO(args.yolo_weights)

    splits    = json.load(open(args.training_json))
    img_dir   = Path(args.images_dir)
    lbl_dir   = Path(args.labels_dir)

    print("Building cropped training split ...")
    train_ds = CroppedGlottisDataset(
        splits["training"], img_dir, lbl_dir, yolo,
        crop_size=args.crop_size, padding=args.padding, augment=True,
    )
    print("Building cropped validation split ...")
    val_ds = CroppedGlottisDataset(
        splits["Val"], img_dir, lbl_dir, yolo,
        crop_size=args.crop_size, padding=args.padding, augment=False,
    )

    print(f"Train crops : {len(train_ds)}  |  Val crops : {len(val_ds)}")
    if len(train_ds) == 0:
        raise RuntimeError("No YOLO detections in training split — check yolo-weights.")

    train_ldr = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_ldr   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNet(1, 1, tuple(args.features)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net  : {n_params / 1e6:.2f}M parameters  (crop_size={args.crop_size})\n")

    bce       = nn.BCEWithLogitsLoss()
    optim     = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
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
    print(f"Use --unet-weights {out_path} with eval_girafe.py "
          f"(yolo-crop+unet pipeline) for evaluation.")


if __name__ == "__main__":
    main()
