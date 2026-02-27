"""Train U-Net on cropped glottis patches (ROI + letterbox).

Two ROI modes:
  yolo  — Use YOLO to detect bbox, crop image+GT to that box, letterbox to crop_size.
  gt    — Use GT mask to get tight bbox (with padding), crop image+GT, letterbox to crop_size.

In both cases crops are letterboxed to ``--crop-size``×``--crop-size`` (aspect ratio preserved).
At inference: YOLO detect → crop → letterbox → U-Net → unletterbox → paste into full frame.

Training on GT ROI (+ letterbox) aligns the crop U-Net input with the same preprocessing
used at inference (detect → crop → letterbox).

Example
-------
python scripts/train_unet_crop.py \\
    --images-dir    GIRAFE/Training/imagesTr \\
    --labels-dir    GIRAFE/Training/labelsTr \\
    --training-json GIRAFE/Training/training.json \\
    --yolo-weights  outputs/yolo/girafe/weights/best.pt \\
    --output        outputs/openglottal_unet_crop.pt \\
    --crop-size     256 \\
    --epochs        50 \\
    --device        mps
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm.auto import tqdm
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

import sys
_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from openglottal.models import UNet, TemporalDetector
from openglottal.utils import dice_loss, letterbox_with_info, letterbox_apply_geometry


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _source_hash(fnames: list[str]) -> str:
    return hashlib.sha256(",".join(sorted(fnames)).encode()).hexdigest()[:16]


def _cache_meta_valid(cache_dir: Path, split: str, fnames: list[str], crop_size: int, label_suffix: str) -> bool:
    meta_path = cache_dir / split / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        return (
            meta.get("source_hash") == _source_hash(fnames)
            and meta.get("crop_size") == crop_size
            and meta.get("label_suffix") == label_suffix
        )
    except (json.JSONDecodeError, KeyError):
        return False


# ── Dataset ───────────────────────────────────────────────────────────────────

class CroppedGlottisDataset(Dataset):
    """
    Pre-computes YOLO crops for all split images and serves them as tensors.

    Images where YOLO finds no detection are silently skipped.  The crop is
    padded by ``padding`` pixels (same as TemporalDetector default), then
    letterboxed to ``crop_size`` × ``crop_size`` (aspect ratio preserved).
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
        label_suffix: str = "",
        cache_dir: Path | None = None,
        split: str = "train",
    ) -> None:
        self.crop_size = crop_size
        self.augment = augment
        self.samples: list[tuple[np.ndarray, np.ndarray] | int] = []
        self._from_cache = False
        self._img_cache: Path | None = None
        self._msk_cache: Path | None = None

        cache_dir = Path(cache_dir) if cache_dir else None
        if cache_dir and _cache_meta_valid(cache_dir, split, fnames, crop_size, label_suffix):
            meta = json.loads((cache_dir / split / "meta.json").read_text())
            n = int(meta["n"])
            self.samples = list(range(n))
            self._from_cache = True
            self._img_cache = cache_dir / split / "images"
            self._msk_cache = cache_dir / split / "masks"
            return

        for fname in tqdm(fnames, desc=f"Crop {split}", unit="img"):
            img_bgr = cv2.imread(str(img_dir / fname))
            stem = Path(fname).stem
            lbl_name = f"{stem}{label_suffix}.png" if label_suffix else fname
            lbl_gray = cv2.imread(str(lbl_dir / lbl_name), cv2.IMREAD_GRAYSCALE)
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
            crop_img_raw = gray[y1:y2, x1:x2]
            crop_lbl_raw = lbl_gray[y1:y2, x1:x2]
            boxed_img, pad_t, pad_l, ch, cw = letterbox_with_info(
                crop_img_raw, crop_size, value=0
            )
            boxed_lbl = letterbox_apply_geometry(
                crop_lbl_raw, crop_size, pad_t, pad_l, ch, cw,
                value=0, interp=cv2.INTER_NEAREST,
            )
            self.samples.append((boxed_img, boxed_lbl))

        if cache_dir and len(self.samples) > 0:
            (cache_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (cache_dir / split / "masks").mkdir(parents=True, exist_ok=True)
            for i, (img, msk) in tqdm(enumerate(self.samples), total=len(self.samples), desc=f"Write {split} cache", unit="crop"):
                cv2.imwrite(str(cache_dir / split / "images" / f"{i:06d}.png"), img)
                cv2.imwrite(str(cache_dir / split / "masks" / f"{i:06d}.png"), msk)
            (cache_dir / split / "meta.json").write_text(json.dumps({
                "n": len(self.samples), "crop_size": crop_size, "label_suffix": label_suffix,
                "source_hash": _source_hash(fnames), "roi_mode": "yolo",
            }, indent=2))
            indices = list(range(len(self.samples)))
            self.samples = indices
            self._from_cache = True
            self._img_cache = cache_dir / split / "images"
            self._msk_cache = cache_dir / split / "masks"

    def __len__(self) -> int:
        return len(self.samples)

    def _tensor_and_augment(self, img: np.ndarray, msk: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._from_cache and self._img_cache is not None and self._msk_cache is not None:
            img = cv2.imread(str(self._img_cache / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(str(self._msk_cache / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                raise FileNotFoundError(f"Cache missing {idx:06d}")
            return self._tensor_and_augment(img, msk)
        img, msk = self.samples[idx]
        return self._tensor_and_augment(img, msk)

class GTCroppedGlottisDataset(Dataset):
    """
    Crop image + GT mask using the GT mask's tight bounding box (with padding),
    then letterbox to crop_size×crop_size. No YOLO — uses ground truth ROI.

    Matches inference preprocessing (crop → letterbox) so the crop U-Net is
    trained on the same geometry it will see at test time when fed YOLO crops.
    """

    def __init__(
        self,
        fnames: list[str],
        img_dir: Path,
        lbl_dir: Path,
        crop_size: int = 256,
        padding: int = 8,
        augment: bool = False,
        label_suffix: str = "",
        cache_dir: Path | None = None,
        split: str = "train",
    ) -> None:
        self.crop_size = crop_size
        self.augment = augment
        self.samples: list[tuple[np.ndarray, np.ndarray] | int] = []
        self._from_cache = False
        self._img_cache: Path | None = None
        self._msk_cache: Path | None = None

        cache_dir = Path(cache_dir) if cache_dir else None
        if cache_dir and _cache_meta_valid(cache_dir, split, fnames, crop_size, label_suffix):
            meta = json.loads((cache_dir / split / "meta.json").read_text())
            n = int(meta["n"])
            self.samples = list(range(n))
            self._from_cache = True
            self._img_cache = cache_dir / split / "images"
            self._msk_cache = cache_dir / split / "masks"
            return

        for fname in tqdm(fnames, desc=f"Crop {split}", unit="img"):
            img_bgr = cv2.imread(str(img_dir / fname))
            stem = Path(fname).stem
            lbl_name = f"{stem}{label_suffix}.png" if label_suffix else fname
            lbl_gray = cv2.imread(str(lbl_dir / lbl_name), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or lbl_gray is None:
                continue
            if lbl_gray.max() == 0:
                continue

            ys, xs = np.where(lbl_gray > 0)
            H, W = img_bgr.shape[:2]
            x1 = max(0, int(xs.min()) - padding)
            y1 = max(0, int(ys.min()) - padding)
            x2 = min(W, int(xs.max()) + 1 + padding)
            y2 = min(H, int(ys.max()) + 1 + padding)
            if x2 <= x1 or y2 <= y1:
                continue

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            crop_img_raw = gray[y1:y2, x1:x2]
            crop_lbl_raw = lbl_gray[y1:y2, x1:x2]
            boxed_img, pad_t, pad_l, ch, cw = letterbox_with_info(
                crop_img_raw, crop_size, value=0
            )
            boxed_lbl = letterbox_apply_geometry(
                crop_lbl_raw, crop_size, pad_t, pad_l, ch, cw,
                value=0, interp=cv2.INTER_NEAREST,
            )
            self.samples.append((boxed_img, boxed_lbl))

        if cache_dir and len(self.samples) > 0:
            (cache_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (cache_dir / split / "masks").mkdir(parents=True, exist_ok=True)
            for i, (img, msk) in tqdm(enumerate(self.samples), total=len(self.samples), desc=f"Write {split} cache", unit="crop"):
                cv2.imwrite(str(cache_dir / split / "images" / f"{i:06d}.png"), img)
                cv2.imwrite(str(cache_dir / split / "masks" / f"{i:06d}.png"), msk)
            (cache_dir / split / "meta.json").write_text(json.dumps({
                "n": len(self.samples), "crop_size": crop_size, "label_suffix": label_suffix,
                "source_hash": _source_hash(fnames), "roi_mode": "gt",
            }, indent=2))
            self.samples = list(range(len(self.samples)))
            self._from_cache = True
            self._img_cache = cache_dir / split / "images"
            self._msk_cache = cache_dir / split / "masks"

    def __len__(self) -> int:
        return len(self.samples)

    def _tensor_and_augment(self, img: np.ndarray, msk: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
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
                t_img = TF.resize(t_img, [new_size, new_size], interpolation=TF.InterpolationMode.BILINEAR)
                t_msk = TF.resize(t_msk, [new_size, new_size], interpolation=TF.InterpolationMode.NEAREST)
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
                t_img = torch.clamp(t_img + torch.randn_like(t_img) * random.uniform(0.01, 0.05), 0.0, 1.0)
            if random.random() > 0.5:
                t_img = TF.gaussian_blur(t_img, kernel_size=random.choice([3, 5]), sigma=random.uniform(0.5, 1.5))
            if random.random() > 0.5:
                t_img = torch.clamp(t_img * random.uniform(0.7, 1.3), 0.0, 1.0)
            if random.random() > 0.5:
                t_img = TF.adjust_contrast(t_img, random.uniform(0.7, 1.3))
        return t_img, t_msk

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._from_cache and self._img_cache is not None and self._msk_cache is not None:
            img = cv2.imread(str(self._img_cache / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(str(self._msk_cache / f"{idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                raise FileNotFoundError(f"Cache missing {idx:06d}")
            return self._tensor_and_augment(img, msk)
        img, msk = self.samples[idx]
        return self._tensor_and_augment(img, msk)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train U-Net on YOLO-cropped glottis patches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir",    required=True)
    p.add_argument("--labels-dir",    required=True)
    p.add_argument("--training-json", required=True)
    p.add_argument("--roi-mode",      choices=("yolo", "gt"), default="yolo",
                   help="yolo: crop using YOLO bbox. gt: crop using GT mask bbox (then letterbox).")
    p.add_argument("--yolo-weights",  default=None,
                   help="YOLO weights for --roi-mode yolo. Not used for gt.")
    p.add_argument("--output",        default="outputs/openglottal_unet_crop.pt",
                   help="Path to save best checkpoint (outputs/ is gitignored).")
    p.add_argument("--output-suffix", default=None,
                   help="If set, save as <output_stem>_<suffix>.pt (e.g. letterbox, bagls).")
    p.add_argument("--crop-size",     type=int,   default=256,
                   help="Side length (px) for letterboxed YOLO crops (aspect ratio preserved).")
    p.add_argument("--padding",       type=int,   default=8,
                   help="Extra pixels added to each side of the YOLO bbox.")
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--batch",         type=int,   default=16)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--features", nargs="+", type=int, default=[32, 64, 128, 256])
    p.add_argument("--device",        default=None,
                   help="torch device (mps / cuda / cpu). Auto-detected if omitted.")
    p.add_argument("--label-suffix",  default="",
                   help="Mask filename is stem+label_suffix+.png (e.g. _seg for BAGLS).")
    p.add_argument("--log-dir",       default="runs",
                   help="TensorBoard log directory. Logs go to <log-dir>/<run_name>.")
    p.add_argument("--num-workers",   type=int, default=0,
                   help="DataLoader workers. Use 4–8 to overlap data prep with training (can help MPS/GPU utilization).")
    p.add_argument("--cache-dir",     default=None,
                   help="Disk cache for pre-cropped images. First run builds cache; later runs load from cache (fast startup, lower RAM). e.g. outputs/crop_cache")
    p.add_argument("--cache-only",    action="store_true",
                   help="Only build the crop cache (requires --cache-dir), then exit. Use before training for fast restarts.")
    p.add_argument("--patience",       type=int, default=0,
                   help="Early stopping: stop if val loss does not improve for this many epochs (0 = disabled). e.g. 5")
    p.add_argument("--resume",        default=None,
                   help="Resume: load model weights from this checkpoint and continue training (epoch 1, fresh optimizer).")
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
    print(f"Device : {device}", flush=True)
    print(f"ROI mode : {args.roi_mode} (crop → letterbox to {args.crop_size}×{args.crop_size})", flush=True)

    splits  = json.load(open(args.training_json))
    img_dir = Path(args.images_dir)
    lbl_dir = Path(args.labels_dir)

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if args.cache_only:
        if not cache_dir:
            raise RuntimeError("--cache-only requires --cache-dir.")
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache dir : {cache_dir}  (building only, then exit)\n")
    elif cache_dir:
        print(f"Cache dir : {cache_dir}  (load or build pre-cropped images)")

    if args.roi_mode == "gt":
        print("Building GT-ROI cropped training split (bbox from mask + letterbox) ...", flush=True)
        train_ds = GTCroppedGlottisDataset(
            splits["training"], img_dir, lbl_dir,
            crop_size=args.crop_size, padding=args.padding, augment=True,
            label_suffix=args.label_suffix,
            cache_dir=cache_dir, split="train",
        )
        print("Building GT-ROI cropped validation split ...", flush=True)
        val_ds = GTCroppedGlottisDataset(
            splits["Val"], img_dir, lbl_dir,
            crop_size=args.crop_size, padding=args.padding, augment=False,
            label_suffix=args.label_suffix,
            cache_dir=cache_dir, split="val",
        )
    else:
        if not args.yolo_weights:
            raise RuntimeError("--roi-mode yolo requires --yolo-weights.")
        from ultralytics import YOLO as _YOLO
        print(f"Loading YOLO from {args.yolo_weights} ...")
        yolo = _YOLO(args.yolo_weights)
        print("Building YOLO-cropped training split ...")
        train_ds = CroppedGlottisDataset(
            splits["training"], img_dir, lbl_dir, yolo,
            crop_size=args.crop_size, padding=args.padding, augment=True,
            label_suffix=args.label_suffix,
            cache_dir=cache_dir, split="train",
        )
        print("Building YOLO-cropped validation split ...")
        val_ds = CroppedGlottisDataset(
            splits["Val"], img_dir, lbl_dir, yolo,
            crop_size=args.crop_size, padding=args.padding, augment=False,
            label_suffix=args.label_suffix,
            cache_dir=cache_dir, split="val",
        )

    print(f"Train crops : {len(train_ds)}  |  Val crops : {len(val_ds)}", flush=True)
    if len(train_ds) == 0:
        raise RuntimeError("No crops in training split (empty masks or no YOLO detections).")

    if args.cache_only:
        print("\nCache ready. Run training with the same --cache-dir for fast startup.")
        return

    train_ldr = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers,
        **(dict(persistent_workers=True) if args.num_workers > 0 else {}),
    )
    val_ldr = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers,
        **(dict(persistent_workers=True) if args.num_workers > 0 else {}),
    )

    model = UNet(1, 1, tuple(args.features)).to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        model.load_state_dict(state, strict=True)
        print(f"Resumed from {args.resume}", flush=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"U-Net  : {n_params / 1e6:.2f}M parameters  (crop_size={args.crop_size})\n", flush=True)

    bce       = nn.BCEWithLogitsLoss()
    optim     = torch.optim.AdamW(model.parameters(), lr=args.lr)
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

        lr = scheduler.get_last_lr()[0]
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("lr", lr, epoch)

        print(f"Epoch {epoch:3d}/{args.epochs}  train={train_loss:.4f}  val={val_loss:.4f}", flush=True)

        # Save latest every epoch (so you have a checkpoint even if run stops early)
        latest_path = out_path.parent / f"{out_path.stem}_latest{out_path.suffix}"
        torch.save(model.state_dict(), latest_path)

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
    print(f"At inference: YOLO detect → crop → letterbox → this U-Net → unletterbox.", flush=True)
    print(f"Use --crop-weights {out_path} with eval_bagls.py / eval_girafe.py (yolo-crop+unet pipeline).", flush=True)


if __name__ == "__main__":
    main()
