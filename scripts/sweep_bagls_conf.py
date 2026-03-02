"""Single-pass YOLO confidence threshold sweep on BAGLS.

Runs YOLO (at conf=0.01) and U-Net inference once per frame, then applies
multiple confidence thresholds in post-processing to measure the effect on
detection recall and segmentation accuracy.

Usage
-----
python scripts/sweep_bagls_conf.py \
    --bagls-dir      BAGLS/test \
    --unet-weights   outputs/openglottal_unet.pt \
    --crop-weights   outputs/openglottal_unet_crop.pt \
    --yolo-weights   outputs/openglottal_yolo.pt \
    --device         mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openglottal.models import UNet
from openglottal.utils import unet_segment_frame, letterbox_with_info, unletterbox, resolve_weights_path

CONF_THRESHOLDS = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25]
PIPELINES = ["unet-only", "yolo+unet", "yolo-crop+unet"]
RAW_CONF = 0.001  # run YOLO at minimum conf to capture all possible detections
PADDING = 8


def letterbox(img: np.ndarray, size: int = 256, value: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_h, pad_w = size - new_h, size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    if img.ndim == 3:
        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(value, value, value))
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=value)


def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p = (pred > 0).astype(np.float32).ravel()
    g = (gt > 0).astype(np.float32).ravel()
    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    dice = float(2 * tp / denom_dice) if denom_dice > 0 else 1.0
    iou = float(tp / denom_iou) if denom_iou > 0 else 1.0
    return dice, iou


def unet_on_crop(
    gray: np.ndarray, box: tuple, model: torch.nn.Module,
    device: torch.device, crop_size: int = 256,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros_like(gray)
    crop_h, crop_w = crop.shape[:2]
    # Letterbox crop to preserve aspect ratio
    boxed, pad_t, pad_l, content_h, content_w = letterbox_with_info(
        crop, crop_size, value=0
    )
    mask_cs = unet_segment_frame(boxed, model, device)
    mask_orig = unletterbox(
        mask_cs, pad_t, pad_l, content_h, content_w,
        crop_h, crop_w, interp=cv2.INTER_NEAREST,
    )
    full = np.zeros_like(gray)
    full[y1:y2, x1:x2] = mask_orig
    return full


def raw_yolo_detect(
    yolo_model: YOLO, frame_bgr: np.ndarray, padding: int = PADDING,
) -> tuple[tuple[int, int, int, int] | None, float]:
    """Run YOLO at RAW_CONF and return (box, confidence) or (None, 0.0)."""
    H, W = frame_bgr.shape[:2]
    results = yolo_model(frame_bgr, conf=RAW_CONF, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    idx = int(boxes.conf.argmax())
    conf = float(boxes.conf[idx].cpu())
    x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw = int(x2 - x1) + 2 * padding
    bh = int(y2 - y1) + 2 * padding
    hw, hh = bw // 2, bh // 2
    cx = int(np.clip(cx, hw, W - hw))
    cy = int(np.clip(cy, hh, H - hh))
    return (cx - hw, cy - hh, cx + hw, cy + hh), conf


def main() -> None:
    p = argparse.ArgumentParser(
        description="Single-pass YOLO conf sweep on BAGLS.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bagls-dir", required=True)
    p.add_argument("--unet-weights", required=True)
    p.add_argument("--crop-weights", default=None)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--canvas", type=int, default=256)
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--output-json", default=None)
    p.add_argument("--output-per-frame-dice", default=None,
                   help="Save per-frame Dice (yolo-crop+unet, tau=0.02) as JSON list for fig_bagls_sweep waveforms.")
    args = p.parse_args()

    device = torch.device(args.device)

    unet_path = resolve_weights_path(args.unet_weights)
    crop_path = resolve_weights_path(args.crop_weights) if args.crop_weights else None
    yolo_path = resolve_weights_path(args.yolo_weights)

    unet = UNet(1, 1, (32, 64, 128, 256)).to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    unet.eval()
    print(f"Loaded full-frame U-Net : {unet_path}")

    crop_model = None
    if crop_path is not None:
        crop_model = UNet(1, 1, (32, 64, 128, 256)).to(device)
        crop_model.load_state_dict(
            torch.load(crop_path, map_location=device, weights_only=True))
        crop_model.eval()
        print(f"Loaded crop U-Net       : {crop_path}")

    yolo_model = YOLO(str(yolo_path))
    print(f"Loaded YOLO             : {yolo_path}")

    test_dir = Path(args.bagls_dir)
    img_files = sorted(
        f for f in test_dir.iterdir()
        if f.suffix == ".png" and not f.name.endswith("_seg.png")
    )
    if args.max_images:
        img_files = img_files[:args.max_images]
    print(f"\nBAGLS test frames: {len(img_files)}\n")
    print("Running single-pass inference (YOLO + U-Net + crop U-Net) ...")

    # -- Single pass: collect per-frame data ----------------------------------
    per_frame = []
    for i, img_path in enumerate(img_files):
        seg_path = img_path.with_name(img_path.stem + "_seg.png")
        if not seg_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        gt_raw = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt_raw is None:
            continue

        img_lb = letterbox(img_bgr, args.canvas)
        gt_lb = letterbox(gt_raw, args.canvas)
        gray_lb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2GRAY)

        box, conf = raw_yolo_detect(yolo_model, img_lb)

        mask_unet_full = unet_segment_frame(gray_lb, unet, device)

        mask_crop_full = None
        if crop_model is not None and box is not None:
            mask_crop_full = unet_on_crop(gray_lb, box, crop_model, device)

        per_frame.append({
            "box": box,
            "conf": conf,
            "mask_unet": mask_unet_full,
            "mask_crop": mask_crop_full,
            "gt": gt_lb,
        })

        if (i + 1) % 500 == 0:
            print(f"  [{i+1}/{len(img_files)}]")

    print(f"Inference done: {len(per_frame)} frames.\n")

    # -- Sweep thresholds -----------------------------------------------------
    results = {}
    for thr in CONF_THRESHOLDS:
        agg = {pipe: {"dice": [], "iou": [], "n_det": 0, "n_total": 0}
               for pipe in PIPELINES}

        for f in per_frame:
            gt = f["gt"]
            box_valid = f["box"] is not None and f["conf"] >= thr

            # unet-only (threshold-independent)
            agg["unet-only"]["n_total"] += 1
            d, iu = frame_metrics(f["mask_unet"], gt)
            agg["unet-only"]["dice"].append(d)
            agg["unet-only"]["iou"].append(iu)

            # yolo+unet
            agg["yolo+unet"]["n_total"] += 1
            if box_valid:
                agg["yolo+unet"]["n_det"] += 1
                x1, y1, x2, y2 = f["box"]
                mask_yu = np.zeros_like(f["mask_unet"])
                mask_yu[y1:y2, x1:x2] = f["mask_unet"][y1:y2, x1:x2]
            else:
                mask_yu = np.zeros_like(f["mask_unet"])
            d, iu = frame_metrics(mask_yu, gt)
            agg["yolo+unet"]["dice"].append(d)
            agg["yolo+unet"]["iou"].append(iu)

            # yolo-crop+unet
            if crop_model is not None:
                agg["yolo-crop+unet"]["n_total"] += 1
                if box_valid and f["mask_crop"] is not None:
                    agg["yolo-crop+unet"]["n_det"] += 1
                    mask_c = f["mask_crop"]
                else:
                    mask_c = np.zeros_like(gt)
                d, iu = frame_metrics(mask_c, gt)
                agg["yolo-crop+unet"]["dice"].append(d)
                agg["yolo-crop+unet"]["iou"].append(iu)

        results[thr] = agg

    # -- Print summary table --------------------------------------------------
    sep = "=" * 90
    print(sep)
    print(f"  {'Conf':>5}  {'Pipeline':<20}  {'Det.Recall':>10}  "
          f"{'Dice':>8}  {'IoU':>8}  {'Dice>=0.5':>10}")
    print(sep)

    for thr in CONF_THRESHOLDS:
        agg = results[thr]
        for pipe in PIPELINES:
            data = agg[pipe]
            if not data["dice"]:
                continue
            n_det = data["n_det"]
            n_tot = data["n_total"]
            det_rec = n_det / n_tot if n_tot > 0 else float("nan")
            mean_dice = np.mean(data["dice"])
            mean_iou = np.mean(data["iou"])
            d50 = np.mean([d >= 0.5 for d in data["dice"]]) * 100
            dr_str = "1.000 *" if pipe == "unet-only" else f"{det_rec:.3f}"
            print(f"  {thr:>5.2f}  {pipe:<20}  {dr_str:>10}  "
                  f"{mean_dice:>8.3f}  {mean_iou:>8.3f}  {d50:>9.1f}%")
        if thr != CONF_THRESHOLDS[-1]:
            print("  " + "─" * 86)

    print(sep)
    print(f"  * unet-only is threshold-independent (no YOLO gate).")
    print(f"  Evaluated on {len(per_frame)} BAGLS frames.")
    print()

    if args.output_json:
        serialisable = {}
        for thr, agg in results.items():
            serialisable[str(thr)] = {
                pipe: {
                    "dice_mean": float(np.mean(data["dice"])) if data["dice"] else None,
                    "iou_mean": float(np.mean(data["iou"])) if data["iou"] else None,
                    "dice_ge_05": float(np.mean([d >= 0.5 for d in data["dice"]]) * 100)
                    if data["dice"] else None,
                    "det_recall": data["n_det"] / data["n_total"]
                    if data["n_total"] > 0 else None,
                    "n_det": data["n_det"],
                    "n_total": data["n_total"],
                }
                for pipe, data in agg.items()
                if data["dice"]
            }
        with open(args.output_json, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Results saved → {args.output_json}")

    if args.output_per_frame_dice and crop_model is not None:
        thr = 0.02
        per_frame_dice = []
        for f in per_frame:
            gt = f["gt"]
            box_valid = f["box"] is not None and f["conf"] >= thr
            if box_valid and f["mask_crop"] is not None:
                mask_c = f["mask_crop"]
            else:
                mask_c = np.zeros_like(gt)
            d, _ = frame_metrics(mask_c, gt)
            per_frame_dice.append(d)
        with open(args.output_per_frame_dice, "w") as f:
            json.dump(per_frame_dice, f)
        print(f"Per-frame dice (yolo-crop+unet, tau=0.02) saved → {args.output_per_frame_dice}")


if __name__ == "__main__":
    main()
