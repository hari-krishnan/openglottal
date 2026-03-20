"""Evaluate BAGLS with histogram-matched preprocessing.

Each BAGLS grayscale frame is histogram-matched to the aggregate GIRAFE
training histogram before being fed to YOLO and U-Net.  This maps the
BAGLS pixel distribution into the GIRAFE domain at inference time.

Usage
-----
python scripts/eval_bagls_histmatch.py \
    --bagls-dir      /path/to/BAGLS/test \
    --unet-weights   outputs/openglottal_unet_crop.pt \
    --yolo-weights   weights/openglottal_yolo.pt \
    --girafe-hist    outputs/girafe_histogram.npy \
    --device         mps \
    --max-images     100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_repo_root = str(Path(__file__).resolve().parents[1])
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from openglottal.models import UNet, TemporalDetector
from openglottal.utils import unet_segment_frame, letterbox_with_info, unletterbox, resolve_weights_path


# ── Letterbox ─────────────────────────────────────────────────────────────────

def letterbox(img: np.ndarray, size: int = 256, value: int = 0) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    pad_h = size - new_h
    pad_w = size - new_w
    top,   bottom = pad_h // 2, pad_h - pad_h // 2
    left,  right  = pad_w // 2, pad_w - pad_w // 2
    if img.ndim == 3:
        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(value, value, value))
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=value)


# ── Histogram matching ─────────────────────────────────────────────────────────

def build_ref_cdf(histogram: np.ndarray) -> np.ndarray:
    """Normalised cumulative distribution function from a 256-bin histogram."""
    cdf = histogram.astype(np.float64).cumsum()
    cdf /= cdf[-1]
    return cdf


def histmatch(gray: np.ndarray, ref_cdf: np.ndarray) -> np.ndarray:
    """Map uint8 grayscale image to match reference CDF.

    Computes the source CDF, then builds a 256-entry lookup table that maps
    each source intensity to the reference intensity with the closest CDF value.
    """
    src_hist, _ = np.histogram(gray.ravel(), bins=256, range=(0, 256))
    src_cdf = src_hist.astype(np.float64).cumsum()
    src_cdf /= src_cdf[-1]

    # For each source value s, find the ref value r where ref_cdf[r] ≈ src_cdf[s]
    lut = np.searchsorted(ref_cdf, src_cdf).clip(0, 255).astype(np.uint8)
    return lut[gray]


# ── Helpers ───────────────────────────────────────────────────────────────────

def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    p = (pred > 0).astype(np.float32).ravel()
    g = (gt   > 0).astype(np.float32).ravel()
    tp = (p * g).sum()
    fp = (p * (1 - g)).sum()
    fn = ((1 - p) * g).sum()
    denom_dice = 2 * tp + fp + fn
    denom_iou  = tp + fp + fn
    dice = float(2 * tp / denom_dice) if denom_dice > 0 else 1.0
    iou  = float(tp / denom_iou)      if denom_iou  > 0 else 1.0
    return dice, iou


def unet_on_crop(
    gray: np.ndarray,
    box: tuple,
    model: torch.nn.Module,
    device: torch.device,
    crop_size: int = 256,
) -> np.ndarray:
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros_like(gray)
    crop_h, crop_w = crop.shape[:2]
    boxed, pad_t, pad_l, content_h, content_w = letterbox_with_info(crop, crop_size, value=0)
    mask_cs = unet_segment_frame(boxed, model, device)
    mask_orig = unletterbox(mask_cs, pad_t, pad_l, content_h, content_w,
                            crop_h, crop_w, interp=cv2.INTER_NEAREST)
    full = np.zeros_like(gray)
    full[y1:y2, x1:x2] = mask_orig
    return full


# ── Evaluation ────────────────────────────────────────────────────────────────

PIPELINES = ["unet-only", "yolo+unet", "yolo-crop+unet"]


def evaluate(
    test_dir: Path,
    unet_model: torch.nn.Module,
    crop_model: torch.nn.Module | None,
    detector: TemporalDetector | None,
    device: torch.device,
    ref_cdf: np.ndarray,
    max_images: int = 0,
    canvas: int = 256,
    crop_pad: int = 0,
) -> tuple[dict, dict]:
    agg = {p: {"dice": [], "iou": [], "n_det": 0, "n_total": 0} for p in PIPELINES}
    det_stats: dict = {"tp": 0, "fp": 0, "fn": 0, "n_pos_gt": 0}

    img_files = sorted(
        f for f in test_dir.iterdir()
        if f.suffix == ".png" and not f.name.endswith("_seg.png")
    )
    if max_images:
        img_files = img_files[:max_images]

    for i, img_path in enumerate(img_files):
        seg_path = img_path.with_name(img_path.stem + "_seg.png")
        if not seg_path.exists():
            continue

        img_bgr = cv2.imread(str(img_path))
        gt_raw  = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt_raw is None:
            continue

        img_lb = letterbox(img_bgr, canvas)
        gt_lb  = letterbox(gt_raw,  canvas)
        gray_lb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2GRAY)

        # Apply histogram matching: BAGLS → GIRAFE distribution
        gray_matched = histmatch(gray_lb, ref_cdf)

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(img_files)}] ...")

        # YOLO detection on histogram-matched frame
        if detector is not None:
            detector.reset()
            img_for_det = cv2.cvtColor(gray_matched, cv2.COLOR_GRAY2BGR)
            box = detector.detect(img_for_det)
        else:
            box = None

        if detector is not None:
            gt_pos = (gt_lb > 0).any()
            if gt_pos:
                det_stats["n_pos_gt"] += 1
            if box is not None:
                x1, y1, x2, y2 = (max(0, min(canvas, int(v))) for v in box)
                if gt_lb[y1:y2, x1:x2].any():
                    det_stats["tp"] += 1
                else:
                    det_stats["fp"] += 1
            else:
                if gt_pos:
                    det_stats["fn"] += 1

        # ── unet-only ──────────────────────────────────────────────────────
        agg["unet-only"]["n_total"] += 1
        mask_u = unet_segment_frame(gray_matched, unet_model, device)
        d, iu = frame_metrics(mask_u, gt_lb)
        agg["unet-only"]["dice"].append(d)
        agg["unet-only"]["iou"].append(iu)

        # ── yolo+unet ──────────────────────────────────────────────────────
        agg["yolo+unet"]["n_total"] += 1
        if box is not None:
            agg["yolo+unet"]["n_det"] += 1
            x1, y1, x2, y2 = box
            mask_yu = np.zeros_like(mask_u)
            mask_yu[y1:y2, x1:x2] = mask_u[y1:y2, x1:x2]
        else:
            mask_yu = np.zeros_like(mask_u)
        d, iu = frame_metrics(mask_yu, gt_lb)
        agg["yolo+unet"]["dice"].append(d)
        agg["yolo+unet"]["iou"].append(iu)

        # ── yolo-crop+unet ─────────────────────────────────────────────────
        if crop_model is not None:
            agg["yolo-crop+unet"]["n_total"] += 1
            if box is not None:
                agg["yolo-crop+unet"]["n_det"] += 1
                if crop_pad:
                    x1, y1, x2, y2 = box
                    box = (
                        max(0, x1 - crop_pad), max(0, y1 - crop_pad),
                        min(canvas, x2 + crop_pad), min(canvas, y2 + crop_pad),
                    )
                mask_c = unet_on_crop(gray_matched, box, crop_model, device)
            else:
                mask_c = np.zeros_like(gray_matched)
            d, iu = frame_metrics(mask_c, gt_lb)
            agg["yolo-crop+unet"]["dice"].append(d)
            agg["yolo-crop+unet"]["iou"].append(iu)

    return agg, det_stats


# ── Pretty-print ──────────────────────────────────────────────────────────────

def print_table(agg: dict, has_yolo: bool, has_crop: bool, det_stats: dict | None = None) -> None:
    label_map = {
        "unet-only":      "U-Net only",
        "yolo+unet":      "YOLO+UNet",
        "yolo-crop+unet": "YOLO-Crop+UNet",
    }
    active = ["unet-only"]
    if has_yolo:
        active.append("yolo+unet")
    if has_crop and has_yolo:
        active.append("yolo-crop+unet")

    sep = "─" * 76
    print(f"\n{sep}")
    print(f"  {'Method':<25}  {'Det.Recall':>10}  {'Dice':>8}  {'IoU':>8}  {'Dice≥0.5':>10}")
    print(sep)
    for pipe in active:
        data  = agg[pipe]
        dices = data["dice"]
        ious  = data["iou"]
        n_det = data["n_det"]
        n_tot = data["n_total"]
        det_rec   = n_det / n_tot if n_tot > 0 else float("nan")
        mean_dice = np.mean(dices) if dices else float("nan")
        mean_iou  = np.mean(ious)  if ious  else float("nan")
        d50       = np.mean([d >= 0.5 for d in dices]) * 100 if dices else float("nan")
        label     = label_map[pipe]
        dr_str = f"{det_rec:.3f}" if pipe != "unet-only" else "1.000 *"
        print(f"  {label:<25}  {dr_str:>10}  {mean_dice:>8.3f}  {mean_iou:>8.3f}  {d50:>9.1f}%")
    print(sep)
    print("  * U-Net only: no YOLO gate — always processes 100% of frames.")
    print(f"  Evaluated on {agg['unet-only']['n_total']} BAGLS test frames")
    print()

    if has_yolo and det_stats is not None:
        tp = det_stats["tp"]
        fp = det_stats["fp"]
        fn = det_stats["fn"]
        prec = float(tp) / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec  = float(tp) / (tp + fn) if (tp + fn) > 0 else float("nan")
        print(f"  YOLO detection — Precision: {prec:.3f}  Recall: {rec:.3f}  (TP={tp} FP={fp} FN={fn})")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BAGLS evaluation with GIRAFE histogram-matched preprocessing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bagls-dir",    required=True)
    p.add_argument("--unet-weights", required=True)
    p.add_argument("--crop-weights", default=None)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--girafe-hist",  required=True,
                   help="Precomputed GIRAFE aggregate histogram (.npy from compute_girafe_histogram.py).")
    p.add_argument("--device",       default="cpu")
    p.add_argument("--canvas",       type=int, default=256)
    p.add_argument("--max-images",   type=int, default=0)
    p.add_argument("--conf",         type=float, default=0.25)
    p.add_argument("--crop-pad",     type=int, default=0)
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    unet_path = resolve_weights_path(args.unet_weights)
    crop_path = resolve_weights_path(args.crop_weights) if args.crop_weights else None
    yolo_path = resolve_weights_path(args.yolo_weights)

    unet = UNet(1, 1, (32, 64, 128, 256)).to(device)
    unet.load_state_dict(torch.load(unet_path, map_location=device, weights_only=True))
    unet.eval()
    print(f"Loaded U-Net    : {unet_path}")

    crop_model = None
    if crop_path is not None:
        crop_model = UNet(1, 1, (32, 64, 128, 256)).to(device)
        crop_model.load_state_dict(torch.load(crop_path, map_location=device, weights_only=True))
        crop_model.eval()
        print(f"Loaded crop U-Net: {crop_path}")

    detector = TemporalDetector(str(yolo_path), conf=args.conf)
    print(f"Loaded YOLO     : {yolo_path}")

    ref_hist = np.load(args.girafe_hist)
    ref_cdf  = build_ref_cdf(ref_hist)
    print(f"Loaded GIRAFE reference histogram: {args.girafe_hist}  (total pixels: {ref_hist.sum():,})")

    test_dir = Path(args.bagls_dir)
    n_avail  = sum(1 for f in test_dir.iterdir()
                   if f.suffix == ".png" and not f.name.endswith("_seg.png"))
    n_eval   = args.max_images if args.max_images else n_avail
    print(f"\nBAGLS test frames: {n_avail} available, evaluating {n_eval}\n")

    agg, det_stats = evaluate(
        test_dir   = test_dir,
        unet_model = unet,
        crop_model = crop_model,
        detector   = detector,
        device     = device,
        ref_cdf    = ref_cdf,
        max_images = args.max_images,
        canvas     = args.canvas,
        crop_pad   = args.crop_pad,
    )

    print_table(agg, has_yolo=True, has_crop=crop_model is not None, det_stats=det_stats)


if __name__ == "__main__":
    main()
