"""Cross-dataset evaluation on BAGLS (Benchmark for Automatic Glottis Segmentation).

BAGLS images come in many sizes (256×256, 512×256, 512×128, 352×208, …).  Each
image ``N.png`` is paired with a binary GT mask ``N_seg.png``.

Pre-processing (applied identically to image and GT mask):
    Letterbox to 256×256 — scale to fit the longest side to 256, then pad
    the shorter dimension symmetrically with zeros.  This preserves aspect
    ratio and places the glottis at the correct scale without distortion.

Pipelines evaluated:
    unet-only         — U-Net on full (letterboxed) frame, no YOLO gate
    yolo+unet         — YOLO bbox → U-Net pixels inside box  (full-frame model)
    yolo-crop+unet    — YOLO crop → resize → U-Net → project back  (crop model)

Usage
-----
python scripts/eval_bagls.py \\
    --bagls-dir      BAGLS/test \\
    --unet-weights   outputs/openglottal_unet.pt \\
    --crop-weights   outputs/openglottal_unet_crop.pt \\
    --yolo-weights   outputs/openglottal_yolo.pt \\
    --device         mps \\
    --max-images     500
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
    """
    Scale img so its longest side = ``size``, then pad the shorter side
    symmetrically with ``value`` to produce a square ``size``×``size`` array.

    Works for both 2-D (grayscale / mask) and 3-D (BGR) arrays.
    """
    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    interp = cv2.INTER_LINEAR if img.ndim == 3 else cv2.INTER_NEAREST
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    pad_h = size - new_h
    pad_w = size - new_w
    top,    bottom = pad_h // 2, pad_h - pad_h // 2
    left,   right  = pad_w // 2, pad_w - pad_w // 2

    if img.ndim == 3:
        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(value, value, value))
    else:
        return cv2.copyMakeBorder(resized, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=value)


# ── Segmentation helpers ──────────────────────────────────────────────────────

def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """(Dice, IoU) for binary uint8 masks."""
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
    # Letterbox crop to crop_size×crop_size to preserve aspect ratio
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


# ── Evaluation loop ───────────────────────────────────────────────────────────

PIPELINES = ["unet-only", "yolo+unet", "yolo-crop+unet"]


def evaluate(
    test_dir: Path,
    unet_model: torch.nn.Module,
    crop_model: torch.nn.Module | None,
    detector: TemporalDetector | None,
    device: torch.device,
    max_images: int = 0,
    canvas: int = 256,
) -> tuple[dict[str, dict], dict]:
    agg = {p: {"dice": [], "iou": [], "n_det": 0, "n_total": 0}
           for p in PIPELINES}
    # Detection stats (TP/FP/FN) using presence of GT mask inside predicted bbox
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

        img_bgr  = cv2.imread(str(img_path))
        gt_raw   = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt_raw is None:
            continue

        # Letterbox both image and GT identically
        img_lb  = letterbox(img_bgr,  canvas)
        gt_lb   = letterbox(gt_raw,   canvas)
        gray_lb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2GRAY)

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(img_files)}] ...")

        # YOLO detection on letterboxed BGR frame.
        # Reset per frame: BAGLS frames are not a temporal sequence (mixed
        # patients/clips), so the TemporalDetector's centre-clamp must not
        # carry state across unrelated frames.
        if detector is not None:
            detector.reset()
        box = detector.detect(img_lb) if detector is not None else None

        # Detection precision/recall bookkeeping: consider a frame positive
        # when the ground-truth mask has any positive pixel. A predicted
        # bbox counts as TP if the GT mask contains any positive pixel
        # inside the predicted box; otherwise it's an FP. If no box is
        # predicted but GT is positive, that's an FN.
        if detector is not None:
            gt_pos = (gt_lb > 0).any()
            if gt_pos:
                det_stats["n_pos_gt"] += 1
            if box is not None:
                x1, y1, x2, y2 = box
                # clamp box to image
                x1 = max(0, min(canvas, int(x1)))
                x2 = max(0, min(canvas, int(x2)))
                y1 = max(0, min(canvas, int(y1)))
                y2 = max(0, min(canvas, int(y2)))
                # check if any GT pixels inside predicted bbox
                if gt_lb[y1:y2, x1:x2].any():
                    det_stats["tp"] += 1
                else:
                    det_stats["fp"] += 1
            else:
                if gt_pos:
                    det_stats["fn"] += 1

        # ── unet-only ──────────────────────────────────────────────────────
        agg["unet-only"]["n_total"] += 1
        mask_u = unet_segment_frame(gray_lb, unet_model, device)
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
                mask_c = unet_on_crop(gray_lb, box, crop_model, device)
            else:
                mask_c = np.zeros_like(gray_lb)
            d, iu = frame_metrics(mask_c, gt_lb)
            agg["yolo-crop+unet"]["dice"].append(d)
            agg["yolo-crop+unet"]["iou"].append(iu)

    return agg, det_stats


# ── Pretty-print ──────────────────────────────────────────────────────────────

def print_table(agg: dict, has_yolo: bool, has_crop: bool, det_stats: dict | None = None) -> None:
    label_map = {
        "unet-only":       "U-Net only",
        "yolo+unet":       "YOLO+UNet",
        "yolo-crop+unet":  "YOLO-Crop+UNet",
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
        # unet-only always processes every frame (no gate), so recall = 1.000
        dr_str = f"{det_rec:.3f}" if pipe != "unet-only" else "1.000 *"
        print(f"  {label:<25}  {dr_str:>10}  {mean_dice:>8.3f}  "
              f"{mean_iou:>8.3f}  {d50:>9.1f}%")

    print(sep)
    print("  * U-Net only: no YOLO gate — always processes 100% of frames.")
    print(f"  Evaluated on {agg['unet-only']['n_total']} BAGLS test frames")
    print()

    # If YOLO was enabled, print detection precision/recall based on
    # presence of GT mask inside predicted bboxes (TP/FP/FN counts).
    if has_yolo and det_stats is not None:
        tp = det_stats.get("tp", 0)
        fp = det_stats.get("fp", 0)
        fn = det_stats.get("fn", 0)
        prec = float(tp) / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = float(tp) / (tp + fn) if (tp + fn) > 0 else float("nan")
        print(f"  YOLO detection — Precision: {prec:.3f}  Recall: {rec:.3f}  (TP={tp} FP={fp} FN={fn})")
        print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-dataset evaluation on BAGLS test set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bagls-dir",     required=True,
                   help="BAGLS test directory containing N.png + N_seg.png pairs.")
    p.add_argument("--unet-weights",  required=True,
                   help="Full-frame U-Net weights (unet_glottis_v2.pt).")
    p.add_argument("--crop-weights",  default=None,
                   help="Crop-mode U-Net weights (unet_glottis_crop.pt). Optional.")
    p.add_argument("--yolo-weights",  default=None,
                   help="YOLO weights. Enables yolo+unet and yolo-crop+unet pipelines.")
    p.add_argument("--device",        default="cpu")
    p.add_argument("--canvas",        type=int, default=256,
                   help="Letterbox target size (px).")
    p.add_argument("--max-images",    type=int, default=0,
                   help="Evaluate only the first N images (0 = all 3502).")
    p.add_argument("--conf",          type=float, default=0.25,
                   help="YOLO confidence threshold.")
    p.add_argument("--output-json",   default=None,
                   help="Save per-frame metrics to this JSON path.")
    p.add_argument("--no-timestamp",  action="store_true",
                   help="Save to exact --output-json path (no timestamp suffix).")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    # Resolve weight paths (support weights/ from in-progress or pre-migration runs)
    unet_path = resolve_weights_path(args.unet_weights)
    crop_path = resolve_weights_path(args.crop_weights) if args.crop_weights else None
    yolo_path = resolve_weights_path(args.yolo_weights) if args.yolo_weights else None

    # Full-frame U-Net
    unet = UNet(1, 1, (32, 64, 128, 256)).to(device)
    unet.load_state_dict(
        torch.load(unet_path, map_location=device, weights_only=True))
    unet.eval()
    print(f"Loaded full-frame U-Net : {unet_path}")

    # Crop-mode U-Net (optional)
    crop_model = None
    if crop_path is not None:
        crop_model = UNet(1, 1, (32, 64, 128, 256)).to(device)
        crop_model.load_state_dict(
            torch.load(crop_path, map_location=device, weights_only=True))
        crop_model.eval()
        print(f"Loaded crop U-Net       : {crop_path}")

    # YOLO (optional)
    detector = None
    if yolo_path is not None:
        detector = TemporalDetector(str(yolo_path), conf=args.conf)
        print(f"Loaded YOLO (conf={args.conf:.2f}): {yolo_path}")

    test_dir = Path(args.bagls_dir)
    n_avail  = sum(1 for f in test_dir.iterdir()
                   if f.suffix == ".png" and not f.name.endswith("_seg.png"))
    n_eval   = args.max_images if args.max_images else n_avail
    print(f"\nBAGLS test frames : {n_avail} available, evaluating {n_eval}\n")

    agg, det_stats = evaluate(
        test_dir   = test_dir,
        unet_model = unet,
        crop_model = crop_model,
        detector   = detector,
        device     = device,
        max_images = args.max_images,
        canvas     = args.canvas,
    )

    print_table(agg, has_yolo=detector is not None, has_crop=crop_model is not None,
                det_stats=det_stats if detector is not None else None)

    if args.output_json:
        import json
        from datetime import datetime
        serialisable = {
            pipe: {k: (v if isinstance(v, (int, float)) else [float(x) for x in v])
                   for k, v in data.items()}
            for pipe, data in agg.items()
        }
        out_path = Path(args.output_json)
        if args.no_timestamp:
            save_path = out_path
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = out_path.parent / f"{out_path.stem}_{ts}{out_path.suffix}"
        serialisable["_meta"] = {
            "bagls_dir": args.bagls_dir,
            "crop_letterbox": True,
            "written_at": datetime.now().isoformat(),
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"Raw results saved → {save_path}")


if __name__ == "__main__":
    main()
