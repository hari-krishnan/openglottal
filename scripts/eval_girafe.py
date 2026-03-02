"""Per-patient evaluation on the GIRAFE test split.

Reproduces the format of Table 3 in Andrade-Miranda et al. (Data in Brief, 2025)
so results can be directly compared against their reported baselines:

    InP        — traditional inpainting + active contours
    Loh        — semi-automatic GlottisExplorer
    UNet       — their U-Net trained from scratch
    SwinUNetV2 — their transformer baseline

This script evaluates five pipelines:
    unet-only   — U-Net on full frame, no YOLO gate
    yolo+otsu   — YOLO bbox → Otsu threshold within box   (training-free baseline)
    yolo+unet   — YOLO bbox → U-Net pixels inside box
    yolo+motion — YOLO bbox → YOLOGuidedVFT (temporal)

Metrics per frame:
    Det.Recall  — fraction of frames where the detector fired (YOLO found a box)
    Dice        — 2·TP / (2·TP + FP + FN)
    IoU         — TP / (TP + FP + FN)
    Dice≥0.5    — fraction of frames with Dice ≥ 0.5

Example
-------
python scripts/eval_girafe.py \\
    --images-dir  GIRAFE/Training/imagesTr \\
    --labels-dir  GIRAFE/Training/labelsTr \\
    --training-json GIRAFE/Training/training.json \\
    --unet-weights outputs/openglottal_unet.pt \\
    --yolo-weights outputs/openglottal_yolo.pt \\
    --device mps
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openglottal.models import UNet, TemporalDetector
from openglottal.models.tracker import YOLOGuidedVFT
from openglottal.features import YGVFT_PARAMS, YGVFT_INIT
from openglottal.utils import unet_segment_frame, letterbox_with_info, unletterbox, resolve_weights_path

# ── GIRAFE published baselines (mean across 4 test patients) ──────────────────
GIRAFE_BASELINE = [
    ("InP (GIRAFE paper)",   None,  0.713, None, None),
    ("U-Net (GIRAFE paper)", None,  0.643, None, None),
    ("SwinUNetV2 (paper)",   None,  0.621, None, None),
]

TEST_PATIENTS = ["patient57A3", "patient61", "patient63", "patient64"]
OUR_PIPELINES = ["unet-only", "yolo+otsu", "yolo+unet", "yolo-crop+unet", "yolo+motion"]


def load_patient_to_pathology(raw_data_dir: Path) -> dict[str, str]:
    """Load patient ID -> disorder status from GIRAFE Raw_Data patient*/metadata.json."""
    out: dict[str, str] = {}
    for pdir in sorted(raw_data_dir.iterdir()):
        if not pdir.is_dir():
            continue
        meta_file = pdir / "metadata.json"
        if not meta_file.exists():
            continue
        meta = json.load(open(meta_file))
        out[pdir.name] = meta.get("disorder status", "Unknown")
    return out


def dice_by_pathology(
    patient_dice: dict[str, dict[str, list[float]]],
    patient_to_pathology: dict[str, str],
) -> dict[str, dict[str, list[float]]]:
    """Group per-patient dice lists by pathology label. Returns pathology -> pipe -> list of dices."""
    by_patho: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for patient, pipe_dice in patient_dice.items():
        patho = patient_to_pathology.get(patient, "Unknown")
        for pipe, dices in pipe_dice.items():
            by_patho[patho][pipe].extend(dices)
    return dict(by_patho)


def print_per_pathology_dice(dice_by_pathology: dict[str, dict[str, list[float]]]) -> None:
    """Print mean Dice per pathology per pipeline."""
    label_map = {
        "unet-only": "U-Net only", "yolo+otsu": "YOLO+OTSU", "yolo+unet": "YOLO+UNet",
        "yolo-crop+unet": "YOLO-Crop+UNet", "yolo+motion": "YOLO+Motion",
    }
    pathos = sorted(dice_by_pathology.keys())
    pipes = [p for p in OUR_PIPELINES]
    print("\nDice by pathology (mean over frames):")
    print("  " + "".join(f"{label_map.get(p, p):>12}" for p in pipes))
    for patho in pathos:
        row = []
        for pipe in pipes:
            dices = dice_by_pathology[patho].get(pipe, [])
            row.append(f"{np.mean(dices):.3f}" if dices else "  n/a  ")
        print(f"  {patho:<12}" + "".join(f"{r:>12}" for r in row))

UNET_CROP_SIZE = 256  # resize YOLO crop to this before U-Net inference


# ── Segmentation helpers ──────────────────────────────────────────────────────

def frame_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Return (Dice, IoU) for binary uint8 masks."""
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
    unet_model: torch.nn.Module,
    device: torch.device,
    crop_size: int = UNET_CROP_SIZE,
) -> np.ndarray:
    """
    Crop YOLO bbox from a grayscale frame, letterbox to ``crop_size``×``crop_size``
    (preserving aspect ratio), run U-Net, then project the output mask back to
    full-frame coordinates.

    This gives U-Net higher effective resolution on the glottis region.
    NOTE: the existing U-Net was trained on full frames, so performance reflects
    domain shift.  Re-train with ``train_unet_crop.py`` to close the gap.
    """
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros_like(gray)
    crop_h, crop_w = crop.shape[:2]
    # Letterbox crop to preserve aspect ratio
    boxed, pad_t, pad_l, content_h, content_w = letterbox_with_info(
        crop, crop_size, value=0
    )
    mask_crop_sz = unet_segment_frame(boxed, unet_model, device)
    mask_orig = unletterbox(
        mask_crop_sz, pad_t, pad_l, content_h, content_w,
        crop_h, crop_w, interp=cv2.INTER_NEAREST,
    )
    full_mask = np.zeros_like(gray)
    full_mask[y1:y2, x1:x2] = mask_orig
    return full_mask


def otsu_in_box(gray: np.ndarray, box: tuple) -> np.ndarray:
    """Otsu threshold (inverted — glottis is dark) within YOLO bbox."""
    x1, y1, x2, y2 = box
    mask = np.zeros_like(gray)
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return mask
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask[y1:y2, x1:x2] = thresh
    return mask


# ── Per-patient motion pipeline ───────────────────────────────────────────────

def evaluate_patient_motion(
    fnames: list[str],
    images_dir: Path,
    labels_dir: Path,
    detector: TemporalDetector,
) -> tuple[dict[str, list[float]], int]:
    """Run YOLOGuidedVFT on a patient's sorted frame sequence.

    Returns (metric_lists, n_detected) where n_detected counts frames where
    the detector fired (including init frames).
    """
    detector.reset()
    tracker: YOLOGuidedVFT | None = None
    init_buf: list[np.ndarray] = []
    first_box = None
    per_frame: dict[str, list[float]] = defaultdict(list)
    n_detected = 0

    for fname in sorted(fnames):
        img_bgr = cv2.imread(str(images_dir / fname))
        gt_mask = cv2.imread(str(labels_dir / fname), cv2.IMREAD_GRAYSCALE)
        if img_bgr is None or gt_mask is None:
            continue

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        box  = detector.detect(img_bgr)
        if box is not None:
            n_detected += 1

        if tracker is None:
            init_buf.append(gray)
            if first_box is None and box is not None:
                first_box = box
            if len(init_buf) >= YGVFT_INIT:
                tracker = YOLOGuidedVFT(**YGVFT_PARAMS)
                tracker.initialize(init_buf, bbox=first_box)
                init_buf = []
            continue  # init frames excluded from metrics

        mask = tracker.process_frame(gray, box)
        dice, iou = frame_metrics(mask, gt_mask)
        per_frame["dice"].append(dice)
        per_frame["iou"].append(iou)

    return per_frame, n_detected


# ── Main evaluation loop ──────────────────────────────────────────────────────

def evaluate(
    test_fnames: list[str],
    images_dir: Path,
    labels_dir: Path,
    unet_model: torch.nn.Module,
    device: torch.device,
    detector: TemporalDetector | None,
) -> tuple[dict[str, dict], dict[str, dict[str, list[float]]]]:
    """Return (agg, patient_dice). agg: pipeline → {dice, iou, n_detected, n_total}; patient_dice: patient → pipeline → list of dices."""
    agg: dict[str, dict] = {p: {"dice": [], "iou": [], "n_det": 0, "n_total": 0}
                             for p in OUR_PIPELINES}
    patient_dice: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    by_patient: dict[str, list[str]] = defaultdict(list)
    for fname in sorted(test_fnames):
        patient = "_".join(fname.split("_")[:-1])
        by_patient[patient].append(fname)

    for patient, fnames in by_patient.items():
        print(f"  Processing {patient} ({len(fnames)} frames)...")

        if detector is not None:
            detector.reset()

        for fname in fnames:
            img_bgr = cv2.imread(str(images_dir / fname))
            gt_mask = cv2.imread(str(labels_dir / fname), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or gt_mask is None:
                print(f"    WARNING: could not read {fname}, skipping.")
                continue

            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            if detector is not None:
                box = detector.detect(img_bgr)
            else:
                box = None

            # ── UNet-only (no YOLO gate) ─────────────────────────────────────
            agg["unet-only"]["n_total"] += 1
            mask_unet_full = unet_segment_frame(gray, unet_model, device)
            d, i = frame_metrics(mask_unet_full, gt_mask)
            agg["unet-only"]["dice"].append(d)
            agg["unet-only"]["iou"].append(i)
            patient_dice[patient]["unet-only"].append(d)

            # ── YOLO+OTSU ─────────────────────────────────────────────────────
            agg["yolo+otsu"]["n_total"] += 1
            if box is not None:
                agg["yolo+otsu"]["n_det"] += 1
                mask_otsu = otsu_in_box(gray, box)
            else:
                mask_otsu = np.zeros_like(gray)
            d, i = frame_metrics(mask_otsu, gt_mask)
            agg["yolo+otsu"]["dice"].append(d)
            agg["yolo+otsu"]["iou"].append(i)
            patient_dice[patient]["yolo+otsu"].append(d)

            # ── YOLO+UNet ─────────────────────────────────────────────────────
            agg["yolo+unet"]["n_total"] += 1
            mask_unet = unet_segment_frame(gray, unet_model, device)
            if box is not None:
                agg["yolo+unet"]["n_det"] += 1
                x1, y1, x2, y2 = box
                mask_yu = np.zeros_like(mask_unet)
                mask_yu[y1:y2, x1:x2] = mask_unet[y1:y2, x1:x2]
            else:
                mask_yu = np.zeros_like(mask_unet)
            d, i = frame_metrics(mask_yu, gt_mask)
            agg["yolo+unet"]["dice"].append(d)
            agg["yolo+unet"]["iou"].append(i)
            patient_dice[patient]["yolo+unet"].append(d)

            # ── YOLO-Crop+UNet ────────────────────────────────────────────────
            agg["yolo-crop+unet"]["n_total"] += 1
            if box is not None:
                agg["yolo-crop+unet"]["n_det"] += 1
                mask_crop = unet_on_crop(gray, box, unet_model, device)
            else:
                mask_crop = np.zeros_like(gray)
            d, i = frame_metrics(mask_crop, gt_mask)
            agg["yolo-crop+unet"]["dice"].append(d)
            agg["yolo-crop+unet"]["iou"].append(i)
            patient_dice[patient]["yolo-crop+unet"].append(d)

        # ── YOLO+motion (temporal, per patient) ───────────────────────────────
        if detector is not None:
            motion_data, n_det_motion = evaluate_patient_motion(
                fnames, images_dir, labels_dir, detector
            )
            n_eval = len(motion_data.get("dice", []))
            print(f"    yolo+motion: {n_eval}/{len(fnames)} frames evaluated "
                  f"(first {YGVFT_INIT} used for init)")
            agg["yolo+motion"]["dice"].extend(motion_data.get("dice", []))
            agg["yolo+motion"]["iou"].extend(motion_data.get("iou", []))
            agg["yolo+motion"]["n_det"] += n_det_motion
            agg["yolo+motion"]["n_total"] += len(fnames)
            patient_dice[patient]["yolo+motion"].extend(motion_data.get("dice", []))

    return agg, dict(patient_dice)


# ── Pretty-print table ────────────────────────────────────────────────────────

def print_table(agg: dict, has_yolo: bool) -> None:
    pipes = ["unet-only"]
    if has_yolo:
        pipes += [p for p in OUR_PIPELINES if p != "unet-only"]
    label_map = {
        "unet-only":      "U-Net only",
        "yolo+otsu":      "YOLO+OTSU",
        "yolo+unet":      "YOLO+UNet",
        "yolo-crop+unet": "YOLO-Crop+UNet *",
        "yolo+motion":    "YOLO+Motion",
    }

    sep = "─" * 76
    print(f"\n{sep}")
    print(f"  {'Method':<25}  {'Det.Recall':>10}  {'Dice':>8}  {'IoU':>8}  {'Dice≥0.5':>10}")
    print(sep)

    # Published baselines (no detection recall or Dice≥0.5 available)
    for label, det_rec, dice, iou, d50 in GIRAFE_BASELINE:
        dr  = f"{det_rec:.3f}" if det_rec is not None else "  n/a  "
        d   = f"{dice:.3f}"   if dice    is not None else "  n/a  "
        io  = f"{iou:.3f}"    if iou     is not None else "  n/a  "
        d5  = f"{d50:.1f}%"   if d50     is not None else "  n/a  "
        print(f"  {label:<25}  {dr:>10}  {d:>8}  {io:>8}  {d5:>10}")

    print("  " + "· " * 37)

    for pipe in pipes:
        data = agg[pipe]
        dices = data["dice"]
        ious  = data["iou"]
        n_det = data["n_det"]
        n_tot = data["n_total"]
        det_rec = n_det / n_tot if n_tot > 0 else float("nan")
        mean_dice = np.mean(dices) if dices else float("nan")
        mean_iou  = np.mean(ious)  if ious  else float("nan")
        d50 = np.mean([d >= 0.5 for d in dices]) * 100 if dices else float("nan")
        label = label_map[pipe]
        dr_str = "1.000 *" if pipe == "unet-only" else f"{det_rec:.3f}"
        print(f"  {label:<25}  {dr_str:>10}  {mean_dice:>8.3f}  "
              f"{mean_iou:>8.3f}  {d50:>9.1f}%")

    print(sep)
    print("  * U-Net only: no YOLO gate — always processes 100% of frames.")
    print("  * YOLO-Crop+UNet: use --unet-weights unet_glottis_crop.pt (crop-trained).")
    print("    With full-frame weights the domain shift collapses this pipeline to ~0.")
    print("    Run scripts/train_unet_crop.py once to produce the crop-trained weights.")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GIRAFE test evaluation with Det.Recall, Dice, IoU, Dice≥0.5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--images-dir",    required=True)
    p.add_argument("--labels-dir",    required=True)
    p.add_argument("--training-json", required=True)
    p.add_argument("--raw-data-dir",  default=None,
                   help="GIRAFE/Raw_Data dir for per-pathology Dice (patient*/metadata.json).")
    p.add_argument("--unet-weights",  required=True)
    p.add_argument("--yolo-weights",  default=None,
                   help="Enables YOLO+OTSU, YOLO+UNet, YOLO+Motion pipelines.")
    p.add_argument("--max-hold-frames", type=int, default=3,
                   help="Max frames to hold YOLO box when YOLO misses; after this detection is zeroed. Use a large value (e.g. 999999) for hold-forever (original) behavior.")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-json", default=None,
                   help="Save raw per-frame results as JSON.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # Resolve weight paths (support weights/ from in-progress or pre-migration runs)
    unet_path = resolve_weights_path(args.unet_weights)
    yolo_path = resolve_weights_path(args.yolo_weights) if args.yolo_weights else None

    unet = UNet(1, 1, (32, 64, 128, 256)).to(device)
    unet.load_state_dict(
        torch.load(unet_path, map_location=device, weights_only=True)
    )
    unet.eval()
    print(f"Loaded U-Net  : {unet_path}")

    detector = None
    if yolo_path is not None:
        detector = TemporalDetector(
            str(yolo_path),
            max_hold_frames=args.max_hold_frames,
        )
        print(f"Loaded YOLO   : {yolo_path}  (max_hold_frames={args.max_hold_frames})")

    splits = json.load(open(args.training_json))
    test_fnames = splits["test"]
    print(f"Test frames   : {len(test_fnames)} across {len(TEST_PATIENTS)} patients\n")

    agg, patient_dice = evaluate(
        test_fnames=test_fnames,
        images_dir=Path(args.images_dir),
        labels_dir=Path(args.labels_dir),
        unet_model=unet,
        device=device,
        detector=detector,
    )

    print_table(agg, has_yolo=detector is not None)

    raw_dir = Path(args.raw_data_dir) if args.raw_data_dir else None
    by_patho = None
    if raw_dir is not None:
        if raw_dir.is_dir():
            patient_to_pathology = load_patient_to_pathology(raw_dir)
            by_patho = dice_by_pathology(patient_dice, patient_to_pathology)
            print_per_pathology_dice(by_patho)
        else:
            print(f"  --raw-data-dir not found: {raw_dir}", file=sys.stderr)

    if args.output_json:
        serialisable = {
            pipe: {k: (v if isinstance(v, (int, float)) else [float(x) for x in v])
                   for k, v in data.items()}
            for pipe, data in agg.items()
        }
        out_data = {"aggregate": serialisable}
        if by_patho is not None:
            out_data["dice_by_pathology"] = {
                patho: {pipe: dices for pipe, dices in pipe_dice.items()}
                for patho, pipe_dice in by_patho.items()
            }
        with open(args.output_json, "w") as f:
            json.dump(out_data, f, indent=2)
        print(f"Raw results saved to {args.output_json}")


if __name__ == "__main__":
    main()
