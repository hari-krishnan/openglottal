"""Batch inference script for OpenGlottal.

Two input modes
---------------
AVI mode   -- directory of .avi files; each produces its own _out.avi
Image mode -- directory of images (.png / .jpg); frames read in sorted
              order and treated as a single video; produces one _out.avi

Output
------
For every input (avi or image sequence) an annotated ``*_out.avi`` is
written to ``--output-dir`` containing:
  - original frame (BGR)
  - glottis mask overlay (style controlled by --overlay-style)
  - YOLO bounding box (yellow)
  - per-frame area text

Overlay styles (--overlay-style)
---------------------------------
fill     -- semi-transparent green fill + contour outline (default)
contour  -- contour outline only, no fill
none     -- bbox and area text only; no mask drawn

A ``features.csv`` is also written with one row per input file / sequence,
containing all scalar kinematic features.

Examples
--------
# AVI directory (full-frame U-Net)
python scripts/infer.py \\
    --input  GIRAFE/Raw_Data/ \\
    --yolo-weights outputs/openglottal_yolo.pt \\
    --unet-weights outputs/openglottal_unet.pt \\
    --pipeline unet \\
    --output-dir results/

# Crop U-Net (e.g. model from Kaggle)
python scripts/infer.py \\
    --input  GIRAFE/Raw_Data/ \\
    --yolo-weights outputs/openglottal_yolo.pt \\
    --crop-weights outputs/openglottal_unet_crop.pt \\
    --pipeline yolo-crop+unet \\
    --output-dir results/

# Image sequence directory
python scripts/infer.py \\
    --input  my_frames/ \\
    --mode   images \\
    --yolo-weights outputs/openglottal_yolo.pt \\
    --unet-weights outputs/openglottal_unet.pt \\
    --pipeline unet \\
    --output-dir results/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openglottal.models import TemporalDetector, UNet
from openglottal.models.tracker import VocalFoldTracker, YOLOGuidedVFT
from openglottal.features import (
    VFT_PARAMS, VFT_INIT,
    YGVFT_PARAMS, YGVFT_INIT,
    _kinematic_features,
)
from openglottal.utils import (
    unet_segment_frame,
    letterbox_with_info,
    unletterbox,
    _silence_stderr,
    resolve_weights_path,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
FEATURE_COLS = ["area_mean", "area_std", "area_range",
                "open_quotient", "f0", "periodicity", "cv"]
GIRAFE_CAPTURE_FPS = 4000.0

# ── Overlay drawing ───────────────────────────────────────────────────────────

def _draw_overlay(
    frame_bgr: np.ndarray,
    mask: np.ndarray | None,
    box: tuple | None,
    area: float,
    overlay_style: str = "fill",   # "fill" | "contour" | "none"
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with mask + bbox + area text burned in.

    Parameters
    ----------
    overlay_style:
        ``"fill"``    — semi-transparent green fill + contour outline.
        ``"contour"`` — contour outline only, no fill.
        ``"none"``    — bbox and area label only; mask is ignored.
    """
    out = frame_bgr.copy()

    if mask is not None and mask.any() and overlay_style != "none":
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if overlay_style == "fill":
            green = np.zeros_like(out)
            green[:, :, 1] = mask
            out = cv2.addWeighted(out, 1.0, green, 0.4, 0)
        cv2.drawContours(out, cs, -1, (0, 255, 0), 1)  # outline always in fill+contour

    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 1)  # yellow bbox

    label = f"area={int(area)}"
    cv2.putText(out, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


# ── Per-frame inference (returns frames + waveform) ───────────────────────────

def _run_pipeline(
    frames_bgr: list[np.ndarray],
    detector: TemporalDetector | None,
    pipeline: str,
    model: torch.nn.Module | None,
    device: torch.device,
    overlay_style: str = "fill",
) -> tuple[list[np.ndarray], list[float]]:
    """Run the chosen pipeline on a list of BGR frames.

    Returns
    -------
    annotated_frames : list of BGR uint8 frames with overlay
    area_wave        : per-frame glottal area (pixels)
    """
    if detector is not None:
        detector.reset()
    annotated: list[np.ndarray] = []
    area_wave: list[float] = []

    if pipeline == "vft":
        tracker: VocalFoldTracker | None = None
        init_buf: list[np.ndarray] = []
        target_hw: tuple[int, int] | None = None

        for frm in frames_bgr:
            box = detector.detect(frm)
            mask = None
            area = 0.0

            if box is not None:
                x1, y1, x2, y2 = box
                crop = frm[y1:y2, x1:x2]
                if crop.size > 0:
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    if target_hw is None:
                        target_hw = (gray.shape[1], gray.shape[0])
                    elif gray.shape != (target_hw[1], target_hw[0]):
                        gray = cv2.resize(gray, target_hw, interpolation=cv2.INTER_LINEAR)

                    if tracker is None:
                        init_buf.append(gray)
                        if len(init_buf) >= VFT_INIT:
                            tracker = VocalFoldTracker(**VFT_PARAMS)
                            tracker.initialize(init_buf)
                            init_buf = []
                    else:
                        crop_mask = tracker.process_frame(gray)
                        area = float(np.sum(crop_mask > 0))
                        # Project crop mask back to full-frame coords for display
                        mask = np.zeros(frm.shape[:2], np.uint8)
                        mh = min(crop_mask.shape[0], y2 - y1)
                        mw = min(crop_mask.shape[1], x2 - x1)
                        mask[y1:y1 + mh, x1:x1 + mw] = crop_mask[:mh, :mw]

            area_wave.append(area)
            annotated.append(_draw_overlay(frm, mask, box, area, overlay_style))

    elif pipeline == "guided-vft":
        tracker_g: YOLOGuidedVFT | None = None
        init_buf_g: list[np.ndarray] = []
        first_box = None

        for frm in frames_bgr:
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            box = detector.detect(frm)
            mask = None
            area = 0.0

            if tracker_g is None:
                init_buf_g.append(gray)
                if first_box is None and box is not None:
                    first_box = box
                if len(init_buf_g) >= YGVFT_INIT:
                    tracker_g = YOLOGuidedVFT(**YGVFT_PARAMS)
                    tracker_g.initialize(init_buf_g, bbox=first_box)
                    init_buf_g = []
            else:
                mask = tracker_g.process_frame(gray, box)
                area = float(np.sum(mask > 0))

            area_wave.append(area)
            annotated.append(_draw_overlay(frm, mask, box, area, overlay_style))

    elif pipeline == "unet-only":
        # U-Net on every frame; no YOLO gate; full-frame mask and area always.
        for frm in frames_bgr:
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            mask_full = unet_segment_frame(gray, model, device)
            area = float(np.sum(mask_full > 0))
            area_wave.append(area)
            annotated.append(_draw_overlay(frm, mask_full, None, area, overlay_style))

    elif pipeline == "yolo-crop+unet":
        # YOLO bbox → crop → letterbox 256×256 → crop U-Net → unletterbox → paste
        crop_size = 256
        for frm in frames_bgr:
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            box = detector.detect(frm)
            area = 0.0
            mask_display = None
            if box is not None and model is not None:
                x1, y1, x2, y2 = box
                crop = gray[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_h, crop_w = crop.shape[:2]
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
                    mask_display = full
                    area = float(np.sum(mask_orig > 0))
            area_wave.append(area)
            annotated.append(_draw_overlay(frm, mask_display, box, area, overlay_style))

    else:  # unet (YOLO+full-frame U-Net: detection-gated; U-Net on full frame)
        for frm in frames_bgr:
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            box = detector.detect(frm)
            area = 0.0
            mask_display = None
            if box is not None:
                mask_full = unet_segment_frame(gray, model, device)
                x1, y1, x2, y2 = box
                area = float(np.sum(mask_full[y1:y2, x1:x2] > 0))
                mask_display = mask_full

            area_wave.append(area)
            annotated.append(_draw_overlay(frm, mask_display, box, area, overlay_style))

    return annotated, area_wave


# ── Video writer ──────────────────────────────────────────────────────────────

def _write_avi(path: Path, frames: list[np.ndarray], fps: float = 25.0) -> None:
    if not frames:
        return
    H, W = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    for f in frames:
        writer.write(f)
    writer.release()


# ── Source loaders ────────────────────────────────────────────────────────────

def _load_avi(avi_path: Path) -> tuple[list[np.ndarray], float]:
    """Returns (frames_bgr, container_fps)."""
    with _silence_stderr():
        cap = cv2.VideoCapture(str(avi_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = []
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            frames.append(frm)
        cap.release()
    return frames, fps


def _load_images(img_dir: Path) -> list[np.ndarray]:
    """Load images from a directory in sorted filename order."""
    paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if not paths:
        raise ValueError(f"No images found in {img_dir}")
    frames = []
    for p in paths:
        frm = cv2.imread(str(p))
        if frm is not None:
            frames.append(frm)
    return frames


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch glottal segmentation inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True,
                   help="Directory of .avi files or directory of images.")
    p.add_argument("--mode", choices=["avi", "images"], default="avi",
                   help="'avi': one video per file; 'images': whole dir = one sequence.")
    p.add_argument("--yolo-weights", default=None,
                   help="Required for pipelines: vft, guided-vft, unet. Not used for unet-only.")
    p.add_argument("--unet-weights", default=None,
                   help="Full-frame U-Net weights (for unet, unet-only).")
    p.add_argument("--crop-weights", default=None,
                   help="Crop U-Net weights (for yolo-crop+unet). Use with --pipeline yolo-crop+unet.")
    p.add_argument("--pipeline", choices=["vft", "guided-vft", "unet", "unet-only", "yolo-crop+unet"], default="unet")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--fps", type=float, default=None,
                   help="Output FPS (images mode only; AVI mode reads FPS from file).")
    p.add_argument("--capture-fps", type=float, default=GIRAFE_CAPTURE_FPS,
                   help="Actual capture frame rate in Hz for f0 conversion (GIRAFE HSV = 4000).")
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-hold-frames", type=int, default=3,
                   help="Max frames to retain last YOLO box when YOLO misses (0 = no hold).")
    p.add_argument(
        "--overlay-style",
        choices=["fill", "contour", "none"],
        default="fill",
        help=(
            "Mask overlay style in output video: "
            "'fill' = semi-transparent green fill + outline (default), "
            "'contour' = outline only, "
            "'none' = no mask drawn."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pipeline in ("unet", "unet-only") and not args.unet_weights:
        print("--unet-weights is required for the unet and unet-only pipelines.", file=sys.stderr)
        sys.exit(1)
    if args.pipeline == "yolo-crop+unet" and not args.crop_weights:
        print("--crop-weights is required for the yolo-crop+unet pipeline.", file=sys.stderr)
        sys.exit(1)
    if args.pipeline in ("vft", "guided-vft", "unet", "yolo-crop+unet") and not args.yolo_weights:
        print("--yolo-weights is required for the vft, guided-vft, unet, and yolo-crop+unet pipelines.", file=sys.stderr)
        sys.exit(1)

    # ── Load models ───────────────────────────────────────────────────────────
    device = torch.device(args.device)
    yolo_path = resolve_weights_path(args.yolo_weights) if args.yolo_weights else None
    unet_path = resolve_weights_path(args.unet_weights) if args.unet_weights else None
    crop_path = resolve_weights_path(args.crop_weights) if args.crop_weights else None
    detector: TemporalDetector | None = None
    if yolo_path is not None:
        detector = TemporalDetector(
            str(yolo_path),
            max_hold_frames=args.max_hold_frames,
        )

    model = None
    if args.pipeline == "yolo-crop+unet" and crop_path is not None:
        model = UNet(1, 1, (32, 64, 128, 256)).to(device)
        model.load_state_dict(
            torch.load(crop_path, map_location=device, weights_only=True)
        )
        model.eval()
        print(f"Loaded crop U-Net: {crop_path}")
    elif unet_path is not None:
        model = UNet(1, 1, (32, 64, 128, 256)).to(device)
        model.load_state_dict(
            torch.load(unet_path, map_location=device, weights_only=True)
        )
        model.eval()

    # ── Collect jobs: list of (stem, frames_bgr, fps) ────────────────────────
    jobs: list[tuple[str, list[np.ndarray], float]] = []

    if args.mode == "avi":
        avi_files = sorted(input_dir.glob("*.avi"))
        if not avi_files:
            avi_files = sorted(input_dir.rglob("*.avi"))
        if not avi_files:
            print(f"No .avi files found in {input_dir}", file=sys.stderr)
            sys.exit(1)
        for avi in avi_files:
            frames, fps = _load_avi(avi)
            jobs.append((avi.stem, frames, fps))
    else:  # images
        frames = _load_images(input_dir)
        fps = args.fps or 25.0
        jobs.append((input_dir.name, frames, fps))

    # ── Process each job ─────────────────────────────────────────────────────
    csv_path = output_dir / "features.csv"
    csv_file = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=["source"] + FEATURE_COLS)
    writer.writeheader()

    for stem, frames, fps in jobs:
        print(f"\n[{stem}]  {len(frames)} frames @ {fps:.1f} fps")

        if not frames:
            print("  WARNING: no frames loaded, skipping.")
            continue

        annotated, area_wave = _run_pipeline(
            frames, detector, args.pipeline, model, device, args.overlay_style
        )

        out_avi = output_dir / f"{stem}_out.avi"
        _write_avi(out_avi, annotated, fps)
        print(f"  Wrote {out_avi}")

        feats = _kinematic_features(area_wave)
        if feats is None:
            print("  WARNING: silent waveform — no glottis detected.")
            writer.writerow({"source": stem, **{c: "" for c in FEATURE_COLS}})
        else:
            feats["f0"] = feats["f0"] * args.capture_fps  # cycles/frame → Hz
            row = {"source": stem}
            for col in FEATURE_COLS:
                row[col] = f"{feats[col]:.4f}" if isinstance(feats[col], float) else feats[col]
            writer.writerow(row)
            for col in FEATURE_COLS:
                v = feats[col]
                print(f"  {col}: {v:.4f}" if isinstance(v, float) else f"  {col}: {v}")

    csv_file.close()
    print(f"\nFeatures saved to {csv_path}")


if __name__ == "__main__":
    main()
