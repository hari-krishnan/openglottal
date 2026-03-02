"""Glottal Area Waveform (GAW) feature analysis across all GIRAFE patients.

Runs the YOLO+UNet pipeline on every patient's full AVI video, extracts
kinematic features from the resulting area waveform, then compares Healthy
vs Pathological groups using box plots and Mann-Whitney U tests.

Patients with disorder status "Unknown" are excluded from the group comparison
but their features are still computed and saved.

Features extracted per video
----------------------------
    area_mean       — mean glottal area (px²) across open frames
    area_std        — standard deviation of area
    area_range      — max − min area (vibratory excursion)
    open_quotient   — fraction of cycle where glottis is open
    f0              — fundamental frequency (Hz), estimated from area waveform
    periodicity     — autocorrelation periodicity score  [0, 1]
    cv              — coefficient of variation (area_std / area_mean)

Usage
-----
python scripts/analyze_gaw.py \\
    --raw-data-dir  GIRAFE/Raw_Data \\
    --yolo-weights  outputs/yolo/girafe/weights/best.pt \\
    --unet-weights  outputs/openglottal_unet.pt \\
    --device        mps \\
    --output-dir    results/gaw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openglottal.models import UNet, TemporalDetector
from openglottal.features import _kinematic_features
from openglottal.utils import unet_segment_frame, _silence_stderr, resolve_weights_path

# Collapse rare conditions into broad groups
HEALTHY_LABEL = "Healthy"
PATHOLOGICAL_LABELS = {
    "Paresis", "Polyps", "Diplophonia", "Nodules",
    "Paralysis", "Cysts", "Carcinoma", "Multinodular Goiter", "Other",
}


# ── Video loading ─────────────────────────────────────────────────────────────

GIRAFE_CAPTURE_FPS = 4000.0


def load_avi(avi_path: Path) -> list[np.ndarray]:
    with _silence_stderr():
        cap = cv2.VideoCapture(str(avi_path))
        frames: list[np.ndarray] = []
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            frames.append(frm)
        cap.release()
    return frames


# ── Per-video feature extraction ──────────────────────────────────────────────

def extract_gaw_features(
    frames: list[np.ndarray],
    capture_fps: float,
    detector: TemporalDetector,
    unet_model: torch.nn.Module,
    device: torch.device,
) -> dict | None:
    """Run YOLO+UNet on all frames and return kinematic features, or None."""
    detector.reset()
    area_wave: list[float] = []

    for frm in frames:
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        box  = detector.detect(frm)
        if box is not None:
            mask = unet_segment_frame(gray, unet_model, device)
            x1, y1, x2, y2 = box
            area = float(np.sum(mask[y1:y2, x1:x2] > 0))
        else:
            area = 0.0
        area_wave.append(area)

    feats = _kinematic_features(area_wave)
    if feats is not None and feats.get("f0") is not None:
        feats["f0"] = feats["f0"] * capture_fps  # convert cycles/frame → Hz
    return feats


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GAW feature analysis: Healthy vs Pathological.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-data-dir",  required=True,
                   help="GIRAFE/Raw_Data containing patient*/patient*.avi")
    p.add_argument("--yolo-weights",  required=True)
    p.add_argument("--unet-weights",  required=True)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--output-dir",    default="results/gaw")
    p.add_argument("--fps", type=float, default=GIRAFE_CAPTURE_FPS,
                   help="Actual capture frame rate in Hz (GIRAFE HSV = 4000).")
    p.add_argument("--max-frames",    type=int, default=0,
                   help="Truncate each AVI to this many frames (0 = full video).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device     = torch.device(args.device)
    raw_dir    = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ───────────────────────────────────────────────────────────
    yolo_path = resolve_weights_path(args.yolo_weights)
    unet_path = resolve_weights_path(args.unet_weights)
    detector = TemporalDetector(str(yolo_path))

    unet = UNet(1, 1, (32, 64, 128, 256)).to(device)
    unet.load_state_dict(
        torch.load(unet_path, map_location=device, weights_only=True)
    )
    unet.eval()
    print(f"Loaded YOLO : {yolo_path}")
    print(f"Loaded UNet : {unet_path}\n")

    # ── Process each patient ──────────────────────────────────────────────────
    records: list[dict] = []

    patient_dirs = sorted(raw_dir.iterdir())
    for pdir in patient_dirs:
        avi_files = list(pdir.glob("*.avi"))
        meta_file = pdir / "metadata.json"
        if not avi_files or not meta_file.exists():
            continue

        meta   = json.load(open(meta_file))
        status = meta.get("disorder status", "Unknown")
        avi    = avi_files[0]

        print(f"  {pdir.name:20s}  [{status}]  {avi.name}")
        frames = load_avi(avi)
        if args.max_frames and len(frames) > args.max_frames:
            frames = frames[:args.max_frames]

        feats = extract_gaw_features(frames, args.fps, detector, unet, device)
        if feats is None:
            print(f"    → silent / no glottis detected, skipping.")
            continue

        record = {"patient": pdir.name, "disorder": status,
                  **{k: (float(v) if v is not None else None) for k, v in feats.items() if not k.startswith("_")}}
        records.append(record)
        f0_str = f"{feats['f0']:.1f}Hz" if feats.get("f0") is not None else "N/A"
        print(f"    area_mean={feats['area_mean']:.1f}  OQ={feats['open_quotient']:.3f}  "
              f"f0={f0_str}  periodicity={feats['periodicity']:.3f}")

    # ── Save all features ─────────────────────────────────────────────────────
    out_json = output_dir / "gaw_features.json"
    with open(out_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nAll features saved → {out_json}")

    # ── Group comparison: Healthy vs Pathological ─────────────────────────────
    healthy = [r for r in records if r["disorder"] == HEALTHY_LABEL]
    patho   = [r for r in records if r["disorder"] in PATHOLOGICAL_LABELS]
    unknown = [r for r in records if r["disorder"] not in
               {HEALTHY_LABEL} | PATHOLOGICAL_LABELS]

    print(f"\nGroup sizes: Healthy={len(healthy)}, Pathological={len(patho)}, "
          f"Unknown/excluded={len(unknown)}")

    if not healthy or not patho:
        print("Not enough data for group comparison.")
        return

    feat_cols = ["area_mean", "area_std", "area_range",
                 "open_quotient", "f0", "periodicity", "cv"]

    # Mann-Whitney U test (non-parametric, no normality assumption)
    try:
        from scipy.stats import mannwhitneyu
        has_scipy = True
    except ImportError:
        has_scipy = False
        print("scipy not installed — skipping p-values.")

    print(f"\n{'─'*72}")
    print(f"  {'Feature':<18} {'Healthy (mean±std)':>22} {'Patho (mean±std)':>22}"
          + ("  p-value" if has_scipy else ""))
    print(f"{'─'*72}")

    for feat in feat_cols:
        h_vals = [r[feat] for r in healthy if feat in r and r[feat] is not None]
        p_vals = [r[feat] for r in patho   if feat in r and r[feat] is not None]
        if not h_vals or not p_vals:
            continue
        h_str = f"{np.mean(h_vals):.3f} ± {np.std(h_vals):.3f}"
        p_str = f"{np.mean(p_vals):.3f} ± {np.std(p_vals):.3f}"
        line  = f"  {feat:<18} {h_str:>22} {p_str:>22}"
        if has_scipy:
            _, pval = mannwhitneyu(h_vals, p_vals, alternative="two-sided")
            sig = " *" if pval < 0.05 else ("  †" if pval < 0.10 else "")
            line += f"  {pval:.4f}{sig}"
        print(line)

    print(f"{'─'*72}")
    print("  * p < 0.05   † p < 0.10\n")

    # ── Box plots ─────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        axes = axes.ravel()
        for ax, feat in zip(axes, feat_cols):
            h_vals = [r[feat] for r in healthy if feat in r and r[feat] is not None]
            p_vals = [r[feat] for r in patho   if feat in r and r[feat] is not None]
            ax.boxplot([h_vals, p_vals], labels=["Healthy", "Pathological"],
                       patch_artist=True,
                       boxprops=dict(facecolor="#AED6F1"),
                       medianprops=dict(color="navy", linewidth=2))
            ax.set_title(feat, fontsize=9)
            ax.tick_params(labelsize=8)
        axes[-1].set_visible(False)  # hide spare subplot
        fig.suptitle("Glottal Area Waveform Features: Healthy vs Pathological",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plot_path = output_dir / "gaw_boxplots.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Box plots saved → {plot_path}")
    except ImportError:
        print("matplotlib not installed — skipping plots.")


if __name__ == "__main__":
    main()
