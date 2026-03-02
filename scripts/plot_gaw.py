"""Plot Glottal Area Waveform (GAW) for all patients in a folder.

Runs the YOLO+UNet pipeline on each patient's AVI, extracts the area-vs-frame
waveform, and saves one plot per patient to an output directory.

Usage
-----
python scripts/plot_gaw.py \\
    --raw-data-dir  GIRAFE/Raw_Data \\
    --yolo-weights  weights/openglottal_yolo.pt \\
    --unet-weights  weights/openglottal_unet.pt \\
    --output-dir    results/gaw_plots \\
    --device        mps
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


def extract_area_waveform(
    frames: list[np.ndarray],
    detector: TemporalDetector,
    unet_model: torch.nn.Module,
    device: torch.device,
) -> list[float]:
    """Run YOLO+UNet on all frames; return list of glottal area per frame."""
    detector.reset()
    area_wave: list[float] = []
    for frm in frames:
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        box = detector.detect(frm)
        if box is not None:
            mask = unet_segment_frame(gray, unet_model, device)
            x1, y1, x2, y2 = box
            area = float(np.sum(mask[y1:y2, x1:x2] > 0))
        else:
            area = 0.0
        area_wave.append(area)
    return area_wave


def plot_gaw(
    area_wave: list[float],
    patient_id: str,
    disorder: str,
    fps: float,
    out_path: Path,
    title: str | None = None,
) -> None:
    """Plot GAW (time vs area) and save to out_path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames = len(area_wave)
    time_s = np.arange(n_frames) / fps if fps > 0 else np.arange(n_frames)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(time_s, area_wave, alpha=0.4)
    ax.plot(time_s, area_wave, linewidth=0.8, color="C0")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Glottal area (px²)")
    if title is None:
        title = f"{patient_id}  [{disorder}]"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    feats = _kinematic_features(area_wave)
    if feats is not None:
        f0_hz = (feats["f0"] * fps) if feats.get("f0") is not None else None
        text = f"mean={feats['area_mean']:.0f}  range={feats['area_range']:.0f}  OQ={feats['open_quotient']:.2f}"
        if f0_hz is not None:
            text += f"  f0={f0_hz:.0f} Hz"
        text += f"  cv={feats['cv']:.2f}"
        ax.text(0.99, 0.97, text, transform=ax.transAxes, fontsize=8,
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot GAW for all patients in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw-data-dir", required=True,
                   help="Folder containing patient*/ with *.avi and metadata.json")
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--unet-weights", required=True)
    p.add_argument("--output-dir", default="results/gaw_plots",
                   help="Folder to save one PNG per patient")
    p.add_argument("--device", default="cpu")
    p.add_argument("--fps", type=float, default=GIRAFE_CAPTURE_FPS,
                   help="Capture frame rate (Hz) for time axis")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Truncate each AVI to this many frames (0 = full)")
    p.add_argument("--resume", action="store_true",
                   help="Skip patients that already have a plot in output-dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    raw_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yolo_path = resolve_weights_path(args.yolo_weights)
    unet_path = resolve_weights_path(args.unet_weights)
    detector = TemporalDetector(str(yolo_path))
    unet = UNet(1, 1, (32, 64, 128, 256)).to(device)
    unet.load_state_dict(
        torch.load(unet_path, map_location=device, weights_only=True)
    )
    unet.eval()

    print(f"YOLO:  {yolo_path}")
    print(f"UNet:  {unet_path}")
    print(f"Out:   {output_dir}\n")

    for pdir in sorted(raw_dir.iterdir()):
        if not pdir.is_dir():
            continue
        avi_files = list(pdir.glob("*.avi"))
        meta_file = pdir / "metadata.json"
        if not avi_files:
            continue
        meta = json.load(open(meta_file)) if meta_file.exists() else {}
        disorder = meta.get("disorder status", "Unknown")
        avi = avi_files[0]

        out_path = output_dir / f"gaw_{pdir.name}.png"
        if args.resume and out_path.exists():
            print(f"  {pdir.name}  [{disorder}]  skip (exists)")
            continue
        print(f"  {pdir.name}  [{disorder}]  {avi.name}")
        frames = load_avi(avi)
        if args.max_frames and len(frames) > args.max_frames:
            frames = frames[: args.max_frames]

        area_wave = extract_area_waveform(frames, detector, unet, device)
        if not area_wave or max(area_wave) == 0:
            print("    → no glottis / silent, skip")
            continue

        plot_gaw(
            area_wave,
            pdir.name,
            disorder,
            args.fps,
            out_path,
        )
        print(f"    → {out_path}")

    print(f"\nPlots saved in {output_dir}")


if __name__ == "__main__":
    main()
