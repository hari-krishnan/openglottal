"""Generate U-Net-only and YOLO+UNet demo videos and store them.

Runs the inference script twice:
  1. Pipeline "unet-only" -> output-base/videos_unet_only/
  2. Pipeline "unet" (YOLO+UNet) -> output-base/videos_yolo_unet/

Paths for --input, --output-base, and weights are relative to the current
working directory (or pass absolute paths). Run from the repo root so that
the openglottal package is importable when infer.py is invoked.

Usage (with GIRAFE data and weights available):
  python scripts/generate_demo_videos.py --input GIRAFE/Raw_Data

Or with custom weights and output base:
  UNET_WEIGHTS=path/to/unet.pt YOLO_WEIGHTS=path/to/yolo.pt \\
  python scripts/generate_demo_videos.py --input GIRAFE/Raw_Data --output-base results
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Package root (parent of scripts/) — used as cwd when calling infer.py so openglottal is importable
OPENGLOB_ROOT = SCRIPT_DIR.parents[1]

DEFAULT_UNET = "glottal_detector/unet_glottis_v2.pt"
DEFAULT_YOLO = "runs/detect/glottal_detector/yolov8n_girafe/weights/best.pt"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate U-Net-only and YOLO+UNet videos and store in results/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, default=Path("GIRAFE/Raw_Data"),
                   help="Directory containing patient .avi files.")
    p.add_argument("--output-base", type=Path, default=Path("results"),
                   help="Base directory for outputs; videos go into output-base/videos_*.")
    p.add_argument("--unet-weights", type=Path,
                   default=Path(os.environ.get("UNET_WEIGHTS", DEFAULT_UNET)),
                   help="Path to U-Net weights (or set UNET_WEIGHTS).")
    p.add_argument("--yolo-weights", type=Path,
                   default=Path(os.environ.get("YOLO_WEIGHTS", DEFAULT_YOLO)),
                   help="Path to YOLO weights (or set YOLO_WEIGHTS).")
    p.add_argument("--device", default="mps",
                   help="Device for inference (e.g. cpu, cuda, mps).")
    p.add_argument("--max-hold-frames", type=int, default=3,
                   help="Max frames to retain last YOLO box when YOLO misses.")
    args = p.parse_args()

    cwd = Path.cwd()
    input_dir = args.input if args.input.is_absolute() else cwd / args.input
    base = args.output_base if args.output_base.is_absolute() else cwd / args.output_base
    unet_w = args.unet_weights if args.unet_weights.is_absolute() else cwd / args.unet_weights
    yolo_w = args.yolo_weights if args.yolo_weights.is_absolute() else cwd / args.yolo_weights

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)
    if not unet_w.is_file():
        print(f"U-Net weights not found: {unet_w}", file=sys.stderr)
        sys.exit(1)
    if not yolo_w.is_file():
        print(f"YOLO weights not found: {yolo_w}", file=sys.stderr)
        sys.exit(1)

    out_unet_only = base / "videos_unet_only"
    out_yolo_unet = base / "videos_yolo_unet"
    out_unet_only.mkdir(parents=True, exist_ok=True)
    out_yolo_unet.mkdir(parents=True, exist_ok=True)

    # Run infer.py from repo root so openglottal package is importable
    infer_script = SCRIPT_DIR / "infer.py"
    cmd_base = [
        sys.executable, str(infer_script),
        "--input", str(input_dir),
        "--device", args.device,
        "--max-hold-frames", str(args.max_hold_frames),
    ]

    print("Generating U-Net-only videos ...")
    subprocess.run(
        cmd_base + ["--unet-weights", str(unet_w), "--pipeline", "unet-only", "--output-dir", str(out_unet_only)],
        cwd=str(OPENGLOB_ROOT),
        check=True,
    )
    print(f"  Stored in {out_unet_only}\n")

    print("Generating YOLO+UNet videos ...")
    subprocess.run(
        cmd_base + ["--yolo-weights", str(yolo_w), "--unet-weights", str(unet_w), "--pipeline", "unet", "--output-dir", str(out_yolo_unet)],
        cwd=str(OPENGLOB_ROOT),
        check=True,
    )
    print(f"  Stored in {out_yolo_unet}\n")

    print("Done. Videos:")
    for d in (out_unet_only, out_yolo_unet):
        n = len(list(d.glob("*_out.avi")))
        print(f"  {d}: {n} file(s)")


if __name__ == "__main__":
    main()
