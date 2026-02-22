"""Build a montage image from annotated frames for a given patient.

Reads an existing annotated AVI (e.g. patient66_out.avi from infer.py),
samples frames evenly, and arranges them in a grid. Run infer.py first to
create the annotated video if needed.

Usage
-----
python scripts/make_montage.py --input results/videos_yolo_unet/patient66_out.avi \\
    --output openglottal/paper/patient66_montage.png --num-frames 12 --cols 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

def load_annotated_frames_from_avi(avi_path: Path) -> list[np.ndarray]:
    """Load all frames from an existing annotated AVI."""
    frames = []
    cap = cv2.VideoCapture(str(avi_path))
    while True:
        ret, frm = cap.read()
        if not ret:
            break
        frames.append(frm)
    cap.release()
    return frames


def sample_frames(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    """Return n frames evenly spaced over the sequence."""
    if not frames or n <= 0:
        return []
    if n >= len(frames):
        return list(frames)
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


def make_montage(
    frames: list[np.ndarray],
    cols: int = 4,
    border: int = 4,
    border_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Arrange frames in a grid. Frames are resized to the same size (min H, min W)."""
    if not frames:
        raise ValueError("No frames to montage")
    n = len(frames)
    rows = (n + cols - 1) // cols
    h, w = frames[0].shape[:2]
    for f in frames[1:]:
        h = min(h, f.shape[0])
        w = min(w, f.shape[1])
    # Resize all to (w, h)
    resized = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames]
    # Pad to full grid
    while len(resized) < rows * cols:
        resized.append(np.full((h, w, 3), border_color, dtype=np.uint8))
    # Build grid
    cell_h = h + 2 * border
    cell_w = w + 2 * border
    out_h = rows * cell_h
    out_w = cols * cell_w
    out = np.full((out_h, out_w, 3), border_color, dtype=np.uint8)
    for idx, f in enumerate(resized[: rows * cols]):
        r, c = idx // cols, idx % cols
        y1 = r * cell_h + border
        x1 = c * cell_w + border
        out[y1 : y1 + h, x1 : x1 + w] = f
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a montage of annotated frames for a patient.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", type=Path, required=True,
                   help="Path to annotated _out.avi, or to raw patient .avi to run inference.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output montage image path. Default: <patient>_montage.png in current dir.")
    p.add_argument("--num-frames", type=int, default=12,
                   help="Number of frames to include in the montage.")
    p.add_argument("--cols", type=int, default=4,
                   help="Number of columns in the grid.")
    args = p.parse_args()

    inp = args.input
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    frames = load_annotated_frames_from_avi(inp)

    if not frames:
        print("No frames to montage.", file=sys.stderr)
        sys.exit(1)

    sampled = sample_frames(frames, args.num_frames)
    montage = make_montage(sampled, cols=args.cols)
    out = args.output
    if out is None:
        stem = inp.stem.replace("_out", "")
        out = Path.cwd() / f"{stem}_montage.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), montage)
    print(f"Montage saved: {out} ({len(sampled)} frames, {montage.shape[1]}x{montage.shape[0]} px)")


if __name__ == "__main__":
    main()
