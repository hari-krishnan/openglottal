#!/usr/bin/env python3
"""Benchmark U-Net pipeline throughput (frames/s) for paper claim validation.

Validates: "502-frame GIRAFE patient video in ~11 s (~47 frames/s) on Apple M-series, MPS."
Usage:
  python scripts/benchmark_video_speed.py --frames 502 --device mps --unet-weights weights/openglottal_unet.pt
  python scripts/benchmark_video_speed.py --frames 502 --device mps --unet-weights weights/openglottal_unet.pt --yolo-weights weights/openglottal_yolo.pt  # YOLO+UNet
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch

from openglottal.utils import resolve_weights_path, unet_segment_frame


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark U-Net video processing speed.")
    p.add_argument("--frames", type=int, default=502, help="Number of frames to simulate (default: 502, GIRAFE median).")
    p.add_argument("--device", type=str, default="mps", help="Device: mps, cuda, or cpu.")
    p.add_argument("--unet-weights", type=str, default="weights/openglottal_unet.pt", help="Path to U-Net weights.")
    p.add_argument("--yolo-weights", type=str, default=None, help="If set, run YOLO+UNet (slower); else U-Net only.")
    p.add_argument("--warmup", type=int, default=20, help="Warmup frames before timing.")
    p.add_argument("--video", type=str, default=None, help="Optional: path to AVI to use real frames and include load time.")
    args = p.parse_args()

    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu") if args.device == "mps" else torch.device(args.device)
    if args.device == "mps" and device.type == "cpu":
        print("Warning: MPS not available, using CPU. Claim in paper is for MPS.")

    # Load U-Net
    from openglottal.models.unet import UNet
    unet_path = resolve_weights_path(args.unet_weights)
    if not unet_path.exists():
        print(f"Error: U-Net weights not found at {unet_path}")
        return
    model = UNet(1, 1, (32, 64, 128, 256)).to(device)
    model.load_state_dict(torch.load(str(unet_path), map_location=device, weights_only=True))
    model.eval()

    detector = None
    if args.yolo_weights:
        from openglottal.models.detector import TemporalDetector
        yolo_path = resolve_weights_path(args.yolo_weights)
        if not yolo_path.exists():
            print(f"Error: YOLO weights not found at {yolo_path}")
            return
        detector = TemporalDetector(str(yolo_path))
        print("Pipeline: YOLO+UNet (detection-gated)")
    else:
        print("Pipeline: U-Net only")

    # Get frames: from video or synthetic
    if args.video:
        from openglottal.utils import load_frames_bgr
        t0 = time.perf_counter()
        frames_bgr = load_frames_bgr(args.video)
        load_s = time.perf_counter() - t0
        n_frames = min(len(frames_bgr), args.frames)
        frames_bgr = frames_bgr[:n_frames]
        print(f"Loaded {n_frames} frames from {args.video} in {load_s:.2f} s")
    else:
        n_frames = args.frames
        frames_bgr = [np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8) for _ in range(n_frames)]
        load_s = 0.0
        print(f"Using {n_frames} synthetic 256×256 frames (no load time)")

    if detector:
        detector.reset()

    # Warmup
    for frm in frames_bgr[: args.warmup]:
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        unet_segment_frame(gray, model, device)
        if detector is not None:
            detector.detect(frm)

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    # Timed run (same loop as features.extract_features_unet)
    t0 = time.perf_counter()
    for frm_bgr in frames_bgr:
        gray_full = cv2.cvtColor(frm_bgr, cv2.COLOR_BGR2GRAY)
        mask_full = unet_segment_frame(gray_full, model, device)
        if detector is None:
            _ = float(np.sum(mask_full > 0))
        else:
            box = detector.detect(frm_bgr)
            if box is None:
                _ = 0.0
            else:
                x1, y1, x2, y2 = box
                _ = float(np.sum(mask_full[y1:y2, x1:x2] > 0))

    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    fps = n_frames / elapsed
    total_s = load_s + elapsed
    total_fps = n_frames / total_s if total_s > 0 else 0

    print(f"\nResults ({n_frames} frames, device={device}):")
    print(f"  Inference time: {elapsed:.2f} s  →  {fps:.1f} frames/s")
    if load_s > 0:
        print(f"  Video load time: {load_s:.2f} s")
        print(f"  Total (load + inference): {total_s:.2f} s  →  {total_fps:.1f} frames/s")

    # Validate paper claim: 502 frames in ~11 s → ≥ ~45.6 fps
    target_fps = 47.0
    target_s = 11.0
    ok = fps >= (502 / target_s)  # need at least 502/11 ≈ 45.6 fps
    print(f"\nPaper claim: 502 frames in ~11 s (~47 frames/s) on MPS.")
    print(f"  Inference: {fps:.1f} fps  →  502 frames in {502/fps:.1f} s  {'✓ within claim' if ok and n_frames >= 502 else '(run with --frames 502 for exact check)'}")
    if load_s > 0 and n_frames >= 502:
        total_502 = load_s + (502 / fps) if n_frames >= 502 else total_s
        print(f"  Total (load + 502-frame inference): ~{total_502:.1f} s  {'✓' if total_502 <= 12 else '—'} vs ~11 s claim")


if __name__ == "__main__":
    main()
