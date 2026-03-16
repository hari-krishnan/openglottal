"""Command-line interface for OpenGlottal."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="openglottal",
        description="Automated glottal area segmentation from high-speed videoendoscopy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── openglottal run ───────────────────────────────────────────────────────
    run_p = sub.add_parser("run", help="Run inference on a video file.")
    run_p.add_argument("video", help="Path to input .avi / .mp4 video.")
    run_p.add_argument("--yolo-weights", help="Path to YOLO .pt weights (required for vft, guided-vft, unet).")
    run_p.add_argument("--unet-weights", help="Path to U-Net .pt weights (required for unet, unet-only).")
    run_p.add_argument(
        "--pipeline",
        choices=["vft", "guided-vft", "unet", "unet-only"],
        default="unet",
        help="Pipeline: vft, guided-vft, unet (YOLO+UNet), or unet-only (no YOLO gate).",
    )
    run_p.add_argument("--output", "-o", default="results", help="Output directory.")
    run_p.add_argument("--device", default="cpu", help="Torch device (cpu / cuda / mps).")

    # ── openglottal displacement ─────────────────────────────────────────────
    disp_p = sub.add_parser(
        "displacement",
        help="Compute L/R displacement and kinematics from a video using U-Net (optionally YOLO-gated).",
    )
    disp_p.add_argument("video", help="Path to input .avi / .mp4 video.")
    disp_p.add_argument("--yolo-weights", help="Path to YOLO .pt weights (optional gate).")
    disp_p.add_argument("--unet-weights", required=True, help="Path to U-Net .pt weights.")
    disp_p.add_argument("--start", type=int, help="Start frame index (default 0).")
    disp_p.add_argument("--end", type=int, help="End frame index (inclusive; default = last frame).")
    disp_p.add_argument(
        "--mode",
        choices=["lr", "differential", "area"],
        default="lr",
        help="Which waveform(s) to write: lr = L/R + diff + area; differential = L-R only; area = area only.",
    )
    disp_p.add_argument(
        "--lr-position",
        type=float,
        default=0.5,
        help="L/R position along medial line (0 = AC, 1 = PC, default 0.5).",
    )
    disp_p.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="Axes learning beta for medial line smoothing (default 0.25).",
    )
    disp_p.add_argument(
        "--fps",
        type=float,
        default=4000.0,
        help="Frame rate (Hz) used for f0 and periodicity features (default 4000).",
    )
    disp_p.add_argument(
        "--output",
        "-o",
        default="results",
        help="Output directory; CSV and JSON will be written here.",
    )
    disp_p.add_argument("--device", default="cpu", help="Torch device (cpu / cuda / mps).")

    # ── openglottal build-dataset ─────────────────────────────────────────────
    bd_p = sub.add_parser("build-dataset", help="Build YOLO dataset from GIRAFE masks.")
    bd_p.add_argument("--images-dir", required=True)
    bd_p.add_argument("--labels-dir", required=True)
    bd_p.add_argument("--training-json", required=True)
    bd_p.add_argument("--output-dir", default="yolo_data")
    bd_p.add_argument("--force", action="store_true", help="Rebuild if already exists.")

    args = parser.parse_args(argv)

    if args.command == "run":
        _cmd_run(parser, args)
    elif args.command == "displacement":
        _cmd_displacement(parser, args)
    elif args.command == "build-dataset":
        _cmd_build_dataset(args)


def _cmd_run(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    import torch
    from pathlib import Path
    from .models import TemporalDetector, UNet
    from .features import (
        extract_features_detector,
        extract_features_yolo_guided_vft,
        extract_features_unet,
    )

    device = torch.device(args.device)

    if args.pipeline == "unet-only":
        if not args.unet_weights:
            parser.error("--unet-weights is required for the unet-only pipeline.")
        model = UNet(1, 1, (32, 64, 128, 256)).to(device)
        model.load_state_dict(
            torch.load(args.unet_weights, map_location=device, weights_only=True)
        )
        model.eval()
        feats = extract_features_unet(args.video, None, model, device)
    elif args.pipeline == "vft":
        if not args.yolo_weights:
            parser.error("--yolo-weights is required for the vft pipeline.")
        detector = TemporalDetector(args.yolo_weights)
        feats = extract_features_detector(args.video, detector)
    elif args.pipeline == "guided-vft":
        if not args.yolo_weights:
            parser.error("--yolo-weights is required for the guided-vft pipeline.")
        detector = TemporalDetector(args.yolo_weights)
        feats = extract_features_yolo_guided_vft(args.video, detector)
    else:  # unet (YOLO+UNet)
        if not args.yolo_weights:
            parser.error("--yolo-weights is required for the unet pipeline.")
        if not args.unet_weights:
            parser.error("--unet-weights is required for the unet pipeline.")
        detector = TemporalDetector(args.yolo_weights)
        model = UNet(1, 1, (32, 64, 128, 256)).to(device)
        model.load_state_dict(
            torch.load(args.unet_weights, map_location=device, weights_only=True)
        )
        model.eval()
        feats = extract_features_unet(args.video, detector, model, device)

    if feats is None:
        print("No glottis detected — check your weights or input video.")
        sys.exit(1)

    import json, os
    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "features.json")
    save = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in feats.items()}
    with open(out_path, "w") as f:
        json.dump(save, f, indent=2)
    print(f"Features saved to {out_path}")
    for k, v in feats.items():
        if not k.startswith("_"):
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


def _cmd_build_dataset(args: argparse.Namespace) -> None:
    from .data import build_yolo_dataset

    yaml_path = build_yolo_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        training_json=args.training_json,
        output_dir=args.output_dir,
        force=args.force,
    )
    print(f"YAML config written to {yaml_path}")


def _cmd_displacement(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """CLI entrypoint for displacement analysis and waveform export."""
    import os
    import json
    from pathlib import Path

    import torch

    from .models import TemporalDetector, UNet
    from .metadata import get_video_frame_count
    from .displacement import compute_displacement_series

    device = torch.device(args.device)

    if not args.unet_weights:
        parser.error("--unet-weights is required for the displacement command.")

    # Load models
    detector = None
    if args.yolo_weights:
        detector = TemporalDetector(args.yolo_weights)

    model = UNet(1, 1, (32, 64, 128, 256)).to(device)
    model.load_state_dict(
        torch.load(args.unet_weights, map_location=device, weights_only=True)
    )
    model.eval()

    # Frame range
    total_frames = get_video_frame_count(args.video)
    start = args.start if args.start is not None else 0
    end = args.end if args.end is not None else max(0, total_frames - 1)
    if start < 0 or end < start or start >= total_frames:
        parser.error(f"Invalid frame range: start={start}, end={end}, total_frames={total_frames}.")

    rows, feats = compute_displacement_series(
        args.video,
        start,
        end,
        model,
        device,
        detector=detector,
        beta=args.beta,
        lr_position=args.lr_position,
        fps=args.fps,
    )
    if not rows:
        print("No valid displacement samples produced — check weights or input video.")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    base = Path(args.video).stem
    csv_path = os.path.join(args.output, f"{base}_displacement_{args.mode}.csv")
    json_path = os.path.join(args.output, f"{base}_displacement_features.json")

    # Write CSV
    import csv

    lr_pos = float(args.lr_position)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# lr_position_0_1", lr_pos])
        if feats:
            writer.writerow(["# open_quotient", feats.get("open_quotient")])
            writer.writerow(["# f0_hz", feats.get("f0_hz")])
            writer.writerow(["# periodicity", feats.get("periodicity")])
            writer.writerow(["# cv", feats.get("cv")])
            writer.writerow(["# fps_used", feats.get("fps_used")])
            writer.writerow(["# mean_opening", feats.get("mean_opening")])
            writer.writerow(["# std_opening", feats.get("std_opening")])

        if args.mode == "lr":
            writer.writerow(
                ["frame", "left_disp", "right_disp", "diff_L_minus_R", "area", "lr_position_0_1"]
            )
            for frame_idx, l, r, area in rows:
                writer.writerow([frame_idx, l, r, l - r, area, lr_pos])
        elif args.mode == "differential":
            writer.writerow(["frame", "diff_L_minus_R", "lr_position_0_1"])
            for frame_idx, l, r, _ in rows:
                writer.writerow([frame_idx, l - r, lr_pos])
        else:  # "area"
            writer.writerow(["frame", "area", "lr_position_0_1"])
            for frame_idx, _, _, area in rows:
                writer.writerow([frame_idx, area, lr_pos])

    # Write features JSON (LR-based kinematics)
    feats_out = feats or {}
    with open(json_path, "w") as f:
        json.dump(feats_out, f, indent=2)

    print(f"Displacement CSV saved to {csv_path}")
    print(f"Kinematic features (LR-based) saved to {json_path}")
