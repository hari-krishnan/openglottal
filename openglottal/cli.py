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
