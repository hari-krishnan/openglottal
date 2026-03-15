"""Sweep crop_pad parameter (0-50 px) on BAGLS test set using GIRAFE models.

This evaluates the yolo-crop+unet pipeline with varying padding around
YOLO-detected bounding boxes to find the optimal padding for cross-dataset
generalization (GIRAFE training → BAGLS evaluation).

Usage
-----
python scripts/sweep_crop_pad_girafe.py \\
    --bagls-dir      /path/to/BAGLS/test \\
    --device         mps \\
    --max-images     500 \\
    --output-file    results/crop_pad_sweep_girafe.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import numpy as np


def run_evaluation(
    bagls_dir: str,
    unet_weights: str,
    crop_weights: str,
    yolo_weights: str,
    crop_pad: int,
    device: str,
    max_images: int,
    canvas: int = 256,
) -> dict:
    """Run eval_bagls.py with specified parameters and return aggregated metrics."""
    cmd = [
        sys.executable, "scripts/eval_bagls.py",
        "--bagls-dir", bagls_dir,
        "--unet-weights", unet_weights,
        "--crop-weights", crop_weights,
        "--yolo-weights", yolo_weights,
        "--crop-pad", str(crop_pad),
        "--device", device,
        "--canvas", str(canvas),
        "--max-images", str(max_images),
    ]
    
    print(f"  Running with crop_pad={crop_pad:2d}...", end=" ", flush=True)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"ERROR")
            print(result.stderr)
            return None
        
        # Parse stdout to extract metrics from the table
        output = result.stdout
        lines = output.split("\n")
        
        # Find the YOLO-Crop+UNet line in the table
        metrics = None
        for line in lines:
            if "YOLO-Crop+UNet" in line:
                parts = line.split()
                # Format: Method Det.Recall Dice IoU Dice≥0.5
                # Example: YOLO-Crop+UNet  0.950  0.847  0.742  94.2%
                try:
                    det_recall = float(parts[1])
                    dice = float(parts[2])
                    iou = float(parts[3])
                    dice_50_pct = float(parts[4].rstrip('%'))
                    metrics = {
                        "crop_pad": crop_pad,
                        "det_recall": det_recall,
                        "dice": dice,
                        "iou": iou,
                        "dice_50_pct": dice_50_pct,
                    }
                    break
                except (ValueError, IndexError):
                    pass
        
        if metrics is None:
            print("PARSE ERROR")
            return None
        
        print(f"✓ DICE={metrics['dice']:.3f}")
        return metrics
        
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        return None
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sweep crop_pad parameter on BAGLS using GIRAFE models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--bagls-dir", required=True,
                   help="BAGLS test directory (N.png + N_seg.png pairs).")
    p.add_argument("--device", default="cpu",
                   help="Device: cpu, cuda, mps, etc.")
    p.add_argument("--max-images", type=int, default=0,
                   help="Max images to evaluate (0 = all).")
    p.add_argument("--canvas", type=int, default=256,
                   help="Letterbox target size (px).")
    p.add_argument("--min-pad", type=int, default=0,
                   help="Minimum crop_pad value (px).")
    p.add_argument("--max-pad", type=int, default=10,
                   help="Maximum crop_pad value (px).")
    p.add_argument("--step", type=int, default=1,
                   help="Step size between crop_pad values (px).")
    p.add_argument("--output-file", default="results/crop_pad_sweep_girafe.json",
                   help="Save sweep results to JSON file.")
    args = p.parse_args()

    # GIRAFE model paths
    repo_root = Path(__file__).resolve().parents[1]
    unet_weights = repo_root / "weights" / "og_girafe_unet_full.pt"
    crop_weights = repo_root / "weights" / "og_girafe_unet_crop.pt"
    yolo_weights = repo_root / "weights" / "og_girafe_yolo.pt"

    # Verify models exist
    for path, name in [
        (unet_weights, "Full-frame U-Net"),
        (crop_weights, "Crop U-Net"),
        (yolo_weights, "YOLO"),
    ]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            sys.exit(1)

    print(f"\n{'='*70}")
    print(f"GIRAFE Models → BAGLS Evaluation: crop_pad Sweep")
    print(f"{'='*70}")
    print(f"  Full-frame U-Net : {unet_weights.name}")
    print(f"  Crop U-Net       : {crop_weights.name}")
    print(f"  YOLO             : {yolo_weights.name}")
    print(f"  BAGLS dir        : {args.bagls_dir}")
    print(f"  Device           : {args.device}")
    print(f"  Canvas           : {args.canvas}×{args.canvas}")
    print(f"  Max images       : {args.max_images if args.max_images else 'all'}")
    print(f"\n  Sweeping crop_pad: {args.min_pad} to {args.max_pad} px (step={args.step})")
    print(f"{'='*70}\n")

    results = []
    pad_values = range(args.min_pad, args.max_pad + 1, args.step)
    
    for i, crop_pad in enumerate(pad_values, 1):
        print(f"[{i}/{len(list(pad_values))}]", end=" ")
        metrics = run_evaluation(
            bagls_dir=args.bagls_dir,
            unet_weights=str(unet_weights),
            crop_weights=str(crop_weights),
            yolo_weights=str(yolo_weights),
            crop_pad=crop_pad,
            device=args.device,
            max_images=args.max_images,
            canvas=args.canvas,
        )
        if metrics:
            results.append(metrics)

    if not results:
        print("\nERROR: No successful evaluations!")
        sys.exit(1)

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Crop Pad':>10}  {'DetRecall':>10}  {'DICE':>8}  {'IoU':>8}  {'DICE≥0.5':>10}")
    print(f"{'-'*70}")
    
    best_dice = max(results, key=lambda x: x["dice"])
    best_iou = max(results, key=lambda x: x["iou"])
    
    for res in results:
        pad = res["crop_pad"]
        dice = res["dice"]
        iou = res["iou"]
        det = res["det_recall"]
        dice_50 = res["dice_50_pct"]
        
        marker = " ← BEST DICE" if res == best_dice else ""
        print(f"  {pad:8d}px  {det:10.3f}  {dice:8.3f}  {iou:8.3f}  {dice_50:9.1f}%{marker}")
    
    print(f"{'-'*70}")
    print(f"  Best DICE: {best_dice['dice']:.3f} at crop_pad={best_dice['crop_pad']}px")
    print(f"  Best IoU:  {best_iou['iou']:.3f} at crop_pad={best_iou['crop_pad']}px")
    print(f"{'='*70}\n")

    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "_meta": {
            "sweep_type": "crop_pad",
            "models": "GIRAFE",
            "evaluated_on": "BAGLS",
            "min_pad": args.min_pad,
            "max_pad": args.max_pad,
            "step": args.step,
            "canvas": args.canvas,
            "device": args.device,
            "written_at": datetime.now().isoformat(),
        },
        "results": results,
        "best": {
            "best_dice": {
                "crop_pad": best_dice["crop_pad"],
                "dice": float(best_dice["dice"]),
                "iou": float(best_dice["iou"]),
                "det_recall": float(best_dice["det_recall"]),
            },
            "best_iou": {
                "crop_pad": best_iou["crop_pad"],
                "dice": float(best_iou["dice"]),
                "iou": float(best_iou["iou"]),
                "det_recall": float(best_iou["det_recall"]),
            },
        },
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
