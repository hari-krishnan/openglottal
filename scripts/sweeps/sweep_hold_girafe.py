"""Sweep temporal hold (1--20 frames, plus 0 and ∞) on GIRAFE and plot hold vs DSC.

Generates paper/hold_ablation.pdf (line graph). Run from repo root:
  ./run scripts/sweep_hold_girafe.py --images-dir GIRAFE/Training/imagesTr \\
    --labels-dir GIRAFE/Training/labelsTr --training-json GIRAFE/Training/training.json \\
    --unet-weights weights/openglottal_unet.pt --yolo-weights weights/openglottal_yolo.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo))
sys.path.insert(0, str(_repo / "scripts"))

from openglottal.models import UNet, TemporalDetector
from openglottal.utils import resolve_weights_path

from eval_girafe import evaluate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep hold 1--20 (and 0, ∞) on GIRAFE, plot hold vs DSC.")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--labels-dir", required=True)
    p.add_argument("--training-json", required=True)
    p.add_argument("--unet-weights", required=True)
    p.add_argument("--yolo-weights", required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--output", default="paper/hold_ablation.pdf", help="Output figure path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    unet_path = resolve_weights_path(args.unet_weights)
    yolo_path = resolve_weights_path(args.yolo_weights)

    unet = UNet(1, 1, (32, 64, 128, 256)).to(device)
    unet.load_state_dict(
        torch.load(unet_path, map_location=device, weights_only=True)
    )
    unet.eval()

    splits = json.load(open(args.training_json))
    test_fnames = splits["test"]
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    # Hold values: 0, 1..20, and a large value for "infinity"
    hold_values = [0] + list(range(1, 21)) + [999999]
    holds = []
    dscs = []
    det_recalls = []
    d50s = []

    for hold in hold_values:
        detector = TemporalDetector(str(yolo_path), max_hold_frames=hold)
        agg, _ = evaluate(
            test_fnames=test_fnames,
            images_dir=images_dir,
            labels_dir=labels_dir,
            unet_model=unet,
            device=device,
            detector=detector,
        )
        data = agg["yolo+unet"]
        n_det, n_tot = data["n_det"], data["n_total"]
        dices = data["dice"]
        mean_dice = float(np.mean(dices)) if dices else float("nan")
        det_rec = n_det / n_tot if n_tot > 0 else float("nan")
        d50 = 100 * np.mean([d >= 0.5 for d in dices]) if dices else float("nan")

        dscs.append(mean_dice)
        det_recalls.append(det_rec)
        d50s.append(d50)
        print(f"  hold={hold if hold != 999999 else '∞'}: Det.Recall={det_rec:.3f}, DSC={mean_dice:.3f}, DSC≥0.5={d50:.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; saving raw data to paper/hold_ablation_data.json")
        out_data = {
            "hold": hold_values,
            "hold_display": [h if h != 999999 else "inf" for h in hold_values],
            "dsc": dscs,
            "det_recall": det_recalls,
            "d50": d50s,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json_path = Path(args.output).with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(out_data, f, indent=2)
        return

    fig, ax1 = plt.subplots(figsize=(6, 4))
    # x positions: 0,1,...,20, 21 for infinity
    x_numeric = list(range(0, 22))  # 0..21
    # Map hold_values to indices: 0->0, 1->1, ..., 20->20, 999999->21
    idx_for_hold = {h: i for i, h in enumerate(hold_values)}
    x_vals = list(range(len(hold_values)))
    ax1.plot(x_vals, dscs, "o-", color="C0", label="DSC", markersize=4)
    ax1.set_xlabel("Temporal hold (frames)")
    ax1.set_ylabel("DSC", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.set_ylim(0.65, 0.80)
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels([str(h) if h != 999999 else "∞" for h in hold_values])
    # Mark default 4 frames (index 5 in hold_values: 0,1,2,3,4,...)
    ax1.axvline(x=4, color="green", linestyle=":", alpha=0.8, label="4 frames (1 ms)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.plot(x_vals, [r * 100 for r in det_recalls], "s--", color="C1", label="Det.Recall (%)", markersize=3)
    ax2.set_ylabel("Det.Recall (%)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")
    ax2.set_ylim(75, 105)
    ax2.legend(loc="center right")

    plt.title("YOLO+UNet on GIRAFE test set: hold duration vs DSC and Det.Recall")
    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
