"""Plot BAGLS YOLO threshold sweep (replaces Table 5 in the paper).

With --per-frame-dice PATH: adds a second row with 3 waveform panels (Dice vs
frame index for Patient 14, 50, 46B1). Generate the JSON by running:
  sweep_bagls_conf.py ... --output-per-frame-dice PATH
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "paper" / "bagls_sweep.pdf"

# Same 3 patients as fig_gaw_examples (GAW figure)
WAVEFORM_PANELS = [
    ("Patient 14", "Healthy"),
    ("Patient 50", "Paresis"),
    ("Patient 46B1", "Paralysis"),
]

# Table 5 data: tau, Det.Recall, DSC, IoU, DSC>=0.5 (%)
tau = np.array([0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.25])
det_recall = np.array([0.943, 0.917, 0.895, 0.859, 0.842, 0.819, 0.773, 0.688])
dsc = np.array([0.646, 0.652, 0.654, 0.659, 0.656, 0.652, 0.641, 0.609])
iou = np.array([0.553, 0.561, 0.563, 0.568, 0.567, 0.565, 0.558, 0.533])
dsc_ge_50 = np.array([75.0, 75.7, 75.8, 76.3, 76.0, 75.6, 74.3, 70.3])


def main() -> None:
    p = argparse.ArgumentParser(description="Plot BAGLS sweep and optional 3 waveform panels.")
    p.add_argument("--per-frame-dice", type=Path, default=None,
                   help="JSON list of per-frame Dice (from sweep_bagls_conf --output-per-frame-dice).")
    p.add_argument("--output", type=Path, default=OUT, help="Output PDF path.")
    args = p.parse_args()

    per_frame_dice = None
    if args.per_frame_dice is not None and args.per_frame_dice.exists():
        with open(args.per_frame_dice) as f:
            per_frame_dice = json.load(f)
        if not isinstance(per_frame_dice, list):
            per_frame_dice = None

    if per_frame_dice is not None:
        n = len(per_frame_dice)
        fig = plt.figure(figsize=(6, 5.5))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1.2, 1], hspace=0.35)
        ax_sweep = fig.add_subplot(gs[0, :])
        wave_axes = [fig.add_subplot(gs[1, j]) for j in range(3)]
    else:
        fig, ax_sweep = plt.subplots(figsize=(5.5, 3.5))
        wave_axes = None

    # ── Sweep plot ───────────────────────────────────────────────────────────
    ax_sweep.set_xscale("log")
    ax_sweep.set_xlabel(r"YOLO confidence threshold $\tau$")
    ax_sweep.set_ylabel("DSC / IoU / Det.Recall", color="C0")
    ax_sweep.tick_params(axis="y", labelcolor="C0")
    ax_sweep.set_xticks(tau)
    ax_sweep.set_xticklabels([str(t) for t in tau])
    ax_sweep.set_ylim(0.5, 1.0)
    ax_sweep.plot(tau, det_recall, "o-", color="C0", label="Det.Recall", markersize=6)
    ax_sweep.plot(tau, dsc, "s-", color="C1", label="DSC", markersize=6)
    ax_sweep.plot(tau, iou, "^-", color="C2", label="IoU", markersize=6)

    ax2 = ax_sweep.twinx()
    ax2.set_ylabel(r"DSC$\geq$0.5 (\%)", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax2.set_ylim(68, 78)
    ax2.plot(tau, dsc_ge_50, "d-", color="C3", label=r"DSC$\geq$0.5", markersize=6)

    ax_sweep.axvline(0.02, color="gray", linestyle="--", alpha=0.7)
    ax_sweep.annotate(r"default $\tau{=}0.02$", (0.02, 0.52), fontsize=8, color="gray")

    lines1, labels1 = ax_sweep.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_sweep.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    # ── Waveforms (3 patients / segments) ────────────────────────────────────
    if per_frame_dice is not None and n >= 3 and wave_axes is not None:
        seg_size = n // 3
        segments = [
            per_frame_dice[0:seg_size],
            per_frame_dice[seg_size:2 * seg_size],
            per_frame_dice[2 * seg_size:n],
        ]
        for ax, dice_list, (name, status) in zip(wave_axes, segments, WAVEFORM_PANELS):
            label = f"{name} ({status})"
            x = np.arange(len(dice_list))
            ax.plot(x, dice_list, color="C1", linewidth=0.8, alpha=0.9)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Frame index")
            ax.set_ylabel("Dice" if ax == wave_axes[0] else "")
            ax.set_title(label, fontsize=10)
            ax.grid(True, alpha=0.3)
        fig.align_ylabels(wave_axes)

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight")
    plt.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
