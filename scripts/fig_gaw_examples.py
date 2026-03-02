"""Build a 3-panel figure for the paper: one Healthy GAW and two disorder GAWs
(Patient 50, Paresis; Patient 46B1, Paralysis). Reads existing PNGs from
results/gaw_plots/. Run from repo root after plot_gaw.py has been run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
GAW_PLOTS = REPO / "results" / "gaw_plots"
OUT = REPO / "paper" / "gaw_examples.png"

# One healthy, two disorders (Paresis, Paralysis)
PANELS = [
    ("patient14", "Healthy"),
    ("patient50", "Paresis"),
    ("patient46B1", "Paralysis"),
]


def main() -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    axes = axes.ravel()
    for ax, (stem, label) in zip(axes, PANELS):
        path = GAW_PLOTS / f"gaw_{stem}.png"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run plot_gaw.py first.")
        img = plt.imread(path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label, fontsize=12)
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
