"""Plot YOLO confidence sweep comparison: GIRAFE U-Net vs BAGLS U-Net."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = Path("outputs")
GIRAFE_JSON = OUT_DIR / "sweep_bagls_yolo_girafe_unet_full.json"
BAGLS_JSON  = OUT_DIR / "sweep_bagls_yolo_bagls_unet_full.json"

def load(path: Path) -> tuple[list, dict]:
    with open(path) as f:
        data = json.load(f)
    taus = sorted(float(k) for k in data)
    return taus, {float(k): v for k, v in data.items()}

taus_g, girafe = load(GIRAFE_JSON)
taus_b, bagls  = load(BAGLS_JSON)
taus = taus_g  # same for both

# Extract series
def series(d, taus, pipe, metric):
    return [d[t][pipe][metric] for t in taus]

# GIRAFE U-Net
g_unet_only   = series(girafe, taus, "unet-only",      "dice_mean")
g_full        = series(girafe, taus, "yolo+unet",      "dice_mean")
g_crop        = series(girafe, taus, "yolo-crop+unet", "dice_mean")
g_full_recall = series(girafe, taus, "yolo+unet",      "det_recall")
g_crop_recall = series(girafe, taus, "yolo-crop+unet", "det_recall")

# BAGLS U-Net
b_unet_only   = series(bagls,  taus, "unet-only",      "dice_mean")
b_full        = series(bagls,  taus, "yolo+unet",      "dice_mean")
b_crop        = series(bagls,  taus, "yolo-crop+unet", "dice_mean")
b_full_recall = series(bagls,  taus, "yolo+unet",      "det_recall")
b_crop_recall = series(bagls,  taus, "yolo-crop+unet", "det_recall")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=False)
fig.suptitle("BAGLS YOLO Confidence Threshold Sweep (3500 test frames)", fontsize=13, fontweight="bold")

# Okabe-Ito colorblind-safe palette — distinct luminance for grayscale printing
# Each series also has a unique marker + line style for B&W distinguishability
styles = {
    "girafe_crop": dict(color="#0072B2", marker="o", ls="-",   lw=2.0, ms=5),   # deep blue
    "girafe_full": dict(color="#56B4E9", marker="s", ls="--",  lw=1.5, ms=5),   # sky blue
    "bagls_crop":  dict(color="#D55E00", marker="^", ls="-",   lw=2.0, ms=5),   # vermillion
    "bagls_full":  dict(color="#E69F00", marker="D", ls="-.",  lw=1.5, ms=4),   # orange
    "recall":      dict(color="#009E73", marker="o", ls="-",   lw=2.5, ms=5),   # green
    "nogate":      dict(color="#555555", marker="",  ls=":",   lw=1.2, ms=0),   # dark gray dotted
}

# ── Left: DSC vs τ ────────────────────────────────────────────────────────────
ax = axes[0]
ax.axhline(g_unet_only[0], color=styles["nogate"]["color"], ls=":",  lw=1.2,
           label="U-Net only — GIRAFE (no gate)")
ax.axhline(b_unet_only[0], color=styles["nogate"]["color"], ls="--", lw=1.2,
           label="U-Net only — BAGLS (no gate)")

ax.plot(taus, g_crop, **{k: v for k, v in styles["girafe_crop"].items()},
        label="BAGLS YOLO + GIRAFE U-Net (crop)")
ax.plot(taus, g_full, **{k: v for k, v in styles["girafe_full"].items()},
        label="BAGLS YOLO + GIRAFE U-Net (full)")
ax.plot(taus, b_crop, **{k: v for k, v in styles["bagls_crop"].items()},
        label="BAGLS YOLO + BAGLS U-Net (crop)")
ax.plot(taus, b_full, **{k: v for k, v in styles["bagls_full"].items()},
        label="BAGLS YOLO + BAGLS U-Net (full)")

ax.set_xlabel("Confidence threshold τ", fontsize=11)
ax.set_ylabel("Mean DSC", fontsize=11)
ax.set_title("DSC vs. Confidence Threshold", fontsize=11)
ax.set_xlim(-0.01, 0.76)
ax.set_ylim(0.55, 0.90)
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax.grid(True, alpha=0.3)

# Mark peaks
g_crop_peak_tau = taus[int(np.argmax(g_crop))]
g_crop_peak_dsc = max(g_crop)
b_full_peak_tau = taus[int(np.argmax(b_full))]
b_full_peak_dsc = max(b_full)
bbox_style = dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85)
gc_col = styles["girafe_crop"]["color"]
bf_col = styles["bagls_full"]["color"]
ax.annotate(f"τ={g_crop_peak_tau:.2f}\nDSC={g_crop_peak_dsc:.3f}",
            xy=(g_crop_peak_tau, g_crop_peak_dsc),
            xytext=(g_crop_peak_tau + 0.08, g_crop_peak_dsc - 0.025),
            fontsize=7.5, color=gc_col, bbox=bbox_style,
            arrowprops=dict(arrowstyle="->", color=gc_col, lw=0.8))
ax.annotate(f"τ={b_full_peak_tau:.2f}\nDSC={b_full_peak_dsc:.3f}",
            xy=(b_full_peak_tau, b_full_peak_dsc),
            xytext=(b_full_peak_tau + 0.08, b_full_peak_dsc + 0.008),
            fontsize=7.5, color=bf_col, bbox=bbox_style,
            arrowprops=dict(arrowstyle="->", color=bf_col, lw=0.8))

# ── Right: Det.Recall vs τ ────────────────────────────────────────────────────
# Recall depends only on YOLO — one curve shared across all U-Net combinations
ax2 = axes[1]

rc_col = styles["recall"]["color"]
ax2.axhline(1.0, color=styles["nogate"]["color"], ls=":", lw=1.2,
            label="No gate (recall = 1)")
ax2.plot(taus, g_crop_recall, color=rc_col,
         marker=styles["recall"]["marker"], ls=styles["recall"]["ls"],
         lw=styles["recall"]["lw"], ms=styles["recall"]["ms"],
         label="BAGLS YOLO detection recall")

# Shade safe zone (recall >= 0.85)
safe_taus = [t for t in taus if g_crop_recall[taus.index(t)] >= 0.85]
if safe_taus:
    ax2.axvspan(min(safe_taus) - 0.01, max(safe_taus) + 0.01,
                alpha=0.10, color=rc_col, label="Recall ≥ 0.85 zone")

# Annotate cliff
cliff_tau = 0.55
ax2.annotate("Recall cliff\n(τ > 0.50)",
             xy=(cliff_tau, g_crop_recall[taus.index(cliff_tau)]),
             xytext=(0.40, 0.72),
             fontsize=8.5, color=styles["nogate"]["color"],
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
             arrowprops=dict(arrowstyle="->", color=styles["nogate"]["color"], lw=0.8))

# Annotate peak safe point
ax2.annotate(f"τ=0.35\nrecall={g_crop_recall[taus.index(0.35)]:.3f}",
             xy=(0.35, g_crop_recall[taus.index(0.35)]),
             xytext=(0.38, g_crop_recall[taus.index(0.35)] + 0.05),
             fontsize=7.5, color=rc_col,
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
             arrowprops=dict(arrowstyle="->", color=rc_col, lw=0.8))

ax2.set_xlabel("Confidence threshold τ", fontsize=11)
ax2.set_ylabel("Detection Recall", fontsize=11)
ax2.set_title("BAGLS YOLO Detection Recall vs. τ\n(shared across all U-Net variants)", fontsize=10)
ax2.set_xlim(-0.01, 0.76)
ax2.set_ylim(0.50, 1.05)
ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
ax2.grid(True, alpha=0.3)

# ── Shared legend below both panels, horizontal ───────────────────────────────
handles_l, labels_l = axes[0].get_legend_handles_labels()
handles_r, labels_r = axes[1].get_legend_handles_labels()
# Deduplicate recall-panel entries already in left legend
all_handles = handles_l + [h for h, lb in zip(handles_r, labels_r) if lb not in labels_l]
all_labels  = labels_l  + [lb for lb in labels_r if lb not in labels_l]
fig.legend(all_handles, all_labels,
           loc="lower center",
           ncol=4,
           fontsize=8.5,
           frameon=True,
           bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.10, 1, 1])
out = OUT_DIR / "conf_sweep_comparison.pdf"
plt.savefig(out, bbox_inches="tight", dpi=150)
plt.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
print(f"Saved → {out}")

# ── Table ─────────────────────────────────────────────────────────────────────
print()
print("=" * 100)
print(f"  {'τ':>5}  {'Det.Recall':>10}  "
      f"{'GIRAFE crop':>12}  {'GIRAFE full':>12}  "
      f"{'BAGLS crop':>11}  {'BAGLS full':>11}  {'BAGLS unet-only':>15}")
print("=" * 100)

highlight_taus = [0.001, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75]
for t in taus:
    if t not in highlight_taus:
        continue
    gc = girafe[t]["yolo-crop+unet"]["dice_mean"]
    gf = girafe[t]["yolo+unet"]["dice_mean"]
    bc = bagls[t]["yolo-crop+unet"]["dice_mean"]
    bf = bagls[t]["yolo+unet"]["dice_mean"]
    bu = bagls[t]["unet-only"]["dice_mean"]
    rec = girafe[t]["yolo-crop+unet"]["det_recall"]
    print(f"  {t:>5.3f}  {rec:>10.3f}  {gc:>12.3f}  {gf:>12.3f}  {bc:>11.3f}  {bf:>11.3f}  {bu:>15.3f}")

print("=" * 100)
print(f"  {'no-gate':>5}  {'1.000':>10}  "
      f"  {girafe[taus[0]]['unet-only']['dice_mean']:>10.3f}  "
      f"{'(same)':>13}  "
      f"  {bagls[taus[0]]['unet-only']['dice_mean']:>9.3f}  "
      f"{'(same)':>11}  {'':>15}")
