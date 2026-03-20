#!/usr/bin/env python3
"""Compare results/gaw/gaw_features.json table and cv row to paper (main.tex tab:gaw).

The script prints the pooled table (Healthy vs Pathological) as produced from
the JSON (same logic as analyze_gaw.py), then the paper's stratified values
for side-by-side comparison. The paper reports by sex (Female 12H/11P, Male 3H/14P);
without sex in the JSON we cannot reproduce the stratified table exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

def _mean(vals):
    n = len(vals)
    return sum(vals) / n if n else 0.0

def _std(vals):
    n = len(vals)
    if n < 2:
        return 0.0
    m = _mean(vals)
    return (sum((x - m) ** 2 for x in vals) / (n - 1)) ** 0.5

try:
    from scipy.stats import mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Same grouping as analyze_gaw.py
HEALTHY_LABEL = "Healthy"
PATHOLOGICAL_LABELS = {
    "Paresis", "Polyps", "Diplophonia", "Nodules",
    "Paralysis", "Cysts", "Carcinoma", "Multinodular Goiter", "Other",
}

FEAT_COLS = ["area_mean", "area_std", "area_range", "open_quotient", "f0", "periodicity", "cv"]

# Paper (main.tex tab:gaw) — stratified by sex
# Female (12 H / 11 P), Male (3 H / 14 P)
PAPER_TABLE = {
    "area_mean":   {"F_H": (125.2, 43.1), "F_P": (247.8, 204.6), "F_p": 0.230,
                    "M_H": (192.1, 18.3), "M_P": (172.7, 94.0),  "M_p": 0.768},
    "area_std":    {"F_H": (112.9, 32.2), "F_P": (118.9, 96.0), "F_p": 0.406,
                    "M_H": (142.7, 35.0), "M_P": (92.0, 66.9),   "M_p": 0.197},
    "area_range":  {"F_H": (336.7, 97.6), "F_P": (375.5, 272.2), "F_p": 0.559,
                    "M_H": (439.7, 86.7), "M_P": (343.1, 212.3), "M_p": 0.488},
    "open_quotient": {"F_H": (0.760, 0.207), "F_P": (0.874, 0.131), "F_p": 0.192,
                      "M_H": (0.860, 0.145), "M_P": (0.843, 0.186), "M_p": 1.000},
    "f0":          {"F_H": (241.7, 34.8), "F_P": (203.5, 73.6), "F_p": 0.156,
                    "M_H": (183.3, 75.0), "M_P": (82.5, 79.3),   "M_p": 0.169},
    "periodicity": {"F_H": (0.955, 0.008), "F_P": (0.946, 0.013), "F_p": 0.255,
                    "M_H": (0.962, 0.001), "M_P": (0.900, 0.116), "M_p": 0.068},
    "cv":          {"F_H": (0.95, 0.20), "F_P": (0.57, 0.29), "F_p": 0.006,
                    "M_H": (0.75, 0.19), "M_P": (0.63, 0.40), "M_p": 0.509},
}


def main() -> None:
    here = Path(__file__).resolve().parent
    repo = here.parent
    json_path = repo / "results" / "gaw" / "gaw_features.json"
    if not json_path.exists():
        print(f"Missing {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        records = json.load(f)

    healthy = [r for r in records if r["disorder"] == HEALTHY_LABEL]
    patho   = [r for r in records if r["disorder"] in PATHOLOGICAL_LABELS]
    unknown = [r for r in records if r["disorder"] not in
               {HEALTHY_LABEL} | PATHOLOGICAL_LABELS]

    n_h, n_p = len(healthy), len(patho)
    print(f"Group sizes: Healthy={n_h}, Pathological={n_p}, Unknown/excluded={len(unknown)}")
    print(f"Paper: 15 Healthy, 25 Pathological (40 total for table); "
          f"stratified Female 12H/11P, Male 3H/14P.\n")

    # Pooled table from JSON (same as analyze_gaw.py printed output)
    print("=" * 72)
    print("POOLED TABLE (from gaw_features.json — same as script printed table)")
    print("=" * 72)
    print(f"  {'Feature':<18} {'Healthy (mean±std)':>22} {'Patho (mean±std)':>22}"
          + ("  p-value" if HAS_SCIPY else ""))
    print("-" * 72)

    pooled = {}
    for feat in FEAT_COLS:
        h_vals = [r[feat] for r in healthy if feat in r and r[feat] is not None]
        p_vals = [r[feat] for r in patho   if feat in r and r[feat] is not None]
        if not h_vals or not p_vals:
            continue
        h_mean, h_std = _mean(h_vals), _std(h_vals)
        p_mean, p_std = _mean(p_vals), _std(p_vals)
        pooled[feat] = {"H": (h_mean, h_std), "P": (p_mean, p_std)}
        if HAS_SCIPY:
            _, pval = mannwhitneyu(h_vals, p_vals, alternative="two-sided")
            pooled[feat]["p"] = float(pval)
            sig = " *" if pval < 0.05 else ("  †" if pval < 0.10 else "")
            line = f"  {feat:<18} {h_mean:.3f} ± {h_std:.3f}   {p_mean:.3f} ± {p_std:.3f}   {pval:.4f}{sig}"
        else:
            line = f"  {feat:<18} {h_mean:.3f} ± {h_std:.3f}   {p_mean:.3f} ± {p_std:.3f}"
        print(line)
    print("-" * 72)
    if HAS_SCIPY:
        print("  * p < 0.05   † p < 0.10\n")

    # Paper table (stratified) — cv row highlighted
    print("=" * 72)
    print("PAPER TABLE (main.tex tab:gaw — stratified by sex)")
    print("=" * 72)
    print("  Female (12 H / 11 P)                    Male (3 H / 14 P)")
    print(f"  {'Feature':<14} H              P        p     H              P        p")
    print("-" * 72)
    for feat in FEAT_COLS:
        row = PAPER_TABLE[feat]
        fmt = ".3f" if feat in ("open_quotient", "periodicity", "cv") else ".1f"
        f_H = f"{row['F_H'][0]:{fmt}}±{row['F_H'][1]:{fmt}}"
        f_P = f"{row['F_P'][0]:{fmt}}±{row['F_P'][1]:{fmt}}"
        m_H = f"{row['M_H'][0]:{fmt}}±{row['M_H'][1]:{fmt}}"
        m_P = f"{row['M_P'][0]:{fmt}}±{row['M_P'][1]:{fmt}}"
        bold = "** " if feat == "cv" else "   "
        print(f"  {bold}{feat:<14} {f_H:>12} {f_P:>12} {row['F_p']:.3f}   {m_H:>12} {m_P:>12} {row['M_p']:.3f}")
    print("-" * 72)
    print("  ** cv row: paper reports Female H=0.95±0.20, P=0.57±0.29, p=0.006;")
    print("     Male H=0.75±0.19, P=0.63±0.40, p=0.509.\n")

    # Comparison: cv row
    print("=" * 72)
    print("CV ROW COMPARISON")
    print("=" * 72)
    if "cv" in pooled:
        h_mean, h_std = pooled["cv"]["H"]
        p_mean, p_std = pooled["cv"]["P"]
        pval = pooled["cv"].get("p")
        print(f"  From JSON (pooled):  Healthy = {h_mean:.3f} ± {h_std:.3f},  "
              f"Patho = {p_mean:.3f} ± {p_std:.3f}" +
              (f",  p = {pval:.4f}" if pval is not None else ""))
    print("  Paper (female):      Healthy = 0.95 ± 0.20,  Patho = 0.57 ± 0.29,  p = 0.006")
    print("  Paper (male):        Healthy = 0.75 ± 0.19,  Patho = 0.63 ± 0.40,  p = 0.509")
    print()
    print("  Note: Paper table is stratified by sex (no pooled cv in paper).")
    print("  Pooled cv from this run can differ from female/male subgroup means.")
    print("  To reproduce the paper table exactly, run analyze_gaw with sex in")
    print("  metadata and stratify (Female 12H/11P, Male 3H/14P).")
    print("=" * 72)


if __name__ == "__main__":
    main()
