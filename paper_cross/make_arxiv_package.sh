#!/bin/bash
# Create an arXiv submission package for the paper_cross paper.
# Run from repo root: ./paper_cross/make_arxiv_package.sh
# Output: paper_cross/openglottal_arxiv.zip

set -e
PAPER="paper_cross"
OUT="paper_cross/arxiv_submit"
ZIP="paper_cross/openglottal_arxiv.zip"

rm -rf "$OUT"
mkdir -p "$OUT"

# Source files (figures are in paper_cross/; arXiv package is flat)
cp "$PAPER/main.tex" "$OUT/main.tex"
cp "$PAPER/refs.bib" "$OUT/"

# Figures (only those referenced in main.tex)
cp "$PAPER/pipeline.png"          "$OUT/"
cp "$PAPER/pipeline.pdf"          "$OUT/" 2>/dev/null || true
cp "$PAPER/hold_ablation.pdf"     "$OUT/"
cp "$PAPER/patient1_montage.png"  "$OUT/"
cp "$PAPER/bagls_sweep.pdf"       "$OUT/"
cp "$PAPER/gaw_examples.png"      "$OUT/"
cp "$PAPER/conf_sweep_comparison.pdf" "$OUT/" 2>/dev/null || true

# Optional: include .bbl so arXiv uses your bibliography if their BibTeX run differs
cp "$PAPER/main.bbl" "$OUT/" 2>/dev/null || true

cd "$OUT"
zip -r "../openglottal_arxiv.zip" .
cd -
rm -rf "$OUT"

echo "Created $ZIP"
echo "Upload $ZIP at https://arxiv.org/submit (or use tar.gz)."
echo "Select main.tex as the main file when prompted."
