# OpenGlottal — Repository Evaluation (current state)

**Date:** 2026-02-24

---

## Summary

The repo is **ready for open-source release**. Clone → install → run inference with bundled weights; data can be downloaded via script; training and evaluation are documented and script-driven. Repository URL is set to https://github.com/hari-krishnan/openglottal.

---

## What’s in place

| Area | Status |
|------|--------|
| **License** | MIT (`LICENSE`) |
| **Package** | `pyproject.toml` with deps, `openglottal` package, CLI entry point |
| **Requirements** | README: Python ≥3.9, PyTorch, Ultralytics, OpenCV. Full list in `pyproject.toml` (torch, torchvision, ultralytics, opencv-python, numpy, scipy, matplotlib, tqdm) |
| **Weights** | `weights/openglottal_yolo.pt` and `weights/openglottal_unet.pt` committed; README Quick Start and GIRAFE eval use these paths |
| **Documentation** | README: install, four pipelines, Quick Start (API + CLI), evaluation (GIRAFE + BAGLS), training, dataset (with `download_datasets.py`), repo structure, kinematic features, citation |
| **Paths** | All scripts take paths as arguments; no hardcoded absolute paths; relative paths are cwd-relative |
| **Data** | No bundled data; `scripts/download_datasets.py` for GIRAFE/BAGLS; `data/` gitignored |
| **Artifacts** | `results/`, `eval_results*.json`, LaTeX build products in `.gitignore` |
| **Paper** | `paper/` with `main.tex`, `refs.bib`, `fig1_pipeline.mmd`, montage PNGs; builds with latexmk |

---

## Optional / before public link

- **Clone URL / Paper:** Set to https://github.com/hari-krishnan/openglottal in README and paper.
- **BAGLS eval:** README uses `weights/openglottal_*.pt` for unet and yolo; `--crop-weights` still points to `glottal_detector/unet_glottis_crop.pt` (train with `train_unet_crop.py` if you need YOLO-Crop+UNet).

---

## Layout (tracked)

```
openglottal/
├── weights/           # openglottal_yolo.pt, openglottal_unet.pt
├── openglottal/       # package: cli, data, features, utils, models/
├── scripts/           # train_*, eval_*, infer, analyze_gaw, download_datasets, make_montage, sweep_bagls_conf, generate_demo_videos
├── configs/default.yaml
├── paper/             # main.tex, refs.bib, fig1_pipeline.mmd, montage PNGs
├── README.md, LICENSE, pyproject.toml, .gitignore
├── OPEN_SOURCE_READINESS.md, FILE_EVALUATION.md
```

No `data/`, `docs/`, or `tests/` directories; no references to them in the main README.

---

## Usability

- **Inference:** `pip install -e .` then `openglottal run video.avi --yolo-weights weights/openglottal_yolo.pt --unet-weights weights/openglottal_unet.pt --pipeline unet` works with the committed weights.
- **Reproducibility:** Eval/training use CLI args; GIRAFE/BAGLS obtained via `download_datasets.py` or linked Zenodo DOIs.
- **Verdict:** Repo is easy to use for inference and for reproducing training/evaluation clone and repository URLs are set to https://github.com/hari-krishnan/openglottal.
