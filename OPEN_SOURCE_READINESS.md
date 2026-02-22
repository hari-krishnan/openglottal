# OpenGlottal — Open Source Readiness Evaluation

Evaluation of the OpenGlottal codebase for public release, aligned with the paper and current work.

---

## Summary

| Area | Status | Notes |
|------|--------|--------|
| **License** | ✅ Ready | MIT, clear copyright (2024 OpenGlottal Contributors) |
| **Documentation** | ✅ Good | README has install, quick start, pipelines, training, GAW |
| **Reproducibility** | ✅ Good | Scripts take paths/weights; GIRAFE/BAGLS URLs in paper |
| **No secrets/paths** | ✅ Clean | No hardcoded `/Users/`, API keys, or tokens in code |
| **Package** | ✅ Ready | `pyproject.toml`, `openglottal` package, CLI entry point |
| **Paper vs README** | ⚠️ Align | README said 63 pts / Det.Recall 1.0; paper uses 65 pts, 0.95 — **fixed** |
| **Repository URL** | ✅ Set | Paper and README use https://github.com/hari-krishnan/openglottal |
| **Tests** | — | No tests directory; add later if desired. |

---

## What’s in good shape

- **License**: MIT with standard grant and disclaimer.
- **Structure**: Clear split between library (`openglottal/`), scripts (`scripts/`), configs, and paper.
- **Dependencies**: Declared in `pyproject.toml` with version lower bounds; optional `dev` and `docs` extras.
- **CLI**: `openglottal` command wired via `pyproject.toml`; run/inference paths documented.
- **Data**: GIRAFE and BAGLS referenced by DOI/ Zenodo in paper; README describes splits and paths.
- **.gitignore**: Excludes weights, data, runs, venv, and build artifacts.
- **Citation**: README has BibTeX for Patel et al.; paper has full refs and data/code availability section.

---

## Fixes applied for release

1. **README ↔ paper alignment**
   - GIRAFE results table: Det.Recall set to **0.95** (not 1.000) for YOLO pipelines; YOLO+UNet Dice **0.746** to match paper.
   - GAW / cohort: **65 patients** (15 Healthy, 25 Pathological, 25 Unknown) and related text updated from 63.
2. **Repository URL**
   - Paper and README use `https://github.com/hari-krishnan/openglottal`.
3. **Tests**
   - No tests directory in the repo; add `tests/` and pytest later if desired.
4. **Clone URL**
   - README clone URL set to `https://github.com/hari-krishnan/openglottal`.

---

## Recommended before/after push

- [x] **Repository URL**: Set to `https://github.com/hari-krishnan/openglottal` in paper and README.
- [ ] **Weights / Zenodo**: Decide whether to publish trained weights (e.g. YOLO + U-Net) on Zenodo and link from README/paper; document in README if “download weights from …”.
- [ ] **CI (optional)**: Add GitHub Actions (or similar) to run `ruff`/`mypy` (and `pytest` if tests are added) on push.
- [ ] **CHANGELOG**: Add a minimal `CHANGELOG.md` or “Releases” section for version 0.1.0 at first public tag.

---

## File checklist (no sensitive content)

- No `*.pt` / `*.pth` committed (handled by `.gitignore`).
- No raw data in repo; use `scripts/download_datasets.py` for GIRAFE and BAGLS.
- Paper and README use the repository URL; dataset URLs are concrete.
- Scripts use `argparse` and paths from CLI; no embedded absolute or user-specific paths.

---

*Evaluation date: 2026-02-23. Re-evaluate after setting public repo URL and any Zenodo release.*
