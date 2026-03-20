# Contributing to OpenGlottal

Thanks for your interest. This project is research-oriented; small, focused PRs are easiest to review.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional: `./run scripts/...` runs commands with the repo venv (see [README](README.md)).

## Checks before a PR

- **Tests:** `pytest` (from repo root with the venv active).
- **Lint:** `ruff check .` and `ruff format --check .` (fix with `ruff format .`).

## Style

- Prefer type hints for new public functions.
- Keep scripts runnable from the repository root; avoid hardcoded machine-specific paths in examples (use placeholders like `/path/to/BAGLS/test`).

## Issues & questions

Open a GitHub issue for bugs, feature ideas, or clarification on reproducing paper numbers.
