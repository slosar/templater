# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `py/`: `spectra_loader.py` handles FITS ingestion, `template_model.py` contains the baseline JAX model, and `neural_template_model.py` holds the experimental neural variant. Training entrypoints live in `scripts/`, primarily `scripts/train.py`. Analysis and exploration notebooks are under `notebooks/`. Shared metadata is in `metadata/`. Large generated artifacts such as `checkpoints/`, `checkpoints-neural/`, `checkpoints-templates/`, and plots belong to runs and analysis, not hand-edited source.

## Build, Test, and Development Commands
Install the environment with:

```bash
pip install -e .
```

Run a fast smoke test with a tiny training job:

```bash
python scripts/train.py --n-spectra 100 --zmin 0.4 --zmax 1.1 --Nt 3 --Nz 50 --n-epochs 2 --batch-size 32 --checkpoint-dir /tmp/ckpt/
```

Run a standard training pass:

```bash
python scripts/train.py --n-spectra 5000 --zmin 0.4 --zmax 1.1 --Nt 5 --Nz 200 --n-epochs 20
```

Use `jupyter notebook` to open `notebooks/analyze.ipynb` for checkpoint inspection and fit plots.

## Coding Style & Naming Conventions
Target Python 3.10+ and keep code compatible with the dependencies declared in `pyproject.toml`. Follow the existing style: 4-space indentation, type hints on public APIs, `snake_case` for functions and variables, and `PascalCase` for classes such as `TemplateModel` and `SpectraDataset`. Keep modules focused by responsibility. Prefer short docstrings and direct argparse help text over dense inline commentary.

## Testing Guidelines
There is no dedicated `tests/` suite yet. Treat the small `scripts/train.py` run above as the required smoke test for model or loader changes. When changing data loading, verify one batch loads cleanly; when changing model logic, confirm training produces checkpoints and the analysis notebook can read them. Put reproducibility-sensitive checks behind explicit CLI flags rather than notebook-only edits.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `add n(z) shaping via rejection sampling (--shape-nofz)` and `frozen templates, n(z) acceleration, template resolution, spectra shuffle`. Keep commit subjects concise, lowercase where natural, and focused on one logical change. Pull requests should state the training scenario affected, list commands used for validation, and attach plots or screenshots when notebook outputs or recovered spectra change.
