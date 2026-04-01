# Spectral Template Fitter

Learns a small set of spectral templates and a redshift distribution n(z) from
DESI galaxy spectra, using GPU-accelerated JAX optimisation.

## Model

Each galaxy spectrum is modelled as a linear combination of Nt templates,
shifted by the galaxy's redshift:

```
flux_if ≈ Σ_k  alpha_ik · T_k(wave_obs[f] / (1+z_i))
```

The redshift and per-galaxy amplitudes alpha are not fixed — the loss
marginalises over a discrete z search grid:

```
p_i  = Σ_z  exp( -½·chi2_iz  -  ½·(z - z_prior_i)² / zerr_i² )  · n(z)
loss = -mean_i log(p_i)
```

At each redshift grid point, alpha is solved analytically (linear least squares),
so the only learned parameters are the template array **T** and the redshift
distribution **n(z)**.

## Requirements

Python ≥ 3.10. Install dependencies:

```bash
pip install "jax[cuda12]>=0.4" optax torch fitsio numpy matplotlib notebook
```

For CPU-only JAX replace `jax[cuda12]` with `jax`.

## Data layout

```
spectra/              spec-*.fits files (HDUs: wave, flux, flux_ivar, cat)
metadata/
  desi-galaxy-cat-zerr.fits    per-galaxy zerr, joined on targetid
```

The wavelength grid is fixed at 3600–9824 Å (7781 pixels).

## Training

```bash
# Phase 1 — template recovery (catalog redshifts as prior)

 python scripts/train.py   --n-spectra 5000 --zmin 0.3 --zmax 0.8  --zmin-loader 0.4 --zmax-loader 0.7  --Nt 5 --Nz 1000 --n-epochs 200 --batch-size=2048

# Phase 2 — blind redshift recovery (uninformative prior)

 python scripts/train.py   --n-spectra 5000 --zmin 0.3 --zmax 0.8  --zmin-loader 0.4 --zmax-loader 0.7  --Nt 5 --Nz 1000 --n-epochs 200 --batch-size=2048 --disable-z-prior 0.99

# Quick smoke test
python scripts/train.py \
  --n-spectra 100 --zmin 0.4 --zmax 1.1 \
  --Nt 3 --Nz 50 --n-epochs 2 --batch-size 32 \
  --checkpoint-dir /tmp/ckpt/
```

All CLI options:

| Flag | Default | Description |
|------|---------|-------------|
| `--spectra-dir` | `spectra/` | Directory of `spec-*.fits` files |
| `--zerr-catalog` | `metadata/desi-galaxy-cat-zerr.fits` | FITS file with `targetid` + `zerr` |
| `--n-spectra` | all | Cap number of spectra loaded |
| `--zmin` / `--zmax` | 0.4 / 1.1 | Redshift filter and template grid range |
| `--desi-target-mask` | none | Bitmask filter on `desi_target` |
| `--Nt` | 5 | Number of templates |
| `--Nz` | 200 | z search grid size |
| `--Nnz` | 50 | n(z) bin count |
| `--n-epochs` | 20 | Training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--batch-size` | 256 | Batch size (fixed via `drop_last`) |
| `--zerr-override` | — | Override catalog zerr (e.g. `1.0` for uninformative prior) |
| `--zerr-floor` | 1e-4 | Minimum zerr |
| `--checkpoint-dir` | `checkpoints/` | Where to save `.npz` checkpoints |
| `--checkpoint-interval` | 5 | Save every N epochs |
| `--resume` | — | Path to checkpoint to continue from |

## Analysis notebook

After training, open `notebooks/analyze.ipynb`. It will:

- Load the latest checkpoint from `checkpoints/`
- Compute the z posterior for each galaxy and find the best-fit redshift
- Plot observed spectra overlaid with template fits
- Plot the recovered n(z) against the true redshift histogram
- Plot the learned template shapes

## Code structure

```
py/
  spectra_loader.py   PyTorch Dataset — reads FITS, joins zerr, filters by z / target
  template_model.py   JAX TemplateModel — interpolation, loss, inference, checkpointing
scripts/
  train.py            CLI training script
notebooks/
  explore_spectra.ipynb   quick flux-vs-wavelength explorer
  analyze.ipynb           post-training analysis and visualisation
```

## Notes

- JAX JIT compiles on the first training step; subsequent steps are fast.
- The first epoch is slower than the rest due to XLA compilation.
- Templates cover `[wave_min/(1+zmax), wave_max/(1+zmin)]` in rest frame so they
  always overlap the observed range at any z in the search grid.
- Checkpoints are `.npz` files containing `T`, `log_nz_raw`, and a JSON config string.
