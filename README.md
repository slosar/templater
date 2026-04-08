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
so the only learned parameters are the templates and the redshift distribution **n(z)**.

### TemplateModel

Free-form template array `T` (Nt × Nft). Simple and fast. Works well for
phase 1 (catalog-z training). For blind z recovery, unconstrained alpha can
create false chi2 minima at wrong redshifts.

### NeuralTemplateModel (experimental)

Each template is an MLP-modelled continuum plus explicit Gaussian emission/absorption
lines at 21 known rest-frame wavelengths ([OII], Hα, [OIII], Ca H&K, etc.):

```
T_k(λ) = MLP(λ_norm)[k]  +  Σ_j  A_{kj} · exp(-(λ - λ_j)² / 2σ_j²)
```

A Gaussian Mixture Model (GMM) prior on alpha is added to the training loss to
discourage unphysical template combinations. Enable with `--neural-templates`.

## Requirements

Python ≥ 3.10. Install dependencies:

```bash
pip install "jax[cuda12]>=0.4" optax torch fitsio numpy matplotlib notebook
```

For CPU-only JAX replace `jax[cuda12]` with `jax`.

## Data layout

```
spectra/              original spec-*.fits files (HDUs: wave, flux, flux_ivar, cat)
spectra_shuffled/     shuffled copy for faster sequential I/O during training
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

# Neural template model
python scripts/train.py \
  --neural-templates --Nt 4 --mlp-hidden 64 \
  --n-spectra 50000 --zmin 0.75 --zmax 1.0 --Nz 1000 \
  --n-epochs 200 --batch-size 8192
```

All CLI options:

| Flag | Default | Description |
|------|---------|-------------|
| `--spectra-dir` | `spectra_shuffled/` | Directory of `spec-*.fits` files |
| `--zerr-catalog` | `metadata/desi-galaxy-cat-zerr.fits` | FITS file with `targetid` + `zerr` |
| `--n-spectra` | all | Cap number of spectra loaded |
| `--zmin` / `--zmax` | 0.4 / 1.1 | Redshift filter and template grid range |
| `--zmin-loader` / `--zmax-loader` | same as zmin/zmax | Separate loader z cuts |
| `--desi-target-mask` | none | Bitmask filter on `desi_target` |
| `--target-noise` | 0 | Add noise to cap SNR at this value (0 = disabled) |
| `--shape-nofz` | off | Rejection-sample to a target n(z) shape |
| `--nofz-z0` / `--nofz-alpha` / `--nofz-beta` | `0.88 / 40 / 40` | Parameters for `--shape-nofz` |
| `--Nt` | 5 | Number of templates |
| `--Nz` | 200 | z search grid size |
| `--Nnz` | 50 | n(z) bin count |
| `--template-res-boost` | 1.0 | Increase template resolution by this factor |
| `--neural-templates` | off | Use NeuralTemplateModel instead of TemplateModel |
| `--mlp-hidden` | 64 | MLP hidden width (neural only) |
| `--gmm-components` | 5 | GMM components for alpha prior (neural only) |
| `--gmm-weight` | 0.1 | Weight on GMM log-prob in loss (neural only) |
| `--line-noise-init` | 0.01 | Initial noise scale for non-[OII] line amplitudes (neural only) |
| `--nz-sigma` | 0.4 | Width of the initial n(z) Gaussian; `0` gives uniform init |
| `--n-epochs` | 20 | Training epochs |
| `--lr` | 1e-3 | Adam learning rate |
| `--batch-size` | 256 | Batch size (fixed via `drop_last`) |
| `--seed` | 42 | Random seed |
| `--num-workers` | 4 | PyTorch DataLoader workers |
| `--disable-z-prior` | 0.0 | Fraction of batch with flat z prior per step |
| `--z-prior-warmup` | 0 | Epochs to ramp up `--disable-z-prior` |
| `--zerr-override` | — | Override catalog zerr (e.g. `1.0` for uninformative prior) |
| `--zerr-floor` | 1e-4 | Minimum zerr |
| `--freeze-templates` | off | Only optimise n(z), hold templates fixed |
| `--nz-steps` | 0 | Extra n(z)-only gradient steps per batch |
| `--nz-lr` | `--lr` | Learning rate for `--nz-steps` |
| `--t0-init` | `mean_flux` | Initialisation for template 0 (`mean_flux` or `flat`) |
| `--template-l2` | 0 | L2 penalty on template amplitudes |
| `--template-ortho` | 0 | Penalise template cross-correlation to reduce degeneracy |
| `--checkpoint-dir` | `checkpoints/` | Where to save `.npz` checkpoints |
| `--checkpoint-interval` | 5 | Save every N epochs |
| `--resume` | — | Resume from checkpoint |
| `--resume-templates-only` | — | Load templates from checkpoint, reinit n(z) |
| `--log-interval` | 10 | Print loss every N gradient steps |

## Analysis notebook

After training, open `notebooks/analyze.ipynb`. It will:

- Load the latest checkpoint from `checkpoints/`
- Compute the z posterior for each galaxy and find the best-fit redshift
- Plot observed spectra overlaid with template fits
- Plot the recovered n(z) against the true redshift histogram
- Plot the learned template shapes

The notebook auto-detects checkpoint family from the saved config:

- `TemplateModel` checkpoints store `T`, `log_nz_raw`, and `config`
- `NeuralTemplateModel` checkpoints store MLP weights, line parameters, GMM parameters, and `config["model_type"] == "neural"`

## Direct Neural Fits

`py/neural_template_fit.py` fits a `NeuralTemplateModel` directly to an existing
`TemplateModel` checkpoint, rather than training on survey spectra. This is useful
for checking how well the neural parameterisation can reproduce already-learned
rest-frame templates before spending time on full survey training.

The accompanying notebook `notebooks/fit_neural_to_templates.ipynb` compares the
fitted neural templates against the source template checkpoint and reports compact
fit metrics.

## Code structure

```
py/
  spectra_loader.py        PyTorch Dataset — reads FITS, joins zerr, filters by z / target
  template_model.py        JAX TemplateModel — free-form templates, loss, inference, checkpointing
  neural_template_model.py JAX NeuralTemplateModel — MLP continuum + explicit lines + GMM prior
  neural_template_fit.py   Helpers for fitting neural templates to saved template checkpoints
scripts/
  train.py            CLI training script
notebooks/
  explore_spectra.ipynb   quick flux-vs-wavelength explorer
  analyze.ipynb           post-training analysis and visualisation
  fit_neural_to_templates.ipynb  compare direct neural fits against saved templates
```

## Notes

- JAX JIT compiles on the first training step; subsequent steps are fast.
- The first epoch is slower than the rest due to XLA compilation.
- `--target-noise` modifies spectra on read to cap total SNR. Keep it at `0` in
  analysis unless you explicitly want to evaluate a noise-degraded setting.
- Templates cover `[wave_min/(1+zmax), wave_max/(1+zmin)]` in rest frame so they
  always overlap the observed range at any z in the search grid.
- Checkpoints are `.npz` files with arrays plus a JSON config string.
