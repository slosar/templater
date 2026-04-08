# CLAUDE.md — Notes for Claude Code

## Project overview

Spectral template fitting for DESI galaxy spectra. Goal: learn a small set of
spectral templates T (Nt × Nft) and a redshift distribution n(z) that jointly
describe a galaxy survey, by marginalising over redshift.

## Repository layout

```
py/
  spectra_loader.py        — PyTorch Dataset for FITS spectra (fitsio-based)
  template_model.py        — JAX model: TemplateModel class, loss, inference helpers
  neural_template_model.py — JAX model: NeuralTemplateModel (MLP continuum + explicit lines + GMM alpha prior)
scripts/
  train.py            — CLI training utility (argparse, optax, DataLoader bridge)
notebooks/
  explore_spectra.ipynb  — basic spectra exploration (flux vs wavelength)
  analyze.ipynb          — load checkpoint, plot fits, n(z) comparison
metadata/
  desi-galaxy-cat.fits        — symlink to shared data
  desi-galaxy-cat-zerr.fits   — external catalog with zerr column (joined on targetid)
spectra/                       — symlink to ~1685 spec-*.fits files (~500 spectra each)
spectra_shuffled/              — shuffled copy of spectra (faster sequential I/O)
checkpoint_templates/          — checkpoints from TemplateModel training runs
checkpoints/                   — checkpoints from current/latest training run
instructions/problem.md        — original problem statement
```

## Data facts

- Spectra files: `spectra/spec-*.fits`, each ~500 rows
- HDUs per file: `wave` (7781,), `flux` (N, 7781), `flux_ivar` (N, 7781), `cat` (table)
- Wavelength grid: fixed 3600–9824 Å, 7781 points, ~0.8 Å/pixel
- `cat` columns include: `targetid`, `z`, `desi_target` (bitmask)
- `zerr` lives only in `metadata/desi-galaxy-cat-zerr.fits`, joined on `targetid`
- `metadata/desi-galaxy-cat.fits` does NOT have zerr (same 27 columns as per-file cat)
- Full targetid overlap between spectra and zerr catalog (842k rows in zerr cat)

## Model architectures

There are two model classes sharing the same z-marginalisation and n(z) learning.

### TemplateModel (`py/template_model.py`)
- **Parameters**: `T` (Nt, Nft) + `log_nz_raw` (Nnz,). Free-form template array.
- **Template grid**: Nft points spanning `[wave_min/(1+zmax), wave_max/(1+zmin)]`
  — always covers observed wavelengths when shifted. Nft ≈ 6630 for z=[0.4, 1.1].
- **Alpha**: analytically solved per galaxy per z via `jnp.linalg.solve`; not a parameter.
- **Loss**: negative mean log-likelihood with z marginalisation (online logsumexp in scan).
- **chi2 identity**: `chi2_min = Σ(flux²·ivar) - bᵀα` — avoids storing residuals.
- **Checkpoint keys**: `T`, `log_nz_raw`, `config` (no `model_type` field).

### NeuralTemplateModel (`py/neural_template_model.py`)
- **Template shape**: `T_k(λ) = MLP(λ_norm)[k] + Σ_j A_{kj} · Gauss(λ; λ_j, σ_j)`
  — shared-trunk MLP (1 → hidden → hidden → Nt, GELU, He-init) for smooth continuum,
  plus explicit Gaussian emission/absorption lines at fixed rest-frame wavelengths.
- **Line catalog**: 21 lines including [OII] doublet, Hα/β/γ/δ, [OIII], Ca H&K, Na D, etc.
  Lines outside `[t_wave_min, t_wave_max]` are automatically excluded.
- **Parameters**: `mlp_weights`, `line_A` (Nt, N_lines), `line_sigma_raw` (N_lines),
  `log_nz_raw` (Nnz,), plus GMM params `gmm_log_pi`, `gmm_mu`, `gmm_L_raw`.
- **GMM alpha prior**: K-component GMM on the alpha amplitude vector, added to the
  per-galaxy per-z log-weight during training only (`gmm_weight · log p_GMM(α)`).
  NOT applied in `compute_z_posterior` — it shapes what templates learn, not inference.
- **Checkpoint keys**: includes `model_type: 'neural'` in config for auto-detection.
- **Status**: under development; currently does not outperform TemplateModel.

## Framework choices

- **JAX** for model, loss, gradients (`jax.lax.scan`, `jax.remat`, `jax.vmap`)
- **optax** for optimisation (adam)
- **PyTorch DataLoader** for data I/O (bridge: `.numpy()` → `jnp.array()`)
- **fitsio** for FITS reading (not astropy)

## Key implementation details

- `TemplateModel.__init__` precomputes `(lo_idx, weight)` shape `(Nz, Nf)` — no
  per-step interpolation overhead. These are JAX arrays captured as compile-time
  constants inside `jax.jit`.
- `jax.lax.scan` over z keeps peak memory at O(B×Nt×Nf) per step (not Nz×B×Nf).
- `jax.remat` on the scan body trades recomputation for activation memory in backprop.
- `drop_last=True` in DataLoader ensures fixed batch size → no JAX retracing.
- Regularise A matrix with `1e-6 * I` before `linalg.solve` to handle masked pixels.
- zerr floor `1e-4` guards against zero-division in the z prior.
- `log_nz_raw` parameterised in unconstrained space; `log_softmax` normalises it.

## Two training phases

- **Phase 1** (template recovery, known z): pass `--zerr-catalog` with tight catalog
  zerr (~5e-5). The z prior collapses to the catalog z; effectively fits templates at
  known redshifts.
- **Phase 2** (blind z recovery): use `--zerr-override 1.0` to make z prior
  uninformative. Templates and n(z) must explain the data without a z hint.

## Installed versions (as of 2026-03-31)

- Python 3.x, JAX 0.9.2, optax 0.2.8, torch 2.11.0+cu130, fitsio>=1.3

## Common commands

```bash
# Quick smoke test (2 epochs, 100 spectra, small grid)
python scripts/train.py \
  --spectra-dir spectra/ \
  --zerr-catalog metadata/desi-galaxy-cat-zerr.fits \
  --n-spectra 100 --zmin 0.4 --zmax 1.1 \
  --Nt 3 --Nz 50 --n-epochs 2 --batch-size 32 \
  --checkpoint-dir /tmp/ckpt/

# Phase 1: template recovery with catalog zerr
python scripts/train.py \
  --spectra-dir spectra/ \
  --zerr-catalog metadata/desi-galaxy-cat-zerr.fits \
  --n-spectra 5000 --zmin 0.4 --zmax 1.1 \
  --Nt 5 --Nz 200 --n-epochs 20

# Phase 2: blind z (uninformative prior)
python scripts/train.py \
  --spectra-dir spectra/ \
  --n-spectra 5000 --zmin 0.4 --zmax 1.1 \
  --Nt 5 --Nz 200 --n-epochs 20 \
  --zerr-override 1.0

# Neural template model
python scripts/train.py \
  --neural-templates \
  --Nt 4 --mlp-hidden 64 --gmm-components 5 --gmm-weight 0.1 \
  --n-spectra 50000 --zmin 0.75 --zmax 1.0 --Nz 1000 \
  --n-epochs 200 --batch-size 8192
```

## Notes on fitsio slice syntax

`f['flux'][row, :]` returns shape `(1, Nf)` not `(Nf,)` — use
`f['flux'][row:row+1, :][0]` to get a 1D array.

## Checkpoint auto-detection (analyze.ipynb)

The notebook detects model type via `config['model_type'] == 'neural'`. TemplateModel
checkpoints have no `model_type` field; NeuralTemplateModel checkpoints have
`model_type: 'neural'`. Both share `log_nz_raw` and `config` keys; TemplateModel adds
`T`; NeuralTemplateModel adds `mlp_weights`, `line_A`, `line_sigma_raw`, and GMM keys.

## `target_noise` and z recovery

`SpectraDataset` accepts `target_noise` (float, default 0): adds white noise to bring
each spectrum's SNR down to `sqrt(sum(flux² * ivar)) = target_noise`. Setting this in
the analysis notebook degrades SNR and can completely break z recovery even when
templates look fine (O2 doublet visible). **Always set `target_noise=0` in the notebook
when checking z recovery, unless the training was also done with the same noise level.**

## Potential future work

- PCA initialisation of templates from training data
- Chunked vmap over z (z_batch_size parameter) for GPU parallelism within a z block
- Per-galaxy regularisation on alpha (prior on template amplitudes)
- Extend to photometry or multi-band data
