#!/usr/bin/env python3
"""
Train spectral templates and n(z) on DESI galaxy spectra.

The model fits Nt spectral templates to galaxy spectra by marginalising over a
redshift search grid, simultaneously learning the template shapes and the
redshift distribution n(z).

Examples
--------
# Phase 1 — template recovery with known redshifts (tight catalog zerr):
python scripts/train.py \\
    --n-spectra 500 --zmin 0.4 --zmax 1.1 \\
    --Nt 5 --Nz 200 --n-epochs 20

# Phase 2 — blind redshift recovery (uninformative z prior):
python scripts/train.py \\
    --n-spectra 500 --zmin 0.4 --zmax 1.1 \\
    --Nt 5 --Nz 200 --n-epochs 20 \\
    --zerr-override 1.0

# Smoke test (fast):
python scripts/train.py \\
    --n-spectra 100 --zmin 0.4 --zmax 1.1 \\
    --Nt 3 --Nz 50 --n-epochs 2 --batch-size 32 \\
    --checkpoint-dir /tmp/ckpt/
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

# Make imports work whether running from repo root or scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'py'))

from spectra_loader import SpectraDataset
from template_model import TemplateModel

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError("PyTorch is required for the DataLoader. Install with: pip install torch")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train spectral template model on DESI spectra",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Data ----------------------------------------------------------------
    data = p.add_argument_group("Data")
    data.add_argument('--spectra-dir', default='spectra_shuffled/',
                      help='Directory containing spec-*.fits files')
    data.add_argument('--zerr-catalog', default='metadata/desi-galaxy-cat-zerr.fits',
                      help='Path to FITS catalog with targetid and zerr columns')
    data.add_argument('--n-spectra', type=int, default=None,
                      help='Cap dataset at this many spectra')
    data.add_argument('--zmin-loader', type=float, default=None,
                      help='Minimum redshift cut when loading spectra (defaults to --zmin)')
    data.add_argument('--zmax-loader', type=float, default=None,
                      help='Maximum redshift cut when loading spectra (defaults to --zmax)')
    data.add_argument('--shape-nofz', action='store_true',
                      help='Rejection-sample galaxies to shape the loaded n(z) to '
                           '(z/z0)^alpha * exp(-(z/z0)^beta), normalised to peak=1.')
    data.add_argument('--nofz-z0', type=float, default=0.88,
                      help='z0 parameter for --shape-nofz target distribution')
    data.add_argument('--nofz-alpha', type=float, default=40.0,
                      help='alpha parameter for --shape-nofz target distribution')
    data.add_argument('--nofz-beta', type=float, default=40.0,
                      help='beta parameter for --shape-nofz target distribution')
    data.add_argument('--noise-mult', type=float, default=0.0,
                      help='Add noise per batch to bring each spectrum down to this '
                           'total SNR = sqrt(sum(flux^2*ivar)).  Spectra already '
                           'below this SNR are left unchanged.  0 = disabled.')

    # ---- Model ---------------------------------------------------------------
    model_g = p.add_argument_group("Model")
    model_g.add_argument('--zmin', type=float, default=0.4,
                         help='Minimum redshift for template grid and z search')
    data.add_argument('--desi-target-mask', type=int, default=None,
                      help='Keep spectra where desi_target & mask != 0')

    model_g.add_argument('--zmax', type=float, default=1.1,
                         help='Maximum redshift for template grid and z search')
    model_g.add_argument('--Nt', type=int, default=5,
                         help='Number of spectral templates')
    model_g.add_argument('--Nz', type=int, default=200,
                         help='Number of points in the z search grid')
    model_g.add_argument('--Nnz', type=int, default=50,
                         help='Number of bins in the n(z) distribution')
    model_g.add_argument('--template-res-boost', type=float, default=1.0,
                         help='Divide the template pixel spacing by this factor, '
                              'increasing template resolution. 1.0 = native pixel scale.')
    # Neural template options
    model_g.add_argument('--neural-templates', action='store_true',
                         help='Use NeuralTemplateModel: MLP continuum + explicit spectral '
                              'lines + a normalizing-flow prior on template amplitudes.')
    model_g.add_argument('--mlp-hidden', type=int, default=64,
                         help='Hidden layer width of the continuum MLP '
                              '(only used with --neural-templates).')
    model_g.add_argument('--alpha-flow-layers', type=int, default=4,
                         help='Number of affine coupling layers in the alpha prior flow.')
    model_g.add_argument('--alpha-flow-hidden', type=int, default=64,
                         help='Hidden width of the alpha-flow conditioner MLP.')
    model_g.add_argument('--alpha-prior-weight', type=float, default=0.1,
                         help='Weight on log p_flow(alpha) added to each per-z log-weight. '
                              'Set 0.0 to disable the alpha prior.')
    model_g.add_argument('--line-noise-init', type=float, default=0.01,
                         help='Std of the random noise for non-[OII] line amplitudes at init. '
                              'The [OII] doublet is always initialised to 1.0. '
                              '(only used with --neural-templates).')

    model_g.add_argument('--nz-sigma', type=float, default=0.4,
                         help='Width of initial n(z) Gaussian as a fraction of the z range. '
                              'E.g. 0.05 with z=[0.4,1.1] gives sigma=0.035 in redshift. '
                              'Set 0 for uniform initialisation.')

    # ---- Training ------------------------------------------------------------
    train = p.add_argument_group("Training")
    train.add_argument('--n-epochs', type=int, default=20, help='Number of training epochs')
    train.add_argument('--freeze-templates', action='store_true',
                       help='Hold templates fixed; only optimise n(z) and the alpha prior. '
                            'Precomputes and reuses the full chi2/alpha grid.')
    train.add_argument('--nz-steps', type=int, default=0,
                       help='Extra n(z)-only gradient steps per batch after the main step. '
                            'Reuses cached chi2 values when templates are frozen; otherwise '
                            'reuses the chi2 matrix from the main step. Use with --nz-lr '
                            'for a separate rate.')
    train.add_argument('--nz-lr', type=float, default=None,
                       help='Adam learning rate for the extra n(z)-only steps '
                            '(defaults to --lr when not set).')
    train.add_argument('--t0-init', choices=['mean_flux', 'flat'], default='mean_flux',
                       help='Initialisation for template 0: mean_flux (default) or '
                            'flat (1.0 + noise, avoids edge artefacts)')
    train.add_argument('--template-l2', type=float, default=0.0,
                       help='L2 regularisation on templates. Pushes unconstrained '
                            'edge regions to zero. Try 1e-4 to 1e-2.')
    train.add_argument('--template-ortho', type=float, default=0.0,
                       help='Orthogonality regularisation weight. Penalises pairwise '
                            'correlations between templates to prevent degeneracy. '
                            'Try 0.1 to 1.0.')
    train.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    train.add_argument('--batch-size', type=int, default=256, help='Batch size (fixed via drop_last)')
    train.add_argument('--seed', type=int, default=42, help='Random seed')
    train.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader worker processes')

    # ---- Redshift prior ------------------------------------------------------
    prior = p.add_argument_group("Redshift prior")
    prior.add_argument('--zerr-override', type=float, default=None,
                       help='Override catalog zerr with this value for all galaxies. '
                            'Use e.g. 1.0 for an uninformative prior.')
    prior.add_argument('--zerr-floor', type=float, default=1e-4,
                       help='Minimum zerr (guards against division by zero)')
    prior.add_argument('--disable-z-prior', type=float, default=0.0,
                       metavar='FRACTION',
                       help='Fraction of galaxies per batch whose z prior is disabled '
                            '(z_prior set to (zmin+zmax)/2, zerr=100). '
                            '0.0 = all use catalog prior; 1.0 = all disabled. '
                            'E.g. 0.99 disables 99%% of galaxies each step.')
    prior.add_argument('--z-prior-warmup', type=int, default=0,
                       metavar='EPOCHS',
                       help='Linearly ramp blind fraction from 0 → --disable-z-prior '
                            'over this many epochs. Prevents templates settling into '
                            'wrong redshift convention before anchoring takes effect.')

    # ---- Checkpointing -------------------------------------------------------
    ckpt = p.add_argument_group("Checkpoints")
    ckpt.add_argument('--checkpoint-dir', default='checkpoints',
                      help='Directory to save checkpoints')
    ckpt.add_argument('--checkpoint-interval', type=int, default=5,
                      help='Save every N epochs')
    ckpt.add_argument('--resume', default=None,
                      help='Path to .npz checkpoint to resume from')
    ckpt.add_argument('--resume-templates-only', default=None,
                      metavar='CHECKPOINT',
                      help='Load only the T array from this checkpoint; '
                           'reinitialise log_nz_raw fresh (using --nz-sigma). '
                           'Starts from epoch 0. Useful with --freeze-templates '
                           'to optimise n(z) on a new dataset with fixed templates.')

    # ---- Logging -------------------------------------------------------------
    log = p.add_argument_group("Logging")
    log.add_argument('--log-interval', type=int, default=10,
                     help='Print loss every N gradient steps')

    return p.parse_args()


def _compute_flux_mean(ds: SpectraDataset, n_sample: int = 500) -> np.ndarray:
    """Compute mean flux over a random subset (for template initialisation)."""
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=min(n_sample, len(ds)), replace=False)
    fluxes = [ds[int(i)]['flux'].numpy() for i in indices]
    return np.mean(fluxes, axis=0)


def _compute_snr_stats(ds: SpectraDataset, label: str = "SNR stats",
                       n_sample: int = 2000) -> None:
    """Sample spectra and print SNR statistics. Respects ds._noise_mult."""
    rng = np.random.default_rng(0)
    indices = rng.choice(len(ds), size=min(n_sample, len(ds)), replace=False)
    snrs = []
    for i in indices:
        item = ds[int(i)]
        flux = item['flux'].numpy()
        ivar = item['ivar'].numpy()
        w = np.where(ivar>0)[0]
        Np = len(w)
        snr2 = (flux**2 * ivar)[w].sum() - Np
        snr = np.sqrt(snr2) if snr2 > 0 else 0.0
        snrs.append(float(snr))
    snrs = np.array(snrs)
    print(f"  {label} ({len(snrs)} spectra): "
          f"min={snrs.min():.1f}  median={np.median(snrs):.1f}  "
          f"mean={snrs.mean():.1f}  max={snrs.max():.1f}")


def _get_zerr(batch: dict, args: argparse.Namespace) -> jax.Array:
    """Return zerr JAX array for the current batch, applying override/floor."""
    B = batch['flux'].shape[0]
    if args.zerr_override is not None:
        zerr = jnp.full((B,), args.zerr_override)
    elif 'zerr' in batch:
        zerr = jnp.array(batch['zerr'].numpy())
    else:
        # No zerr provided and no override → fall back to uninformative prior
        print("Warning: no zerr in batch and no --zerr-override; using zerr=1.0")
        zerr = jnp.ones((B,))
    return jnp.maximum(zerr, args.zerr_floor)


def _get_zerr_numpy(batch: dict, args: argparse.Namespace) -> np.ndarray:
    """NumPy equivalent of _get_zerr for precomputed frozen-template batches."""
    B = batch['flux'].shape[0]
    if args.zerr_override is not None:
        zerr = np.full((B,), args.zerr_override, dtype=np.float32)
    elif 'zerr' in batch:
        zerr = batch['zerr'].numpy().astype(np.float32)
    else:
        print("Warning: no zerr in batch and no --zerr-override; using zerr=1.0")
        zerr = np.ones((B,), dtype=np.float32)
    return np.maximum(zerr, args.zerr_floor).astype(np.float32)


def _make_mask_like(tree, keep_keys: set[str]) -> dict:
    """Build an optax mask pytree that keeps the requested top-level keys trainable."""
    mask = {}
    for key, value in tree.items():
        if key in keep_keys:
            if isinstance(value, dict):
                mask[key] = jax.tree_util.tree_map(lambda _: True, value)
            elif isinstance(value, list):
                mask[key] = jax.tree_util.tree_map(lambda _: True, value)
            elif isinstance(value, tuple):
                mask[key] = jax.tree_util.tree_map(lambda _: True, value)
            else:
                mask[key] = True
        else:
            if isinstance(value, dict):
                mask[key] = jax.tree_util.tree_map(lambda _: False, value)
            elif isinstance(value, list):
                mask[key] = jax.tree_util.tree_map(lambda _: False, value)
            elif isinstance(value, tuple):
                mask[key] = jax.tree_util.tree_map(lambda _: False, value)
            else:
                mask[key] = False
    return mask


def _load_dataset_batch(ds: SpectraDataset, indices: np.ndarray) -> dict[str, torch.Tensor]:
    """Materialise a dataset chunk into the same structure produced by DataLoader."""
    items = [ds[int(i)] for i in indices]
    batch: dict[str, torch.Tensor] = {
        'flux': torch.stack([item['flux'] for item in items]),
        'ivar': torch.stack([item['ivar'] for item in items]),
        'z': torch.stack([item['z'] for item in items]),
    }
    if 'zerr' in items[0]:
        batch['zerr'] = torch.stack([item['zerr'] for item in items])
    return batch


def _precompute_frozen_cache(
    ds: SpectraDataset,
    model,
    params: dict,
    args: argparse.Namespace,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    """Precompute chi2(z) and alpha(z) once when templates are frozen."""
    n_obj = len(ds)
    chi2_cache = np.empty((n_obj, model.Nz), dtype=np.float32)
    alpha_cache = np.empty((n_obj, model.Nz, model.Nt), dtype=np.float32)
    z_cache = np.empty((n_obj,), dtype=np.float32)
    zerr_cache = np.empty((n_obj,), dtype=np.float32)
    compute_cache = jax.jit(model.compute_chi2_alpha_grid)

    print(f"Precomputing frozen-template chi2/alpha cache ({n_obj} spectra)...")
    for start in range(0, n_obj, chunk_size):
        stop = min(start + chunk_size, n_obj)
        indices = np.arange(start, stop)
        batch = _load_dataset_batch(ds, indices)
        flux = jnp.array(batch['flux'].numpy())
        ivar = jnp.array(batch['ivar'].numpy())
        chi2_chunk, alpha_chunk = compute_cache(params, flux, ivar)
        chi2_cache[start:stop] = np.array(chi2_chunk)
        alpha_cache[start:stop] = np.array(alpha_chunk)
        z_cache[start:stop] = batch['z'].numpy().astype(np.float32)
        zerr_cache[start:stop] = _get_zerr_numpy(batch, args)
        if start == 0 or stop == n_obj or ((start // chunk_size + 1) % 10 == 0):
            print(f"  cached {stop}/{n_obj}")

    return {
        'chi2': chi2_cache,
        'alpha': alpha_cache,
        'z': z_cache,
        'zerr': zerr_cache,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"JAX devices: {jax.devices()}")

    # ---- Build dataset -------------------------------------------------------
    # Loader cuts default to the model search grid range when not set explicitly.
    zmin_loader = args.zmin_loader if args.zmin_loader is not None else args.zmin
    zmax_loader = args.zmax_loader if args.zmax_loader is not None else args.zmax

    print("Loading dataset...")
    ds = SpectraDataset(
        args.spectra_dir,
        zerr_catalog=args.zerr_catalog,
        n_spectra=args.n_spectra,
        zmin=zmin_loader,
        zmax=zmax_loader,
        desi_target_mask=args.desi_target_mask,
        shape_nofz=args.shape_nofz,
        nofz_z0=args.nofz_z0,
        nofz_alpha=args.nofz_alpha,
        nofz_beta=args.nofz_beta,
        seed=args.seed,
        noise_mult=args.noise_mult,
    )
    print(f"  {len(ds)} spectra | loader z in [{zmin_loader}, {zmax_loader}]")
    if args.noise_mult > 0:
        ds._noise_mult = 0.0
        _compute_snr_stats(ds, label="SNR stats (raw)")
        ds._noise_mult = args.noise_mult
        print(f"  noise_mult={args.noise_mult:.1f}")
        _compute_snr_stats(ds, label="SNR stats (noisy)")
    else:
        _compute_snr_stats(ds)

    if len(ds) < args.batch_size:
        raise ValueError(
            f"Dataset ({len(ds)} spectra) is smaller than batch_size ({args.batch_size}). "
            "Lower --batch-size or increase --n-spectra."
        )

    wave_obs = ds.wave.numpy()

    # ---- Build model ---------------------------------------------------------
    if args.neural_templates:
        from neural_template_model import NeuralTemplateModel
        model = NeuralTemplateModel(
            Nt=args.Nt,
            wave_obs=wave_obs,
            zmin=args.zmin,
            zmax=args.zmax,
            Nz=args.Nz,
            Nnz=args.Nnz,
            template_res_boost=args.template_res_boost,
            mlp_hidden=args.mlp_hidden,
            alpha_flow_layers=args.alpha_flow_layers,
            alpha_flow_hidden=args.alpha_flow_hidden,
            alpha_prior_weight=args.alpha_prior_weight,
        )
        print(f"  NeuralTemplateModel: MLP hidden={args.mlp_hidden}, "
              f"flow layers={args.alpha_flow_layers}, flow hidden={args.alpha_flow_hidden}")
        print(f"  Active lines ({model.N_lines}): {', '.join(model.active_line_names)}")
    else:
        model = TemplateModel(
            Nt=args.Nt,
            wave_obs=wave_obs,
            zmin=args.zmin,
            zmax=args.zmax,
            Nz=args.Nz,
            Nnz=args.Nnz,
            template_res_boost=args.template_res_boost,
            alpha_flow_layers=args.alpha_flow_layers,
            alpha_flow_hidden=args.alpha_flow_hidden,
            alpha_prior_weight=args.alpha_prior_weight,
        )
    print(f"  Template grid: {model.Nft} pts, "
          f"{float(model.t_wave[0]):.1f}–{float(model.t_wave[-1]):.1f} Å")
    print(f"  Search grid:   Nz={args.Nz}, z in [{args.zmin}, {args.zmax}]")
    if zmin_loader != args.zmin or zmax_loader != args.zmax:
        print(f"  (loader cut:   z in [{zmin_loader}, {zmax_loader}])")
    print(f"  n(z) bins:     Nnz={args.Nnz}")
    print(f"  Alpha flow prior: layers={model.alpha_flow_layers}, "
          f"hidden={model.alpha_flow_hidden}, weight={model.alpha_prior_weight}")

    # ---- Initialise or resume ------------------------------------------------
    key = jax.random.PRNGKey(args.seed)
    start_epoch = 0

    if args.resume is not None:
        print(f"Resuming from {args.resume}")
        if args.neural_templates:
            _, params, start_epoch = NeuralTemplateModel.load_checkpoint(args.resume, wave_obs)
        else:
            _, params, start_epoch = TemplateModel.load_checkpoint(args.resume, wave_obs)
        start_epoch += 1
        print(f"  Continuing from epoch {start_epoch}")
    elif args.resume_templates_only is not None:
        print(f"Loading templates from {args.resume_templates_only} (n(z) reinitialised)")
        if args.neural_templates:
            _, ckpt_params, ckpt_epoch = NeuralTemplateModel.load_checkpoint(
                args.resume_templates_only, wave_obs
            )
            fresh = model.init_params(key, nz_sigma=args.nz_sigma)
            params = {
                'mlp_weights':    ckpt_params['mlp_weights'],
                'line_A':         ckpt_params['line_A'],
                'line_sigma_raw': ckpt_params['line_sigma_raw'],
                'log_nz_raw':     fresh['log_nz_raw'],
                'alpha_flow':     fresh['alpha_flow'],
            }
        else:
            _, ckpt_params, ckpt_epoch = TemplateModel.load_checkpoint(
                args.resume_templates_only, wave_obs
            )
            fresh = model.init_params(key, nz_sigma=args.nz_sigma)
            params = {
                "T": ckpt_params["T"],
                "log_nz_raw": fresh["log_nz_raw"],
                "alpha_flow": fresh["alpha_flow"],
            }
        print(f"  Templates from epoch {ckpt_epoch}, n(z) fresh (nz_sigma={args.nz_sigma})")
    else:
        if args.neural_templates:
            params = model.init_params(key, nz_sigma=args.nz_sigma,
                                       line_amp_init=args.line_noise_init)
        else:
            if args.t0_init == 'mean_flux':
                print("Computing flux mean for template initialisation...")
                flux_mean = _compute_flux_mean(ds)
            else:
                flux_mean = None
            params = model.init_params(key, flux_mean=flux_mean, t0_init=args.t0_init,
                                       nz_sigma=args.nz_sigma)

    if args.freeze_templates:
        cache_chunk_size = max(args.batch_size, 64)
        frozen_cache = _precompute_frozen_cache(ds, model, params, args, cache_chunk_size)
        train_mask = _make_mask_like(params, {'log_nz_raw', 'alpha_flow'})
        optimizer = optax.masked(optax.adam(args.lr), train_mask)
    else:
        frozen_cache = None
        optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(params)

    if args.nz_steps > 0:
        nz_lr = args.nz_lr if args.nz_lr is not None else args.lr
        nz_mask = _make_mask_like(params, {'log_nz_raw'})
        nz_optimizer = optax.masked(optax.adam(nz_lr), nz_mask)
        nz_opt_state = nz_optimizer.init(params)

    # ---- JIT-compiled training step ------------------------------------------
    template_l2 = args.template_l2
    template_ortho = args.template_ortho

    if args.freeze_templates:
        @jax.jit
        def train_step(params, opt_state, chi2_matrix, alpha_matrix, z_prior, zerr):
            loss_val, grads = jax.value_and_grad(model.alpha_nz_loss)(
                params, chi2_matrix, alpha_matrix, z_prior, zerr
            )
            updates, new_opt_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val
    else:
        @jax.jit
        def train_step(params, opt_state, flux, ivar, z_prior, zerr):
            loss_val, grads = jax.value_and_grad(model.loss)(
                params, flux, ivar, z_prior, zerr, template_l2, template_ortho
            )
            updates, new_opt_state = optimizer.update(grads, opt_state)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_opt_state, loss_val

    # ---- Extra n(z) step functions (only built when needed) ------------------
    if args.nz_steps > 0:
        if not args.freeze_templates:
            @jax.jit
            def compute_chi2(params, flux, ivar):
                return model.compute_chi2_matrix(params, flux, ivar)

        @jax.jit
        def nz_step(params, nz_opt_state, chi2_matrix, z_prior, zerr):
            loss_val, grads = jax.value_and_grad(model.nz_loss)(
                params, chi2_matrix, z_prior, zerr
            )
            updates, new_state = nz_optimizer.update(grads, nz_opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state, loss_val

    # ---- Training loop -------------------------------------------------------
    if args.freeze_templates:
        n_steps_per_epoch = len(ds) // args.batch_size
    else:
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=False,
        )
        n_steps_per_epoch = len(loader)

    print(f"\nTraining: {args.n_epochs - start_epoch} epochs, "
          f"{n_steps_per_epoch} steps/epoch, batch={args.batch_size}")
    if args.freeze_templates:
        print("  freeze_templates=True  (templates fixed; reusing cached chi2/alpha grids)")
    if args.nz_steps > 0:
        nz_lr_eff = args.nz_lr if args.nz_lr is not None else args.lr
        print(f"  nz_steps={args.nz_steps}  nz_lr={nz_lr_eff}")
    print("(First step triggers JAX JIT compilation — may be slow)\n")

    total_steps = 0
    for epoch in range(start_epoch, args.n_epochs):
        epoch_losses: list[float] = []
        t0 = time.time()

        # Curriculum: ramp blind fraction from 0 → target over warmup epochs
        if args.disable_z_prior > 0.0 and args.z_prior_warmup > 0:
            t = min(epoch / args.z_prior_warmup, 1.0)
            blind_frac = args.disable_z_prior * t
        else:
            blind_frac = args.disable_z_prior
        print(f"  blind_frac={blind_frac:.3f}")

        if args.freeze_templates:
            n_full = (len(ds) // args.batch_size) * args.batch_size
            perm = np.random.permutation(len(ds))[:n_full].reshape(-1, args.batch_size)
            for batch_idx in perm:
                chi2_mat = jnp.array(frozen_cache['chi2'][batch_idx])
                alpha_mat = jnp.array(frozen_cache['alpha'][batch_idx])
                z_prior = jnp.array(frozen_cache['z'][batch_idx])
                zerr = jnp.array(frozen_cache['zerr'][batch_idx])
                if blind_frac > 0.0:
                    mask = np.random.random(args.batch_size) < blind_frac
                    z_center = (args.zmin + args.zmax) / 2.0
                    z_prior = jnp.where(mask, z_center, z_prior)
                    zerr = jnp.where(mask, 100.0, zerr)

                params, opt_state, loss_val = train_step(
                    params, opt_state, chi2_mat, alpha_mat, z_prior, zerr
                )
                loss_scalar = float(loss_val)

                if args.nz_steps > 0:
                    for _ in range(args.nz_steps):
                        params, nz_opt_state, _ = nz_step(
                            params, nz_opt_state, chi2_mat, z_prior, zerr
                        )
                epoch_losses.append(loss_scalar)

                if total_steps % args.log_interval == 0:
                    print(f"  epoch {epoch:3d} step {total_steps:5d}:  loss={loss_scalar:.4f}")
                total_steps += 1
        else:
            for batch in loader:
                flux = jnp.array(batch['flux'].numpy())
                ivar = jnp.array(batch['ivar'].numpy())
                z_prior = jnp.array(batch['z'].numpy())
                zerr = _get_zerr(batch, args)
                if blind_frac > 0.0:
                    mask = np.random.random(args.batch_size) < blind_frac
                    z_center = (args.zmin + args.zmax) / 2.0
                    z_prior = jnp.where(mask, z_center, z_prior)
                    zerr = jnp.where(mask, 100.0, zerr)

                params, opt_state, loss_val = train_step(
                    params, opt_state, flux, ivar, z_prior, zerr
                )
                loss_scalar = float(loss_val)

                if args.nz_steps > 0:
                    chi2_mat = compute_chi2(params, flux, ivar)
                    for _ in range(args.nz_steps):
                        params, nz_opt_state, _ = nz_step(
                            params, nz_opt_state, chi2_mat, z_prior, zerr
                        )
                epoch_losses.append(loss_scalar)

                if total_steps % args.log_interval == 0:
                    print(f"  epoch {epoch:3d} step {total_steps:5d}:  loss={loss_scalar:.4f}")
                total_steps += 1

        elapsed = time.time() - t0
        mean_loss = float(np.mean(epoch_losses))
        print(f"Epoch {epoch:3d}:  mean loss={mean_loss:.4f}  ({elapsed:.1f}s)")

        should_save = (
            ((epoch + 1) % args.checkpoint_interval == 0)
            or (epoch == args.n_epochs - 1)
        )
        if should_save:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch{epoch:04d}.npz"
            )
            model.save_checkpoint(params, epoch, ckpt_path)
            print(f"  → Saved {ckpt_path}")

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
