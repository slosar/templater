"""
Neural spectral template model for DESI galaxy spectra.

Architecture
------------
Each template k is:

    T_k(λ_rest) = MLP(λ_norm)[k]  +  Σ_j  A_{kj} · exp(-(λ_rest - λ_j)² / 2σ_j²)

where:
  - MLP  : shared-trunk, Nt-output MLP that models the smooth continuum per template.
             Input is normalised rest-frame wavelength.  He-initialised, GELU activations.
  - λ_j  : fixed rest-frame line positions (known physics, accurate wavelengths).
  - A_{kj}: learnable amplitude of line j in template k (positive = emission, negative = absorption).
  - σ_j  : learnable per-line width (softplus + 1 Å floor), shared across templates.

A Gaussian Mixture Model (GMM) on the amplitude vector α acts as a learned prior on
galaxy type, adding  gmm_weight · log p_GMM(α)  to the per-galaxy per-z log-weight.
This penalises unlikely combinations of templates and encourages discrete galaxy populations
to emerge.

The z-marginalisation, n(z) learning, and analytical α solve are unchanged from TemplateModel.
"""

from __future__ import annotations

import json
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Spectral line catalog — rest-frame wavelengths (Å)
# ---------------------------------------------------------------------------

LINE_WAVELENGTHS = np.array([
    3727.092, 3729.875,          # [OII] doublet
    3869.86,                     # [NeIII]
    3889.05,                     # H8 / HeI blend
    3933.664, 3968.470,          # Ca K, Ca H (absorption)
    4101.734,                    # Hδ
    4304.4,                      # G-band (Ca+Fe+CH molecular, absorption)
    4340.464,                    # Hγ
    4861.325,                    # Hβ
    4958.911, 5006.843,          # [OIII] doublet
    5175.4,                      # Mg b (absorption triplet, effective centre)
    5889.950, 5895.924,          # Na D doublet (absorption)
    6300.304,                    # [OI]
    6548.050, 6562.801, 6583.460,  # [NII], Hα, [NII]
    6716.440, 6730.816,          # [SII] doublet
], dtype=np.float32)

LINE_NAMES = [
    '[OII]3727', '[OII]3730',
    '[NeIII]3870', 'H8',
    'CaK', 'CaH',
    'Hd', 'G-band', 'Hg', 'Hb',
    '[OIII]4959', '[OIII]5007',
    'Mgb',
    'NaD1', 'NaD2',
    '[OI]6300',
    '[NII]6548', 'Ha', '[NII]6583',
    '[SII]6716', '[SII]6731',
]

# ---------------------------------------------------------------------------
# MLP helpers
# ---------------------------------------------------------------------------

def _mlp_init(key: jax.Array, layer_sizes: list[int]) -> list[tuple]:
    """He-initialise an MLP.  Returns list of (W, b) tuples."""
    weights = []
    for i in range(len(layer_sizes) - 1):
        key, k = jax.random.split(key)
        fan_in = layer_sizes[i]
        W = jax.random.normal(k, (fan_in, layer_sizes[i + 1])) * jnp.sqrt(2.0 / fan_in)
        b = jnp.zeros(layer_sizes[i + 1])
        weights.append((W, b))
    return weights


def _mlp_apply(weights: list[tuple], x: jax.Array) -> jax.Array:
    """Forward pass with GELU activations on all but the last layer.

    Parameters
    ----------
    weights : list of (W, b)
    x       : (batch, fan_in) or (fan_in,)

    Returns
    -------
    (batch, fan_out) or (fan_out,)
    """
    for i, (W, b) in enumerate(weights):
        x = x @ W + b
        if i < len(weights) - 1:
            x = jax.nn.gelu(x)
    return x


# ---------------------------------------------------------------------------
# GMM helpers
# ---------------------------------------------------------------------------

def _make_L(L_raw: jax.Array) -> jax.Array:
    """Convert unconstrained (K, Nt, Nt) → lower-triangular with positive diagonal.

    Off-diagonal elements are zeroed; diagonal is passed through softplus + 1e-3.
    """
    Nt = L_raw.shape[-1]
    eye = jnp.eye(Nt)
    L = jnp.tril(L_raw)                                         # zero upper triangle
    diag_raw = (L_raw * eye[None]).sum(axis=-1)                  # (K, Nt) raw diagonal
    diag_pos = jax.nn.softplus(diag_raw) + 1e-3                  # (K, Nt) positive
    # Replace diagonal: off-diagonal from L, diagonal from diag_pos
    diag_mat = jnp.einsum('ki,ij->kij', diag_pos, eye)           # (K, Nt, Nt)
    off_mask = (1.0 - eye)[None]                                  # (1, Nt, Nt)
    return L * off_mask + diag_mat


def _gmm_log_prob(
    alpha: jax.Array,
    log_pi: jax.Array,
    mu: jax.Array,
    L_raw: jax.Array,
) -> jax.Array:
    """Log probability of alpha under a GMM with Cholesky-parameterised covariances.

    Parameters
    ----------
    alpha  : (Nt,)
    log_pi : (K,)  unnormalised log mixture weights
    mu     : (K, Nt)
    L_raw  : (K, Nt, Nt)  unconstrained → L via _make_L (Σ_k = L_k @ L_k^T)

    Returns
    -------
    Scalar log p(alpha).
    """
    K, Nt = mu.shape
    log_pi_norm = jax.nn.log_softmax(log_pi)    # (K,)
    L = _make_L(L_raw)                           # (K, Nt, Nt)

    def component_lp(log_pi_k, mu_k, L_k):
        diff = alpha - mu_k                      # (Nt,)
        # Solve L_k z = diff  →  z = L_k^{-1} diff,  Mahalanobis = ||z||²
        z = jax.scipy.linalg.solve_triangular(L_k, diff, lower=True)
        mahal = jnp.sum(z ** 2)
        log_det_L = jnp.sum(jnp.log(jnp.diag(L_k)))   # log|L| = Σ log diag(L)
        # log N(alpha; mu_k, L_k L_k^T)
        return log_pi_k - 0.5 * (Nt * jnp.log(2.0 * jnp.pi) + 2.0 * log_det_L + mahal)

    log_components = jax.vmap(component_lp)(log_pi_norm, mu, L)  # (K,)
    return jax.nn.logsumexp(log_components)


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class NeuralTemplateModel:
    """Spectral template model with MLP continuum + explicit spectral lines.

    Parameters
    ----------
    Nt               : Number of spectral templates.
    wave_obs         : (Nf,) observed wavelength grid in Å.
    zmin, zmax       : Redshift search range.
    Nz               : Points in z search grid.
    Nnz              : Bins in the learnable n(z) distribution.
    template_res_boost: Divide template pixel spacing by this factor.
    mlp_hidden       : Hidden layer width of the continuum MLP.
    gmm_components   : Number of GMM components for the α prior.
    gmm_weight       : Weight on log p_GMM(α) added to per-galaxy log-weight.
                       Set to 0.0 to disable the GMM prior entirely.
    """

    def __init__(
        self,
        Nt: int,
        wave_obs: np.ndarray,
        zmin: float,
        zmax: float,
        Nz: int,
        Nnz: int = 50,
        template_res_boost: float = 1.2,
        mlp_hidden: int = 64,
        gmm_components: int = 5,
        gmm_weight: float = 0.1,
    ) -> None:
        self.Nt = Nt
        self.Nf = int(len(wave_obs))
        self.Nz = Nz
        self.Nnz = Nnz
        self.zmin = float(zmin)
        self.zmax = float(zmax)
        self.wave_min = float(wave_obs.min())
        self.wave_max = float(wave_obs.max())
        self.template_res_boost = float(template_res_boost)
        self.mlp_hidden = int(mlp_hidden)
        self.gmm_components = int(gmm_components)
        self.gmm_weight = float(gmm_weight)

        # ---- Template rest-frame wavelength grid ----------------------------
        t_wave_min = self.wave_min / (1.0 + zmax)
        t_wave_max = self.wave_max / (1.0 + zmin)
        delta_wave = ((float(wave_obs[-1]) - float(wave_obs[0]))
                      / (len(wave_obs) - 1) / template_res_boost)
        self.Nft = int(round((t_wave_max - t_wave_min) / delta_wave)) + 1
        t_wave = np.linspace(t_wave_min, t_wave_max, self.Nft).astype(np.float32)
        self.delta_wave = float(delta_wave)   # template pixel spacing in Å

        # ---- z search grid -------------------------------------------------
        zgrid = np.linspace(zmin, zmax, Nz).astype(np.float32)

        # ---- Precompute interpolation weights (same as TemplateModel) -------
        rest_waves = wave_obs[None, :].astype(np.float32) / (1.0 + zgrid[:, None])
        lo = (np.searchsorted(t_wave, rest_waves, side='right') - 1).astype(np.int32)
        lo = np.clip(lo, 0, self.Nft - 2)
        weight = ((rest_waves - t_wave[lo]) /
                  (t_wave[lo + 1] - t_wave[lo])).astype(np.float32)
        weight = np.clip(weight, 0.0, 1.0)

        # ---- n(z) grid and interpolation weights ----------------------------
        z_nz_grid = np.linspace(zmin, zmax, Nnz).astype(np.float32)
        nz_lo = (np.searchsorted(z_nz_grid, zgrid, side='right') - 1).astype(np.int32)
        nz_lo = np.clip(nz_lo, 0, Nnz - 2)
        nz_weight = ((zgrid - z_nz_grid[nz_lo]) /
                     (z_nz_grid[nz_lo + 1] - z_nz_grid[nz_lo])).astype(np.float32)
        nz_weight = np.clip(nz_weight, 0.0, 1.0)

        # ---- Filter line catalog to template rest-frame range ---------------
        line_mask = (LINE_WAVELENGTHS >= t_wave_min) & (LINE_WAVELENGTHS <= t_wave_max)
        self.active_lines = LINE_WAVELENGTHS[line_mask]
        self.active_line_names = [n for n, m in zip(LINE_NAMES, line_mask) if m]
        self.N_lines = int(line_mask.sum())

        # ---- MLP configuration ----------------------------------------------
        # Input: 1 (normalised λ), hidden layers, output: Nt (one value per template)
        self.mlp_layer_sizes = [1, mlp_hidden, mlp_hidden, Nt]

        # Wavelength normalisation: map t_wave range to approximately [-1, 1]
        self.wave_norm_mean = float((t_wave_min + t_wave_max) / 2.0)
        self.wave_norm_scale = float((t_wave_max - t_wave_min) / 2.0)

        # Store as JAX arrays (captured as compile-time constants in jit closures)
        self.t_wave       = jnp.array(t_wave)
        self.wave_obs     = jnp.array(wave_obs, dtype=jnp.float32)
        self.zgrid        = jnp.array(zgrid)
        self.z_nz_grid    = jnp.array(z_nz_grid)
        self.lo_idx       = jnp.array(lo)          # (Nz, Nf) int32
        self.weight       = jnp.array(weight)      # (Nz, Nf) float32
        self.nz_lo_idx    = jnp.array(nz_lo)       # (Nz,)    int32
        self.nz_weight    = jnp.array(nz_weight)   # (Nz,)    float32
        self.line_wav_jax = jnp.array(self.active_lines)   # (N_lines,)

    # -------------------------------------------------------------------------
    # Parameter initialisation
    # -------------------------------------------------------------------------

    def init_params(
        self,
        key: jax.Array,
        flux_mean: Optional[np.ndarray] = None,   # accepted for API compat, not used
        nz_sigma: float = 0.05,
        line_amp_init: float = 0.1,
    ) -> dict:
        """Return initial parameter pytree.

        Keys
        ----
        mlp_weights    : list of (W, b) tuples — continuum MLP
        line_A         : (Nt, N_lines) — line amplitudes
        line_sigma_raw : (N_lines,) — unconstrained line widths (σ = softplus + 0.1 Å)
        log_nz_raw     : (Nnz,) — unnormalised log n(z)
        gmm_log_pi     : (K,) — unnormalised log GMM mixture weights
        gmm_mu         : (K, Nt) — GMM component means
        gmm_L_raw      : (K, Nt, Nt) — unconstrained Cholesky factors
        """
        key, k_mlp, k_lines, k_gmm = jax.random.split(key, 4)

        # MLP: He init
        mlp_weights = _mlp_init(k_mlp, self.mlp_layer_sizes)

        # Line amplitudes: start the [OII] doublet (indices 0,1) large and positive
        # so the model anchors on the strongest z-discriminating feature from the start.
        # All other lines initialised as small noise (line_amp_init std).
        line_A = jax.random.normal(k_lines, (self.Nt, self.N_lines)) * line_amp_init
        line_A = line_A.at[0, 0].set(1.0)   # [OII] 3727 — template 0 only
        line_A = line_A.at[0, 1].set(1.0)   # [OII] 3730 — template 0 only

        # Line widths: init to one template-grid pixel (delta_wave).
        # σ = softplus(x) + 0.1;  solve x = softplus_inv(delta_wave - 0.1)
        _target_sp = float(self.delta_wave) - 0.1
        _sigma_raw_init = float(np.log(np.exp(_target_sp) - 1))
        line_sigma_raw = jnp.full((self.N_lines,), _sigma_raw_init)

        # n(z): Gaussian initialisation centred at (zmin+zmax)/2
        z_nz = np.array(self.z_nz_grid)
        z_center = (self.zmin + self.zmax) / 2.0
        if nz_sigma > 0.0:
            sigma = nz_sigma * (self.zmax - self.zmin)
            log_nz_raw = jnp.array(
                -0.5 * ((z_nz - z_center) / sigma) ** 2, dtype=jnp.float32
            )
        else:
            log_nz_raw = jnp.zeros(self.Nnz)

        # GMM: uniform weights, randomly scattered means, identity covariance
        # softplus^{-1}(1.0) = log(e - 1) ≈ 0.5413  →  diagonal starts at 1.0
        # Means must be initialised randomly to break symmetry — if all K components
        # start at the same point they receive identical gradients and never diverge.
        K, Nt = self.gmm_components, self.Nt
        diag_init = float(np.log(np.exp(10.0) - 1.0))  # softplus^{-1}(10.0) → σ≈10
        gmm_log_pi = jnp.zeros(K)
        gmm_mu     = jax.random.normal(k_gmm, (K, Nt)) * 30.0
        gmm_L_raw  = jnp.zeros((K, Nt, Nt)) + diag_init * jnp.eye(Nt)[None]

        return {
            'mlp_weights':    mlp_weights,
            'line_A':         line_A,
            'line_sigma_raw': line_sigma_raw,
            'log_nz_raw':     log_nz_raw,
            'gmm_log_pi':     gmm_log_pi,
            'gmm_mu':         gmm_mu,
            'gmm_L_raw':      gmm_L_raw,
        }

    # -------------------------------------------------------------------------
    # Template construction
    # -------------------------------------------------------------------------

    def _build_templates(self, params: dict) -> jax.Array:
        """Evaluate all Nt templates on the rest-frame grid.

        Returns
        -------
        T : (Nt, Nft)
        """
        # Continuum via MLP: (Nft, 1) → (Nft, Nt)
        lambda_norm = ((self.t_wave - self.wave_norm_mean) / self.wave_norm_scale)[:, None]
        continuum = _mlp_apply(params['mlp_weights'], lambda_norm)   # (Nft, Nt)
        T = continuum.T                                               # (Nt, Nft)

        # Line profiles: Gaussians at known rest-frame positions
        sigma = jax.nn.softplus(params['line_sigma_raw']) + 0.1      # (N_lines,) in Å
        dwave = self.t_wave[:, None] - self.line_wav_jax[None, :]    # (Nft, N_lines)
        profiles = jnp.exp(-0.5 * (dwave / sigma[None, :]) ** 2)    # (Nft, N_lines)

        # T_lines[k, f] = Σ_j A[k,j] * profiles[f,j]
        T_lines = params['line_A'] @ profiles.T                      # (Nt, Nft)

        return T + T_lines

    # -------------------------------------------------------------------------
    # Interpolation (identical to TemplateModel)
    # -------------------------------------------------------------------------

    def _interpolate_templates(self, T: jax.Array) -> jax.Array:
        """Interpolate T onto the observed grid for every z in zgrid.

        Parameters
        ----------
        T : (Nt, Nft)

        Returns
        -------
        T_all : (Nz, Nt, Nf)
        """
        T_lo = T[:, self.lo_idx]       # (Nt, Nz, Nf)
        T_hi = T[:, self.lo_idx + 1]
        w = self.weight                 # (Nz, Nf)
        T_all = T_lo * (1.0 - w) + T_hi * w   # (Nt, Nz, Nf)
        return T_all.transpose(1, 0, 2)         # (Nz, Nt, Nf)

    def _interp_at_z(self, T: jax.Array, z: jax.Array) -> jax.Array:
        """Interpolate T at a single (possibly traced) redshift z.

        Parameters
        ----------
        T : (Nt, Nft)
        z : scalar

        Returns
        -------
        T_z : (Nt, Nf)
        """
        rest_wave = self.wave_obs / (1.0 + z)
        lo = jnp.searchsorted(self.t_wave, rest_wave, side='right') - 1
        lo = jnp.clip(lo, 0, self.Nft - 2)
        w = (rest_wave - self.t_wave[lo]) / (self.t_wave[lo + 1] - self.t_wave[lo])
        w = jnp.clip(w, 0.0, 1.0)
        return T[:, lo] * (1.0 - w) + T[:, lo + 1] * w

    # -------------------------------------------------------------------------
    # Loss
    # -------------------------------------------------------------------------

    def loss(
        self,
        params: dict,
        flux: jax.Array,
        ivar: jax.Array,
        z_prior: jax.Array,
        zerr: jax.Array,
        template_l2: float = 0.0,
        template_ortho: float = 0.0,
    ) -> jax.Array:
        """Negative mean log-likelihood over a batch.

        Parameters
        ----------
        params         : param pytree (see init_params)
        flux           : (B, Nf)
        ivar           : (B, Nf)
        z_prior        : (B,)  catalog redshift
        zerr           : (B,)  catalog redshift uncertainty
        template_l2    : float  L2 regularisation on template values (T matrix)
        template_ortho : float  Orthogonality regularisation weight. Penalises
                                pairwise correlations between templates. Try 0.1–1.0.

        Returns
        -------
        Scalar loss.
        """
        T     = self._build_templates(params)      # (Nt, Nft)
        T_all = self._interpolate_templates(T)     # (Nz, Nt, Nf)

        log_nz_norm = jax.nn.log_softmax(params['log_nz_raw'])
        log_nz_at_z = (
            log_nz_norm[self.nz_lo_idx] * (1.0 - self.nz_weight)
            + log_nz_norm[self.nz_lo_idx + 1] * self.nz_weight
        )   # (Nz,)

        zerr_safe   = jnp.maximum(zerr, 1e-8)[:, None]
        dz          = self.zgrid[None, :] - z_prior[:, None]
        log_prior_z = -0.5 * (dz / zerr_safe) ** 2            # (B, Nz)

        flux_ivar_flux = (flux * ivar * flux).sum(axis=-1)     # (B,)
        Nt     = self.Nt
        eye_Nt = jnp.eye(Nt)
        gmm_weight = self.gmm_weight   # Python float → traced as compile-time constant

        def scan_body(carry, x):
            log_max, sum_exp = carry                 # (B,)
            T_z, log_nz_z, log_prior_z_col = x      # (Nt,Nf), scalar, (B,)

            # Linear solve for optimal alpha per (galaxy, z)
            T_ivar = T_z[None, :, :] * ivar[:, None, :]         # (B, Nt, Nf)
            A = jnp.einsum('bkf,lf->bkl', T_ivar, T_z) + 1e-6 * eye_Nt[None]
            b = (T_ivar * flux[:, None, :]).sum(axis=-1)         # (B, Nt)
            alpha = jax.vmap(jnp.linalg.solve)(A, b)            # (B, Nt)

            # chi2 identity: chi2_min = flux²·ivar - b^T α
            chi2 = flux_ivar_flux - (b * alpha).sum(axis=-1)    # (B,)

            log_w = log_nz_z + log_prior_z_col - 0.5 * chi2    # (B,)

            # GMM prior on alpha (compiled away when gmm_weight == 0.0)
            if gmm_weight > 0.0:
                log_p_alpha = jax.vmap(
                    lambda a: _gmm_log_prob(
                        a, params['gmm_log_pi'], params['gmm_mu'], params['gmm_L_raw']
                    )
                )(alpha)                                          # (B,)
                log_w = log_w + gmm_weight * log_p_alpha

            # Numerically stable online logsumexp
            new_max = jnp.maximum(log_max, log_w)
            new_sum = sum_exp * jnp.exp(log_max - new_max) + jnp.exp(log_w - new_max)
            return (new_max, new_sum), None

        B = flux.shape[0]
        init_carry = (jnp.full((B,), -1e30), jnp.zeros((B,)))
        xs = (T_all, log_nz_at_z, log_prior_z.T)   # log_prior_z.T: (Nz, B)

        (log_max, sum_exp), _ = jax.lax.scan(jax.remat(scan_body), init_carry, xs)

        log_p    = log_max + jnp.log(sum_exp)        # (B,)
        loss_val = -jnp.mean(log_p)

        if template_l2 > 0.0:
            loss_val = loss_val + template_l2 * jnp.mean(T ** 2)

        if template_ortho > 0.0:
            gram = T @ T.T    # (Nt, Nt)
            diag = jnp.diag(gram) + 1e-8
            norm = jnp.sqrt(diag[:, None] * diag[None, :])
            corr = gram / norm   # correlation matrix, entries in [-1, 1]
            off_diag = corr * (1.0 - jnp.eye(self.Nt))
            loss_val = loss_val + template_ortho * jnp.mean(off_diag ** 2)

        return loss_val

    def compute_chi2_matrix(
        self,
        params: dict,
        flux: jax.Array,
        ivar: jax.Array,
    ) -> jax.Array:
        """Per-galaxy chi2_min at every z.  Returns (B, Nz)."""
        T     = self._build_templates(params)
        T_all = self._interpolate_templates(T)
        flux_ivar_flux = (flux * ivar * flux).sum(axis=-1)
        eye_Nt = jnp.eye(self.Nt)

        def scan_body(_, T_z):
            T_ivar = T_z[None, :, :] * ivar[:, None, :]
            A = jnp.einsum('bkf,lf->bkl', T_ivar, T_z) + 1e-6 * eye_Nt[None]
            b = (T_ivar * flux[:, None, :]).sum(axis=-1)
            alpha = jax.vmap(jnp.linalg.solve)(A, b)
            chi2 = flux_ivar_flux - (b * alpha).sum(axis=-1)
            return _, chi2

        _, chi2_mat = jax.lax.scan(jax.remat(scan_body), None, T_all)  # (Nz, B)
        return chi2_mat.T   # (B, Nz)

    def nz_loss(
        self,
        params: dict,
        chi2_matrix: jax.Array,
        z_prior: jax.Array,
        zerr: jax.Array,
    ) -> jax.Array:
        """n(z)-only loss given a precomputed chi2 matrix.

        Gradient flows only through params['log_nz_raw'].
        """
        log_nz_norm = jax.nn.log_softmax(params['log_nz_raw'])
        log_nz_at_z = (
            log_nz_norm[self.nz_lo_idx] * (1.0 - self.nz_weight)
            + log_nz_norm[self.nz_lo_idx + 1] * self.nz_weight
        )
        zerr_safe = jnp.maximum(zerr, 1e-8)[:, None]
        dz = self.zgrid[None, :] - z_prior[:, None]
        log_prior = -0.5 * (dz / zerr_safe) ** 2
        log_w = log_nz_at_z[None, :] + log_prior - 0.5 * chi2_matrix
        return -jnp.mean(jax.nn.logsumexp(log_w, axis=-1))

    # -------------------------------------------------------------------------
    # Inference helpers
    # -------------------------------------------------------------------------

    def compute_z_posterior(
        self,
        params: dict,
        flux: jax.Array,
        ivar: jax.Array,
        z_prior: jax.Array,
        zerr: jax.Array,
    ) -> jax.Array:
        """Unnormalised log-posterior over the z search grid.  Returns (B, Nz)."""
        T     = self._build_templates(params)
        T_all = self._interpolate_templates(T)

        log_nz_norm = jax.nn.log_softmax(params['log_nz_raw'])
        log_nz_at_z = (
            log_nz_norm[self.nz_lo_idx] * (1.0 - self.nz_weight)
            + log_nz_norm[self.nz_lo_idx + 1] * self.nz_weight
        )
        zerr_safe = jnp.maximum(zerr, 1e-8)[:, None]
        dz = self.zgrid[None, :] - z_prior[:, None]
        log_prior = -0.5 * (dz / zerr_safe) ** 2
        flux_ivar_flux = (flux * ivar * flux).sum(axis=-1)
        eye_Nt = jnp.eye(self.Nt)

        def scan_body(_, x):
            T_z, log_nz_z, log_prior_z = x
            T_ivar = T_z[None, :, :] * ivar[:, None, :]
            A = jnp.einsum('bkf,lf->bkl', T_ivar, T_z) + 1e-6 * eye_Nt[None]
            b = (T_ivar * flux[:, None, :]).sum(axis=-1)
            alpha = jax.vmap(jnp.linalg.solve)(A, b)
            chi2  = flux_ivar_flux - (b * alpha).sum(axis=-1)
            return _, log_nz_z + log_prior_z - 0.5 * chi2

        xs = (T_all, log_nz_at_z, log_prior.T)
        _, log_weights = jax.lax.scan(scan_body, None, xs)   # (Nz, B)
        return log_weights.T   # (B, Nz)

    def predict_alpha(
        self,
        params: dict,
        flux: jax.Array,
        ivar: jax.Array,
        z_vals: jax.Array,
    ) -> jax.Array:
        """Optimal linear amplitudes per galaxy at its redshift.  Returns (B, Nt)."""
        T  = self._build_templates(params)
        Nt = self.Nt

        def single(flux_i, ivar_i, z_i):
            T_z   = self._interp_at_z(T, z_i)
            T_ivar = T_z * ivar_i[None, :]
            A = T_ivar @ T_z.T + 1e-6 * jnp.eye(Nt)
            b = (T_ivar * flux_i[None, :]).sum(axis=-1)
            return jnp.linalg.solve(A, b)

        return jax.vmap(single)(flux, ivar, z_vals)

    # -------------------------------------------------------------------------
    # Checkpoint helpers
    # -------------------------------------------------------------------------

    def config(self) -> dict:
        return {
            'model_type':         'neural',
            'Nt':                 self.Nt,
            'Nft':                self.Nft,
            'Nz':                 self.Nz,
            'Nnz':                self.Nnz,
            'zmin':               self.zmin,
            'zmax':               self.zmax,
            'wave_min':           self.wave_min,
            'wave_max':           self.wave_max,
            'Nf':                 self.Nf,
            'template_res_boost': self.template_res_boost,
            'mlp_hidden':         self.mlp_hidden,
            'gmm_components':     self.gmm_components,
            'gmm_weight':         self.gmm_weight,
            'N_lines':            self.N_lines,
        }

    def save_checkpoint(self, params: dict, epoch: int, path: str) -> None:
        """Save params and config to a .npz file."""
        flat: dict = {
            'line_A':         np.array(params['line_A']),
            'line_sigma_raw': np.array(params['line_sigma_raw']),
            'log_nz_raw':     np.array(params['log_nz_raw']),
            'gmm_log_pi':     np.array(params['gmm_log_pi']),
            'gmm_mu':         np.array(params['gmm_mu']),
            'gmm_L_raw':      np.array(params['gmm_L_raw']),
            'config':         json.dumps(self.config() | {'epoch': epoch}),
        }
        for i, (W, b) in enumerate(params['mlp_weights']):
            flat[f'mlp_W{i}'] = np.array(W)
            flat[f'mlp_b{i}'] = np.array(b)
        np.savez(path, **flat)

    @staticmethod
    def load_checkpoint(
        path: str,
        wave_obs: np.ndarray,
    ) -> tuple['NeuralTemplateModel', dict, int]:
        """Load checkpoint.  Returns (model, params, epoch)."""
        ckpt = np.load(path, allow_pickle=True)
        cfg  = json.loads(str(ckpt['config']))
        model = NeuralTemplateModel(
            Nt=cfg['Nt'],
            wave_obs=wave_obs,
            zmin=cfg['zmin'],
            zmax=cfg['zmax'],
            Nz=cfg['Nz'],
            Nnz=cfg['Nnz'],
            template_res_boost=cfg.get('template_res_boost', 1.0),
            mlp_hidden=cfg.get('mlp_hidden', 64),
            gmm_components=cfg.get('gmm_components', 5),
            gmm_weight=cfg.get('gmm_weight', 0.1),
        )
        # Reconstruct MLP weights list
        mlp_weights = []
        i = 0
        while f'mlp_W{i}' in ckpt:
            mlp_weights.append((jnp.array(ckpt[f'mlp_W{i}']),
                                jnp.array(ckpt[f'mlp_b{i}'])))
            i += 1
        params = {
            'mlp_weights':    mlp_weights,
            'line_A':         jnp.array(ckpt['line_A']),
            'line_sigma_raw': jnp.array(ckpt['line_sigma_raw']),
            'log_nz_raw':     jnp.array(ckpt['log_nz_raw']),
            'gmm_log_pi':     jnp.array(ckpt['gmm_log_pi']),
            'gmm_mu':         jnp.array(ckpt['gmm_mu']),
            'gmm_L_raw':      jnp.array(ckpt['gmm_L_raw']),
        }
        return model, params, int(cfg['epoch'])
