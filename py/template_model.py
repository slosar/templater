"""
Spectral template fitting model for DESI galaxy spectra.

Model
-----
prediction_if = sum_k alpha_ik * T_k(wave_obs[f] / (1+z_i))

where T_k are spectral templates (learned), alpha_ik are per-galaxy linear
amplitudes (solved analytically at each z), and z_i are redshifts.

Loss
----
Marginalizes over a discrete z search grid, weighted by a learnable n(z)
distribution and a Gaussian redshift prior from the catalog:

  p_i = sum_z  exp(-0.5*chi2_iz - 0.5*(z-z_prior_i)^2/zerr_i^2) * n(z)
  loss = -mean_i log(p_i)

chi2_iz is evaluated at the analytically optimal alpha (linear solve per z).
The chi2 identity chi2_min = flux^T V flux - b^T alpha avoids storing residuals.

Usage
-----
    from template_model import TemplateModel
    import jax, jax.numpy as jnp, numpy as np

    model = TemplateModel(Nt=5, wave_obs=wave, zmin=0.4, zmax=1.1, Nz=200)
    params = model.init_params(jax.random.PRNGKey(0), flux_mean=flux_mean)

    loss_and_grad = jax.jit(jax.value_and_grad(model.loss))
    loss_val, grads = loss_and_grad(params, flux, ivar, z_prior, zerr)
"""

from __future__ import annotations

import json
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


class TemplateModel:
    """JAX spectral template model.

    Parameters
    ----------
    Nt : int
        Number of spectral templates.
    wave_obs : np.ndarray, shape (Nf,)
        Observed wavelength grid in Angstroms (fixed, shared across spectra).
    zmin, zmax : float
        Redshift search range.
    Nz : int
        Number of points in the z search grid.
    Nnz : int
        Number of bins in the learnable n(z) distribution.
    """

    def __init__(
        self,
        Nt: int,
        wave_obs: np.ndarray,
        zmin: float,
        zmax: float,
        Nz: int,
        Nnz: int = 50,
    ) -> None:
        self.Nt = Nt
        self.Nf = int(len(wave_obs))
        self.Nz = Nz
        self.Nnz = Nnz
        self.zmin = float(zmin)
        self.zmax = float(zmax)
        self.wave_min = float(wave_obs.min())
        self.wave_max = float(wave_obs.max())

        # ---- Template rest-frame wavelength grid --------------------------------
        # Must cover wave_obs[f]/(1+z) for all z in [zmin, zmax].
        t_wave_min = self.wave_min / (1.0 + zmax)
        t_wave_max = self.wave_max / (1.0 + zmin)
        # Spacing: match the approximate pixel scale of the observed grid.
        delta_wave = (float(wave_obs[-1]) - float(wave_obs[0])) / (len(wave_obs) - 1)
        self.Nft = int(round((t_wave_max - t_wave_min) / delta_wave)) + 1
        t_wave = np.linspace(t_wave_min, t_wave_max, self.Nft).astype(np.float32)

        # ---- z search grid ------------------------------------------------------
        zgrid = np.linspace(zmin, zmax, Nz).astype(np.float32)

        # ---- Precompute interpolation weights -----------------------------------
        # For each (z, observed wavelength), find where wave_obs/(1+z) lands in t_wave.
        # rest_waves[j, f] = wave_obs[f] / (1 + zgrid[j])  → (Nz, Nf)
        rest_waves = wave_obs[None, :].astype(np.float32) / (1.0 + zgrid[:, None])
        lo = (np.searchsorted(t_wave, rest_waves, side='right') - 1).astype(np.int32)
        lo = np.clip(lo, 0, self.Nft - 2)
        # Fractional weight for the upper neighbour
        weight = ((rest_waves - t_wave[lo]) /
                  (t_wave[lo + 1] - t_wave[lo])).astype(np.float32)
        weight = np.clip(weight, 0.0, 1.0)

        # ---- n(z) grid and its mapping from search grid -------------------------
        z_nz_grid = np.linspace(zmin, zmax, Nnz).astype(np.float32)
        nz_lo = (np.searchsorted(z_nz_grid, zgrid, side='right') - 1).astype(np.int32)
        nz_lo = np.clip(nz_lo, 0, Nnz - 2)
        nz_weight = ((zgrid - z_nz_grid[nz_lo]) /
                     (z_nz_grid[nz_lo + 1] - z_nz_grid[nz_lo])).astype(np.float32)
        nz_weight = np.clip(nz_weight, 0.0, 1.0)

        # Store everything as JAX arrays.
        # These are captured as compile-time constants inside jax.jit closures.
        self.t_wave = jnp.array(t_wave)
        self.wave_obs = jnp.array(wave_obs, dtype=jnp.float32)
        self.zgrid = jnp.array(zgrid)
        self.z_nz_grid = jnp.array(z_nz_grid)
        self.lo_idx = jnp.array(lo)       # (Nz, Nf)  int32
        self.weight = jnp.array(weight)   # (Nz, Nf)  float32
        self.nz_lo_idx = jnp.array(nz_lo)      # (Nz,)  int32
        self.nz_weight = jnp.array(nz_weight)  # (Nz,)  float32

    # -------------------------------------------------------------------------
    # Parameter initialisation
    # -------------------------------------------------------------------------

    def init_params(
        self,
        key: jax.Array,
        flux_mean: Optional[np.ndarray] = None,
        t0_init: str = 'mean_flux',
    ) -> dict:
        """Return initial parameter pytree {"T": (Nt, Nft), "log_nz_raw": (Nnz,)}.

        T[0] initialisation is controlled by t0_init:
          'mean_flux' — interpolate mean observed flux onto template grid (default)
          'flat'      — 1.0 + small random noise; avoids edge artefacts from
                        flux extrapolation when templates span a wider range than data

        Remaining templates always start as small Gaussian noise.
        log_nz_raw is initialised to zeros (uniform n(z) after log_softmax).
        """
        key_t0, key_noise = jax.random.split(key)

        if t0_init == 'flat':
            t0 = np.ones(self.Nft, dtype=np.float32)
            t0 += np.array(jax.random.normal(key_t0, shape=(self.Nft,))) * 0.01
        else:  # 'mean_flux'
            t_wave_np = np.array(self.t_wave)
            wave_obs_np = np.array(self.wave_obs)
            if flux_mean is not None:
                t0 = np.interp(t_wave_np, wave_obs_np, flux_mean).astype(np.float32)
                t0 = np.maximum(t0, 0.0) / self.Nt
            else:
                t0 = np.ones(self.Nft, dtype=np.float32) * 0.1

        noise_scale = float(np.abs(t0).mean()) * 0.01
        noise = jax.random.normal(key_noise, shape=(self.Nt - 1, self.Nft)) * noise_scale

        T = jnp.concatenate(
            [jnp.array(t0)[None, :], noise],  # (Nt, Nft)
            axis=0,
        )
        log_nz_raw = jnp.zeros(self.Nnz)

        return {"T": T, "log_nz_raw": log_nz_raw}

    # -------------------------------------------------------------------------
    # Interpolation
    # -------------------------------------------------------------------------

    def _interpolate_templates(self, T: jax.Array) -> jax.Array:
        """Interpolate templates onto the observed grid for every z in zgrid.

        Parameters
        ----------
        T : (Nt, Nft)

        Returns
        -------
        T_all : (Nz, Nt, Nf)
        """
        # lo_idx: (Nz, Nf) — advanced indexing along axis 1 of T
        T_lo = T[:, self.lo_idx]      # (Nt, Nz, Nf)
        T_hi = T[:, self.lo_idx + 1]  # (Nt, Nz, Nf)
        w = self.weight               # (Nz, Nf) — broadcasts over Nt dim
        T_all = T_lo * (1.0 - w) + T_hi * w   # (Nt, Nz, Nf)
        return T_all.transpose(1, 0, 2)         # (Nz, Nt, Nf)

    def _interp_at_z(self, T: jax.Array, z: jax.Array) -> jax.Array:
        """Interpolate templates at a single (possibly traced) redshift z.

        Parameters
        ----------
        T : (Nt, Nft)
        z : scalar JAX array

        Returns
        -------
        T_z : (Nt, Nf)
        """
        rest_wave = self.wave_obs / (1.0 + z)                      # (Nf,)
        lo = jnp.searchsorted(self.t_wave, rest_wave, side='right') - 1
        lo = jnp.clip(lo, 0, self.Nft - 2)
        w = (rest_wave - self.t_wave[lo]) / (self.t_wave[lo + 1] - self.t_wave[lo])
        w = jnp.clip(w, 0.0, 1.0)
        return T[:, lo] * (1.0 - w) + T[:, lo + 1] * w            # (Nt, Nf)

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
    ) -> jax.Array:
        """Negative mean log-likelihood over a batch.

        Parameters
        ----------
        params      : {"T": (Nt, Nft), "log_nz_raw": (Nnz,)}
        flux        : (B, Nf)  float32
        ivar        : (B, Nf)  float32
        z_prior     : (B,)     float32 — catalog redshift
        zerr        : (B,)     float32 — catalog redshift uncertainty
        template_l2 : float    L2 regularisation weight on T. Pushes template
                               values in unconstrained (edge) regions to zero.

        Returns
        -------
        Scalar loss (differentiable w.r.t. params).
        """
        T_all = self._interpolate_templates(params["T"])   # (Nz, Nt, Nf)

        # Log-normalised n(z) interpolated onto search grid
        log_nz_norm = jax.nn.log_softmax(params["log_nz_raw"])   # (Nnz,)
        log_nz_at_z = (
            log_nz_norm[self.nz_lo_idx] * (1.0 - self.nz_weight)
            + log_nz_norm[self.nz_lo_idx + 1] * self.nz_weight
        )   # (Nz,)

        # Gaussian z-prior log-weight, shape (B, Nz)
        zerr_safe = jnp.maximum(zerr, 1e-8)[:, None]
        dz = self.zgrid[None, :] - z_prior[:, None]
        log_prior = -0.5 * (dz / zerr_safe) ** 2   # (B, Nz)

        # Data-only term: sum_f flux^2 * ivar, shape (B,)
        flux_ivar_flux = (flux * ivar * flux).sum(axis=-1)

        Nt = self.Nt
        eye_Nt = jnp.eye(Nt)

        def scan_body(carry, x):
            log_max, sum_exp = carry          # each (B,)
            T_z, log_nz_z, log_prior_z = x   # (Nt,Nf), scalar, (B,)

            # A[b,k,l] = sum_f T_z[k,f] * ivar[b,f] * T_z[l,f]   shape (B,Nt,Nt)
            T_ivar = T_z[None, :, :] * ivar[:, None, :]        # (B, Nt, Nf)
            A = jnp.einsum('bkf,lf->bkl', T_ivar, T_z)        # (B, Nt, Nt)
            A = A + 1e-6 * eye_Nt[None]                         # regularise

            # b[b,k] = sum_f T_z[k,f] * ivar[b,f] * flux[b,f]   shape (B,Nt)
            b = (T_ivar * flux[:, None, :]).sum(axis=-1)

            # Solve A @ alpha = b per galaxy
            alpha = jax.vmap(jnp.linalg.solve)(A, b)           # (B, Nt)

            # chi2_min = flux_ivar_flux - b^T alpha  (algebraic identity)
            chi2 = flux_ivar_flux - (b * alpha).sum(axis=-1)   # (B,)

            # Log weight at this z for each galaxy
            log_w = log_nz_z + log_prior_z - 0.5 * chi2       # (B,)

            # Numerically stable online logsumexp update
            new_max = jnp.maximum(log_max, log_w)
            new_sum = sum_exp * jnp.exp(log_max - new_max) + jnp.exp(log_w - new_max)
            return (new_max, new_sum), None

        B = flux.shape[0]
        init_carry = (jnp.full((B,), -1e30), jnp.zeros((B,)))
        # log_prior.T shape: (Nz, B)
        xs = (T_all, log_nz_at_z, log_prior.T)

        (log_max, sum_exp), _ = jax.lax.scan(
            jax.remat(scan_body), init_carry, xs
        )

        log_p = log_max + jnp.log(sum_exp)   # (B,)
        return -jnp.mean(log_p) + template_l2 * jnp.mean(params["T"] ** 2)

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
        """Compute unnormalised log-posterior over the z search grid.

        Parameters
        ----------
        (same as loss)

        Returns
        -------
        log_posterior : (B, Nz)
            log_w[i, j] = log n(z_j) + log p(z_j | z_prior_i) - 0.5 * chi2_ij
        """
        T_all = self._interpolate_templates(params["T"])

        log_nz_norm = jax.nn.log_softmax(params["log_nz_raw"])
        log_nz_at_z = (
            log_nz_norm[self.nz_lo_idx] * (1.0 - self.nz_weight)
            + log_nz_norm[self.nz_lo_idx + 1] * self.nz_weight
        )

        zerr_safe = jnp.maximum(zerr, 1e-8)[:, None]
        dz = self.zgrid[None, :] - z_prior[:, None]
        log_prior = -0.5 * (dz / zerr_safe) ** 2

        flux_ivar_flux = (flux * ivar * flux).sum(axis=-1)
        Nt = self.Nt
        eye_Nt = jnp.eye(Nt)

        def scan_body(_, x):
            T_z, log_nz_z, log_prior_z = x
            T_ivar = T_z[None, :, :] * ivar[:, None, :]
            A = jnp.einsum('bkf,lf->bkl', T_ivar, T_z) + 1e-6 * eye_Nt[None]
            b = (T_ivar * flux[:, None, :]).sum(axis=-1)
            alpha = jax.vmap(jnp.linalg.solve)(A, b)
            chi2 = flux_ivar_flux - (b * alpha).sum(axis=-1)
            log_w = log_nz_z + log_prior_z - 0.5 * chi2    # (B,)
            return _, log_w

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
        """Compute optimal linear amplitudes for each galaxy at its redshift.

        Parameters
        ----------
        z_vals : (B,) — one redshift per galaxy

        Returns
        -------
        alpha : (B, Nt)
        """
        Nt = self.Nt

        def single(flux_i, ivar_i, z_i):
            T_z = self._interp_at_z(params["T"], z_i)           # (Nt, Nf)
            T_ivar = T_z * ivar_i[None, :]                      # (Nt, Nf)
            A = T_ivar @ T_z.T + 1e-6 * jnp.eye(Nt)            # (Nt, Nt)
            b = (T_ivar * flux_i[None, :]).sum(axis=-1)         # (Nt,)
            return jnp.linalg.solve(A, b)

        return jax.vmap(single)(flux, ivar, z_vals)   # (B, Nt)

    # -------------------------------------------------------------------------
    # Checkpoint helpers
    # -------------------------------------------------------------------------

    def config(self) -> dict:
        """Return JSON-serialisable model configuration."""
        return {
            "Nt": self.Nt,
            "Nft": self.Nft,
            "Nz": self.Nz,
            "Nnz": self.Nnz,
            "zmin": self.zmin,
            "zmax": self.zmax,
            "wave_min": self.wave_min,
            "wave_max": self.wave_max,
            "Nf": self.Nf,
        }

    def save_checkpoint(self, params: dict, epoch: int, path: str) -> None:
        """Save params and config to a .npz file."""
        np.savez(
            path,
            T=np.array(params["T"]),
            log_nz_raw=np.array(params["log_nz_raw"]),
            config=json.dumps(self.config() | {"epoch": epoch}),
        )

    @staticmethod
    def load_checkpoint(
        path: str,
        wave_obs: np.ndarray,
    ) -> tuple["TemplateModel", dict, int]:
        """Load checkpoint. Returns (model, params, epoch)."""
        ckpt = np.load(path, allow_pickle=True)
        cfg = json.loads(str(ckpt["config"]))
        model = TemplateModel(
            Nt=cfg["Nt"],
            wave_obs=wave_obs,
            zmin=cfg["zmin"],
            zmax=cfg["zmax"],
            Nz=cfg["Nz"],
            Nnz=cfg["Nnz"],
        )
        params = {
            "T": jnp.array(ckpt["T"]),
            "log_nz_raw": jnp.array(ckpt["log_nz_raw"]),
        }
        return model, params, int(cfg["epoch"])
