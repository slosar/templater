"""Helpers for fitting neural templates directly to saved template checkpoints.

This is intentionally separate from survey-data training.  It answers a simpler
question: can the current NeuralTemplateModel parameterisation reproduce the
rest-frame template shapes learned by TemplateModel?
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any

import fitsio
import jax
import jax.numpy as jnp
import numpy as np
import optax

from neural_template_model import NeuralTemplateModel
from template_model import TemplateModel


def latest_checkpoint(checkpoint_dir: str) -> str:
    """Return the latest checkpoint in a directory."""
    ckpts = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch*.npz")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir!r}")
    return ckpts[-1]


def load_wave_obs(spectra_dir: str = "spectra") -> np.ndarray:
    """Load the shared observed wavelength grid from the first spectra file."""
    spec_files = sorted(glob.glob(os.path.join(spectra_dir, "spec-*.fits")))
    if not spec_files:
        raise FileNotFoundError(f"No spec-*.fits files found in {spectra_dir!r}")
    with fitsio.FITS(spec_files[0]) as f:
        return f["wave"].read().astype(np.float32)


def build_template_array(model: Any, params: dict) -> np.ndarray:
    """Return rest-frame templates as a NumPy array for either model family."""
    if isinstance(model, NeuralTemplateModel):
        return np.array(model._build_templates(params))
    return np.array(params["T"])


def _copy_auxiliary_params(
    source_params: dict,
    target_params: dict,
    *,
    gmm_mode: str,
) -> dict:
    """Copy checkpoint parameters that are not part of the direct template fit."""
    out = dict(target_params)
    if "log_nz_raw" in source_params:
        out["log_nz_raw"] = source_params["log_nz_raw"]

    if gmm_mode not in {"copy_if_present", "reinit"}:
        raise ValueError(f"Unsupported gmm_mode={gmm_mode!r}")

    if gmm_mode == "copy_if_present":
        for key in ("gmm_log_pi", "gmm_mu", "gmm_L_raw"):
            if key in source_params and key in out:
                out[key] = source_params[key]
    return out


def _fit_loss(model: NeuralTemplateModel, target_templates: jax.Array, variance: float):
    def loss_fn(params: dict) -> jax.Array:
        pred = model._build_templates(params)
        mse = jnp.mean((pred - target_templates) ** 2)
        return mse / (variance + 1e-8)

    return loss_fn


def _per_template_metrics(target: np.ndarray, fit: np.ndarray) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    for k in range(target.shape[0]):
        diff = fit[k] - target[k]
        target_centered = target[k] - target[k].mean()
        fit_centered = fit[k] - fit[k].mean()
        denom = np.sqrt((target_centered ** 2).sum() * (fit_centered ** 2).sum())
        corr = float((target_centered * fit_centered).sum() / denom) if denom > 0 else np.nan
        metrics.append(
            {
                "template": float(k),
                "rmse": float(np.sqrt(np.mean(diff ** 2))),
                "max_abs_err": float(np.max(np.abs(diff))),
                "corr": corr,
            }
        )
    return metrics


def summarise_fit_metrics(target: np.ndarray, fit: np.ndarray) -> dict[str, Any]:
    """Return compact metrics describing the reconstruction quality."""
    diff = fit - target
    centered = target - target.mean(axis=1, keepdims=True)
    variance = float(np.mean(centered ** 2))
    return {
        "relative_mse": float(np.mean(diff ** 2) / (variance + 1e-8)),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "max_abs_err": float(np.max(np.abs(diff))),
        "per_template": _per_template_metrics(target, fit),
    }


def fit_neural_to_template_checkpoint(
    template_ckpt_path: str,
    wave_obs: np.ndarray,
    *,
    mlp_hidden: int = 64,
    lr: float = 3e-2,
    n_steps: int = 3000,
    seed: int = 0,
    gmm_mode: str = "copy_if_present",
    line_amp_init: float = 0.01,
) -> tuple[NeuralTemplateModel, dict, dict[str, Any]]:
    """Fit NeuralTemplateModel directly to a saved TemplateModel checkpoint."""
    template_model, template_params, epoch = TemplateModel.load_checkpoint(template_ckpt_path, wave_obs)
    target = jnp.array(template_params["T"])
    centered = target - jnp.mean(target, axis=1, keepdims=True)
    variance = float(jnp.mean(centered ** 2))

    gmm_components = template_model.gmm_components if template_model.gmm_components > 0 else 1
    neural_model = NeuralTemplateModel(
        Nt=template_model.Nt,
        wave_obs=wave_obs,
        zmin=template_model.zmin,
        zmax=template_model.zmax,
        Nz=template_model.Nz,
        Nnz=template_model.Nnz,
        template_res_boost=template_model.template_res_boost,
        mlp_hidden=mlp_hidden,
        gmm_components=gmm_components,
        gmm_weight=0.0,
    )
    params = neural_model.init_params(
        jax.random.PRNGKey(seed),
        nz_sigma=0.05,
        line_amp_init=line_amp_init,
    )
    params = _copy_auxiliary_params(template_params, params, gmm_mode=gmm_mode)

    loss_fn = _fit_loss(neural_model, target, variance)
    tx = optax.adam(lr)
    opt_state = tx.init(params)

    @jax.jit
    def step(current_params: dict, current_state: optax.OptState):
        loss, grads = jax.value_and_grad(loss_fn)(current_params)
        updates, current_state = tx.update(grads, current_state, current_params)
        current_params = optax.apply_updates(current_params, updates)
        return current_params, current_state, loss

    best_loss = float("inf")
    best_params = params
    for _ in range(n_steps):
        params, opt_state, loss = step(params, opt_state)
        loss_value = float(loss)
        if loss_value < best_loss:
            best_loss = loss_value
            best_params = jax.tree.map(
                lambda x: x.copy() if hasattr(x, "copy") else x,
                params,
            )

    target_np = np.array(template_params["T"])
    fit_np = np.array(neural_model._build_templates(best_params))
    metrics = summarise_fit_metrics(target_np, fit_np)
    metrics.update(
        {
            "source_checkpoint": template_ckpt_path,
            "source_epoch": epoch,
            "mlp_hidden": mlp_hidden,
            "n_steps": n_steps,
            "lr": lr,
            "seed": seed,
        }
    )
    return neural_model, best_params, metrics


def export_fitted_neural_checkpoint(
    output_path: str,
    model: NeuralTemplateModel,
    params: dict,
    *,
    epoch: int = 0,
    source_checkpoint: str | None = None,
    fit_metrics: dict[str, Any] | None = None,
) -> None:
    """Save a fitted neural checkpoint in the standard loadable format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    arrays: dict[str, Any] = {
        "line_A": np.array(params["line_A"]),
        "line_sigma_raw": np.array(params["line_sigma_raw"]),
        "log_nz_raw": np.array(params["log_nz_raw"]),
        "gmm_log_pi": np.array(params["gmm_log_pi"]),
        "gmm_mu": np.array(params["gmm_mu"]),
        "gmm_L_raw": np.array(params["gmm_L_raw"]),
    }
    for i, (W, b) in enumerate(params["mlp_weights"]):
        arrays[f"mlp_W{i}"] = np.array(W)
        arrays[f"mlp_b{i}"] = np.array(b)

    config = model.config() | {"epoch": epoch}
    if source_checkpoint is not None:
        config["fit_source_checkpoint"] = source_checkpoint
    if fit_metrics is not None:
        config["fit_relative_mse"] = float(fit_metrics["relative_mse"])
        config["fit_rmse"] = float(fit_metrics["rmse"])
        config["fit_max_abs_err"] = float(fit_metrics["max_abs_err"])
    arrays["config"] = json.dumps(config)
    np.savez(output_path, **arrays)


def load_neural_checkpoint_if_compatible(
    checkpoint_path: str,
    wave_obs: np.ndarray,
    *,
    expected_nt: int | None = None,
) -> tuple[NeuralTemplateModel, dict, int]:
    """Load a neural checkpoint and optionally enforce template-count compatibility."""
    model, params, epoch = NeuralTemplateModel.load_checkpoint(checkpoint_path, wave_obs)
    if expected_nt is not None and model.Nt != expected_nt:
        raise ValueError(
            f"Incompatible Nt for comparison: neural checkpoint has Nt={model.Nt}, "
            f"expected Nt={expected_nt}"
        )
    return model, params, epoch
