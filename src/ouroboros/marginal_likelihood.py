from __future__ import annotations

import numpy as np

from ouroboros.model import (
    InterferometerParams,
    _log_gaussian_and_derivs,
    _variance_components,
    response_and_derivs,
)
from ouroboros.quadrature import _build_adaptive_phi_grid


def _marginal_log_likelihood_batch(
    y_a_arr: np.ndarray,
    y_b_arr: np.ndarray,
    delta_phi: float | np.ndarray,
    params: InterferometerParams,
    n_phi_global: int,
    n_phi_local: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi_grid, theta_grid, quad_weights = _build_adaptive_phi_grid(
        y_a_arr,
        y_b_arr,
        delta_phi,
        params.n_atoms_a,
        params.n_atoms_b,
        params.contrast_a,
        params.contrast_b,
        params.p0_a,
        params.p0_b,
        params.sigma_det_a,
        params.sigma_det_b,
        n_phi_global=n_phi_global,
        n_phi_local=n_phi_local,
    )

    p_a, dp_a, ddp_a = response_and_derivs(phi_grid, params.p0_a, params.contrast_a)
    v_a, _, _, dv_a, ddv_a = _variance_components(
        p_a,
        dp_a,
        ddp_a,
        params.n_atoms_a,
        params.sigma_det_a,
        y_a_arr[:, None],
        params.p0_a,
        params.variance_mode,
    )
    log_ga, _, _ = _log_gaussian_and_derivs(y_a_arr[:, None], p_a, dp_a, ddp_a, v_a, dv_a, ddv_a)

    p_b, dp_b, ddp_b = response_and_derivs(theta_grid, params.p0_b, params.contrast_b)
    v_b, _, _, dv_b, ddv_b = _variance_components(
        p_b,
        dp_b,
        ddp_b,
        params.n_atoms_b,
        params.sigma_det_b,
        y_b_arr[:, None],
        params.p0_b,
        params.variance_mode,
    )
    log_gb, dlog_gb, ddlog_gb = _log_gaussian_and_derivs(
        y_b_arr[:, None], p_b, dp_b, ddp_b, v_b, dv_b, ddv_b
    )

    log_joint = log_ga + log_gb + np.log(quad_weights)
    max_log = np.max(log_joint, axis=1, keepdims=True)
    scaled_w = np.exp(log_joint - max_log)
    sum_w = np.sum(scaled_w, axis=1)

    log_l = max_log[:, 0] + np.log(sum_w)
    norm_w = scaled_w / sum_w[:, None]

    dlog_l = np.sum(norm_w * dlog_gb, axis=1)
    second_ratio = np.sum(norm_w * (ddlog_gb + dlog_gb * dlog_gb), axis=1)
    ddlog_l = second_ratio - dlog_l * dlog_l
    return log_l, dlog_l, ddlog_l


def marginal_log_likelihood_and_derivs(
    y_a: np.ndarray,
    y_b: np.ndarray,
    delta_phi: float | np.ndarray,
    params: InterferometerParams,
    n_phi_global: int = 1024,
    n_phi_local: int = 128,
    batch_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_a_arr = np.atleast_1d(np.asarray(y_a, dtype=float))
    y_b_arr = np.atleast_1d(np.asarray(y_b, dtype=float))
    if y_a_arr.shape != y_b_arr.shape or y_a_arr.ndim != 1:
        raise ValueError("y_a and y_b must be 1D arrays of matching shape.")

    dphi_arr = np.asarray(delta_phi, dtype=float)
    is_vec_dphi = dphi_arr.ndim > 0
    if is_vec_dphi and dphi_arr.shape != y_a_arr.shape:
        raise ValueError("Vector delta_phi must match y_a shape.")

    n_shots = y_a_arr.shape[0]
    if n_shots <= batch_size:
        return _marginal_log_likelihood_batch(
            y_a_arr, y_b_arr, delta_phi, params, n_phi_global, n_phi_local
        )

    log_l_out = np.empty(n_shots, dtype=float)
    dlog_l_out = np.empty(n_shots, dtype=float)
    ddlog_l_out = np.empty(n_shots, dtype=float)
    for start in range(0, n_shots, batch_size):
        end = min(start + batch_size, n_shots)
        dp_chunk = dphi_arr[start:end] if is_vec_dphi else delta_phi
        ll, dll, ddll = _marginal_log_likelihood_batch(
            y_a_arr[start:end], y_b_arr[start:end], dp_chunk, params, n_phi_global, n_phi_local
        )
        log_l_out[start:end] = ll
        dlog_l_out[start:end] = dll
        ddlog_l_out[start:end] = ddll
    return log_l_out, dlog_l_out, ddlog_l_out
