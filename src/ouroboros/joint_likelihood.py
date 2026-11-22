from __future__ import annotations

import numpy as np

from ouroboros.model import (
    JointLikelihoodGrads,
    VarianceMode,
    _log_gaussian_mean_score,
    _variance_components,
)
from ouroboros.quadrature import _build_adaptive_phi_grid, _to_column


def _joint_grads_batch(
    y_a_arr: np.ndarray,
    y_b_arr: np.ndarray,
    dphi_arr: float | np.ndarray,
    n_atoms_a: float,
    n_atoms_b: float,
    c_a_1d: float | np.ndarray,
    c_b_1d: float | np.ndarray,
    p0_a_1d: float | np.ndarray,
    p0_b_1d: float | np.ndarray,
    sigma_det_a: float,
    sigma_det_b: float,
    variance_mode: VarianceMode,
    n_phi_global: int,
    n_phi_local: int,
) -> JointLikelihoodGrads:
    n_shots = y_a_arr.shape[0]
    _, c_a_col = _to_column(c_a_1d, n_shots)
    _, c_b_col = _to_column(c_b_1d, n_shots)
    _, p0_a_col = _to_column(p0_a_1d, n_shots)
    _, p0_b_col = _to_column(p0_b_1d, n_shots)

    phi_grid, theta_grid, quad_weights = _build_adaptive_phi_grid(
        y_a_arr,
        y_b_arr,
        dphi_arr,
        n_atoms_a,
        n_atoms_b,
        c_a_1d,
        c_b_1d,
        p0_a_1d,
        p0_b_1d,
        sigma_det_a,
        sigma_det_b,
        n_phi_global=n_phi_global,
        n_phi_local=n_phi_local,
    )

    cos_phi = np.cos(phi_grid)
    p_a = p0_a_col + 0.5 * c_a_col * cos_phi
    dp_a_dummy = np.zeros_like(p_a)
    v_a, vq_a, dv_dp_a, _, _ = _variance_components(
        p_a,
        dp_a_dummy,
        dp_a_dummy,
        n_atoms_a,
        sigma_det_a,
        y_a_arr[:, None],
        p0_a_col,
        variance_mode,
    )
    log_ga, f_a, ra_sq_v = _log_gaussian_mean_score(y_a_arr[:, None], p_a, v_a, dv_dp_a)

    dphi_raw = np.asarray(dphi_arr, dtype=float)
    if phi_grid.shape[0] == 1 and dphi_raw.ndim == 1:
        sin_phi = np.sin(phi_grid)
        cos_dp = np.cos(dphi_raw)[:, None]
        sin_dp = np.sin(dphi_raw)[:, None]
        cos_th = cos_phi * cos_dp - sin_phi * sin_dp
        sin_th = sin_phi * cos_dp + cos_phi * sin_dp
    else:
        cos_th = np.cos(theta_grid)
        sin_th = np.sin(theta_grid)
    p_b = p0_b_col + 0.5 * c_b_col * cos_th
    dp_b_dummy = np.zeros_like(p_b)
    v_b, vq_b, dv_dp_b, _, _ = _variance_components(
        p_b,
        dp_b_dummy,
        dp_b_dummy,
        n_atoms_b,
        sigma_det_b,
        y_b_arr[:, None],
        p0_b_col,
        variance_mode,
    )
    log_gb, f_b, rb_sq_v = _log_gaussian_mean_score(y_b_arr[:, None], p_b, v_b, dv_dp_b)

    log_joint = log_ga + log_gb + np.log(quad_weights)
    max_log = np.max(log_joint, axis=1, keepdims=True)
    scaled_w = np.exp(log_joint - max_log)
    sum_w = np.sum(scaled_w, axis=1)
    log_l = max_log[:, 0] + np.log(sum_w)
    norm_w = scaled_w / sum_w[:, None]

    wf_a = norm_w * f_a
    wf_b = norm_w * f_b

    d_dphi = np.sum(wf_b * (-0.5 * c_b_col * sin_th), axis=1)
    d_lo_a = np.sum(wf_a * (0.5 * (1.0 - cos_phi)), axis=1)
    d_hi_a = np.sum(wf_a * (0.5 * (1.0 + cos_phi)), axis=1)
    d_lo_b = np.sum(wf_b * (0.5 * (1.0 - cos_th)), axis=1)
    d_hi_b = np.sum(wf_b * (0.5 * (1.0 + cos_th)), axis=1)
    d_log_na = np.sum(norm_w * (0.5 * (vq_a / v_a) * (1.0 - ra_sq_v)), axis=1)
    d_log_nb = np.sum(norm_w * (0.5 * (vq_b / v_b) * (1.0 - rb_sq_v)), axis=1)
    return JointLikelihoodGrads(
        log_l=log_l,
        d_delta_phi=d_dphi,
        d_lo_a=d_lo_a,
        d_hi_a=d_hi_a,
        d_lo_b=d_lo_b,
        d_hi_b=d_hi_b,
        d_log_na=d_log_na,
        d_log_nb=d_log_nb,
    )


def joint_marginal_log_likelihood_and_grads(
    y_a: np.ndarray,
    y_b: np.ndarray,
    delta_phi: float | np.ndarray,
    n_atoms_a: float,
    n_atoms_b: float,
    contrast_a: float | np.ndarray,
    contrast_b: float | np.ndarray,
    p0_a: float | np.ndarray = 0.5,
    p0_b: float | np.ndarray = 0.5,
    sigma_det_a: float = 0.0,
    sigma_det_b: float = 0.0,
    variance_mode: VarianceMode = "model",
    n_phi_global: int = 384,
    n_phi_local: int = 128,
    batch_size: int = 2048,
) -> JointLikelihoodGrads:
    y_a_arr = np.atleast_1d(np.asarray(y_a, dtype=float))
    y_b_arr = np.atleast_1d(np.asarray(y_b, dtype=float))
    if y_a_arr.shape != y_b_arr.shape or y_a_arr.ndim != 1:
        raise ValueError("y_a and y_b must be 1D arrays of matching shape.")
    n_shots = y_a_arr.shape[0]
    if variance_mode == "midfringe":
        raise ValueError(
            "Joint nuisance gradients are defined for variance_mode 'model' or 'observed'."
        )

    def _slice(val: float | np.ndarray, st: int, en: int) -> float | np.ndarray:
        arr = np.asarray(val, dtype=float)
        return float(arr) if arr.ndim == 0 else arr[st:en]

    if n_shots <= batch_size:
        return _joint_grads_batch(
            y_a_arr,
            y_b_arr,
            delta_phi,
            n_atoms_a,
            n_atoms_b,
            contrast_a,
            contrast_b,
            p0_a,
            p0_b,
            sigma_det_a,
            sigma_det_b,
            variance_mode,
            n_phi_global,
            n_phi_local,
        )

    out = JointLikelihoodGrads(
        log_l=np.empty(n_shots, dtype=float),
        d_delta_phi=np.empty(n_shots, dtype=float),
        d_lo_a=np.empty(n_shots, dtype=float),
        d_hi_a=np.empty(n_shots, dtype=float),
        d_lo_b=np.empty(n_shots, dtype=float),
        d_hi_b=np.empty(n_shots, dtype=float),
        d_log_na=np.empty(n_shots, dtype=float),
        d_log_nb=np.empty(n_shots, dtype=float),
    )
    for start in range(0, n_shots, batch_size):
        end = min(start + batch_size, n_shots)
        part = _joint_grads_batch(
            y_a_arr[start:end],
            y_b_arr[start:end],
            _slice(delta_phi, start, end),
            n_atoms_a,
            n_atoms_b,
            _slice(contrast_a, start, end),
            _slice(contrast_b, start, end),
            _slice(p0_a, start, end),
            _slice(p0_b, start, end),
            sigma_det_a,
            sigma_det_b,
            variance_mode,
            n_phi_global,
            n_phi_local,
        )
        out.log_l[start:end] = part.log_l
        out.d_delta_phi[start:end] = part.d_delta_phi
        out.d_lo_a[start:end] = part.d_lo_a
        out.d_hi_a[start:end] = part.d_hi_a
        out.d_lo_b[start:end] = part.d_lo_b
        out.d_hi_b[start:end] = part.d_hi_b
        out.d_log_na[start:end] = part.d_log_na
        out.d_log_nb[start:end] = part.d_log_nb
    return out
