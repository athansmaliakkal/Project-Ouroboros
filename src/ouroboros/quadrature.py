from __future__ import annotations

import numpy as np

from ouroboros.model import _MIN_VARIANCE_PROB

_ADAPTIVE_PHASE_WIDTH_CUTOFF = 1.0 / np.sqrt(2.5e4)


def _max_fringe_std(
    p0: float | np.ndarray,
    contrast: float | np.ndarray,
    n_atoms: float,
    sigma_det: float,
) -> np.ndarray:
    p_lo = np.clip(p0 - 0.5 * contrast, _MIN_VARIANCE_PROB, 1.0 - _MIN_VARIANCE_PROB)
    p_hi = np.clip(p0 + 0.5 * contrast, _MIN_VARIANCE_PROB, 1.0 - _MIN_VARIANCE_PROB)
    has_mid = (p_lo <= 0.5) & (0.5 <= p_hi)
    closer_end = np.where(np.abs(p_lo - 0.5) < np.abs(p_hi - 0.5), p_lo, p_hi)
    p_peak = np.where(has_mid, 0.5, closer_end)
    return np.sqrt(p_peak * (1.0 - p_peak) / n_atoms + sigma_det * sigma_det)


def _anchor_half_width(
    y: np.ndarray,
    p0: float | np.ndarray,
    contrast: float | np.ndarray,
    sig_y: float | np.ndarray,
    local_half_width_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    cos_arg = np.clip(2.0 * (y - p0) / contrast, -1.0, 1.0)
    r = np.arccos(cos_arg)
    sin_r = np.sin(r)
    quad_term = 4.0 * local_half_width_sigma * sig_y / contrast
    w = np.sqrt(sin_r * sin_r + quad_term) - sin_r
    return r, np.clip(w, 1e-6, np.pi)


def _build_adaptive_phi_grid(
    y_a: np.ndarray,
    y_b: np.ndarray,
    delta_phi: float | np.ndarray,
    n_atoms_a: float,
    n_atoms_b: float,
    contrast_a: float | np.ndarray,
    contrast_b: float | np.ndarray,
    p0_a: float | np.ndarray,
    p0_b: float | np.ndarray,
    sigma_det_a: float,
    sigma_det_b: float,
    n_phi_global: int,
    n_phi_local: int,
    local_half_width_sigma: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p0_a_1d = np.asarray(p0_a, dtype=float)
    p0_b_1d = np.asarray(p0_b, dtype=float)
    c_a_1d = np.asarray(contrast_a, dtype=float)
    c_b_1d = np.asarray(contrast_b, dtype=float)
    dphi_raw = np.asarray(delta_phi, dtype=float)
    is_scalar_dphi = dphi_raw.ndim == 0

    sig_a = _max_fringe_std(p0_a_1d, c_a_1d, n_atoms_a, sigma_det_a)
    sig_b = _max_fringe_std(p0_b_1d, c_b_1d, n_atoms_b, sigma_det_b)
    min_phase_width = float(np.min(np.minimum(2.0 * sig_a / c_a_1d, 2.0 * sig_b / c_b_1d)))

    if min_phase_width >= _ADAPTIVE_PHASE_WIDTH_CUTOFF:
        phi = np.linspace(-np.pi, np.pi, n_phi_global, endpoint=False)[None, :]
        weights = np.full((1, n_phi_global), 1.0 / n_phi_global, dtype=float)
        dphi_shift = float(dphi_raw) if is_scalar_dphi else dphi_raw[:, None]
        theta = phi + dphi_shift
        return phi, theta, weights

    dphi_col = np.broadcast_to(dphi_raw, (y_a.shape[0],))[:, None]

    r_a, w_a = _anchor_half_width(y_a, p0_a_1d, c_a_1d, sig_a, local_half_width_sigma)
    r_b, w_b = _anchor_half_width(y_b, p0_b_1d, c_b_1d, sig_b, local_half_width_sigma)
    use_b = w_b < w_a
    r = np.where(use_b, r_b, r_a)
    w = np.where(use_b, w_b, w_a)

    n_nodes = 2 * n_phi_local
    gl_nodes, gl_weights = np.polynomial.legendre.leggauss(n_phi_local)
    gl_nodes_2x, gl_weights_2x = np.polynomial.legendre.leggauss(n_nodes)
    u_uniform = np.linspace(-np.pi, np.pi, n_nodes, endpoint=False)[None, :]
    w_uniform = np.full((1, n_nodes), 1.0 / n_nodes, dtype=float)

    span0 = np.minimum(np.pi, r + w)[:, None]
    u_merge0 = span0 * gl_nodes_2x[None, :]
    wt_merge0 = (span0 / (2.0 * np.pi)) * gl_weights_2x[None, :]

    span_pi = np.minimum(np.pi, (np.pi - r) + w)[:, None]
    u_merge_pi = np.pi + span_pi * gl_nodes_2x[None, :]
    wt_merge_pi = (span_pi / (2.0 * np.pi)) * gl_weights_2x[None, :]

    r_col = r[:, None]
    w_col = w[:, None]
    u_split = np.concatenate(
        [-r_col + w_col * gl_nodes[None, :], r_col + w_col * gl_nodes[None, :]], axis=1
    )
    wt_split = (w_col / (2.0 * np.pi)) * np.concatenate([gl_weights, gl_weights])[None, :]

    is_wide = (w >= 0.45 * np.pi)[:, None]
    is_merge0 = (~is_wide) & ((r - w <= 0.0)[:, None])
    is_merge_pi = (~is_wide) & (~is_merge0) & ((r + w >= np.pi)[:, None])

    u_nodes = np.where(
        is_wide,
        u_uniform,
        np.where(is_merge0, u_merge0, np.where(is_merge_pi, u_merge_pi, u_split)),
    )
    quad_weights = np.where(
        is_wide,
        w_uniform,
        np.where(is_merge0, wt_merge0, np.where(is_merge_pi, wt_merge_pi, wt_split)),
    )

    use_b_col = use_b[:, None]
    phi_grid = np.where(use_b_col, u_nodes - dphi_col, u_nodes)
    theta_grid = np.where(use_b_col, u_nodes, u_nodes + dphi_col)
    return phi_grid, theta_grid, quad_weights


def _to_column(val: float | np.ndarray, n_shots: int) -> tuple[float | np.ndarray, np.ndarray]:
    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        return float(arr), arr
    if arr.shape != (n_shots,):
        raise ValueError(f"Per-shot parameter array must have shape ({n_shots},), got {arr.shape}.")
    return arr, arr[:, None]
