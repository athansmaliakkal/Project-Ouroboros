from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ouroboros.asymptotics import _true_sigma
from ouroboros.joint_likelihood import joint_marginal_log_likelihood_and_grads
from ouroboros.model import InterferometerParams, response_and_derivs


@dataclass(frozen=True)
class NuisanceFisherResult:
    fisher_matrix: np.ndarray
    sigma_known_nuisance_rad: float
    sigma_joint_contrast_rad: float
    sigma_joint_5d_rad: float
    inflation_contrast_only: float
    inflation_5d: float
    fixed_nuisance_sensitivity: np.ndarray


def _summarize_nuisance_fisher(f_mat: np.ndarray) -> NuisanceFisherResult:
    cov_5d = np.linalg.pinv(f_mat, rcond=1e-12)
    cov_3d = np.linalg.pinv(f_mat[:3, :3], rcond=1e-12)
    f00 = float(f_mat[0, 0])
    sig_known = float(1.0 / np.sqrt(f00)) if f00 > 0.0 else float("inf")
    sig_3d = float(np.sqrt(max(cov_3d[0, 0], 0.0)))
    sig_5d = float(np.sqrt(max(cov_5d[0, 0], 0.0)))
    sens = -f_mat[0, 1:] / f00 if f00 > 0.0 else np.zeros(4, dtype=float)
    return NuisanceFisherResult(
        fisher_matrix=f_mat,
        sigma_known_nuisance_rad=sig_known,
        sigma_joint_contrast_rad=sig_3d,
        sigma_joint_5d_rad=sig_5d,
        inflation_contrast_only=float(sig_3d / sig_known) if f00 > 0.0 else float("inf"),
        inflation_5d=float(sig_5d / sig_known) if f00 > 0.0 else float("inf"),
        fixed_nuisance_sensitivity=sens,
    )


def _gauss_hermite_3d_grid(
    delta_phi: float,
    params: InterferometerParams,
    n_phi_outer: int,
    n_hermite: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi_outer = np.linspace(-np.pi, np.pi, n_phi_outer, endpoint=False)
    gh_x, gh_w = np.polynomial.hermite.hermgauss(n_hermite)
    gh_w_norm = gh_w / np.sqrt(np.pi)

    p_a_true, _, _ = response_and_derivs(phi_outer, params.p0_a, params.contrast_a)
    p_b_true, _, _ = response_and_derivs(phi_outer + delta_phi, params.p0_b, params.contrast_b)
    sig_a_true = _true_sigma(p_a_true, params.n_atoms_a, params.sigma_det_a)
    sig_b_true = _true_sigma(p_b_true, params.n_atoms_b, params.sigma_det_b)

    sqrt2 = np.sqrt(2.0)
    y_a_grid, y_b_grid = np.broadcast_arrays(
        p_a_true[:, None, None] + sqrt2 * sig_a_true[:, None, None] * gh_x[None, :, None],
        p_b_true[:, None, None] + sqrt2 * sig_b_true[:, None, None] * gh_x[None, None, :],
    )
    weights_3d = (
        np.full((n_phi_outer, 1, 1), 1.0 / n_phi_outer, dtype=float)
        * gh_w_norm[None, :, None]
        * gh_w_norm[None, None, :]
    )
    return y_a_grid.ravel(), y_b_grid.ravel(), weights_3d.ravel()


def asymptotic_nuisance_fisher(
    delta_phi: float,
    params: InterferometerParams,
    n_phi: int = 2048,
) -> NuisanceFisherResult:
    mod_pi = abs(delta_phi % np.pi)
    dist_sing = min(mod_pi, np.pi - mod_pi)
    if dist_sing < 1e-6:
        raise ValueError("Asymptotic Fisher matrix is singular at delta_phi = 0 or pi.")

    n_grid = min(32768, max(n_phi, int(np.ceil(16.0 * 2.0 * np.pi / dist_sing))))
    phi = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    theta = phi + delta_phi
    cos_p, sin_p = np.cos(phi), np.sin(phi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    mu_a = params.p0_a + 0.5 * params.contrast_a * cos_p
    mu_b = params.p0_b + 0.5 * params.contrast_b * cos_t
    m_a = -0.5 * params.contrast_a * sin_p
    m_b = -0.5 * params.contrast_b * sin_t

    p_a_clip = np.clip(mu_a, 1e-9, 1.0 - 1e-9)
    p_b_clip = np.clip(mu_b, 1e-9, 1.0 - 1e-9)
    v_a = p_a_clip * (1.0 - p_a_clip) / params.n_atoms_a + params.sigma_det_a**2
    v_b = p_b_clip * (1.0 - p_b_clip) / params.n_atoms_b + params.sigma_det_b**2
    denom = np.sqrt(v_a * m_b * m_b + v_b * m_a * m_a)

    s_mat = np.vstack(
        [
            (-m_a * m_b) / denom,
            (0.5 * cos_p * m_b) / denom,
            (-0.5 * cos_t * m_a) / denom,
            m_b / denom,
            -m_a / denom,
        ]
    )
    f_inf = (s_mat @ s_mat.T) / float(n_grid)
    return _summarize_nuisance_fisher(f_inf)


def finite_n_nuisance_fisher(
    delta_phi: float,
    params: InterferometerParams,
    n_phi_outer: int = 128,
    n_hermite: int = 11,
    n_phi_inner: int = 512,
    n_phi_local: int = 128,
) -> NuisanceFisherResult:
    ya, yb, weights = _gauss_hermite_3d_grid(delta_phi, params, n_phi_outer, n_hermite)
    grads = joint_marginal_log_likelihood_and_grads(
        ya,
        yb,
        delta_phi,
        params.n_atoms_a,
        params.n_atoms_b,
        params.contrast_a,
        params.contrast_b,
        params.p0_a,
        params.p0_b,
        sigma_det_a=params.sigma_det_a,
        sigma_det_b=params.sigma_det_b,
        variance_mode=params.variance_mode,
        n_phi_global=n_phi_inner,
        n_phi_local=n_phi_local,
    )
    s_mat = np.column_stack(
        [
            grads.d_delta_phi,
            0.5 * (grads.d_hi_a - grads.d_lo_a),
            0.5 * (grads.d_hi_b - grads.d_lo_b),
            grads.d_hi_a + grads.d_lo_a,
            grads.d_hi_b + grads.d_lo_b,
        ]
    )
    f_fin = (s_mat.T * weights) @ s_mat
    return _summarize_nuisance_fisher(f_fin)
