from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from ouroboros.asymptotics import (
    CramerRaoResult,
    _true_sigma,
    asymptotic_optimal_bias,
    known_phase_sql_variance,
)
from ouroboros.marginal_likelihood import marginal_log_likelihood_and_derivs
from ouroboros.model import InterferometerParams, response_and_derivs
from ouroboros.nuisance_fisher import _gauss_hermite_3d_grid


def compute_cramer_rao_quadrature(
    delta_phi: float,
    params: InterferometerParams,
    n_phi_outer: int = 128,
    n_hermite: int = 9,
    n_phi_inner: int = 1024,
    n_phi_local: int = 128,
) -> CramerRaoResult:
    ya_flat, yb_flat, w_flat = _gauss_hermite_3d_grid(delta_phi, params, n_phi_outer, n_hermite)

    _, dlog_l, ddlog_l = marginal_log_likelihood_and_derivs(
        ya_flat,
        yb_flat,
        delta_phi=delta_phi,
        params=params,
        n_phi_global=n_phi_inner,
        n_phi_local=n_phi_local,
    )

    fisher_score = float(np.sum(w_flat * dlog_l * dlog_l))
    fisher_curv = float(-np.sum(w_flat * ddlog_l))

    sigma_crb = 1.0 / np.sqrt(fisher_score) if fisher_score > 0.0 else float("inf")
    sigma_mid = float(np.sqrt(known_phase_sql_variance(params)))
    return CramerRaoResult(
        fisher_score=fisher_score,
        fisher_curvature=fisher_curv,
        sigma_crb_rad=float(sigma_crb),
        sigma_known_midfringe_rad=sigma_mid,
        penalty_factor=float(sigma_crb / sigma_mid),
    )


def compute_cramer_rao_mc(
    delta_phi: float,
    params: InterferometerParams,
    n_samples: int = 20000,
    seed: int = 42,
    binomial_readout: bool = False,
    n_phi_inner: int = 1024,
    n_phi_local: int = 128,
) -> CramerRaoResult:
    rng = np.random.default_rng(seed)
    phi_true = rng.uniform(-np.pi, np.pi, size=n_samples)
    p_a_true, _, _ = response_and_derivs(phi_true, params.p0_a, params.contrast_a)
    p_b_true, _, _ = response_and_derivs(phi_true + delta_phi, params.p0_b, params.contrast_b)

    if binomial_readout:
        n_a_int = round(params.n_atoms_a)
        n_b_int = round(params.n_atoms_b)
        y_a = rng.binomial(n_a_int, np.clip(p_a_true, 0.0, 1.0)) / float(n_a_int)
        y_b = rng.binomial(n_b_int, np.clip(p_b_true, 0.0, 1.0)) / float(n_b_int)
        if params.sigma_det_a > 0.0:
            y_a = y_a + rng.normal(0.0, params.sigma_det_a, size=n_samples)
        if params.sigma_det_b > 0.0:
            y_b = y_b + rng.normal(0.0, params.sigma_det_b, size=n_samples)
    else:
        y_a = rng.normal(p_a_true, _true_sigma(p_a_true, params.n_atoms_a, params.sigma_det_a))
        y_b = rng.normal(p_b_true, _true_sigma(p_b_true, params.n_atoms_b, params.sigma_det_b))

    _, dlog_l, ddlog_l = marginal_log_likelihood_and_derivs(
        y_a,
        y_b,
        delta_phi=delta_phi,
        params=params,
        n_phi_global=n_phi_inner,
        n_phi_local=n_phi_local,
    )
    fisher_score = float(np.mean(dlog_l * dlog_l))
    fisher_curv = float(-np.mean(ddlog_l))
    sigma_crb = 1.0 / np.sqrt(fisher_score) if fisher_score > 0.0 else float("inf")
    sigma_mid = float(np.sqrt(known_phase_sql_variance(params)))
    return CramerRaoResult(
        fisher_score=fisher_score,
        fisher_curvature=fisher_curv,
        sigma_crb_rad=float(sigma_crb),
        sigma_known_midfringe_rad=sigma_mid,
        penalty_factor=float(sigma_crb / sigma_mid),
    )


def find_optimal_bias_exact(
    params: InterferometerParams,
    bounds: tuple[float, float] | None = None,
    n_phi_outer: int = 128,
    n_hermite: int = 9,
    n_phi_inner: int = 1024,
    n_phi_local: int = 128,
    xatol: float = 1e-5,
) -> tuple[float, CramerRaoResult]:
    if bounds is None:
        bounds = (1e-3, 0.5 * np.pi)
        try:
            d_est = asymptotic_optimal_bias(params).delta_phi_opt_rad
            bounds = (max(1e-4, 0.2 * d_est), min(0.5 * np.pi, max(0.25, 5.0 * d_est)))
        except ValueError:
            pass

    def _neg_fisher(dp: float) -> float:
        return -compute_cramer_rao_quadrature(
            float(dp),
            params,
            n_phi_outer=n_phi_outer,
            n_hermite=n_hermite,
            n_phi_inner=n_phi_inner,
            n_phi_local=n_phi_local,
        ).fisher_score

    opt = minimize_scalar(_neg_fisher, bounds=bounds, method="bounded", options={"xatol": xatol})
    best_res = compute_cramer_rao_quadrature(
        float(opt.x),
        params,
        n_phi_outer=n_phi_outer,
        n_hermite=n_hermite,
        n_phi_inner=n_phi_inner,
        n_phi_local=n_phi_local,
    )
    return float(opt.x), best_res
