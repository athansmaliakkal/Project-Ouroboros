import numpy as np
import pytest

from ouroboros import (
    InterferometerParams,
    compute_cramer_rao_mc,
    compute_cramer_rao_quadrature,
    marginal_log_likelihood_and_derivs,
)


def test_likelihood_normalization_2d() -> None:
    params = InterferometerParams(
        n_atoms_a=3100.0,
        n_atoms_b=2040.0,
        contrast_a=0.81,
        contrast_b=0.84,
        variance_mode="model",
    )
    grid = np.linspace(-0.05, 1.05, 276)
    dy = grid[1] - grid[0]
    ya, yb = np.meshgrid(grid, grid, indexing="ij")
    ya_flat = ya.ravel()
    yb_flat = yb.ravel()
    total = 0.0
    batch = 4096
    for start in range(0, ya_flat.size, batch):
        log_l, _, _ = marginal_log_likelihood_and_derivs(
            ya_flat[start : start + batch],
            yb_flat[start : start + batch],
            delta_phi=0.1022,
            params=params,
            n_phi_global=384,
        )
        total += float(np.sum(np.exp(log_l)))
    integral = total * dy * dy
    assert integral == pytest.approx(1.0, rel=1e-4)


def test_analytic_derivatives_match_finite_diff() -> None:
    params = InterferometerParams(
        n_atoms_a=3100.0,
        n_atoms_b=2040.0,
        contrast_a=0.81,
        contrast_b=0.84,
        variance_mode="model",
    )
    y_a = np.array([0.23, 0.51, 0.85])
    y_b = np.array([0.19, 0.56, 0.80])
    dphi = 0.25
    eps = 1e-5

    _, dl, ddl = marginal_log_likelihood_and_derivs(y_a, y_b, dphi, params)
    lp, dlp, _ = marginal_log_likelihood_and_derivs(y_a, y_b, dphi + eps, params)
    lm, dlm, _ = marginal_log_likelihood_and_derivs(y_a, y_b, dphi - eps, params)

    fd_1 = (lp - lm) / (2.0 * eps)
    fd_2 = (dlp - dlm) / (2.0 * eps)
    np.testing.assert_allclose(dl, fd_1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ddl, fd_2, rtol=1e-5, atol=1e-5)


def test_bartlett_identity_and_baynham_baseline() -> None:
    params = InterferometerParams(
        n_atoms_a=3100.0,
        n_atoms_b=2040.0,
        contrast_a=0.81,
        contrast_b=0.84,
        variance_mode="model",
    )
    res = compute_cramer_rao_quadrature(0.1022, params, n_phi_outer=256, n_hermite=15)
    assert res.bartlett_rel_diff < 5e-5
    assert abs(res.sigma_crb_rad * 1e3 - 43.5) <= 1.6


def test_unit_contrast_limit_recovers_pezze_2025() -> None:
    params = InterferometerParams(
        n_atoms_a=1e6,
        n_atoms_b=1e6,
        contrast_a=0.9999,
        contrast_b=0.9999,
        variance_mode="model",
    )
    res = compute_cramer_rao_quadrature(0.5 * np.pi, params, n_phi_outer=256, n_hermite=11)
    assert res.penalty_factor == pytest.approx(1.0, abs=0.02)


def test_zero_bias_degeneracy_limit() -> None:
    params = InterferometerParams(
        n_atoms_a=3100.0,
        n_atoms_b=2040.0,
        contrast_a=0.81,
        contrast_b=0.84,
        variance_mode="model",
    )
    res_zero = compute_cramer_rao_quadrature(1e-4, params)
    res_opt = compute_cramer_rao_quadrature(0.5 * np.pi, params)
    assert res_zero.sigma_crb_rad > 1.3 * res_opt.sigma_crb_rad


def test_mc_agrees_with_quadrature() -> None:
    params = InterferometerParams(
        n_atoms_a=3100.0,
        n_atoms_b=2040.0,
        contrast_a=0.81,
        contrast_b=0.84,
        variance_mode="model",
    )
    q_res = compute_cramer_rao_quadrature(0.1022, params)
    mc_res = compute_cramer_rao_mc(0.1022, params, n_samples=15000, seed=7)
    assert mc_res.sigma_crb_rad == pytest.approx(q_res.sigma_crb_rad, rel=0.03)


def test_midfringe_godambe_sandwich_and_bartlett_violation() -> None:
    p_model = InterferometerParams(
        n_atoms_a=3100.0,
        n_atoms_b=2040.0,
        contrast_a=0.838,
        contrast_b=0.830,
        variance_mode="model",
    )
    p_mid = InterferometerParams(
        n_atoms_a=3100.0,
        n_atoms_b=2040.0,
        contrast_a=0.838,
        contrast_b=0.830,
        variance_mode="midfringe",
    )
    r_mod = compute_cramer_rao_quadrature(0.1022, p_model, n_phi_outer=192, n_hermite=13)
    r_mid = compute_cramer_rao_quadrature(0.1022, p_mid, n_phi_outer=192, n_hermite=13)
    assert r_mod.bartlett_rel_diff < 1e-4
    assert r_mod.sigma_godambe_rad == pytest.approx(r_mod.sigma_crb_rad, rel=1e-4)
    assert r_mid.bartlett_rel_diff > 0.10
    assert r_mid.fisher_curvature > r_mid.fisher_score > 0.0
    assert r_mod.sigma_crb_rad < r_mid.sigma_godambe_rad < r_mid.sigma_crb_rad
    assert r_mid.sigma_godambe_rad * 1e3 == pytest.approx(45.27, abs=0.15)
    assert r_mid.sigma_crb_rad * 1e3 == pytest.approx(53.38, abs=0.15)
