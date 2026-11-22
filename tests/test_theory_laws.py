import numpy as np
import pytest

from ouroboros import (
    InterferometerParams,
    asymptotic_nuisance_fisher,
    asymptotic_optimal_bias,
    compute_cramer_rao_quadrature,
    find_optimal_bias_exact,
    finite_n_nuisance_fisher,
    quadrature_bias_crb_closed_form,
    small_bias_asymptotic_crb,
)


def test_detection_noise_exact_equivalence() -> None:
    n_a, n_b = 200.0, 320.0
    c_a, c_b = 0.78, 0.85
    sd_a, sd_b = 0.015, 0.022
    p_det = InterferometerParams(
        n_atoms_a=n_a,
        n_atoms_b=n_b,
        contrast_a=c_a,
        contrast_b=c_b,
        sigma_det_a=sd_a,
        sigma_det_b=sd_b,
    )
    alpha_a = c_a / np.sqrt(1.0 + 4.0 * n_a * sd_a * sd_a)
    alpha_b = c_b / np.sqrt(1.0 + 4.0 * n_b * sd_b * sd_b)
    p_eq = InterferometerParams(
        n_atoms_a=n_a, n_atoms_b=n_b, contrast_a=alpha_a, contrast_b=alpha_b
    )

    r_det = compute_cramer_rao_quadrature(0.18, p_det, n_phi_outer=96, n_hermite=9, n_phi_inner=384)
    r_eq = compute_cramer_rao_quadrature(0.18, p_eq, n_phi_outer=96, n_hermite=9, n_phi_inner=384)
    assert r_det.fisher_score == pytest.approx(r_eq.fisher_score, rel=1e-12)
    assert r_det.sigma_crb_rad == pytest.approx(r_eq.sigma_crb_rad, rel=1e-12)
    assert r_det.penalty_factor == pytest.approx(r_eq.penalty_factor, rel=1e-12)


def test_high_n_dual_anchor_bartlett_and_quadrature_closed_form() -> None:
    cases = [
        InterferometerParams(1e6, 1e6, 0.825, 0.825),
        InterferometerParams(1e6, 3e6, 0.60, 0.90, sigma_det_a=1e-4, sigma_det_b=2e-4),
    ]
    for params in cases:
        r_num = compute_cramer_rao_quadrature(
            0.5 * np.pi, params, n_phi_outer=128, n_hermite=11, n_phi_local=128
        )
        r_cf = quadrature_bias_crb_closed_form(params)
        assert r_num.bartlett_rel_diff < 1e-6
        assert r_num.sigma_crb_rad == pytest.approx(r_cf.sigma_crb_rad, rel=1e-4)


def test_small_bias_closed_form_and_offset_parity() -> None:
    p_aligned = InterferometerParams(3100.0, 5200.0, 0.75, 0.88, 0.54, 0.52, 0.005, 0.002)
    p_opp = InterferometerParams(3100.0, 5200.0, 0.75, 0.88, 0.54, 0.48, 0.005, 0.002)

    r_cf = small_bias_asymptotic_crb(p_aligned)
    r_proj = asymptotic_nuisance_fisher(1e-3, p_aligned).sigma_known_nuisance_rad
    assert r_cf.sigma_crb_rad == pytest.approx(r_proj, rel=1e-5)

    assert small_bias_asymptotic_crb(p_aligned).sigma_crb_rad < (
        small_bias_asymptotic_crb(p_opp).sigma_crb_rad
    )
    pi2_aligned = asymptotic_nuisance_fisher(0.5 * np.pi, p_aligned).sigma_known_nuisance_rad
    pi2_opp = asymptotic_nuisance_fisher(0.5 * np.pi, p_opp).sigma_known_nuisance_rad
    assert pi2_aligned == pytest.approx(pi2_opp, rel=1e-12)


def test_optimal_bias_scaling_and_baynham_minimum() -> None:
    opt_1e5 = asymptotic_optimal_bias(InterferometerParams(1e5, 1e5, 0.825, 0.825))
    opt_1e6 = asymptotic_optimal_bias(InterferometerParams(1e6, 1e6, 0.825, 0.825))
    opt_asym = asymptotic_optimal_bias(InterferometerParams(3100.0, 2040.0, 0.838, 0.830))
    assert (opt_1e6.delta_phi_opt_rad / opt_1e5.delta_phi_opt_rad) == pytest.approx(
        10.0 ** (-0.3), rel=1e-12
    )
    for opt in (opt_1e5, opt_1e6, opt_asym):
        eps_curv = opt.c2_curvature * (opt.delta_phi_opt_rad**2)
        assert opt.fold_deficit_at_opt == pytest.approx((2.0 / 3.0) * eps_curv, rel=1e-12)
        assert opt.fold_deficit_at_opt > 1e-5

    p_1e6 = InterferometerParams(1e6, 1e6, 0.825, 0.825)
    dp_1e6, res_1e6 = find_optimal_bias_exact(
        p_1e6, bounds=(0.01, 0.04), n_phi_outer=128, n_hermite=9, n_phi_local=96, xatol=2e-4
    )
    assert res_1e6.bartlett_rel_diff < 1e-4
    assert dp_1e6 == pytest.approx(opt_1e6.delta_phi_opt_rad, rel=0.05)
    assert opt_1e6.sigma_crb_pred_rad == pytest.approx(res_1e6.sigma_crb_rad, rel=1e-3)

    p_bay = InterferometerParams(3100.0, 2040.0, 0.81, 0.84)
    dp_bay, res_bay = find_optimal_bias_exact(
        p_bay, bounds=(0.08, 0.18), n_phi_outer=128, n_hermite=9, xatol=5e-4
    )
    assert 0.11 < dp_bay < 0.135
    assert res_bay.sigma_crb_rad * 1e3 < 43.45


def test_nuisance_fisher_z2_orthogonality_and_broken_parity() -> None:
    p_bay = InterferometerParams(3100.0, 2040.0, 0.81, 0.84)
    nf_asymp = asymptotic_nuisance_fisher(0.1022, p_bay)
    np.testing.assert_allclose(nf_asymp.fisher_matrix[0, 3:], 0.0, atol=1e-12)
    assert nf_asymp.inflation_5d == pytest.approx(nf_asymp.inflation_contrast_only, rel=1e-12)
    assert nf_asymp.inflation_5d == pytest.approx(1.031259, rel=1e-4)
    assert nf_asymp.fixed_nuisance_sensitivity[0] < 0.0
    assert nf_asymp.fixed_nuisance_sensitivity[1] < 0.0

    nf_fin = finite_n_nuisance_fisher(0.1022, p_bay, n_phi_outer=192, n_hermite=11)
    assert nf_fin.inflation_5d == pytest.approx(1.016409, rel=5e-4)

    nf_pi2 = asymptotic_nuisance_fisher(0.5 * np.pi, p_bay)
    assert nf_pi2.inflation_5d == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(nf_pi2.fixed_nuisance_sensitivity, 0.0, atol=1e-12)
    f00 = nf_pi2.fisher_matrix[0, 0]
    assert nf_pi2.fisher_matrix[1, 2] == pytest.approx(
        f00 / (p_bay.contrast_a * p_bay.contrast_b), rel=1e-12
    )

    nf_same = asymptotic_nuisance_fisher(
        0.5 * np.pi, InterferometerParams(3100.0, 2040.0, 0.81, 0.84, 0.55, 0.55)
    )
    nf_flip = asymptotic_nuisance_fisher(
        0.5 * np.pi, InterferometerParams(3100.0, 2040.0, 0.81, 0.84, 0.55, 0.45)
    )
    assert nf_same.fixed_nuisance_sensitivity[2] > 0.2
    assert nf_flip.fixed_nuisance_sensitivity[2] == pytest.approx(
        -nf_same.fixed_nuisance_sensitivity[2], rel=1e-12
    )
