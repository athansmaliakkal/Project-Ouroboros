import numpy as np
import pytest

from ouroboros import (
    InterferometerParams,
    fit_ac_mle,
    fit_ellipse_blocks,
    fit_ellipse_fitzgibbon,
    fit_ellipse_pearson,
    fit_harmonic_ols,
    fit_static_mle,
    joint_marginal_log_likelihood_and_grads,
    scan_ac_spectrum,
)


@pytest.mark.parametrize(
    ("n_atoms_a", "n_atoms_b", "sdet_a", "sdet_b"),
    [
        (3100.0, 2040.0, 0.005, 0.004),
        (5e5, 8e5, 0.0, 1e-4),
    ],
)
def test_joint_likelihood_gradients_match_finite_diff(
    n_atoms_a: float, n_atoms_b: float, sdet_a: float, sdet_b: float
) -> None:
    phi_s = np.array([0.4, 1.1, 2.2])
    dp = np.array([0.12, 0.15, 0.08])
    ca = np.array([0.81, 0.78, 0.80])
    cb = np.array([0.84, 0.75, 0.82])
    p0a = np.array([0.51, 0.49, 0.52])
    p0b = np.array([0.48, 0.53, 0.49])
    ya = p0a + 0.5 * ca * np.cos(phi_s) + 0.5 / np.sqrt(n_atoms_a) * np.array([0.4, -0.5, 0.3])
    yb = (
        p0b + 0.5 * cb * np.cos(phi_s + dp) + 0.5 / np.sqrt(n_atoms_b) * np.array([-0.3, 0.6, -0.4])
    )

    grads = joint_marginal_log_likelihood_and_grads(
        ya,
        yb,
        dp,
        n_atoms_a,
        n_atoms_b,
        ca,
        cb,
        p0a,
        p0b,
        sigma_det_a=sdet_a,
        sigma_det_b=sdet_b,
        n_phi_global=384,
        n_phi_local=128,
    )
    eps = 1e-6
    lo_a, hi_a = p0a - 0.5 * ca, p0a + 0.5 * ca
    lo_b, hi_b = p0b - 0.5 * cb, p0b + 0.5 * cb

    def _fd(
        d_dp: float = 0.0,
        d_loa: float = 0.0,
        d_hia: float = 0.0,
        d_lob: float = 0.0,
        d_hib: float = 0.0,
        d_lna: float = 0.0,
        d_lnb: float = 0.0,
    ) -> np.ndarray:
        out = []
        for sgn in (1.0, -1.0):
            la = lo_a + sgn * eps * d_loa
            ha = hi_a + sgn * eps * d_hia
            lb = lo_b + sgn * eps * d_lob
            hb = hi_b + sgn * eps * d_hib
            out.append(
                joint_marginal_log_likelihood_and_grads(
                    ya,
                    yb,
                    dp + sgn * eps * d_dp,
                    n_atoms_a * np.exp(sgn * eps * d_lna),
                    n_atoms_b * np.exp(sgn * eps * d_lnb),
                    ha - la,
                    hb - lb,
                    0.5 * (la + ha),
                    0.5 * (lb + hb),
                    sigma_det_a=sdet_a,
                    sigma_det_b=sdet_b,
                    n_phi_global=384,
                    n_phi_local=128,
                ).log_l
            )
        return (out[0] - out[1]) / (2.0 * eps)

    np.testing.assert_allclose(grads.d_delta_phi, _fd(d_dp=1.0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grads.d_lo_a, _fd(d_loa=1.0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grads.d_hi_a, _fd(d_hia=1.0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grads.d_lo_b, _fd(d_lob=1.0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grads.d_hi_b, _fd(d_hib=1.0), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(grads.d_log_na, _fd(d_lna=1.0), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grads.d_log_nb, _fd(d_lnb=1.0), rtol=1e-5, atol=1e-6)


def test_static_mle_and_smooth_extreme_offset_box() -> None:
    rng = np.random.default_rng(11)
    n_shots = 2500
    phi = rng.uniform(-np.pi, np.pi, n_shots)
    true_dphi = 0.22
    pa = 0.68 + 0.5 * 0.22 * np.cos(phi)
    pb = 0.34 + 0.5 * 0.26 * np.cos(phi + true_dphi)
    ya = rng.binomial(3000, pa) / 3000.0
    yb = rng.binomial(2200, pb) / 2200.0

    init_p = InterferometerParams(3000.0, 2200.0, 0.30, 0.30, 0.50, 0.50)
    res_5d = fit_static_mle(ya, yb, init_p, init_delta_phi=0.15, free_nuisance=True, n_blocks=1)
    assert res_5d.success
    assert res_5d.max_abs_grad < 1e-6
    assert abs(res_5d.delta_phi_rad - true_dphi) < 4.0 * res_5d.delta_phi_se_rad
    assert res_5d.p0_a == pytest.approx(0.68, abs=0.015)
    assert res_5d.contrast_a == pytest.approx(0.22, abs=0.015)

    res_7d = fit_static_mle(
        ya, yb, init_p, init_delta_phi=0.15, free_nuisance=True, free_atom_numbers=True
    )
    assert res_7d.success
    assert 2400.0 < res_7d.n_atoms_a < 3700.0
    assert 1700.0 < res_7d.n_atoms_b < 2800.0


def test_fitzgibbon_exactness_and_seven_sigma_bias_law() -> None:
    phi = np.linspace(-np.pi, np.pi, 4000, endpoint=False)
    for dphi_true in (0.05, 0.1022, 0.5 * np.pi, np.pi - 0.25):
        ya = 0.52 + 0.5 * 0.81 * np.cos(phi)
        yb = 0.47 + 0.5 * 0.84 * np.cos(phi + dphi_true)
        assert fit_ellipse_fitzgibbon(ya, yb) == pytest.approx(dphi_true, abs=1e-10)
        assert fit_ellipse_pearson(ya, yb) == pytest.approx(dphi_true, abs=1e-10)

    rng = np.random.default_rng(99)
    sig_delta = 0.005
    phi_grid = np.linspace(-np.pi, np.pi, 2000, endpoint=False)
    eps = rng.normal(0.0, 1.0, (25, phi_grid.size))
    eps = (eps - np.mean(eps, axis=0, keepdims=True)) / np.std(eps, axis=0, keepdims=True)
    phi_mc = np.broadcast_to(phi_grid, eps.shape).ravel()
    delta_i = 0.1022 + sig_delta * eps.ravel()
    ya_jit = 0.5 + 0.5 * 0.81 * np.cos(phi_mc)
    yb_jit = 0.5 + 0.5 * 0.84 * np.cos(phi_mc + delta_i)
    d_fitz = fit_ellipse_fitzgibbon(ya_jit, yb_jit)
    d_pear = fit_ellipse_pearson(ya_jit, yb_jit)
    ratio_fitz = (d_fitz**2 - 0.1022**2) / (sig_delta**2)
    ratio_pear = (d_pear**2 - 0.1022**2) / (sig_delta**2)
    expected_fitz_slope = 7.0 * 0.1022 / np.tan(0.1022)
    expected_pear_slope = 0.1022 / np.tan(0.1022)
    assert ratio_fitz == pytest.approx(expected_fitz_slope, rel=0.02)
    assert ratio_pear == pytest.approx(expected_pear_slope, rel=0.01)


def test_unbinned_ac_beats_fitzgibbon_under_drift_supernyquist_and_gaps() -> None:
    rng = np.random.default_rng(7)
    on_idx = np.concatenate([np.arange(k * 200, k * 200 + 100) for k in range(14)])
    dt = 3.0
    ts = on_idx * dt + rng.normal(0.0, 0.15 * dt, on_idx.size)
    ts = np.sort(ts)

    f_sig = 0.65 / (141.0 * dt)
    dc_true, amp_true = 0.1022, 0.045
    tau = ts - ts[0]
    dphi_t = dc_true + amp_true * np.cos(2.0 * np.pi * f_sig * tau - 0.3)

    frac = tau / tau[-1]
    ca_t = 0.83 - 0.10 * frac
    cb_t = 0.84 - 0.14 * frac
    p0a_t = 0.50 + 0.025 * np.sin(2.0 * np.pi * frac)
    p0b_t = 0.49 - 0.03 * frac
    phi = rng.uniform(-np.pi, np.pi, ts.size)
    ya = rng.binomial(3100, np.clip(p0a_t + 0.5 * ca_t * np.cos(phi), 0.0, 1.0)) / 3100.0
    yb = rng.binomial(2040, np.clip(p0b_t + 0.5 * cb_t * np.cos(phi + dphi_t), 0.0, 1.0)) / 2040.0

    init_p = InterferometerParams(3100.0, 2040.0, 0.81, 0.84)
    blk_gapped = fit_ellipse_blocks(ts, ya, yb, block_size=141, max_gap_s=20.0)
    assert blk_gapped.retained_fraction == 0.0

    blk_naive = fit_ellipse_blocks(ts, ya, yb, block_size=141, max_gap_s=None)
    ols_naive = fit_harmonic_ols(blk_naive.block_times_s, blk_naive.block_delta_phi_rad, f_sig)
    assert ols_naive.amplitude < 0.45 * amp_true

    res_1b = fit_ac_mle(ts, ya, yb, f_sig, init_p, n_blocks=1)
    res_5b = fit_ac_mle(ts, ya, yb, f_sig, init_p, n_blocks=5)
    assert res_5b.success
    assert res_5b.gram_ac_efficiency > 0.90
    assert 2.0 * (res_5b.log_likelihood - res_1b.log_likelihood) > 50.0
    assert abs(res_5b.dc_rad - dc_true) < 0.015
    assert abs(res_5b.amplitude_rad - amp_true) < 0.015

    scan = scan_ac_spectrum(
        ts, ya, yb, np.array([0.4 * f_sig, f_sig, 1.4 * f_sig]), init_p, n_blocks=5
    )
    assert int(np.argmax(scan.two_delta_log_l)) == 1
    assert scan.two_delta_log_l[1] > 100.0


def test_ac_fold_search_escapes_subfold_basin() -> None:
    rng = np.random.default_rng(314)
    ts = np.arange(1000, dtype=float) * 3.2
    f_sig = 0.25 / (141.0 * 3.2)
    dc_true, amp_true = 0.1022, 0.225
    phi = rng.uniform(-np.pi, np.pi, ts.size)
    ya = rng.binomial(2500, np.clip(0.5 + 0.5 * 0.825 * np.cos(phi), 0.0, 1.0)) / 2500.0
    yb = (
        rng.binomial(
            2500,
            np.clip(
                0.5
                + 0.5
                * 0.825
                * np.cos(phi + dc_true + amp_true * np.cos(2.0 * np.pi * f_sig * ts - 0.35)),
                0.0,
                1.0,
            ),
        )
        / 2500.0
    )
    p = InterferometerParams(2500.0, 2500.0, 0.825, 0.825)
    r_trapped = fit_ac_mle(
        ts, ya, yb, f_sig, p, init_dc=dc_true, free_nuisance=False, fold_search=False
    )
    r_global = fit_ac_mle(
        ts, ya, yb, f_sig, p, init_dc=dc_true, free_nuisance=False, fold_search=True
    )
    assert r_trapped.amplitude_rad < 0.16
    assert r_global.log_likelihood > r_trapped.log_likelihood + 300.0
    assert r_global.amplitude_rad == pytest.approx(amp_true, abs=0.015)
    assert r_global.dc_rad == pytest.approx(dc_true, abs=0.015)


def test_unsorted_timestamps_match_chronological_order() -> None:
    rng = np.random.default_rng(42)
    n_shots = 400
    ts_sorted = np.arange(n_shots, dtype=float) * 3.0
    frac = ts_sorted / ts_sorted[-1]
    ca_t = 0.84 - 0.12 * frac
    cb_t = 0.83 - 0.10 * frac
    f_sig = 0.002
    dphi_t = 0.105 + 0.025 * np.cos(2.0 * np.pi * f_sig * ts_sorted - 0.4)
    phi = rng.uniform(-np.pi, np.pi, n_shots)
    ya_sorted = rng.binomial(2500, np.clip(0.5 + 0.5 * ca_t * np.cos(phi), 0.0, 1.0)) / 2500.0
    yb_sorted = (
        rng.binomial(2500, np.clip(0.5 + 0.5 * cb_t * np.cos(phi + dphi_t), 0.0, 1.0)) / 2500.0
    )

    perm = np.concatenate([np.arange(200, 400), np.arange(0, 200)])
    ts_unsorted = ts_sorted[perm]
    ya_unsorted = ya_sorted[perm]
    yb_unsorted = yb_sorted[perm]

    p = InterferometerParams(2500.0, 2500.0, 0.82, 0.82)
    st_s = fit_static_mle(ya_sorted, yb_sorted, p, n_blocks=4, timestamps_s=ts_sorted)
    st_u = fit_static_mle(ya_unsorted, yb_unsorted, p, n_blocks=4, timestamps_s=ts_unsorted)
    assert st_u.delta_phi_rad == pytest.approx(st_s.delta_phi_rad, abs=1e-10)
    assert st_u.log_likelihood == pytest.approx(st_s.log_likelihood, abs=1e-8)

    ac_s = fit_ac_mle(ts_sorted, ya_sorted, yb_sorted, f_sig, p, n_blocks=4)
    ac_u = fit_ac_mle(ts_unsorted, ya_unsorted, yb_unsorted, f_sig, p, n_blocks=4)
    assert ac_u.amplitude_rad == pytest.approx(ac_s.amplitude_rad, abs=1e-10)
    assert ac_u.dc_rad == pytest.approx(ac_s.dc_rad, abs=1e-10)
    assert ac_u.log_likelihood == pytest.approx(ac_s.log_likelihood, abs=1e-8)

    blk_s = fit_ellipse_blocks(ts_sorted, ya_sorted, yb_sorted, block_size=50, max_gap_s=20.0)
    blk_u = fit_ellipse_blocks(ts_unsorted, ya_unsorted, yb_unsorted, block_size=50, max_gap_s=20.0)
    np.testing.assert_allclose(
        blk_u.block_delta_phi_rad, blk_s.block_delta_phi_rad, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(blk_u.block_times_s, blk_s.block_times_s, rtol=0, atol=1e-12)
