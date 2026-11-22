from __future__ import annotations

import numpy as np

from ouroboros import (
    InterferometerParams,
    asymptotic_nuisance_fisher,
    asymptotic_optimal_bias,
    compute_cramer_rao_quadrature,
    find_optimal_bias_exact,
    fit_ac_mle,
    fit_ellipse_blocks,
    fit_ellipse_fitzgibbon,
    fit_harmonic_ols,
    quadrature_bias_crb_closed_form,
    small_bias_asymptotic_crb,
)


def main() -> None:
    print("=== 1. Closed-Form vs. Exact Finite-N CRB (Asymmetric & Detection Noise) ===")
    p_bay = InterferometerParams(3100.0, 2040.0, 0.81, 0.84)
    p_hi = InterferometerParams(1e6, 3e6, 0.60, 0.90, sigma_det_a=1e-4, sigma_det_b=2e-4)
    for label, params, dphi, cf in [
        (
            "Baynham small-bias vs exact(0.1022)",
            p_bay,
            0.1022,
            small_bias_asymptotic_crb(p_bay),
        ),
        (
            "Baynham quadrature (pi/2)",
            p_bay,
            0.5 * np.pi,
            quadrature_bias_crb_closed_form(p_bay),
        ),
        (
            "High-N asymmetric + sdet (pi/2)",
            p_hi,
            0.5 * np.pi,
            quadrature_bias_crb_closed_form(p_hi),
        ),
    ]:
        num = compute_cramer_rao_quadrature(dphi, params, n_phi_outer=128, n_hermite=11)
        rel = (num.sigma_crb_rad - cf.sigma_crb_rad) / cf.sigma_crb_rad
        print(
            f"  {label:36s}: num={num.sigma_crb_rad * 1e3:.5f} mrad, "
            f"closed={cf.sigma_crb_rad * 1e3:.5f}, rel={rel:+.2e}, bart={num.bartlett_rel_diff:.2e}"
        )

    print("\n=== 2. Finite-N Optimal Bias Scaling dphi* ~ N^{-3/10} ===")
    for label, params, bnds in [
        ("Baynham (3100, 2040)", p_bay, (0.08, 0.18)),
        ("Large-N (1e5, 1e5, C=0.825)", InterferometerParams(1e5, 1e5, 0.825, 0.825), (0.02, 0.06)),
        ("Large-N (1e6, 1e6, C=0.825)", InterferometerParams(1e6, 1e6, 0.825, 0.825), (0.01, 0.03)),
    ]:
        asymp = asymptotic_optimal_bias(params)
        exact_dp, exact_crb = find_optimal_bias_exact(
            params, bounds=bnds, n_phi_outer=128, n_hermite=9, n_phi_local=96, xatol=2e-4
        )
        print(
            f"  {label:28s}: asymp={asymp.delta_phi_opt_rad:.5f} rad, "
            f"exact={exact_dp:.5f} rad, crb_min={exact_crb.sigma_crb_rad * 1e3:.4f} mrad"
        )

    print("\n=== 3. 5x5 Nuisance Fisher Inflation & Fixed-Nuisance Sensitivity (Baynham) ===")
    for dphi in (0.1022, 0.25, 0.60, 0.5 * np.pi):
        nf = asymptotic_nuisance_fisher(dphi, p_bay)
        print(
            f"  dphi={dphi:.4f} rad: sig_known={nf.sigma_known_nuisance_rad * 1e3:.4f} mrad, "
            f"sig_5d={nf.sigma_joint_5d_rad * 1e3:.4f} mrad ({nf.inflation_5d:.4f}x), "
            f"sens_C=({nf.fixed_nuisance_sensitivity[0]:+.4f},{nf.fixed_nuisance_sensitivity[1]:+.4f})"
        )

    print("\n=== 4. Fitzgibbon 7*sigma_Delta^2 Bias & Unbinned vs. Block-Ellipse AC Recovery ===")
    rng = np.random.default_rng(42)
    phi_grid = np.linspace(-np.pi, np.pi, 2000, endpoint=False)
    eps = rng.normal(0.0, 1.0, (25, phi_grid.size))
    eps = (eps - np.mean(eps, axis=0, keepdims=True)) / np.std(eps, axis=0, keepdims=True)
    phi_mc = np.broadcast_to(phi_grid, eps.shape).ravel()
    sig_delta = 0.005
    d_fitz = fit_ellipse_fitzgibbon(
        0.5 + 0.5 * 0.81 * np.cos(phi_mc),
        0.5 + 0.5 * 0.84 * np.cos(phi_mc + 0.1022 + sig_delta * eps.ravel()),
    )
    slope_emp = (d_fitz**2 - 0.1022**2) / (sig_delta**2)
    slope_pred = 7.0 * 0.1022 / np.tan(0.1022)
    print(
        f"  Fitzgibbon jitter slope (dphi^2 - d0^2)/sig^2: emp={slope_emp:.3f} vs theory={slope_pred:.3f}"
    )

    k_shots = 2820
    dt = 3.0
    ts = np.arange(k_shots) * dt
    f_sig = 0.65 / (141.0 * dt)
    frac = ts / ts[-1]
    ca_t = 0.83 - 0.10 * frac
    cb_t = 0.84 - 0.14 * frac
    p0a_t = 0.50 + 0.025 * np.sin(2.0 * np.pi * frac)
    p0b_t = 0.49 - 0.02 * frac
    phi = rng.uniform(-np.pi, np.pi, k_shots)
    dphi_t = 0.1022 + 0.050 * np.cos(2.0 * np.pi * f_sig * ts - 0.4)
    ya = rng.binomial(3100, np.clip(p0a_t + 0.5 * ca_t * np.cos(phi), 0.0, 1.0)) / 3100.0
    yb = rng.binomial(2040, np.clip(p0b_t + 0.5 * cb_t * np.cos(phi + dphi_t), 0.0, 1.0)) / 2040.0

    r_1b = fit_ac_mle(ts, ya, yb, f_sig, p_bay, n_blocks=1)
    r_5b = fit_ac_mle(ts, ya, yb, f_sig, p_bay, n_blocks=5)
    blk = fit_ellipse_blocks(ts, ya, yb, block_size=141)
    ols = fit_harmonic_ols(blk.block_times_s, blk.block_delta_phi_rad, f_sig, t_ref_s=ts[0])
    print(
        f"  Super-Nyquist f*T_blk=0.65 (true dc=102.20, A=50.00 mrad):\n"
        f"    Unbinned 1-blk: dc={r_1b.dc_rad * 1e3:.2f}, A={r_1b.amplitude_rad * 1e3:.2f}+/-{r_1b.amplitude_se_rad * 1e3:.2f}, logL={r_1b.log_likelihood:.1f}\n"
        f"    Unbinned 5-blk: dc={r_5b.dc_rad * 1e3:.2f}, A={r_5b.amplitude_rad * 1e3:.2f}+/-{r_5b.amplitude_se_rad * 1e3:.2f}, logL={r_5b.log_likelihood:.1f}\n"
        f"    M=141 Fitzgibb: dc={ols.dc * 1e3:.2f}, A={ols.amplitude * 1e3:.2f}"
    )


if __name__ == "__main__":
    main()
