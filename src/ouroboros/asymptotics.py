from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ouroboros.model import InterferometerParams

UNIVERSAL_FOLD_MOMENT_M2 = 0.3717295292433912


@dataclass(frozen=True)
class CramerRaoResult:
    fisher_score: float
    fisher_curvature: float
    sigma_crb_rad: float
    sigma_known_midfringe_rad: float
    penalty_factor: float

    @property
    def bartlett_rel_diff(self) -> float:
        denom = max(abs(self.fisher_score), abs(self.fisher_curvature), 1e-30)
        return abs(self.fisher_score - self.fisher_curvature) / denom

    @property
    def sigma_godambe_rad(self) -> float:
        if self.fisher_score <= 0.0 or self.fisher_curvature <= 0.0:
            return float("inf")
        return float(np.sqrt(self.fisher_score) / self.fisher_curvature)


@dataclass(frozen=True)
class OptimalBiasResult:
    delta_phi_opt_rad: float
    sigma_crb_pred_rad: float
    sigma_crb_small_bias_rad: float
    c2_curvature: float
    fold_deficit_at_opt: float


def known_phase_sql_variance(params: InterferometerParams) -> float:
    m_a = 4.0 * params.p0_a * (1.0 - params.p0_a) + 4.0 * params.n_atoms_a * (params.sigma_det_a**2)
    m_b = 4.0 * params.p0_b * (1.0 - params.p0_b) + 4.0 * params.n_atoms_b * (params.sigma_det_b**2)
    return m_a / (params.contrast_a**2 * params.n_atoms_a) + m_b / (
        params.contrast_b**2 * params.n_atoms_b
    )


def _true_sigma(p: np.ndarray, n_atoms: float, sigma_det: float = 0.0) -> np.ndarray:
    p_clip = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.sqrt(p_clip * (1.0 - p_clip) / n_atoms + sigma_det * sigma_det)


def _detector_reduced_scalars(
    params: InterferometerParams,
) -> tuple[float, float, float, float, float, float, float, float]:
    m_a = 4.0 * params.p0_a * (1.0 - params.p0_a) + 4.0 * params.n_atoms_a * (params.sigma_det_a**2)
    m_b = 4.0 * params.p0_b * (1.0 - params.p0_b) + 4.0 * params.n_atoms_b * (params.sigma_det_b**2)
    a2_a = (params.contrast_a**2) / m_a
    a2_b = (params.contrast_b**2) / m_b
    j_a = params.n_atoms_a * a2_a
    j_b = params.n_atoms_b * a2_b
    sig_mid2 = 1.0 / j_a + 1.0 / j_b
    w_a = j_b / (j_a + j_b)
    g_a = 2.0 * (params.p0_a - 0.5) * params.contrast_a / m_a
    g_b = 2.0 * (params.p0_b - 0.5) * params.contrast_b / m_b
    return m_a, m_b, a2_a, a2_b, sig_mid2, w_a, g_a, g_b


def small_bias_asymptotic_crb(params: InterferometerParams) -> CramerRaoResult:
    _, _, a2_a, a2_b, sig_mid2, w_a, g_a, g_b = _detector_reduced_scalars(params)
    w_b = 1.0 - w_a
    ab2 = w_a * a2_a + w_b * a2_b
    gb = w_a * g_a + w_b * g_b
    disc = np.sqrt(max((1.0 - ab2) ** 2 - 4.0 * gb * gb, 0.0))
    num = np.sqrt(max(2.0 * ab2 * (1.0 - ab2) + 4.0 * gb * gb + 2.0 * ab2 * disc, 0.0))
    den = 2.0 * np.sqrt(ab2 + gb * gb)
    f_norm = (1.0 / ab2) * (1.0 - num / den)
    fisher = float(f_norm / sig_mid2)
    sigma_crb = float(1.0 / np.sqrt(fisher))
    sigma_mid = float(np.sqrt(sig_mid2))
    return CramerRaoResult(
        fisher_score=fisher,
        fisher_curvature=fisher,
        sigma_crb_rad=sigma_crb,
        sigma_known_midfringe_rad=sigma_mid,
        penalty_factor=float(sigma_crb / sigma_mid),
    )


def quadrature_bias_crb_closed_form(params: InterferometerParams) -> CramerRaoResult:
    if not (np.isclose(params.p0_a, 0.5, atol=1e-12) and np.isclose(params.p0_b, 0.5, atol=1e-12)):
        raise ValueError(
            "quadrature_bias_crb_closed_form requires p0_a == p0_b == 0.5; "
            "use asymptotic_nuisance_fisher(np.pi / 2, params) for asymmetric offsets."
        )
    m_a, m_b, _, _, sig_mid2, _, _, _ = _detector_reduced_scalars(params)
    sig_ext_a = np.sqrt(
        max(m_a - params.contrast_a**2, 0.0) / (params.n_atoms_a * params.contrast_a**2)
    )
    sig_ext_b = np.sqrt(
        max(m_b - params.contrast_b**2, 0.0) / (params.n_atoms_b * params.contrast_b**2)
    )
    s_term = np.sqrt(sig_mid2 + 2.0 * sig_ext_a * sig_ext_b)
    sigma_crb = float(np.sqrt(s_term * (s_term + sig_ext_a + sig_ext_b)))
    fisher = float(1.0 / (sigma_crb * sigma_crb))
    sigma_mid = float(np.sqrt(sig_mid2))
    return CramerRaoResult(
        fisher_score=fisher,
        fisher_curvature=fisher,
        sigma_crb_rad=sigma_crb,
        sigma_known_midfringe_rad=sigma_mid,
        penalty_factor=float(sigma_crb / sigma_mid),
    )


def asymptotic_optimal_bias(params: InterferometerParams) -> OptimalBiasResult:
    if not (np.isclose(params.p0_a, 0.5, atol=1e-12) and np.isclose(params.p0_b, 0.5, atol=1e-12)):
        raise ValueError("Closed-form asymptotic_optimal_bias requires p0_a == p0_b == 0.5.")
    m_a, m_b, _, _, sig_mid2, _, _, _ = _detector_reduced_scalars(params)
    ext_a2 = max(m_a - params.contrast_a**2, 0.0) / (params.n_atoms_a * params.contrast_a**2)
    ext_b2 = max(m_b - params.contrast_b**2, 0.0) / (params.n_atoms_b * params.contrast_b**2)
    if ext_a2 <= 0.0 or ext_b2 <= 0.0:
        raise ValueError(
            "Asymptotic O(N^{-3/10}) formula requires non-zero extremum variance (alpha_k < 1)."
        )
    ext_tot2 = ext_a2 + ext_b2
    sig_mid = float(np.sqrt(sig_mid2))
    sig_ext = float(np.sqrt(ext_tot2))
    b_bar = sig_ext / sig_mid

    c2 = ((b_bar * b_bar + 3.0) * (1.0 + b_bar) / (8.0 * b_bar)) * (
        4.0 * ext_a2 * ext_b2 / (ext_tot2 * ext_tot2)
    )
    denom_curv = ext_a2 * ext_b2 * (3.0 + b_bar * b_bar)
    dphi_opt = float(
        ((6.0 * UNIVERSAL_FOLD_MOMENT_M2 / np.pi) * sig_mid * (sig_ext**6) / denom_curv) ** 0.2
    )

    f0 = small_bias_asymptotic_crb(params).fisher_score
    eps_fold = (2.0 * UNIVERSAL_FOLD_MOMENT_M2 / (np.pi * f0)) * (sig_ext / (dphi_opt**3))
    eps_curv = c2 * (dphi_opt**2)
    f_pred = f0 * max(1.0 - eps_fold - eps_curv, 1e-6)
    sig_small = float(1.0 / np.sqrt(f0))
    return OptimalBiasResult(
        delta_phi_opt_rad=dphi_opt,
        sigma_crb_pred_rad=float(1.0 / np.sqrt(f_pred)),
        sigma_crb_small_bias_rad=sig_small,
        c2_curvature=float(c2),
        fold_deficit_at_opt=float(eps_fold),
    )
