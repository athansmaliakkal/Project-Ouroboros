from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

VarianceMode = Literal["model", "observed", "midfringe"]

_MIN_VARIANCE_PROB = 1e-9


@dataclass(frozen=True)
class InterferometerParams:
    n_atoms_a: float
    n_atoms_b: float
    contrast_a: float
    contrast_b: float
    p0_a: float = 0.5
    p0_b: float = 0.5
    sigma_det_a: float = 0.0
    sigma_det_b: float = 0.0
    variance_mode: VarianceMode = "model"

    def __post_init__(self) -> None:
        if self.n_atoms_a <= 0 or self.n_atoms_b <= 0:
            raise ValueError("Atom numbers must be strictly positive.")
        if not (0.0 < self.contrast_a <= 1.0 and 0.0 < self.contrast_b <= 1.0):
            raise ValueError("Contrasts must be in (0, 1].")
        if not (0.0 < self.p0_a < 1.0 and 0.0 < self.p0_b < 1.0):
            raise ValueError("Mid-fringe offsets p0 must be in (0, 1).")
        if self.sigma_det_a < 0.0 or self.sigma_det_b < 0.0:
            raise ValueError("Detection noise standard deviations must be non-negative.")
        if self.variance_mode not in ("model", "observed", "midfringe"):
            raise ValueError(f"Unsupported variance_mode: {self.variance_mode}")


@dataclass(frozen=True)
class JointLikelihoodGrads:
    log_l: np.ndarray
    d_delta_phi: np.ndarray
    d_lo_a: np.ndarray
    d_hi_a: np.ndarray
    d_lo_b: np.ndarray
    d_hi_b: np.ndarray
    d_log_na: np.ndarray
    d_log_nb: np.ndarray


def response_and_derivs(
    theta: np.ndarray, p0: float | np.ndarray, contrast: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_c = 0.5 * contrast
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    p = p0 + half_c * cos_t
    dp = -half_c * sin_t
    ddp = -half_c * cos_t
    return p, dp, ddp


def _variance_components(
    p: np.ndarray,
    dp: np.ndarray,
    ddp: np.ndarray,
    n_atoms: float,
    sigma_det: float,
    y: np.ndarray | None,
    p0: float | np.ndarray,
    mode: VarianceMode,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    var_det = sigma_det * sigma_det
    if mode == "model":
        p_clip = np.clip(p, _MIN_VARIANCE_PROB, 1.0 - _MIN_VARIANCE_PROB)
        v_qpn = p_clip * (1.0 - p_clip) / n_atoms
        v = v_qpn + var_det
        active = (p > _MIN_VARIANCE_PROB) & (p < 1.0 - _MIN_VARIANCE_PROB)
        dv_dp = np.where(active, (1.0 - 2.0 * p_clip) / n_atoms, 0.0)
        dv = dv_dp * dp
        ddv = np.where(active, ((1.0 - 2.0 * p_clip) * ddp - 2.0 * dp * dp) / n_atoms, 0.0)
        return v, v_qpn, dv_dp, dv, ddv
    if mode == "observed":
        if y is None:
            raise ValueError("Observed variance mode requires measured excitation fraction y.")
        y_clip = np.clip(y, _MIN_VARIANCE_PROB, 1.0 - _MIN_VARIANCE_PROB)
        v_qpn = y_clip * (1.0 - y_clip) / n_atoms
        v = v_qpn + var_det
        zeros = np.zeros_like(v)
        return v, np.broadcast_to(v_qpn, v.shape), zeros, zeros, zeros
    if mode == "midfringe":
        p0_clip = np.clip(p0, _MIN_VARIANCE_PROB, 1.0 - _MIN_VARIANCE_PROB)
        v_qpn = p0_clip * (1.0 - p0_clip) / n_atoms
        v = np.broadcast_to(v_qpn + var_det, p.shape).copy()
        zeros = np.zeros_like(v)
        return v, np.broadcast_to(v_qpn, p.shape), zeros, zeros, zeros
    raise ValueError(f"Unsupported variance_mode: {mode}")


def _log_gaussian_and_derivs(
    y: np.ndarray,
    p: np.ndarray,
    dp: np.ndarray,
    ddp: np.ndarray,
    v: np.ndarray,
    dv: np.ndarray,
    ddv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = y - p
    r2_over_v = (r * r) / v
    log_g = -0.5 * (np.log(2.0 * np.pi * v) + r2_over_v)
    dlog_g = (r * dp) / v + 0.5 * (dv / v) * (r2_over_v - 1.0)
    ddlog_g = (
        (r * ddp - dp * dp) / v
        - (2.0 * r * dp * dv) / (v * v)
        + 0.5 * (ddv / v) * (r2_over_v - 1.0)
        + 0.5 * (dv / v) ** 2 * (1.0 - 2.0 * r2_over_v)
    )
    return log_g, dlog_g, ddlog_g


def _log_gaussian_mean_score(
    y: np.ndarray,
    p: np.ndarray,
    v: np.ndarray,
    dv_dp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = y - p
    r2_over_v = (r * r) / v
    log_g = -0.5 * (np.log(2.0 * np.pi * v) + r2_over_v)
    dlog_g_dp = r / v + 0.5 * (dv_dp / v) * (r2_over_v - 1.0)
    return log_g, dlog_g_dp, r2_over_v
