from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ouroboros.mle_common import _gram_diagnostics, _validate_shots

EllipseMethod = Literal["fitzgibbon", "pearson"]


@dataclass(frozen=True)
class BlockSeriesResult:
    block_times_s: np.ndarray
    block_delta_phi_rad: np.ndarray
    block_counts: np.ndarray
    retained_fraction: float


@dataclass(frozen=True)
class HarmonicOLSResult:
    dc: float
    c: float
    s: float
    amplitude: float
    phase_rad: float
    power_frac: float
    gram_ac_efficiency: float


def fit_ellipse_pearson(y_a: np.ndarray, y_b: np.ndarray) -> float:
    ya = np.asarray(y_a, dtype=float) - np.mean(y_a)
    yb = np.asarray(y_b, dtype=float) - np.mean(y_b)
    denom = float(np.sqrt(np.dot(ya, ya) * np.dot(yb, yb)))
    if denom <= 0.0:
        return 0.5 * np.pi
    rho = float(np.clip(np.dot(ya, yb) / denom, -1.0, 1.0))
    return float(np.arccos(rho))


def fit_ellipse_fitzgibbon(y_a: np.ndarray, y_b: np.ndarray) -> float:
    x_raw = np.asarray(y_a, dtype=float)
    y_raw = np.asarray(y_b, dtype=float)
    if x_raw.size < 6:
        raise ValueError("Halir-Flusk ellipse fit requires at least 6 points.")
    sx = float(np.std(x_raw))
    sy = float(np.std(y_raw))
    if sx <= 0.0 or sy <= 0.0:
        return 0.5 * np.pi
    x = (x_raw - float(np.mean(x_raw))) / sx
    y = (y_raw - float(np.mean(y_raw))) / sy

    d1 = np.column_stack([x * x, x * y, y * y])
    d2 = np.column_stack([x, y, np.ones_like(x)])
    s1 = d1.T @ d1
    s2 = d1.T @ d2
    s3 = d2.T @ d2
    t_mat = -np.linalg.pinv(s3, rcond=1e-12) @ s2.T
    m_red = s1 + s2 @ t_mat
    m_eig = np.vstack([0.5 * m_red[2, :], -m_red[1, :], 0.5 * m_red[0, :]])
    eigvals, eigvecs = np.linalg.eig(m_eig)
    vecs = eigvecs.real
    cond = 4.0 * vecs[0, :] * vecs[2, :] - vecs[1, :] ** 2
    valid = np.where((cond > 0.0) & (np.abs(eigvals.imag) < 1e-8))[0]
    if valid.size == 0:
        return fit_ellipse_pearson(x_raw, y_raw)
    idx = valid[int(np.argmin(np.abs(eigvals.real[valid])))]
    a_c, b_c, c_c = vecs[:, idx]
    if a_c < 0.0:
        a_c, b_c, c_c = -a_c, -b_c, -c_c
    cos_phi = float(np.clip(-b_c / (2.0 * np.sqrt(max(a_c * c_c, 1e-30))), -1.0, 1.0))
    return float(np.arccos(cos_phi))


def fit_ellipse_blocks(
    timestamps_s: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    block_size: int = 141,
    method: EllipseMethod = "fitzgibbon",
    max_gap_s: float | None = None,
) -> BlockSeriesResult:
    ya, yb, ts = _validate_shots(y_a, y_b, timestamps_s)
    assert ts is not None
    if block_size < 6:
        raise ValueError("block_size must be at least 6.")
    fitter = fit_ellipse_fitzgibbon if method == "fitzgibbon" else fit_ellipse_pearson

    n_shots = ya.size
    if max_gap_s is None:
        splits = [np.arange(n_shots)]
    else:
        cuts = np.where(np.diff(ts) > max_gap_s)[0] + 1
        splits = np.split(np.arange(n_shots), cuts)

    times_out: list[float] = []
    dphis_out: list[float] = []
    counts_out: list[int] = []
    used = 0
    for seg in splits:
        n_full = seg.size // block_size
        for k in range(n_full):
            idx = seg[k * block_size : (k + 1) * block_size]
            times_out.append(float(np.mean(ts[idx])))
            dphis_out.append(fitter(ya[idx], yb[idx]))
            counts_out.append(int(idx.size))
            used += idx.size

    return BlockSeriesResult(
        block_times_s=np.asarray(times_out, dtype=float),
        block_delta_phi_rad=np.asarray(dphis_out, dtype=float),
        block_counts=np.asarray(counts_out, dtype=np.intp),
        retained_fraction=float(used / n_shots),
    )


def fit_harmonic_ols(
    timestamps_s: np.ndarray,
    values: np.ndarray,
    freq_hz: float,
    t_ref_s: float | None = None,
) -> HarmonicOLSResult:
    ts = np.atleast_1d(np.asarray(timestamps_s, dtype=float))
    vals = np.atleast_1d(np.asarray(values, dtype=float))
    if ts.shape != vals.shape or ts.size < 3:
        raise ValueError("Need at least 3 matched points for harmonic OLS.")
    t_ref = float(ts[0]) if t_ref_s is None else float(t_ref_s)
    wt = 2.0 * np.pi * freq_hz * (ts - t_ref)
    design = np.column_stack([np.ones_like(wt), np.cos(wt), np.sin(wt)])
    beta, _, _, _ = np.linalg.lstsq(design, vals, rcond=None)
    resid = vals - design @ beta
    centered = vals - np.mean(vals)
    ss_tot = float(np.dot(centered, centered))
    ss_res = float(np.dot(resid, resid))
    power = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    gram_eff, _ = _gram_diagnostics(ts, freq_hz, t_ref)
    return HarmonicOLSResult(
        dc=float(beta[0]),
        c=float(beta[1]),
        s=float(beta[2]),
        amplitude=float(np.hypot(beta[1], beta[2])),
        phase_rad=float(np.arctan2(-beta[2], beta[1])),
        power_frac=float(max(power, 0.0)),
        gram_ac_efficiency=gram_eff,
    )
