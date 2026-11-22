from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ouroboros.model import InterferometerParams, JointLikelihoodGrads

_BOX_MIN = 1e-4
_BOX_MAX = 1.0 - 1e-4
_S_MIN = 1e-3


@dataclass(frozen=True)
class StaticMLEResult:
    delta_phi_rad: float
    delta_phi_se_rad: float
    contrast_a: float
    contrast_b: float
    p0_a: float
    p0_b: float
    n_atoms_a: float
    n_atoms_b: float
    contrast_a_blocks: np.ndarray
    contrast_b_blocks: np.ndarray
    p0_a_blocks: np.ndarray
    p0_b_blocks: np.ndarray
    log_likelihood: float
    success: bool
    max_abs_grad: float


@dataclass(frozen=True)
class ACMLEResult:
    freq_hz: float
    dc_rad: float
    c_rad: float
    s_rad: float
    amplitude_rad: float
    phase_rad: float
    dc_se_rad: float
    c_se_rad: float
    s_se_rad: float
    amplitude_se_rad: float
    contrast_a_blocks: np.ndarray
    contrast_b_blocks: np.ndarray
    p0_a_blocks: np.ndarray
    p0_b_blocks: np.ndarray
    log_likelihood: float
    gram_ac_efficiency: float
    gram_cond: float
    success: bool
    max_abs_grad: float


@dataclass(frozen=True)
class ACScanResult:
    frequencies_hz: np.ndarray
    log_likelihood_null: float
    log_likelihood: np.ndarray
    two_delta_log_l: np.ndarray
    amplitude_rad: np.ndarray
    dc_rad: np.ndarray
    c_rad: np.ndarray
    s_rad: np.ndarray
    gram_ac_efficiency: np.ndarray


@dataclass(frozen=True)
class _UnpackedNuisance:
    lo_a: np.ndarray
    s_a: np.ndarray
    lo_b: np.ndarray
    s_b: np.ndarray
    ca_blocks: np.ndarray
    cb_blocks: np.ndarray
    pa_blocks: np.ndarray
    pb_blocks: np.ndarray
    ca_shot: float | np.ndarray
    cb_shot: float | np.ndarray
    pa_shot: float | np.ndarray
    pb_shot: float | np.ndarray


def _params_to_box(contrast: float, p0: float) -> tuple[float, float]:
    lo = float(np.clip(p0 - 0.5 * contrast, _BOX_MIN, _BOX_MAX - 0.01))
    hi = float(np.clip(p0 + 0.5 * contrast, lo + 1e-3, _BOX_MAX))
    s = float(np.clip((hi - lo) / (_BOX_MAX - lo), _S_MIN, 1.0))
    return lo, s


def _box_to_contrast_p0(
    lo: np.ndarray | float, s: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray]:
    lo_a = np.asarray(lo, dtype=float)
    s_a = np.asarray(s, dtype=float)
    hi = lo_a + s_a * (_BOX_MAX - lo_a)
    return hi - lo_a, 0.5 * (lo_a + hi)


def _box_chain_grad(
    d_lo: np.ndarray, d_hi: np.ndarray, lo: np.ndarray, s: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    g_lo = d_lo + (1.0 - s) * d_hi
    g_s = (_BOX_MAX - lo) * d_hi
    return g_lo, g_s


def _make_block_ids(n_shots: int, n_blocks: int) -> np.ndarray:
    if n_blocks < 1 or n_blocks > n_shots:
        raise ValueError(f"n_blocks={n_blocks} must satisfy 1 <= n_blocks <= {n_shots}.")
    if n_blocks == 1:
        return np.zeros(n_shots, dtype=np.intp)
    block_ids = np.empty(n_shots, dtype=np.intp)
    for b_idx, idx in enumerate(np.array_split(np.arange(n_shots), n_blocks)):
        block_ids[idx] = b_idx
    return block_ids


def _validate_shots(
    y_a: np.ndarray, y_b: np.ndarray, timestamps_s: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    ya = np.atleast_1d(np.asarray(y_a, dtype=float))
    yb = np.atleast_1d(np.asarray(y_b, dtype=float))
    if ya.shape != yb.shape or ya.ndim != 1 or ya.size == 0:
        raise ValueError("y_a and y_b must be non-empty 1D arrays of matching shape.")
    if timestamps_s is None:
        return ya, yb, None
    ts = np.atleast_1d(np.asarray(timestamps_s, dtype=float))
    if ts.shape != ya.shape:
        raise ValueError("timestamps_s must match y_a shape.")
    if ts.size > 1 and np.any(np.diff(ts) < 0.0):
        order = np.argsort(ts, kind="mergesort")
        ts = ts[order]
        ya = ya[order]
        yb = yb[order]
    return ya, yb, ts


def _gram_diagnostics(
    timestamps_s: np.ndarray, freq_hz: float, t_ref_s: float
) -> tuple[float, float]:
    omega = 2.0 * np.pi * freq_hz
    wt = omega * (timestamps_s - t_ref_s)
    h = np.column_stack([np.ones_like(wt), np.cos(wt), np.sin(wt)])
    gram = (h.T @ h) / float(wt.size)
    g11 = gram[1:, 1:]
    g10 = gram[1:, :1]
    schur = g11 - g10 @ g10.T
    eig_schur = float(np.min(np.linalg.eigvalsh(schur)))
    eig_full = np.linalg.eigvalsh(gram)
    cond = float(eig_full[-1] / max(eig_full[0], 1e-15))
    return float(max(eig_schur / 0.5, 0.0)), cond


def _init_nuisance_box(
    init_params: InterferometerParams,
    n_blocks: int,
    init_nuisance_blocks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[list[float], list[tuple[float, float]]]:
    x0: list[float] = []
    bnds: list[tuple[float, float]] = []
    if init_nuisance_blocks is not None:
        ca_0, cb_0, pa_0, pb_0 = init_nuisance_blocks
        for b in range(n_blocks):
            la, sa = _params_to_box(float(ca_0[b]), float(pa_0[b]))
            lb, sb = _params_to_box(float(cb_0[b]), float(pb_0[b]))
            x0.extend([la, sa, lb, sb])
    else:
        la, sa = _params_to_box(init_params.contrast_a, init_params.p0_a)
        lb, sb = _params_to_box(init_params.contrast_b, init_params.p0_b)
        for _ in range(n_blocks):
            x0.extend([la, sa, lb, sb])
    blk_bnds = [
        (_BOX_MIN, _BOX_MAX - 0.01),
        (_S_MIN, 1.0),
        (_BOX_MIN, _BOX_MAX - 0.01),
        (_S_MIN, 1.0),
    ]
    for _ in range(n_blocks):
        bnds.extend(blk_bnds)
    return x0, bnds


def _unpack_block_nuisance(
    nuis_vec: np.ndarray,
    block_ids: np.ndarray,
    n_blocks: int,
    init_params: InterferometerParams,
    free_nuisance: bool,
) -> _UnpackedNuisance:
    if not free_nuisance:
        z = np.array([], dtype=float)
        return _UnpackedNuisance(
            lo_a=z,
            s_a=z,
            lo_b=z,
            s_b=z,
            ca_blocks=np.full(n_blocks, init_params.contrast_a, dtype=float),
            cb_blocks=np.full(n_blocks, init_params.contrast_b, dtype=float),
            pa_blocks=np.full(n_blocks, init_params.p0_a, dtype=float),
            pb_blocks=np.full(n_blocks, init_params.p0_b, dtype=float),
            ca_shot=init_params.contrast_a,
            cb_shot=init_params.contrast_b,
            pa_shot=init_params.p0_a,
            pb_shot=init_params.p0_b,
        )

    mat = nuis_vec.reshape(n_blocks, 4)
    lo_a, s_a, lo_b, s_b = mat[:, 0], mat[:, 1], mat[:, 2], mat[:, 3]
    ca_b, pa_b = _box_to_contrast_p0(lo_a, s_a)
    cb_b, pb_b = _box_to_contrast_p0(lo_b, s_b)
    return _UnpackedNuisance(
        lo_a=lo_a,
        s_a=s_a,
        lo_b=lo_b,
        s_b=s_b,
        ca_blocks=ca_b,
        cb_blocks=cb_b,
        pa_blocks=pa_b,
        pb_blocks=pb_b,
        ca_shot=float(ca_b[0]) if n_blocks == 1 else ca_b[block_ids],
        cb_shot=float(cb_b[0]) if n_blocks == 1 else cb_b[block_ids],
        pa_shot=float(pa_b[0]) if n_blocks == 1 else pa_b[block_ids],
        pb_shot=float(pb_b[0]) if n_blocks == 1 else pb_b[block_ids],
    )


def _nuisance_block_grad(
    grads: JointLikelihoodGrads,
    block_ids: np.ndarray,
    n_blocks: int,
    nuis: _UnpackedNuisance,
) -> np.ndarray:
    d_loa = np.bincount(block_ids, weights=grads.d_lo_a, minlength=n_blocks)
    d_hia = np.bincount(block_ids, weights=grads.d_hi_a, minlength=n_blocks)
    d_lob = np.bincount(block_ids, weights=grads.d_lo_b, minlength=n_blocks)
    d_hib = np.bincount(block_ids, weights=grads.d_hi_b, minlength=n_blocks)
    ga_l, ga_s = _box_chain_grad(d_loa, d_hia, nuis.lo_a, nuis.s_a)
    gb_l, gb_s = _box_chain_grad(d_lob, d_hib, nuis.lo_b, nuis.s_b)
    return np.column_stack([ga_l, ga_s, gb_l, gb_s]).ravel()


def _compute_bhhh_covariance(
    n_shots: int,
    n_phase_params: int,
    phase_jacobian_rows: np.ndarray,
    grads: JointLikelihoodGrads,
    block_ids: np.ndarray,
    n_blocks: int,
    nuis: _UnpackedNuisance,
    free_nuisance: bool,
    free_atom_numbers: bool,
) -> np.ndarray:
    n_nuis = 4 * n_blocks if free_nuisance else 0
    n_atom = 2 if free_atom_numbers else 0
    dim = n_phase_params + n_nuis + n_atom
    j_mat = np.zeros((n_shots, dim), dtype=float)
    j_mat[:, :n_phase_params] = phase_jacobian_rows

    if free_nuisance:
        g_loa, g_sa = _box_chain_grad(
            grads.d_lo_a, grads.d_hi_a, nuis.lo_a[block_ids], nuis.s_a[block_ids]
        )
        g_lob, g_sb = _box_chain_grad(
            grads.d_lo_b, grads.d_hi_b, nuis.lo_b[block_ids], nuis.s_b[block_ids]
        )
        rows = np.arange(n_shots)
        base = n_phase_params + 4 * block_ids
        j_mat[rows, base] = g_loa
        j_mat[rows, base + 1] = g_sa
        j_mat[rows, base + 2] = g_lob
        j_mat[rows, base + 3] = g_sb
    if free_atom_numbers:
        j_mat[:, -2] = grads.d_log_na
        j_mat[:, -1] = grads.d_log_nb

    fisher = j_mat.T @ j_mat
    return np.linalg.pinv(fisher, rcond=1e-12)
