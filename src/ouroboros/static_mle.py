from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ouroboros.joint_likelihood import joint_marginal_log_likelihood_and_grads
from ouroboros.mle_common import (
    StaticMLEResult,
    _compute_bhhh_covariance,
    _init_nuisance_box,
    _make_block_ids,
    _nuisance_block_grad,
    _unpack_block_nuisance,
    _validate_shots,
)
from ouroboros.model import InterferometerParams


def fit_static_mle(
    y_a: np.ndarray,
    y_b: np.ndarray,
    init_params: InterferometerParams,
    init_delta_phi: float = 0.1,
    free_nuisance: bool = True,
    free_atom_numbers: bool = False,
    n_blocks: int = 1,
    delta_phi_bounds: tuple[float, float] = (1e-4, np.pi - 1e-4),
    atom_number_bounds: tuple[float, float] = (100.0, 1e8),
    n_phi_global: int = 384,
    n_phi_local: int = 128,
    batch_size: int = 2048,
    init_nuisance_blocks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    timestamps_s: np.ndarray | None = None,
) -> StaticMLEResult:
    ya, yb, _ = _validate_shots(y_a, y_b, timestamps_s)
    if init_params.variance_mode == "midfringe" and (free_nuisance or free_atom_numbers):
        raise ValueError("Nuisance estimation requires variance_mode 'model' or 'observed'.")
    n_shots = ya.size
    block_ids = _make_block_ids(n_shots, n_blocks)

    x0_list: list[float] = [
        float(np.clip(init_delta_phi, delta_phi_bounds[0], delta_phi_bounds[1]))
    ]
    bnds: list[tuple[float, float]] = [delta_phi_bounds]

    if free_nuisance:
        x0_n, bnds_n = _init_nuisance_box(init_params, n_blocks, init_nuisance_blocks)
        x0_list.extend(x0_n)
        bnds.extend(bnds_n)

    if free_atom_numbers:
        ln_lo = float(np.log(atom_number_bounds[0]))
        ln_hi = float(np.log(atom_number_bounds[1]))
        x0_list.extend([float(np.log(init_params.n_atoms_a)), float(np.log(init_params.n_atoms_b))])
        bnds.extend([(ln_lo, ln_hi), (ln_lo, ln_hi)])

    n_nuis = 4 * n_blocks if free_nuisance else 0
    inv_k = 1.0 / float(n_shots)

    def _objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        dphi = float(x[0])
        nuis = _unpack_block_nuisance(
            x[1 : 1 + n_nuis], block_ids, n_blocks, init_params, free_nuisance
        )
        na = float(np.exp(x[-2])) if free_atom_numbers else init_params.n_atoms_a
        nb = float(np.exp(x[-1])) if free_atom_numbers else init_params.n_atoms_b

        grads = joint_marginal_log_likelihood_and_grads(
            ya,
            yb,
            dphi,
            na,
            nb,
            nuis.ca_shot,
            nuis.cb_shot,
            nuis.pa_shot,
            nuis.pb_shot,
            init_params.sigma_det_a,
            init_params.sigma_det_b,
            variance_mode=init_params.variance_mode,
            n_phi_global=n_phi_global,
            n_phi_local=n_phi_local,
            batch_size=batch_size,
        )

        grad_out = np.empty_like(x)
        grad_out[0] = np.sum(grads.d_delta_phi)
        if free_nuisance:
            grad_out[1 : 1 + n_nuis] = _nuisance_block_grad(grads, block_ids, n_blocks, nuis)
        if free_atom_numbers:
            grad_out[-2] = np.sum(grads.d_log_na)
            grad_out[-1] = np.sum(grads.d_log_nb)
        return -float(np.sum(grads.log_l)) * inv_k, -grad_out * inv_k

    res = minimize(
        _objective,
        np.asarray(x0_list, dtype=float),
        jac=True,
        method="L-BFGS-B",
        bounds=bnds,
        options={"ftol": 1e-13, "gtol": 1e-7, "maxiter": 400},
    )
    x_opt = res.x
    dphi_hat = float(x_opt[0])
    na_hat = float(np.exp(x_opt[-2])) if free_atom_numbers else init_params.n_atoms_a
    nb_hat = float(np.exp(x_opt[-1])) if free_atom_numbers else init_params.n_atoms_b

    nuis = _unpack_block_nuisance(
        x_opt[1 : 1 + n_nuis], block_ids, n_blocks, init_params, free_nuisance
    )
    grads = joint_marginal_log_likelihood_and_grads(
        ya,
        yb,
        dphi_hat,
        na_hat,
        nb_hat,
        nuis.ca_shot,
        nuis.cb_shot,
        nuis.pa_shot,
        nuis.pb_shot,
        init_params.sigma_det_a,
        init_params.sigma_det_b,
        variance_mode=init_params.variance_mode,
        n_phi_global=n_phi_global,
        n_phi_local=n_phi_local,
        batch_size=batch_size,
    )
    cov = _compute_bhhh_covariance(
        n_shots,
        1,
        grads.d_delta_phi[:, None],
        grads,
        block_ids,
        n_blocks,
        nuis,
        free_nuisance,
        free_atom_numbers,
    )
    _, g_final = _objective(x_opt)
    return StaticMLEResult(
        delta_phi_rad=dphi_hat,
        delta_phi_se_rad=float(np.sqrt(max(cov[0, 0], 0.0))),
        contrast_a=float(np.mean(nuis.ca_blocks)),
        contrast_b=float(np.mean(nuis.cb_blocks)),
        p0_a=float(np.mean(nuis.pa_blocks)),
        p0_b=float(np.mean(nuis.pb_blocks)),
        n_atoms_a=na_hat,
        n_atoms_b=nb_hat,
        contrast_a_blocks=nuis.ca_blocks,
        contrast_b_blocks=nuis.cb_blocks,
        p0_a_blocks=nuis.pa_blocks,
        p0_b_blocks=nuis.pb_blocks,
        log_likelihood=float(np.sum(grads.log_l)),
        success=bool(res.success),
        max_abs_grad=float(np.max(np.abs(g_final))),
    )
