from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from ouroboros.joint_likelihood import joint_marginal_log_likelihood_and_grads
from ouroboros.mle_common import (
    ACMLEResult,
    _compute_bhhh_covariance,
    _gram_diagnostics,
    _init_nuisance_box,
    _make_block_ids,
    _nuisance_block_grad,
    _unpack_block_nuisance,
    _validate_shots,
)
from ouroboros.model import InterferometerParams


def fit_ac_mle(
    timestamps_s: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    freq_hz: float,
    init_params: InterferometerParams,
    init_dc: float = 0.1,
    init_c: float = 0.0,
    init_s: float = 0.0,
    free_nuisance: bool = True,
    n_blocks: int = 1,
    t_ref_s: float | None = None,
    dc_bounds: tuple[float, float] = (0.0, np.pi - 1e-4),
    amp_component_bound: float = 1.5,
    n_phi_global: int = 384,
    n_phi_local: int = 128,
    batch_size: int = 2048,
    init_nuisance_blocks: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    fold_search: bool = True,
) -> ACMLEResult:
    ya, yb, ts = _validate_shots(y_a, y_b, timestamps_s)
    assert ts is not None
    if init_params.variance_mode == "midfringe" and free_nuisance:
        raise ValueError("Nuisance estimation requires variance_mode 'model' or 'observed'.")
    if freq_hz <= 0.0:
        raise ValueError("freq_hz must be strictly positive.")
    t_ref = float(ts[0]) if t_ref_s is None else float(t_ref_s)
    tau = ts - t_ref
    omega = 2.0 * np.pi * freq_hz
    cos_wt = np.cos(omega * tau)
    sin_wt = np.sin(omega * tau)
    gram_eff, gram_cond = _gram_diagnostics(ts, freq_hz, t_ref)

    n_shots = ya.size
    block_ids = _make_block_ids(n_shots, n_blocks)
    inv_k = 1.0 / float(n_shots)

    dc0 = float(np.clip(init_dc, dc_bounds[0], dc_bounds[1]))
    c0 = float(np.clip(init_c, -amp_component_bound, amp_component_bound))
    s0 = float(np.clip(init_s, -amp_component_bound, amp_component_bound))
    if dc0 == 0.0 and c0 == 0.0 and s0 == 0.0:
        dc0 = min(1e-3, dc_bounds[1])

    x0_list: list[float] = [dc0, c0, s0]
    bnds: list[tuple[float, float]] = [
        dc_bounds,
        (-amp_component_bound, amp_component_bound),
        (-amp_component_bound, amp_component_bound),
    ]
    if free_nuisance:
        x0_n, bnds_n = _init_nuisance_box(init_params, n_blocks, init_nuisance_blocks)
        x0_list.extend(x0_n)
        bnds.extend(bnds_n)

    def _objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        dc, c_val, s_val = float(x[0]), float(x[1]), float(x[2])
        dphi_t = dc + c_val * cos_wt + s_val * sin_wt
        nuis = _unpack_block_nuisance(x[3:], block_ids, n_blocks, init_params, free_nuisance)
        grads = joint_marginal_log_likelihood_and_grads(
            ya,
            yb,
            dphi_t,
            init_params.n_atoms_a,
            init_params.n_atoms_b,
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
        grad_out[1] = np.sum(grads.d_delta_phi * cos_wt)
        grad_out[2] = np.sum(grads.d_delta_phi * sin_wt)
        if free_nuisance:
            grad_out[3:] = _nuisance_block_grad(grads, block_ids, n_blocks, nuis)
        return -float(np.sum(grads.log_l)) * inv_k, -grad_out * inv_k

    x_init = np.asarray(x0_list, dtype=float)
    step_dir: tuple[float, float] | None = None
    if init_c == 0.0 and init_s == 0.0 and dc0 > 1e-4:
        nuis0 = _unpack_block_nuisance(x_init[3:], block_ids, n_blocks, init_params, free_nuisance)
        g0 = joint_marginal_log_likelihood_and_grads(
            ya,
            yb,
            dc0,
            init_params.n_atoms_a,
            init_params.n_atoms_b,
            nuis0.ca_shot,
            nuis0.cb_shot,
            nuis0.pa_shot,
            nuis0.pb_shot,
            init_params.sigma_det_a,
            init_params.sigma_det_b,
            variance_mode=init_params.variance_mode,
            n_phi_global=n_phi_global,
            n_phi_local=n_phi_local,
            batch_size=batch_size,
        )
        score_w = np.mean(g0.d_delta_phi * g0.d_delta_phi)
        if score_w > 1e-6:
            h_des = np.column_stack([np.ones_like(cos_wt), cos_wt, sin_wt])
            fish_3 = score_w * (h_des.T @ h_des)
            rhs_3 = h_des.T @ g0.d_delta_phi
            step_3 = np.linalg.pinv(fish_3, rcond=1e-8) @ rhs_3
            norm_ac = float(np.hypot(step_3[1], step_3[2]))
            if norm_ac > 1e-9:
                step_dir = (float(step_3[1] / norm_ac), float(step_3[2] / norm_ac))
            x_cand = x_init.copy()
            x_cand[0] = float(np.clip(dc0 + step_3[0], dc_bounds[0], dc_bounds[1]))
            x_cand[1] = float(np.clip(step_3[1], -amp_component_bound, amp_component_bound))
            x_cand[2] = float(np.clip(step_3[2], -amp_component_bound, amp_component_bound))
            f0, _ = _objective(x_init)
            f1, _ = _objective(x_cand)
            if f1 < f0:
                x_init = x_cand

    lbfgs_opts = {"ftol": 1e-13, "gtol": 1e-7, "maxiter": 500}
    res = minimize(_objective, x_init, jac=True, method="L-BFGS-B", bounds=bnds, options=lbfgs_opts)

    if fold_search and init_c == 0.0 and init_s == 0.0:
        dc_res, c_res, s_res = float(res.x[0]), float(res.x[1]), float(res.x[2])
        amp_res = float(np.hypot(c_res, s_res))
        if amp_res > 0.35 * max(dc_res, 1e-3) or abs(dc_res - dc0) > 0.35 * max(dc0, 1e-2):
            cand_dirs: list[tuple[float, float]] = []
            if amp_res > 1e-6:
                cand_dirs.append((c_res / amp_res, s_res / amp_res))
            if step_dir is not None:
                cand_dirs.append(step_dir)
            base_scale = max(dc0, 0.5 * dc_res, 0.02)
            best_probe = None
            best_probe_fun = float("inf")
            for u_c, u_s in cand_dirs:
                for mult in (1.8, 3.4):
                    amp_p = min(mult * base_scale, 0.9 * amp_component_bound)
                    x_p = x_init.copy()
                    x_p[0] = float(np.clip(0.5 * dc0, dc_bounds[0], dc_bounds[1]))
                    x_p[1] = float(np.clip(amp_p * u_c, -amp_component_bound, amp_component_bound))
                    x_p[2] = float(np.clip(amp_p * u_s, -amp_component_bound, amp_component_bound))
                    if free_nuisance:
                        x_p[3:] = res.x[3:]
                    f_p, _ = _objective(x_p)
                    if f_p < best_probe_fun:
                        best_probe_fun = f_p
                        best_probe = x_p
            if best_probe is not None:
                res_cross = minimize(
                    _objective,
                    best_probe,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bnds,
                    options=lbfgs_opts,
                )
                if res_cross.fun < res.fun:
                    res = res_cross

    x_opt = res.x
    dc_hat, c_hat, s_hat = float(x_opt[0]), float(x_opt[1]), float(x_opt[2])
    amp_hat = float(np.hypot(c_hat, s_hat))
    phase_hat = float(np.arctan2(-s_hat, c_hat))

    nuis = _unpack_block_nuisance(x_opt[3:], block_ids, n_blocks, init_params, free_nuisance)
    dphi_opt = dc_hat + c_hat * cos_wt + s_hat * sin_wt
    grads = joint_marginal_log_likelihood_and_grads(
        ya,
        yb,
        dphi_opt,
        init_params.n_atoms_a,
        init_params.n_atoms_b,
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
    phase_jac = grads.d_delta_phi[:, None] * np.column_stack([np.ones_like(cos_wt), cos_wt, sin_wt])
    cov = _compute_bhhh_covariance(
        n_shots,
        3,
        phase_jac,
        grads,
        block_ids,
        n_blocks,
        nuis,
        free_nuisance,
        False,
    )
    dc_se = float(np.sqrt(max(cov[0, 0], 0.0)))
    c_se = float(np.sqrt(max(cov[1, 1], 0.0)))
    s_se = float(np.sqrt(max(cov[2, 2], 0.0)))
    if amp_hat > 1e-12:
        u_amp = np.array([c_hat / amp_hat, s_hat / amp_hat])
        amp_var = float(u_amp @ cov[1:3, 1:3] @ u_amp)
        amp_se = float(np.sqrt(max(amp_var, 0.0)))
    else:
        amp_se = float(np.sqrt(max(0.5 * (cov[1, 1] + cov[2, 2]), 0.0)))

    _, g_final = _objective(x_opt)
    return ACMLEResult(
        freq_hz=float(freq_hz),
        dc_rad=dc_hat,
        c_rad=c_hat,
        s_rad=s_hat,
        amplitude_rad=amp_hat,
        phase_rad=phase_hat,
        dc_se_rad=dc_se,
        c_se_rad=c_se,
        s_se_rad=s_se,
        amplitude_se_rad=amp_se,
        contrast_a_blocks=nuis.ca_blocks,
        contrast_b_blocks=nuis.cb_blocks,
        p0_a_blocks=nuis.pa_blocks,
        p0_b_blocks=nuis.pb_blocks,
        log_likelihood=float(np.sum(grads.log_l)),
        gram_ac_efficiency=gram_eff,
        gram_cond=gram_cond,
        success=bool(res.success),
        max_abs_grad=float(np.max(np.abs(g_final))),
    )
