from __future__ import annotations

import numpy as np

from ouroboros.ac_mle import fit_ac_mle
from ouroboros.mle_common import ACScanResult, _validate_shots
from ouroboros.model import InterferometerParams
from ouroboros.static_mle import fit_static_mle


def scan_ac_spectrum(
    timestamps_s: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    frequencies_hz: np.ndarray,
    init_params: InterferometerParams,
    init_dc: float = 0.1,
    free_nuisance: bool = True,
    n_blocks: int = 1,
    t_ref_s: float | None = None,
    n_phi_global: int = 384,
    n_phi_local: int = 128,
    batch_size: int = 2048,
) -> ACScanResult:
    ya, yb, ts = _validate_shots(y_a, y_b, timestamps_s)
    assert ts is not None
    freqs = np.atleast_1d(np.asarray(frequencies_hz, dtype=float))
    if np.any(freqs <= 0.0):
        raise ValueError("All scan frequencies must be strictly positive.")
    t_ref = float(ts[0]) if t_ref_s is None else float(t_ref_s)

    null_res = fit_static_mle(
        ya,
        yb,
        init_params=init_params,
        init_delta_phi=max(init_dc, 1e-3),
        free_nuisance=free_nuisance,
        free_atom_numbers=False,
        n_blocks=n_blocks,
        n_phi_global=n_phi_global,
        n_phi_local=n_phi_local,
        batch_size=batch_size,
    )
    nuis_warm = (
        null_res.contrast_a_blocks,
        null_res.contrast_b_blocks,
        null_res.p0_a_blocks,
        null_res.p0_b_blocks,
    )

    n_f = freqs.size
    ll_arr = np.empty(n_f, dtype=float)
    amp_arr = np.empty(n_f, dtype=float)
    dc_arr = np.empty(n_f, dtype=float)
    c_arr = np.empty(n_f, dtype=float)
    s_arr = np.empty(n_f, dtype=float)
    gram_arr = np.empty(n_f, dtype=float)

    for idx, f_hz in enumerate(freqs):
        r = fit_ac_mle(
            ts,
            ya,
            yb,
            freq_hz=float(f_hz),
            init_params=init_params,
            init_dc=null_res.delta_phi_rad,
            free_nuisance=free_nuisance,
            n_blocks=n_blocks,
            t_ref_s=t_ref,
            n_phi_global=n_phi_global,
            n_phi_local=n_phi_local,
            batch_size=batch_size,
            init_nuisance_blocks=nuis_warm if free_nuisance else None,
        )
        ll_arr[idx] = r.log_likelihood
        amp_arr[idx] = r.amplitude_rad
        dc_arr[idx] = r.dc_rad
        c_arr[idx] = r.c_rad
        s_arr[idx] = r.s_rad
        gram_arr[idx] = r.gram_ac_efficiency

    return ACScanResult(
        frequencies_hz=freqs,
        log_likelihood_null=null_res.log_likelihood,
        log_likelihood=ll_arr,
        two_delta_log_l=np.maximum(2.0 * (ll_arr - null_res.log_likelihood), 0.0),
        amplitude_rad=amp_arr,
        dc_rad=dc_arr,
        c_rad=c_arr,
        s_rad=s_arr,
        gram_ac_efficiency=gram_arr,
    )
