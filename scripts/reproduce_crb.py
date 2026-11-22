from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from ouroboros import (
    InterferometerParams,
    compute_cramer_rao_mc,
    compute_cramer_rao_quadrature,
)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "configs" / "baynham_2026_baseline.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    det = cfg["detector"]
    ds = cfg["dataset"]
    pub = cfg["published_targets"]

    print("=== Baynham et al. (Nature 2026) Baseline CRB Reproduction ===")
    print(
        f"Inputs: N_top={det['n_atoms_top']}±{det['n_atoms_top_err']}, "
        f"N_bot={det['n_atoms_bottom']}±{det['n_atoms_bottom_err']}, "
        f"C_top={det['contrast_top']}, C_bot={det['contrast_bottom']}, "
        f"p0=({det['p0_top']}, {det['p0_bottom']}), "
        f"bias={det['delta_phi_bias_rad']:.4f} rad"
    )
    print(
        f"Published target: {pub['cramer_rao_per_shot_mrad']}({int(10 * pub['cramer_rao_per_shot_err_mrad'])}) mrad/shot, "
        f"{pub['whole_dataset_bound_urad']}({int(pub['whole_dataset_bound_err_urad'])}) urad over {ds['shots_per_dataset']} shots\n"
    )

    for mode in ("model", "observed", "midfringe"):
        params = InterferometerParams(
            n_atoms_a=det["n_atoms_top"],
            n_atoms_b=det["n_atoms_bottom"],
            contrast_a=det["contrast_top"],
            contrast_b=det["contrast_bottom"],
            p0_a=det["p0_top"],
            p0_b=det["p0_bottom"],
            variance_mode=mode,
        )
        res = compute_cramer_rao_quadrature(det["delta_phi_bias_rad"], params)
        whole_urad = res.sigma_crb_rad * 1e6 / np.sqrt(ds["shots_per_dataset"])
        print(
            f"[Quadrature, variance_mode={mode:<9s}] "
            f"sigma_CRB={res.sigma_crb_rad * 1e3:.3f} mrad/shot "
            f"(whole={whole_urad:.1f} urad), "
            f"known_mid={res.sigma_known_midfringe_rad * 1e3:.3f} mrad, "
            f"penalty={res.penalty_factor:.4f}x, "
            f"Bartlett_diff={res.bartlett_rel_diff:.2e}"
        )

    params_model = InterferometerParams(
        n_atoms_a=det["n_atoms_top"],
        n_atoms_b=det["n_atoms_bottom"],
        contrast_a=det["contrast_top"],
        contrast_b=det["contrast_bottom"],
        p0_a=det["p0_top"],
        p0_b=det["p0_bottom"],
        variance_mode="model",
    )
    mc_gauss = compute_cramer_rao_mc(
        det["delta_phi_bias_rad"], params_model, n_samples=20000, binomial_readout=False
    )
    mc_binom = compute_cramer_rao_mc(
        det["delta_phi_bias_rad"], params_model, n_samples=20000, binomial_readout=True
    )
    print(
        f"\n[MC Gaussian, 20k] sigma_CRB={mc_gauss.sigma_crb_rad * 1e3:.3f} mrad, Bartlett_diff={mc_gauss.bartlett_rel_diff:.2e}"
    )
    print(
        f"[MC Binomial, 20k] sigma_CRB={mc_binom.sigma_crb_rad * 1e3:.3f} mrad, Bartlett_diff={mc_binom.bartlett_rel_diff:.2e}"
    )

    rel_top = det["n_atoms_top_err"] / det["n_atoms_top"]
    rel_bot = det["n_atoms_bottom_err"] / det["n_atoms_bottom"]
    p_hi = InterferometerParams(
        n_atoms_a=det["n_atoms_top"] * (1.0 + det["atom_calibration_rel_err"]),
        n_atoms_b=det["n_atoms_bottom"] * (1.0 + det["atom_calibration_rel_err"]),
        contrast_a=det["contrast_top"],
        contrast_b=det["contrast_bottom"],
        p0_a=det["p0_top"],
        p0_b=det["p0_bottom"],
        variance_mode="model",
    )
    p_lo = InterferometerParams(
        n_atoms_a=det["n_atoms_top"] * (1.0 - det["atom_calibration_rel_err"]),
        n_atoms_b=det["n_atoms_bottom"] * (1.0 - det["atom_calibration_rel_err"]),
        contrast_a=det["contrast_top"],
        contrast_b=det["contrast_bottom"],
        p0_a=det["p0_top"],
        p0_b=det["p0_bottom"],
        variance_mode="model",
    )
    s_hi = compute_cramer_rao_quadrature(det["delta_phi_bias_rad"], p_hi).sigma_crb_rad * 1e3
    s_lo = compute_cramer_rao_quadrature(det["delta_phi_bias_rad"], p_lo).sigma_crb_rad * 1e3
    print(
        f"\nCalibration uncertainty check (rel_top={rel_top:.1%}, rel_bot={rel_bot:.1%}, "
        f"correlated ±{det['atom_calibration_rel_err']:.0%}): ±{0.5 * (s_lo - s_hi):.2f} mrad"
    )

    print("\n=== Bias phase scan (Baynham contrasts 0.81, 0.84) ===")
    for bias in [0.02, 0.05, 0.1022, 0.2, 0.4, np.pi / 4, np.pi / 2]:
        r = compute_cramer_rao_quadrature(float(bias), params_model)
        print(
            f"  bias={bias:.4f} rad ({np.degrees(bias):5.1f} deg): "
            f"sigma_CRB={r.sigma_crb_rad * 1e3:.3f} mrad, penalty={r.penalty_factor:.4f}x"
        )


if __name__ == "__main__":
    main()
