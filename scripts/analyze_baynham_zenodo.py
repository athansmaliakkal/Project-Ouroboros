from __future__ import annotations

import argparse
import csv
import pathlib
import re

import numpy as np

from ouroboros import (
    InterferometerParams,
    compute_cramer_rao_quadrature,
    fit_ac_mle,
    fit_ellipse_blocks,
    fit_harmonic_ols,
    fit_static_mle,
)

STATIC_FILES = (
    (
        "excitation_fraction_corrected_with_phi_hln-lnl-repeat_"
        "63132_63134_63133_63135_63150_63149_63148_63151_63164_63163_63162_63331_63333_63332_0.csv"
    ),
    (
        "excitation_fraction_corrected_with_phi_hln-lnl-repeat_"
        "63132_63134_63133_63135_63150_63149_63148_63151_63164_63163_63162_63331_63333_63332_2.csv"
    ),
)


def _load_baynham_csv(path: pathlib.Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        lines = (ln for ln in f if ln.strip() and not ln.lstrip().startswith("#"))
        reader = csv.DictReader(lines)
        assert reader.fieldnames is not None
        cols: dict[str, list[float]] = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, buf in cols.items():
                try:
                    buf.append(float(row[k]))
                except (ValueError, TypeError):
                    buf.append(np.nan)
    arrs = {k: np.asarray(v, dtype=float) for k, v in cols.items()}
    if {"timestamp", "excitation_fraction_forward", "excitation_fraction_backward"}.issubset(arrs):
        ok = (
            np.isfinite(arrs["timestamp"])
            & np.isfinite(arrs["excitation_fraction_forward"])
            & np.isfinite(arrs["excitation_fraction_backward"])
        )
        arrs = {k: v[ok] for k, v in arrs.items()}
        order = np.argsort(arrs["timestamp"], kind="mergesort")
        arrs = {k: v[order] for k, v in arrs.items()}
    return arrs


def _parse_signal_freq_hz(filename: str) -> float | None:
    match = re.search(r"signal-([0-9]+(?:_[0-9]+)?)mHz", filename)
    if not match:
        return None
    val_mhz = float(match.group(1).replace("_", "."))
    return val_mhz * 1e-3


def analyze_static_datasets(data_dir: pathlib.Path) -> None:
    print("=== Static HLN/LLN Repeats (56,623 Total Shots) ===")
    p_nom = InterferometerParams(3100.0, 2040.0, 0.81, 0.84, 0.50, 0.50)
    p_swap = InterferometerParams(3100.0, 2040.0, 0.84, 0.81, 0.50, 0.50)

    for fn in STATIC_FILES:
        path = data_dir / fn
        if not path.exists():
            continue
        d = _load_baynham_csv(path)
        ts = d["timestamp"]
        ya = d["excitation_fraction_forward"]
        yb = d["excitation_fraction_backward"]
        r_fix = fit_static_mle(ya, yb, p_nom, init_delta_phi=0.102, free_nuisance=False)
        r_swp = fit_static_mle(ya, yb, p_swap, init_delta_phi=0.102, free_nuisance=False)
        r_5d = fit_static_mle(ya, yb, p_nom, init_delta_phi=0.102, free_nuisance=True, n_blocks=1)
        r_10b = fit_static_mle(
            ya, yb, p_nom, init_delta_phi=0.102, free_nuisance=True, n_blocks=10, timestamps_s=ts
        )
        r_7d = fit_static_mle(
            ya,
            yb,
            p_nom,
            init_delta_phi=0.102,
            free_nuisance=True,
            free_atom_numbers=True,
            n_blocks=1,
            atom_number_bounds=(200.0, 24000.0),
        )
        p_fit = InterferometerParams(
            3100.0, 2040.0, r_5d.contrast_a, r_5d.contrast_b, r_5d.p0_a, r_5d.p0_b
        )
        crb = compute_cramer_rao_quadrature(r_5d.delta_phi_rad, p_fit)
        tag = path.stem[-10:]
        print(f"\n[{tag}] K={ya.size:,}")
        print(
            f"  Fixed(0.81,0.84,0.50): dphi={r_fix.delta_phi_rad * 1e3:.3f} mrad, logL={r_fix.log_likelihood:.1f}"
        )
        print(
            f"  Fixed(0.84,0.81,0.50): dphi={r_swp.delta_phi_rad * 1e3:.3f} mrad, logL={r_swp.log_likelihood:.1f}"
        )
        print(
            f"  Global 5D MLE        : dphi={r_5d.delta_phi_rad * 1e3:.3f}+/-{r_5d.delta_phi_se_rad * 1e3:.3f} mrad, "
            f"C=({r_5d.contrast_a:.4f},{r_5d.contrast_b:.4f}), p0=({r_5d.p0_a:.4f},{r_5d.p0_b:.4f}), "
            f"logL={r_5d.log_likelihood:.1f}, CRB={crb.sigma_crb_rad * 1e3:.3f} mrad/shot"
        )
        print(
            f"  10-Block Tracked     : dphi={r_10b.delta_phi_rad * 1e3:.3f}+/-{r_10b.delta_phi_se_rad * 1e3:.3f} mrad, "
            f"logL={r_10b.log_likelihood:.1f} (2*dLL=+{2 * (r_10b.log_likelihood - r_5d.log_likelihood):.1f})"
        )
        print(
            f"  Global 7D N_eff      : dphi={r_7d.delta_phi_rad * 1e3:.3f} mrad, "
            f"N_eff=({r_7d.n_atoms_a:.0f},{r_7d.n_atoms_b:.0f}), logL={r_7d.log_likelihood:.1f}"
        )


def analyze_ac_datasets(data_dir: pathlib.Path, n_blocks: int) -> None:
    print(
        f"\n=== Injected AC Signal Runs (Time-Blocked {n_blocks}-Block vs. 1-Block vs. M=141 Fitzgibbon) ==="
    )
    p_nom = InterferometerParams(3100.0, 2040.0, 0.81, 0.84, 0.50, 0.50)
    files = sorted(data_dir.glob("excitation_fraction_corrected_signal-*.csv"))
    for path in files:
        f_hz = _parse_signal_freq_hz(path.name)
        if f_hz is None:
            continue
        d = _load_baynham_csv(path)
        ts = d["timestamp"]
        ya = d["excitation_fraction_forward"]
        yb = d["excitation_fraction_backward"]
        r_fix = fit_ac_mle(ts, ya, yb, f_hz, p_nom, free_nuisance=False, n_blocks=1)
        r_1b = fit_ac_mle(ts, ya, yb, f_hz, p_nom, free_nuisance=True, n_blocks=1)
        r_nb = fit_ac_mle(ts, ya, yb, f_hz, p_nom, free_nuisance=True, n_blocks=n_blocks)
        blk = fit_ellipse_blocks(ts, ya, yb, block_size=141, method="fitzgibbon")
        ols = fit_harmonic_ols(blk.block_times_s, blk.block_delta_phi_rad, f_hz, t_ref_s=ts[0])
        two_dll = 2.0 * (r_nb.log_likelihood - r_1b.log_likelihood)
        short = path.name.replace("excitation_fraction_corrected_", "").replace(".csv", "")
        print(
            f"{short:28s} (K={ya.size:5d}, f={f_hz * 1e3:6.2f} mHz, g_eff={r_nb.gram_ac_efficiency:.3f}): "
            f"Fix[dc={r_fix.dc_rad * 1e3:6.2f},A={r_fix.amplitude_rad * 1e3:5.2f}] | "
            f"1-blk[dc={r_1b.dc_rad * 1e3:6.2f},A={r_1b.amplitude_rad * 1e3:5.2f}] | "
            f"{n_blocks}-blk[dc={r_nb.dc_rad * 1e3:6.2f},A={r_nb.amplitude_rad * 1e3:5.2f}+/-{r_nb.amplitude_se_rad * 1e3:4.2f}, 2dLL=+{two_dll:7.1f}] | "
            f"Fitz141[dc={ols.dc * 1e3:6.2f},A={ols.amplitude * 1e3:5.2f}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Zenodo 10.5281/zenodo.19592552 DAI CSVs.")
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        required=True,
        help="Path to directory containing intermediate_data/ or the CSV files directly.",
    )
    parser.add_argument("--n-blocks", type=int, default=10, help="Chronological nuisance blocks.")
    parser.add_argument("--skip-static", action="store_true", help="Skip 56k-shot repeat files.")
    parser.add_argument("--skip-ac", action="store_true", help="Skip AC signal injection files.")
    args = parser.parse_args()
    data_dir = (
        args.data_dir / "intermediate_data"
        if (args.data_dir / "intermediate_data").is_dir()
        else args.data_dir
    )
    if not data_dir.is_dir():
        raise SystemExit(
            f"Directory not found: {data_dir}. Download dataset from Zenodo DOI 10.5281/zenodo.19592552."
        )
    if not args.skip_static:
        analyze_static_datasets(data_dir)
    if not args.skip_ac:
        analyze_ac_datasets(data_dir, n_blocks=args.n_blocks)


if __name__ == "__main__":
    main()
