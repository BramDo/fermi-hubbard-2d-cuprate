#!/usr/bin/env python3
"""Reproduce the exact classical 3x3 Hubbard comparison used for the hardware row.

This script performs no provider or QPU calls. It reuses the repository's exact
fixed-particle-sector implementation and computes the 3x3 one-hole Hubbard
model at T=0.2 and T=0.4.

Model:
    Lx = Ly = 3
    holes = 1
    N_up = N_down = 4
    t = 1
    t'/t = -0.25
    U/t = 8
    mu = 0
    open boundaries
    center_neel initial state

The fixed sector has C(9,4)^2 = 15,876 basis states.

Outputs are written under:
    results_cuprate_2d/classical_3x3_direct_reproduction/

The quantum comparison row is kept as metadata only:
    Readout + TFLO + DD, not validated | 32.078 s | 0.072 | 0.091 | 0.052

The three RMSE values are not recomputed here because an exact classical
solution is itself the continuous-Hamiltonian reference. Against that same
reference its RMSE is zero up to floating-point roundoff. A finite Trotter
circuit must be compared separately if one wants a nonzero Trotter error.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm_multiply

from cuprate_2d_exact_sector_baseline import (
    MODEL_FULL,
    build_hamiltonian,
    diagonal_site_rows,
    initial_state_vector,
    sector_basis,
)
from cuprate_2d_hubbard_pipeline import Cuprate2DConfig, validate_config


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "results_cuprate_2d" / "classical_3x3_direct_reproduction"
TIMES = (0.2, 0.4)
REPEATS = 3

QUANTUM_ROW = {
    "label": "Readout + TFLO + DD, not validated",
    "time_seconds": 32.078,
    "charge_rmse": 0.072,
    "spin_rmse": 0.091,
    "doublon_rmse": 0.052,
}


def config() -> Cuprate2DConfig:
    cfg = Cuprate2DConfig(
        lx=3,
        ly=3,
        holes=1,
        hopping_t=1.0,
        t_prime_over_t=-0.25,
        u_over_t=8.0,
        chemical_potential=0.0,
        steps=2,
        dt=0.2,
        boundary_x="open",
        boundary_y="open",
        initial_state="center_neel",
        shots=0,
    )
    validate_config(cfg)
    return cfg


def summarize_time(cfg: Cuprate2DConfig, basis: list[int], psi: np.ndarray, T: float) -> dict:
    rows = diagonal_site_rows(cfg, basis, psi, step=0, time_value=T, profile="exact_full_hubbard")
    charge = np.array([row["charge"] for row in rows], dtype=float)
    spin = np.array([row["spin_z"] for row in rows], dtype=float)
    doublon = np.array([row["double_occupancy"] for row in rows], dtype=float)

    return {
        "T": T,
        "norm": float(np.vdot(psi, psi).real),
        "total_charge": float(charge.sum()),
        "total_spin_z": float(spin.sum()),
        "mean_abs_spin_z": float(np.mean(np.abs(spin))),
        "mean_double_occupancy": float(np.mean(doublon)),
        "site_observables": rows,
    }


def one_run(cfg: Cuprate2DConfig) -> tuple[dict, float]:
    started = time.perf_counter()

    basis, basis_index, n_up, n_down = sector_basis(cfg)
    hamiltonian, hmeta = build_hamiltonian(cfg, MODEL_FULL)
    psi0 = initial_state_vector(cfg, basis_index)
    generator = -1j * hamiltonian

    results = []
    for T in TIMES:
        psi = expm_multiply(generator * T, psi0)
        results.append(summarize_time(cfg, basis, psi, T))

    elapsed = time.perf_counter() - started
    payload = {
        "basis_dimension": len(basis),
        "n_up": n_up,
        "n_down": n_down,
        "hamiltonian_nnz": int(hamiltonian.nnz),
        "hamiltonian_metadata": hmeta,
        "times": results,
    }
    return payload, elapsed


def write_report(payload: dict) -> None:
    trows = payload["exact_results"]["times"]
    timing = payload["timing"]
    q = payload["quantum_comparison_row"]
    ratio = q["time_seconds"] / timing["median_seconds"]

    lines = [
        "# Direct classical 3x3 reproduction",
        "",
        "This is the exact fixed-sector continuous-Hamiltonian calculation for the same 3x3 one-hole Hubbard model.",
        "",
        "## Model",
        "",
        "- lattice: 3x3, 18 spin orbitals",
        "- N_up = N_down = 4",
        "- fixed-sector dimension: 15,876",
        "- t = 1",
        "- t'/t = -0.25",
        "- U/t = 8",
        "- chemical potential = 0",
        "- open boundaries",
        "- initial state: center_neel with one central hole",
        "- evolution: scipy.sparse.linalg.expm_multiply on the sparse exact Hamiltonian",
        "",
        "## Timing",
        "",
        f"Three end-to-end classical repetitions: {', '.join(f'{x:.6f} s' for x in timing['repetitions_seconds'])}",
        f"Median: **{timing['median_seconds']:.6f} s**",
        f"Quantum-row time / classical median: **{ratio:.2f}x**",
        "",
        "## Exact observables",
        "",
        "| T | total charge | total spin-z | mean |spin-z| | mean doublon | norm |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trows:
        lines.append(
            f"| {row['T']:.1f} | {row['total_charge']:.12f} | {row['total_spin_z']:.12e} | "
            f"{row['mean_abs_spin_z']:.12f} | {row['mean_double_occupancy']:.12f} | {row['norm']:.12f} |"
        )

    lines += [
        "",
        "## Comparison row",
        "",
        "| method | time | charge RMSE | spin RMSE | doublon RMSE |",
        "|---|---:|---:|---:|---:|",
        f"| {q['label']} | {q['time_seconds']:.3f} s | {q['charge_rmse']:.3f} | {q['spin_rmse']:.3f} | {q['doublon_rmse']:.3f} |",
        f"| exact classical continuous-H | {timing['median_seconds']:.6f} s | 0* | 0* | 0* |",
        "",
        "*The exact classical RMSE is zero only when the exact continuous-Hamiltonian result is used as its own reference. This does not mean a finite Trotter circuit has zero Trotter error. Compare that circuit separately against this exact solution.",
        "",
        "## Claim boundary",
        "",
        "3x3 is a validation rung, not a quantum-advantage scale. The exact fixed sector contains only 15,876 basis states, so a sparse classical calculation is straightforward. Hardware/mitigation results remain scientifically useful for validating the quantum workflow, but they do not establish computational advantage at this size.",
        "",
    ]
    (OUTDIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    cfg = config()

    runs: list[float] = []
    exact_payload: dict | None = None
    for _ in range(REPEATS):
        result, elapsed = one_run(cfg)
        runs.append(elapsed)
        exact_payload = result

    assert exact_payload is not None
    expected_dimension = math.comb(9, 4) ** 2
    if exact_payload["basis_dimension"] != expected_dimension:
        raise RuntimeError(
            f"Unexpected sector dimension {exact_payload['basis_dimension']} != {expected_dimension}"
        )

    payload = {
        "status": "CLASSICAL_EXACT_REPRODUCTION",
        "model": {
            "lx": cfg.lx,
            "ly": cfg.ly,
            "holes": cfg.holes,
            "hopping_t": cfg.hopping_t,
            "t_prime_over_t": cfg.t_prime_over_t,
            "u_over_t": cfg.u_over_t,
            "chemical_potential": cfg.chemical_potential,
            "boundary_x": cfg.boundary_x,
            "boundary_y": cfg.boundary_y,
            "initial_state": cfg.initial_state,
            "times": list(TIMES),
        },
        "exact_results": exact_payload,
        "timing": {
            "repetitions_seconds": runs,
            "median_seconds": statistics.median(runs),
            "measurement": "perf_counter end-to-end: sector basis + sparse H + two expm_multiply evolutions + site observables",
        },
        "quantum_comparison_row": QUANTUM_ROW,
        "rmse_note": (
            "Exact classical vs itself is zero up to floating-point roundoff. "
            "Finite-circuit/Trotter error is a separate comparison and must not be conflated with hardware error."
        ),
        "new_QPU_seconds": 0,
    }

    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    write_report(payload)

    print(json.dumps({
        "status": payload["status"],
        "basis_dimension": exact_payload["basis_dimension"],
        "hamiltonian_nnz": exact_payload["hamiltonian_nnz"],
        "timing": payload["timing"],
        "output": str(OUTDIR),
    }, indent=2))


if __name__ == "__main__":
    main()
