#!/usr/bin/env python3
"""IBM hardware smoke test for the 72-qubit 2D cuprate Hubbard target.

The submitted job contains:

  1. main 72-qubit 2D Hubbard Trotter circuit
  2. all-0 readout calibration circuit
  3. all-1 readout calibration circuit

The result postprocessing extracts diagonal Z-basis observables: local
occupation, charge, spin-z, doublons, nearest-neighbor charge correlations,
nearest-neighbor spin correlations, and particle-sector survival. This is a
hardware diagnostic, not a superconductivity claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit, transpile

from advantage_scale_fermi_hubbard_model import (
    bit_from_qiskit_bitstring,
    estimate_local_readout_params,
    mitigated_z_expectation_from_counts,
)
from cuprate_2d_hubbard_pipeline import (
    SPIN_DOWN,
    SPIN_UP,
    Cuprate2DConfig,
    all_hopping_bonds,
    initial_site_rows,
    mode_qubit,
    validate_config,
)
from cuprate_2d_qiskit_circuit import SUPPORTED_CIRCUIT_FAMILIES, build_circuit, load_config
from ibm_staged_fermi_hubbard_runner import (
    backend_name,
    backend_operational,
    backend_pending_jobs,
    build_runtime_service,
    summarize_circuit,
)
from summarize_ibm_staged_results import counts_from_pub_result, iter_pub_results, normalize_counts


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_initial_layout_payload(path: Path | None, qubits: int) -> tuple[list[int] | None, int | None]:
    if path is None:
        return None, None
    payload = json.loads(path.read_text())
    raw = payload.get("initial_layout", payload) if isinstance(payload, dict) else payload
    if isinstance(raw, str):
        layout = [int(item) for item in raw.split()]
    else:
        layout = [int(item) for item in raw]
    if len(layout) != qubits:
        raise ValueError(f"Initial layout {path} has {len(layout)} entries; expected {qubits}")
    if len(set(layout)) != len(layout):
        raise ValueError(f"Initial layout {path} contains duplicate physical qubits")
    seed = None
    if isinstance(payload, dict):
        best = payload.get("best_candidate")
        if isinstance(best, dict) and best.get("seed_transpiler") is not None:
            seed = int(best["seed_transpiler"])
    return layout, seed


def build_readout_calibration_circuits(qubits: int) -> list[QuantumCircuit]:
    all0 = QuantumCircuit(qubits)
    all0.measure_all()
    all1 = QuantumCircuit(qubits)
    for qubit in range(qubits):
        all1.x(qubit)
    all1.measure_all()
    return [all0, all1]


def select_hardware_backend(service: Any, requested_backend: str | None, min_qubits: int) -> Any:
    if requested_backend:
        return service.backend(requested_backend)
    candidates = service.backends(simulator=False, operational=True, min_num_qubits=min_qubits)
    if not candidates:
        candidates = service.backends(simulator=False, min_num_qubits=min_qubits)
    if not candidates:
        raise RuntimeError(f"No IBM hardware backend found with at least {min_qubits} qubits")
    return sorted(
        candidates,
        key=lambda backend: (
            backend_pending_jobs(backend) if backend_pending_jobs(backend) is not None else 10**9,
            backend_name(backend),
        ),
    )[0]


def z_product_raw(counts: Mapping[str, int], cbits: list[int], n_qubits: int) -> float:
    shots = float(sum(counts.values()))
    if shots <= 0.0:
        raise ValueError("Counts are empty")
    total = 0.0
    for bitstring, value in counts.items():
        product = 1.0
        for cbit in cbits:
            bit = bit_from_qiskit_bitstring(bitstring, cbit, n_qubits)
            product *= 1.0 if bit == 0 else -1.0
        total += float(value) * product
    return float(total / shots)


def z_product(
    counts: Mapping[str, int],
    cbits: list[int],
    n_qubits: int,
    *,
    p01: np.ndarray | None,
    p10: np.ndarray | None,
    readout_corrected: bool,
) -> float:
    if readout_corrected and p01 is not None and p10 is not None:
        z_exp, _ = mitigated_z_expectation_from_counts(
            counts,
            qubits=cbits,
            num_qubits=n_qubits,
            p01=p01,
            p10=p10,
        )
        return float(z_exp)
    return z_product_raw(counts, cbits, n_qubits)


def n_from_z(z_value: float) -> float:
    return 0.5 * (1.0 - z_value)


def pair_occupation_from_z(za: float, zb: float, zab: float) -> float:
    return 0.25 * (1.0 - za - zb + zab)


def site_observable_rows(
    cfg: Cuprate2DConfig,
    counts: Mapping[str, int],
    *,
    profile: str,
    p01: np.ndarray | None,
    p10: np.ndarray | None,
    readout_corrected: bool,
) -> list[dict[str, Any]]:
    rows = []
    for site in range(cfg.sites):
        x = site % cfg.lx
        y = site // cfg.lx
        up = mode_qubit(site, SPIN_UP)
        down = mode_qubit(site, SPIN_DOWN)
        zu = z_product(counts, [up], cfg.qubits, p01=p01, p10=p10, readout_corrected=readout_corrected)
        zd = z_product(counts, [down], cfg.qubits, p01=p01, p10=p10, readout_corrected=readout_corrected)
        zud = z_product(counts, [up, down], cfg.qubits, p01=p01, p10=p10, readout_corrected=readout_corrected)
        n_up = n_from_z(zu)
        n_down = n_from_z(zd)
        rows.append(
            {
                "profile": profile,
                "site": site,
                "x": x,
                "y": y,
                "n_up": n_up,
                "n_down": n_down,
                "charge": n_up + n_down,
                "spin_z": n_up - n_down,
                "hole_density": 1.0 - n_up - n_down,
                "double_occupancy": pair_occupation_from_z(zu, zd, zud),
            }
        )
    return rows


def nearest_neighbor_bonds(cfg: Cuprate2DConfig) -> list[dict[str, Any]]:
    return [bond for bond in all_hopping_bonds(cfg) if str(bond["family"]) == "nearest_t"]


def bond_observable_rows(
    cfg: Cuprate2DConfig,
    counts: Mapping[str, int],
    *,
    profile: str,
    p01: np.ndarray | None,
    p10: np.ndarray | None,
    readout_corrected: bool,
) -> list[dict[str, Any]]:
    rows = []
    for bond in nearest_neighbor_bonds(cfg):
        site_a = int(bond["site_a"])
        site_b = int(bond["site_b"])
        charge_corr = 0.0
        spin_corr = 0.0
        for spin_a, sign_a in [(SPIN_UP, 1.0), (SPIN_DOWN, -1.0)]:
            qa = mode_qubit(site_a, spin_a)
            za = z_product(counts, [qa], cfg.qubits, p01=p01, p10=p10, readout_corrected=readout_corrected)
            for spin_b, sign_b in [(SPIN_UP, 1.0), (SPIN_DOWN, -1.0)]:
                qb = mode_qubit(site_b, spin_b)
                zb = z_product(counts, [qb], cfg.qubits, p01=p01, p10=p10, readout_corrected=readout_corrected)
                zab = z_product(counts, [qa, qb], cfg.qubits, p01=p01, p10=p10, readout_corrected=readout_corrected)
                occ_pair = pair_occupation_from_z(za, zb, zab)
                charge_corr += occ_pair
                spin_corr += sign_a * sign_b * occ_pair
        rows.append(
            {
                "profile": profile,
                "orientation": bond["orientation"],
                "site_a": site_a,
                "x_a": bond["x_a"],
                "y_a": bond["y_a"],
                "site_b": site_b,
                "x_b": bond["x_b"],
                "y_b": bond["y_b"],
                "charge_correlation": charge_corr,
                "spin_z_correlation": spin_corr,
            }
        )
    return rows


def expected_particle_numbers(cfg: Cuprate2DConfig) -> tuple[int, int]:
    up = 0
    down = 0
    for row in initial_site_rows(cfg):
        up += int(row["n_up_initial"])
        down += int(row["n_down_initial"])
    return up, down


def particle_numbers_from_bitstring(bitstring: str, cfg: Cuprate2DConfig) -> tuple[int, int]:
    up = 0
    down = 0
    for site in range(cfg.sites):
        up += bit_from_qiskit_bitstring(bitstring, mode_qubit(site, SPIN_UP), cfg.qubits)
        down += bit_from_qiskit_bitstring(bitstring, mode_qubit(site, SPIN_DOWN), cfg.qubits)
    return up, down


def sector_survival(counts: Mapping[str, int], cfg: Cuprate2DConfig) -> dict[str, Any]:
    expected = expected_particle_numbers(cfg)
    total = float(sum(counts.values()))
    exact = 0.0
    abs_up = 0.0
    abs_down = 0.0
    hist: dict[str, int] = {}
    for bitstring, value in counts.items():
        count = int(value)
        observed = particle_numbers_from_bitstring(bitstring, cfg)
        if observed == expected:
            exact += count
        abs_up += count * abs(observed[0] - expected[0])
        abs_down += count * abs(observed[1] - expected[1])
        key = f"{observed[0]},{observed[1]}"
        hist[key] = hist.get(key, 0) + count
    top = [
        {"n_up": int(key.split(",")[0]), "n_down": int(key.split(",")[1]), "shots": shots}
        for key, shots in sorted(hist.items(), key=lambda item: item[1], reverse=True)[:10]
    ]
    return {
        "expected_n_up": expected[0],
        "expected_n_down": expected[1],
        "exact_sector_fraction": 0.0 if total <= 0 else exact / total,
        "mean_abs_n_up_error": 0.0 if total <= 0 else abs_up / total,
        "mean_abs_n_down_error": 0.0 if total <= 0 else abs_down / total,
        "top_particle_sectors": top,
    }


def summarize_site_rows(rows: list[dict[str, Any]], profile: str) -> dict[str, float]:
    selected = [row for row in rows if row["profile"] == profile]
    charges = np.asarray([float(row["charge"]) for row in selected], dtype=float)
    spins = np.asarray([float(row["spin_z"]) for row in selected], dtype=float)
    doubles = np.asarray([float(row["double_occupancy"]) for row in selected], dtype=float)
    return {
        "total_charge": float(np.sum(charges)),
        "total_spin_z": float(np.sum(spins)),
        "mean_charge": float(np.mean(charges)),
        "mean_abs_spin_z": float(np.mean(np.abs(spins))),
        "mean_double_occupancy": float(np.mean(doubles)),
        "min_charge": float(np.min(charges)),
        "max_charge": float(np.max(charges)),
    }


def plot_charge_spin(path: Path, cfg: Cuprate2DConfig, site_rows: list[dict[str, Any]], profile: str) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    selected = [row for row in site_rows if row["profile"] == profile]
    charge = np.zeros((cfg.ly, cfg.lx), dtype=float)
    spin = np.zeros((cfg.ly, cfg.lx), dtype=float)
    double = np.zeros((cfg.ly, cfg.lx), dtype=float)
    for row in selected:
        x = int(row["x"])
        y = int(row["y"])
        charge[y, x] = float(row["charge"])
        spin[y, x] = float(row["spin_z"])
        double[y, x] = float(row["double_occupancy"])

    spin_lim = max(0.25, float(np.max(np.abs(spin))))
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.6), dpi=170, constrained_layout=True)
    panels = [
        (axes[0], charge, f"{profile} charge", "viridis", 0.0, 2.0),
        (axes[1], spin, f"{profile} spin z", "coolwarm", -spin_lim, spin_lim),
        (axes[2], double, f"{profile} doublons", "magma", 0.0, max(0.25, float(np.max(double)))),
    ]
    for ax, data, title, cmap, vmin, vmax in panels:
        image = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xticks(range(cfg.lx))
        ax.set_yticks(range(cfg.ly))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"IBM 72q 2D Hubbard hardware observables: {cfg.lx}x{cfg.ly}, steps={cfg.steps}")
    fig.savefig(path)
    plt.close(fig)
    return path


def job_status_name(job: Any) -> str:
    try:
        return str(job.status()).split(".")[-1]
    except Exception as exc:
        return f"unknown:{type(exc).__name__}:{exc}"


def wait_until_done(job: Any, wait_seconds: int, poll_interval: int) -> str:
    deadline = time.time() + max(0, wait_seconds)
    status = job_status_name(job)
    terminal = {"DONE", "ERROR", "CANCELLED"}
    while status not in terminal and time.time() < deadline:
        time.sleep(max(1, poll_interval))
        status = job_status_name(job)
        print(f"job {job.job_id()} status: {status}", flush=True)
    return status


def transpile_circuits(
    circuits: list[QuantumCircuit],
    backend: Any,
    *,
    optimization_level: int,
    seed_transpiler: int | None,
    initial_layout: Sequence[int] | None = None,
) -> list[QuantumCircuit]:
    result = transpile(
        circuits,
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
        initial_layout=None if initial_layout is None else list(initial_layout),
    )
    return result if isinstance(result, list) else [result]


def submit_job(
    backend: Any,
    circuits: list[QuantumCircuit],
    *,
    shots: int,
) -> Any:
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    return sampler.run(circuits, shots=shots)


def save_plan(
    outdir: Path,
    cfg: Cuprate2DConfig,
    backend: Any,
    logical_circuits: list[QuantumCircuit],
    transpiled_circuits: list[QuantumCircuit],
    roles: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    plan = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": backend_name(backend),
        "backend_num_qubits": int(getattr(backend, "num_qubits", 0) or 0),
        "backend_operational": backend_operational(backend),
        "backend_pending_jobs": backend_pending_jobs(backend),
        "config": asdict(cfg),
        "shots": int(args.shots),
        "submit": bool(args.submit),
        "circuit_family": args.circuit_family,
        "optimization_level": int(args.optimization_level),
        "seed_transpiler": args.seed_transpiler,
        "initial_layout_path": None if args.initial_layout is None else str(args.initial_layout),
        "initial_layout": None if getattr(args, "loaded_initial_layout", None) is None else list(args.loaded_initial_layout),
        "readout_repetitions": int(args.readout_repetitions),
        "readout_shots_requested": int(args.readout_shots) if args.readout_shots is not None else int(args.shots),
        "mitigation_plan": {
            "readout": "all-0/all-1 calibration circuits; repeated calibrations are pooled in postprocessing",
            "readout_shots_note": "IBM SamplerV2 direct runner uses one shots value per run here; different readout_shots requires separate jobs or provider-side support.",
            "sector_profiles": "postselection is intentionally not applied automatically; keep raw/readout-corrected profiles separate.",
        },
        "roles": roles,
        "logical_summaries": [summarize_circuit(circuit) for circuit in logical_circuits],
        "transpiled_summaries": [summarize_circuit(circuit) for circuit in transpiled_circuits],
        "claim_boundary": (
            "72-qubit IBM hardware smoke test for diagonal observables. "
            "Not a d-wave superconductivity or quantum advantage claim."
        ),
    }
    write_json(outdir / "cuprate_2d_ibm_hardware_plan.json", plan)
    return plan


def process_result(
    *,
    outdir: Path,
    cfg: Cuprate2DConfig,
    backend: str,
    job_id: str,
    primitive_result: Any,
    roles: list[str],
) -> dict[str, Any]:
    pubs = []
    for index, pub_result in enumerate(iter_pub_results(primitive_result)):
        counts = counts_from_pub_result(pub_result, cfg.qubits)
        role = roles[index] if index < len(roles) else f"pub_{index}"
        pubs.append(
            {
                "index": index,
                "role": role,
                "shots": int(sum(counts.values())),
                "unique_bitstrings": len(counts),
                "counts": counts,
            }
        )
    raw = {"job_id": job_id, "backend": backend, "pubs": pubs}
    write_json(outdir / "cuprate_2d_ibm_raw_results.json", raw)

    by_role = {pub["role"]: pub for pub in pubs}
    by_prefix: dict[str, list[dict[str, Any]]] = {}
    for pub in pubs:
        role = str(pub["role"])
        if role.startswith("readout_all0"):
            by_prefix.setdefault("readout_all0", []).append(pub)
        elif role.startswith("readout_all1"):
            by_prefix.setdefault("readout_all1", []).append(pub)
    main_counts = normalize_counts(by_role["main"]["counts"], cfg.qubits)
    p01 = None
    p10 = None
    readout_meta = None
    if "readout_all0" in by_prefix and "readout_all1" in by_prefix:
        counts0: dict[str, int] = {}
        counts1: dict[str, int] = {}
        for pub in by_prefix["readout_all0"]:
            for bitstring, count in normalize_counts(pub["counts"], cfg.qubits).items():
                counts0[bitstring] = counts0.get(bitstring, 0) + int(count)
        for pub in by_prefix["readout_all1"]:
            for bitstring, count in normalize_counts(pub["counts"], cfg.qubits).items():
                counts1[bitstring] = counts1.get(bitstring, 0) + int(count)
        p01, p10, readout_meta = estimate_local_readout_params(
            counts0,
            counts1,
            n_qubits=cfg.qubits,
        )
        readout_meta = dict(readout_meta)
        readout_meta["readout_all0_pubs"] = len(by_prefix["readout_all0"])
        readout_meta["readout_all1_pubs"] = len(by_prefix["readout_all1"])
        readout_meta["shots_all0_total"] = int(sum(counts0.values()))
        readout_meta["shots_all1_total"] = int(sum(counts1.values()))

    site_rows = site_observable_rows(
        cfg,
        main_counts,
        profile="raw",
        p01=None,
        p10=None,
        readout_corrected=False,
    )
    bond_rows = bond_observable_rows(
        cfg,
        main_counts,
        profile="raw",
        p01=None,
        p10=None,
        readout_corrected=False,
    )
    profiles = {"raw": summarize_site_rows(site_rows, "raw")}
    if p01 is not None and p10 is not None:
        corrected_sites = site_observable_rows(
            cfg,
            main_counts,
            profile="readout_corrected",
            p01=p01,
            p10=p10,
            readout_corrected=True,
        )
        corrected_bonds = bond_observable_rows(
            cfg,
            main_counts,
            profile="readout_corrected",
            p01=p01,
            p10=p10,
            readout_corrected=True,
        )
        site_rows.extend(corrected_sites)
        bond_rows.extend(corrected_bonds)
        profiles["readout_corrected"] = summarize_site_rows(site_rows, "readout_corrected")

    sector = sector_survival(main_counts, cfg)
    write_csv(outdir / "cuprate_2d_ibm_site_observables.csv", site_rows)
    write_csv(outdir / "cuprate_2d_ibm_bond_observables.csv", bond_rows)
    plot_path = plot_charge_spin(
        outdir / "cuprate_2d_ibm_readout_corrected_observables.png",
        cfg,
        site_rows,
        profile="readout_corrected" if "readout_corrected" in profiles else "raw",
    )

    summary = {
        "job_id": job_id,
        "backend": backend,
        "config": asdict(cfg),
        "shots": int(sum(main_counts.values())),
        "profiles": profiles,
        "particle_sector": sector,
        "readout_calibration": readout_meta,
        "artifacts": {
            "raw_results_json": (outdir / "cuprate_2d_ibm_raw_results.json").as_posix(),
            "site_observables_csv": (outdir / "cuprate_2d_ibm_site_observables.csv").as_posix(),
            "bond_observables_csv": (outdir / "cuprate_2d_ibm_bond_observables.csv").as_posix(),
            "plot": None if plot_path is None else plot_path.as_posix(),
        },
        "claim_boundary": (
            "Diagonal hardware observables only. Pairing observables require separate "
            "basis-rotation/tomography circuits."
        ),
    }
    write_json(outdir / "cuprate_2d_ibm_observable_summary.json", summary)
    write_summary_md(outdir / "cuprate_2d_ibm_hardware_summary.md", summary)
    return summary


def write_summary_md(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# IBM 72q 2D Cuprate Hubbard Hardware Test",
        "",
        f"- backend: `{summary['backend']}`",
        f"- job id: `{summary['job_id']}`",
        f"- shots: {summary['shots']}",
        f"- lattice/qubits: {summary['config']['lx']}x{summary['config']['ly']} / {2 * summary['config']['lx'] * summary['config']['ly']}",
        f"- steps/dt: {summary['config']['steps']}/{summary['config']['dt']}",
        f"- U/t: {summary['config']['u_over_t']}",
        f"- t'/t: {summary['config']['t_prime_over_t']}",
        "",
        "## Local Observables",
        "",
        "| profile | total charge | total spin-z | mean |spin-z| | mean doublon | min charge | max charge |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, metrics in summary["profiles"].items():
        lines.append(
            f"| {profile} | {metrics['total_charge']:.6f} | {metrics['total_spin_z']:.6f} | "
            f"{metrics['mean_abs_spin_z']:.6f} | {metrics['mean_double_occupancy']:.6f} | "
            f"{metrics['min_charge']:.6f} | {metrics['max_charge']:.6f} |"
        )
    sector = summary["particle_sector"]
    lines += [
        "",
        "## Particle Sector",
        "",
        f"- expected N_up/N_down: {sector['expected_n_up']}/{sector['expected_n_down']}",
        f"- exact sector fraction: {sector['exact_sector_fraction']:.6f}",
        f"- mean abs N_up error: {sector['mean_abs_n_up_error']:.6f}",
        f"- mean abs N_down error: {sector['mean_abs_n_down_error']:.6f}",
        "",
        "## Artifacts",
        "",
    ]
    for label, artifact in summary["artifacts"].items():
        if artifact:
            lines.append(f"- {label}: `{artifact}`")
    lines += [
        "",
        "Claim boundary: this is a hardware smoke test of diagonal observables, not a superconductivity claim.",
        "",
    ]
    path.write_text("\n".join(lines))


def load_submitted_job(outdir: Path) -> dict[str, Any]:
    path = outdir / "submitted_jobs.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit or retrieve a 72q 2D cuprate Hubbard IBM hardware test.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--backend", default=None)
    parser.add_argument("--steps", type=int, default=1, help="Hardware smoke-test Trotter steps.")
    parser.add_argument("--shots", type=int, default=512)
    parser.add_argument("--circuit-family", choices=SUPPORTED_CIRCUIT_FAMILIES, default="explicit_jw")
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--seed-transpiler", type=int, default=72)
    parser.add_argument("--initial-layout", type=Path, default=None, help="JSON layout from cuprate_2d_layout_scout.py.")
    parser.add_argument("--use-layout-seed", action="store_true", help="Use best_candidate.seed_transpiler from --initial-layout when present.")
    parser.add_argument("--readout-repetitions", type=int, default=1, help="Repeat all-0/all-1 readout calibration circuits in the submitted job.")
    parser.add_argument("--readout-shots", type=int, default=None, help="Document desired readout shots; direct IBM runner still uses --shots for one Sampler job.")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--retrieve", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--poll-interval", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cfg = replace(cfg, steps=int(args.steps))
    validate_config(cfg)
    args.readout_repetitions = max(1, int(args.readout_repetitions))
    args.loaded_initial_layout, args.loaded_initial_layout_seed = load_initial_layout_payload(args.initial_layout, cfg.qubits)
    if args.use_layout_seed and args.loaded_initial_layout_seed is not None:
        args.seed_transpiler = int(args.loaded_initial_layout_seed)
    if args.readout_shots is not None and int(args.readout_shots) != int(args.shots):
        print(
            "Warning: --readout-shots differs from --shots, but this direct IBM runner submits one Sampler job. "
            "The requested value is recorded in the plan; submitted circuits use --shots.",
            flush=True,
        )
    service = build_runtime_service()

    if args.retrieve:
        submitted = load_submitted_job(args.outdir)
        job_id = str(submitted["job_id"])
        backend = str(submitted["backend"])
        job = service.job(job_id)
        status = wait_until_done(job, args.wait_seconds, args.poll_interval)
        submitted["last_status"] = status
        submitted["last_checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.outdir / "submitted_jobs.json", submitted)
        if status != "DONE":
            print(f"Job {job_id} not done: {status}")
            return
        process_result(outdir=args.outdir, cfg=cfg, backend=backend, job_id=job_id, primitive_result=job.result(), roles=submitted["roles"])
        print(args.outdir / "cuprate_2d_ibm_hardware_summary.md")
        return

    backend = select_hardware_backend(service, args.backend, cfg.qubits)
    main_circuit, _terms = build_circuit(cfg, measured=True, circuit_family=args.circuit_family)
    readout_circuits = []
    roles = ["main"]
    for rep in range(args.readout_repetitions):
        readout_circuits.extend(build_readout_calibration_circuits(cfg.qubits))
        suffix = "" if args.readout_repetitions == 1 else f"_{rep}"
        roles.extend([f"readout_all0{suffix}", f"readout_all1{suffix}"])
    logical_circuits = [main_circuit, *readout_circuits]

    print(f"Selected backend: {backend_name(backend)} pending={backend_pending_jobs(backend)}", flush=True)
    transpiled = transpile_circuits(
        logical_circuits,
        backend,
        optimization_level=int(args.optimization_level),
        seed_transpiler=args.seed_transpiler,
        initial_layout=args.loaded_initial_layout,
    )
    plan = save_plan(args.outdir, cfg, backend, logical_circuits, transpiled, roles, args)

    if not args.submit:
        print(args.outdir / "cuprate_2d_ibm_hardware_plan.json")
        print("Dry-run only. Add --submit to send to IBM Runtime.")
        return

    job = submit_job(backend, transpiled, shots=int(args.shots))
    submitted = {
        "submitted": True,
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": str(job.job_id()),
        "backend": backend_name(backend),
        "circuit_family": args.circuit_family,
        "shots": int(args.shots),
        "readout_repetitions": int(args.readout_repetitions),
        "readout_shots_requested": int(args.readout_shots) if args.readout_shots is not None else int(args.shots),
        "initial_layout_path": None if args.initial_layout is None else str(args.initial_layout),
        "initial_layout": None if args.loaded_initial_layout is None else list(args.loaded_initial_layout),
        "roles": roles,
        "plan_path": (args.outdir / "cuprate_2d_ibm_hardware_plan.json").as_posix(),
        "transpiled_summaries": plan["transpiled_summaries"],
    }
    write_json(args.outdir / "submitted_jobs.json", submitted)
    print(f"Submitted job: {job.job_id()} on {backend_name(backend)}", flush=True)

    status = wait_until_done(job, args.wait_seconds, args.poll_interval)
    submitted["last_status"] = status
    submitted["last_checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(args.outdir / "submitted_jobs.json", submitted)
    if status != "DONE":
        print(args.outdir / "submitted_jobs.json")
        print(f"Job not done within wait window: {status}")
        return
    process_result(outdir=args.outdir, cfg=cfg, backend=backend_name(backend), job_id=str(job.job_id()), primitive_result=job.result(), roles=roles)
    print(args.outdir / "cuprate_2d_ibm_hardware_summary.md")


if __name__ == "__main__":
    main()
