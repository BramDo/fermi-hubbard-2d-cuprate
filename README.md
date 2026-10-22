> **Circuit clarification, 19 October 2026.** This is the historical 2D snapshot, not the later full-fermion Nighthawk experiment. The old one-step `number_preserving_t_only` route omits diagonal hopping and nonlocal fermionic parity strings; onsite gates have no ideal U-dependent measurement effect in that one-step run. See [the circuit/result register](docs/FERMIONIC_ROUTES.md) and [the Nighthawk series](https://edukaizen.nl/2d-local-quantum-advantage-en/). The [full-fermion code repository](https://github.com/BramDo/fermi-hubbard-2d-nighthawk) is private. No matched-accuracy comparison to Google is established.

# 2D Cuprate Fermi-Hubbard Validation

This repository contains the separate 2D cuprate-oriented Fermi-Hubbard route
split out from the original 1D `fermi-hubbard-60q-tdvp` snapshot.

The current focus is a square-lattice one-band Hubbard workflow with `U/t=8`,
`t'/t=-0.25`, hole doping, local diagonal observables, exact small-rung
validation, tensor-network baselines, IBM hardware diagnostics, and Q-CTRL
Fire Opal diagnostics where available.

## Claim Boundary

The hardware output in this repo is diagnostic until it has been compared with
the local baselines and observable-comparison files.

This repo does not claim:

- d-wave superconductivity
- cuprate material behavior
- practical quantum advantage

The validated interpretation is narrower: circuit, noise, sector-survival, and
same-observable comparisons for a staged 2D Hubbard route.

## Main Workflows

**22 October 2026: the old `number_preserving_t_only` route is retired from
new runs.** Its archived results and retrieval scripts remain available.
The existing `explicit_jw` and `number_preserving_t_tprime` alternatives are
unchanged; there is no automatic replacement model. See
[retirement and verification details](docs/T_ONLY_RETIREMENT.md).

Prepare the default 6x6 target:

```powershell
.\scripts\run_cuprate_2d_pipeline_wsl.ps1
```

Run the validation ladder:

```powershell
.\scripts\run_cuprate_2d_validation_ladder_wsl.ps1
```

Run local baselines and comparisons:

```powershell
.\scripts\run_cuprate_2d_tdhf_wsl.ps1
.\scripts\run_cuprate_2d_gaussian_wsl.ps1
.\scripts\compare_cuprate_2d_observables_wsl.ps1
```

The Windows wrappers resolve the repo root from their own location and call the
known WSL Qiskit environment:

```text
/home/bram/.venvs/qiskit/bin/python
```

## GitHub Pages

The static project site lives in `docs/` and is designed for GitHub Pages:

```text
https://bramdo.github.io/fermi-hubbard-2d-cuprate/
```

It summarizes the claim boundary, run paths, curated result locations, and the
canonical Edukaizen 2D cuprate article series. The workflow in
`.github/workflows/pages.yml` publishes `docs/` on pushes to `main` once GitHub
Pages is configured to use GitHub Actions.

## Hardware Safety

Scripts named `submit_*` can submit IBM Runtime or Fire Opal jobs. Other scripts
should be treated as local generation, dry-run, retrieval, comparison, or
analysis unless an explicit `--submit` flag is passed.

Before interpreting a hardware result, compare it against the matching local
baseline under `results_cuprate_2d` and check the sector-conditioned report.

## Repository Layout

```text
cuprate_2d_*.py                 2D target, circuits, baselines, IBM/Fire Opal runners
compare_cuprate_2d_*.py         same-observable comparison tools
analyze_cuprate_2d_*.py         sector-conditioned analysis
scripts/                        PowerShell and Bash workflow wrappers
docs/                           2D report and publication builders
results_cuprate_2d/             curated reproducible 2D result snapshot
output/medium_2d_cuprate_series Medium package for the 2D article series
output/wordpress_2d_cuprate_series WordPress package for the 2D article series
```

## Setup

Base Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional Fire Opal dependencies:

```bash
python -m pip install -r requirements-fire-opal.txt
```

For this machine, prefer the existing WSL environment above when running Qiskit
or Fire Opal workflows.
