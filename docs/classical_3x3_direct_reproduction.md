# Exact classical 3x3 reproduction

This note documents the reproducible classical comparison for the 3x3 one-hole 2D Fermi-Hubbard validation rung.

Run:

```bash
python -B run_classical_3x3_direct_reproduction.py
```

The script reuses `cuprate_2d_exact_sector_baseline.py` and `cuprate_2d_hubbard_pipeline.py` rather than duplicating the Hamiltonian logic.

Model parameters:

- 3x3 lattice, 18 spin orbitals
- one central hole
- `N_up=N_down=4`
- `t=1`
- `t'/t=-0.25`
- `U/t=8`
- `mu=0`
- open boundaries
- `center_neel` initial state
- exact sparse continuous-Hamiltonian evolution at `T=0.2` and `T=0.4`

The fixed particle-number sector has

```text
C(9,4)^2 = 15,876
```

basis states.

Outputs are created in:

```text
results_cuprate_2d/classical_3x3_direct_reproduction/
```

with:

- `results.json`
- `REPORT.md`

The hardware comparison row is recorded as metadata only:

```text
Readout + TFLO + DD, not validated | 32.078 s | 0.072 | 0.091 | 0.052
```

Interpretation boundary: an exact classical solution compared with itself has zero RMSE up to floating-point roundoff. That statement must not be confused with the error of the finite Trotter circuit relative to the exact continuous Hamiltonian. Trotter error is a separate comparison.

No QPU/provider call is made by this script.
