# Retirement of the Old 2D t-only Route

22 October 2026. This is a narrowly scoped change to the public historical
2D snapshot, based on commit `45fd4daf096d6df7e46ef8e97389bf83c7e3e641`.
It does not import the separate local research history or change the 1D
and Nighthawk repositories.

## Active Use

`number_preserving_t_only` is no longer accepted for new construction,
export, MPS runs, layout searches, planning or hardware submission. The 30
dedicated legacy launchers present in this public snapshot stop with a
clear error before starting WSL or contacting a backend.

The existing alternatives `explicit_jw` and `number_preserving_t_tprime`
remain available. The latter is still an XY approximation without the
nonlocal fermionic parity strings; retirement is not a new accuracy claim.
Existing explicit-JW defaults are unchanged. The MPS command previously
defaulted to t-only and now requires an explicit `--circuit-family` choice.
Setting t-prime to zero in an active model remains allowed.

No alternative is silently substituted for a retired request. Use a new
output directory for a different circuit family. Existing labelled or
unlabelled result directories are protected from relabelling at the
generic circuit, MPS, layout and hardware entry points.

## Historical Results

Saved result files, circuit artifacts, job IDs and analysis outputs are
unchanged. The 12 dedicated historical retrieval wrappers are unchanged.
The IBM and Fire Opal Python runners continue to accept
`--circuit-family number_preserving_t_only --retrieve`, but never in
combination with `--submit`. Existing provider-authorization requirements
still apply; this code change does not authorize a provider call.

For an explicitly historical offline comparison, the original circuit is
available through `historical.cuprate_t_only.build_historical_t_only`.
It has no hardware, export or command-line entry point.

## Verification

Run the offline software tests with:

```sh
python -B -m unittest -v test_cuprate_t_only_retirement
```

The tests cover active choices, refusal before provider access, explicit
MPS selection, result-directory protection, mocked historical retrieval,
legacy controls, launcher guards and a real local circuit export.

Before publication, the retained circuit operators and historical control
are compared with the parent commit on small systems. Those checks do not
establish 6x6 convergence or hardware accuracy. No hardware execution is
part of this retirement.

Completed checks: all 17 offline tests passed. All 24 full-operator
comparisons on 2x2 systems have maximum difference 0.0, covering four
parameter choices for the retained builders, historical helper and retained
shallow-scout controls. The configuration and parent-commit evidence are in
[T_ONLY_RETIREMENT_VERIFICATION.json](T_ONLY_RETIREMENT_VERIFICATION.json).
