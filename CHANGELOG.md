# Changelog

All notable changes to this replication package. The concept DOI
`10.5281/zenodo.21229773` always resolves to the latest version; each release
below also has its own version DOI.

## [2.0.1] — 2026-08-24

- Re-mint the half-life goldens (`output/golden/bootstrap_empirical/hl_results_wild.csv`,
  `hl_results.csv`) at the **documented production N** (`nsim = 1,000,000`, `B = 19,999`,
  seed `20260704`). The v2.0.0 goldens had been minted at a modest N (~`nsim 1e4`), so
  `--full` did not reproduce the printed `tab:hl` / Figure S2 values; the re-mint makes
  `--full` reproduce `--quick` and the package match the paper.
- `Figure_S2` regenerated from the re-minted goldens.
- **No code change and no classification change**: the three CI collapses (JPY/NOK/SEK),
  every `includes_one` flag, and JPY's bootstrap-scheme dependence are N-robust and
  identical. Only point/interval values move to their production-N values
  (median MP half-life 7.2; JPY LB CI upper 34.3 yr; per-currency `HL_MP`).
- Added per-file provenance manifests `hl_results_wild.manifest.json` and
  `hl_results.manifest.json`.

## [2.0.0] — 2026-08-23

- Content-replace of the public package on the refactored, self-contained tree: one
  Numba kernel (`src/mlb/kernel_numba.py`), single entry point `replicate.py`
  (`--quick`/`--full`), five doors under `section_replications/`.
- Ships the full finite-sample M-class critical-value table (`P_T` + `MZα`/`MZt`/`MSB`/`MPT`
  at 1/2.5/5/10 %, `c̄ = −7`, K = 20 × 165,000 = 3,300,000 per cell).
- numba-only by design (no NumPy fallback ships); every printed table and figure
  reproduces from files inside the package.
