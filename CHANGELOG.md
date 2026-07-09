# Changelog

All notable changes to this replication package are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/); versioning is
[semantic](https://semver.org/). Version and archival DOIs are recorded in
`CITATION.cff`.

## [1.1.2] — 2026-07-09

Bug fix: `pesaran_cd.py` computed the cross-sectional-dependence statistic over
the full 1970-2024 window (T=55), while the rest of the package (and the paper)
defines the sample as 1973-2024 (T=52, post-Bretton-Woods float; see
`boot_ppp_cbar.py --start-year`). On the correct window the numbers move from
CD=12.88/8.12 (levels/diff) to **CD=14.04/8.65**, signed mean correlation
0.328→0.368, and the count of negatively correlated pairs drops from 6 to 4 of
28 (3 involving the Swiss franc, down from 5). No other module or data file
affected; the conclusion (strong positive dependence) is unchanged and, if
anything, strengthened.

### Fixed
- `pesaran_cd.py` gains a `--start` option (default `1973`, matching
  `boot_ppp_cbar.py --start-year`) that filters `ppp_panel.csv` before
  computing residuals and the CD statistic. Previously the script had no
  window filter and silently used the full 1970-2024 panel.

## [1.1.1] — 2026-07-08

Documentation-completeness fix: the v1.1.0 archive omitted `pesaran_cd.py`
from the shipped files even though it was already used to produce the Section
6 cross-sectional-dependence number reported in the paper. No numerical
result changes; no other module touched. Archived on Zenodo (version DOI
`10.5281/zenodo.21231384`).

### Added
- `pesaran_cd.py` -- a stand-alone Pesaran (2004, 2015) cross-sectional
  dependence diagnostic. Computes the CD statistic on the residuals of the
  strictly univariate Model LB fits of the PPP panel (no panel model is
  estimated); reports the average pairwise residual correlation and the CD on
  first differences as a serial-correlation robustness check. It quantifies the
  dependence induced by the common US-dollar numeraire that is discussed in
  Section 6; inference is not built on it. Validated by
  `python pesaran_cd.py --selftest` (size ~0.05 under independence; power 1.0
  under a common factor; Model LB design gate).
- `README.md` and `.zenodo.json` now mention `pesaran_cd.py` alongside the
  other supporting modules.

### Verified
- All 13 modules byte-compile; every local `import` and `runpy` call resolves to
  an existing module; no orphan CSV string references; no residual Portuguese or
  version suffixes in source.

## [1.1.0] — 2026-07-06

Repository hygiene release: no numerical result changes. Every figure, table,
and calibrated value is identical to v1.0.1; only file names, symbol names, and
documentation were standardized for public reproducibility. Archived on Zenodo
(version DOI `10.5281/zenodo.21231381`).

### Changed
- **File names standardized** (internal version suffixes removed):
  - `robustness_v6.py` → `robustness.py`
  - `boot_ppp_cbar_production_v2.py` → `boot_ppp_cbar.py`
  - `figs_ppp_v2.py` → `figs_ppp.py`
  - `validacao_tab4_fig2.py` → `size_power_cbar_comparison.py`
    (renamed for what it does — a size/power comparison of c-bar specifications —
    rather than for volatile paper table/figure numbers)
- **Data artifact names unified.** The calibration surface is written and read
  under the single name `cbar_surface.csv` throughout (previously the kernel
  wrote an internal `resultados_cbar_ml1_v5.csv` inside a versioned
  `checkpoints_cbar_ml1_v5/` directory, then copied it). The checkpoint directory
  is now `cbar_checkpoints/`. The write→copy step is retained only to relocate
  the surface from the checkpoint directory to the distribution root / `--outdir`.
- **All source is English.** Portuguese docstrings, comments, and the one
  Portuguese function name (`gerar_dgp` → `generate_dgp`, plus internal
  `enumerar_configs`, `ja_processado`, `agregar`, `calibrar_config`,
  `gerar_lambdas`, `salvar_resultado`, `pos_processar` → their English
  equivalents) were translated; all cross-module calls were reconciled.
- Internal version-tracking comments (`v2`/`v3`/`v5`/`v6`, machine-specific
  notes) removed from all sources; the technical content they carried was kept.

### Added
- `mlb_kernel.py` -- the pure-Python (NumPy-only) fallback kernel, so the package
  runs on machines without numba. It is numerically equivalent to `mlb_core.py`
  (design matrix, GLS quasi-differencing, M-class statistics, long-run variance)
  and is imported automatically by `boot_ppp_cbar.py` when numba is absent.
  Validated by `python mlb_kernel.py --selftest` (size ~0.05; exact Lemma 1
  invariance to ~1e-16; power > size). This closes the previously dangling
  fallback reference, which raised SystemExit for a file that was not shipped.
- `CHANGELOG.md` (this file).
- Function-level docstrings for the utility routines in `hl_median_unbiased.py`
  (`detrend`, `load_panel`, `load_dates`, `load_p`, `fmt_ci`).
- `MC_vs_BOOTSTRAP.md` now documents the Section 5 robustness stage (AR(1)
  recalibration, the true-ω² oracle, break-fraction trimming, and the c-bar
  specification comparison) alongside the three resampling procedures.
- `CITATION.cff` now carries the concept DOI (`10.5281/zenodo.21229773`) and the
  v1.0.1 version DOI (`10.5281/zenodo.21229774`).

### Verified
- All 11 modules byte-compile; every local `import` and `runpy` call resolves to
  an existing module; no orphan CSV string references; no residual Portuguese or
  version suffixes in source. See the standardization report in the release notes.

## [1.0.1] — 2026-07-06

- First public archival release; triggered the Zenodo deposit
  (version DOI `10.5281/zenodo.21229774`).

## [1.0.0]

- Initial internal replication package accompanying the manuscript.
