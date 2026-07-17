# Changelog

All notable changes to this replication package are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/); versioning is
[semantic](https://semver.org/). Version and archival DOIs are recorded in
`CITATION.cff`.

## [1.2.0] — 2026-07-17

Recalibration of the AR(p) nuisance persistence and redesign of the empirical
critical values. The pre-1.2.0 `pac1 = 0.4` was traced to an earlier four-lag
diagnostic and lies outside the entire empirical range of the eight Model-LB
residual series (first-lag ADF coefficients −0.01…0.39, median 0.27); the
canonical value is now **0.27**, with per-currency provenance archived. The
empirical config-faithful critical values switch from the common nuisance
family to **sieve-own** — each currency's fitted ADF φ̂ at its BIC order, held
fixed across replications — so the simulated null prices the configuration in
every dimension the data identify (break dates, break count, AR order,
short-run dynamics), matching the Procedure-2 register documented in
`MC_vs_BOOTSTRAP.md`. Manuscript pair: v73.

### Changed
- `boot_ppp_cbar.py`: `pac1` CLI-exposed (`--pac1`, default 0.27; overwrite
  guard for non-canonical values); `--pac-hw` exposed; empirical block
  default **sieve-own** (`estimate_phi_adf`, companion-matrix stationarity
  gate) with `--common-nuisance` reproducing the family design (own `--out`
  required); `pac1`/`pac_hw` recorded in `meta.json`.
- `dependence_bound.py`, `dependence_count_pmf.py`: per-currency feasible
  powers 0.30/0.31 → 0.48/0.45 (Table 8 at pac1=0.27). Consequences:
  P(zero|H1) independence 0.055 → 0.0063; under dependence 0.21–0.23 →
  0.086–0.098; the modal rejection count under dependence is now 3, not 0.
- `boot_out/` regenerated: surface (`cbar*(2,52,p)`: −10.58/−11.10 at p=1/2,
  feasible power 0.48/0.45) and empirical block (sieve-own cvs; new
  `pvalue_cf` and `nuisance` columns). The p=0 cells reproduce v1.1.9
  bit-for-bit (they are pac1-free), validating that the patch touched
  nothing but the nuisance.

### Added
- `pac_diagnostic.py` → `ppp_pac_diagnostic.csv`: per-currency PACF of
  Δ(LB residual) and first-lag ADF coefficients — the provenance of 0.27.
- `boot_ppp_cbar.py --bp`: single-cell calibration at EXACT break positions
  (bypasses the λ grid; e.g. `--bp 12 19` = the empirical 1985+1992
  configuration, spacing 0.135 < the 0.233 averaging-grid minimum).
- `boot_ppp_cbar.py --seed-base`: alternative MC streams (quantifies the
  ≈0.02 resolution of the config-faithful cv).
- Empirical p-values (`pvalue_cf`): fraction of the simulated null below the
  observed statistic (se ≈ 0.0015 at p ≈ 0.05, R = 20,000).
- `boot_out/sensitivity/`: archived designs behind the manuscript's Table-7
  boundary note — common-family 0.27/0.37/legacy-0.40, three seed streams,
  λ-exact cells; see its `README.md`.

### Empirical consequence
The Canadian dollar sits exactly at the 5% boundary of its sieve null
(p = 0.049–0.050 across all archived streams and calibrations; the binary
verdict at the quantile alternates within its ≈0.02 MC resolution). Reported
in the manuscript as a boundary, not a rejection. All other verdicts
unchanged (p ≥ 0.38).

## [1.1.9] — 2026-07-15

Label-integrity pass on the Table 3 / Figure 3 generator, plus a provenance
correction for the manuscript's Table 3 values. Two sites in
`size_power_cbar_comparison.py` survived the v1.1.5 "Model 1" -> "Model LB"
rename because they did not match the original search patterns. With the code
patched, the canonical production run (`SEED_BASE=20260701`, R=10000, MZt CV5)
gives 0.050/0.050/0.521 for the calibrated row; the values previously pasted
into the manuscript (0.049/0.054/0.534) were traced to the `--selftest`
reference tuple in `robustness.py` (a gate-logic check, not a production run)
and are superseded. Seeds and the c-bar path are unchanged (lambda-exact
-8.40); no numerical routine was touched. Archived on Zenodo (version DOI
`10.5281/zenodo.21398025`).

### Fixed
- `size_power_cbar_comparison.py`: (i) remaining "Model 1"/"MODEL 1"
  doc-string, comment, and dict-key sites renamed to "Model LB"; (ii) the
  LaTeX label VALUE printed into `tab_power.tex` still read
  `Calibrated Model~1` (the non-breaking tilde escaped the earlier
  space-separated replace) -> `Calibrated Model~LB`; (iii) a dangling math
  delimiter left by an earlier overlapping replace (`nominal size .05$`) ->
  `nominal size $0.05$`.
- `robustness.py`: the CLI help text and the EXPERIMENT 4 header referenced
  `figuras_v6.py`, which does not exist in the package; rewritten to state
  that `power_comparison.csv` is a standalone data artifact (no other module
  consumes it) and that the canonical tab:power / fig:power generator is the
  self-contained `size_power_cbar_comparison.py`.
- `mlb_core.py`: last stale diagnostic comment surviving the v1.1.7 anchor
  re-adjudication (`the "~0.3"`) -> `the "~0.24"`.
- `boot_ppp_cbar.py`: corrected a stale docstring calling the empirical block
  "PRELIMINARY (placeholder breaks 1973+1985)"; it reads the real per-currency
  dates from `exog_dates.csv` (the placeholder is a fallback that never fires
  in the shipped package). Renamed the output
  `empirical/ppp_empirical_prelim.csv` -> `empirical/ppp_empirical.csv`
  (it backs Tables 7-8; the "prelim" name was misleading).

### Added
- Bundled the two artifacts behind the empirical tables:
  `boot_out/calib/surface_ppp_boot.csv` (Table `pppsurface`) and
  `boot_out/empirical/ppp_empirical.csv` (Table `ppp`), so every table in the
  paper traces to a shipped simulation output.
- `reconcile_tables.py` and `reconcile_boot.py`: package self-checks that
  reconcile the manuscript's table literals against the artifacts.

### Removed
- `cbar_applied_T52.csv`: a superseded 3-cell placeholder (marked
  "Distributed: no") that no code referenced; dropped from the package.

### Changed
- Figure `cbar` (`cbar_surface`): the six break-count curves were drawn on
  matplotlib's default colour cycle. Switched both generators
  (`replicate_section3_4.make_figure`, `mlb_core.make_surface_figure`) to a
  greyscale, colour-blind-safe scheme (m distinguished by marker + line style +
  grey level), matching the strictly-greyscale Section 5-6 figures. Verified
  0 coloured pixels by rasterisation. The other four figures were already
  greyscale.

### Refactored
- All figure generation is centralized in a new single entry point,
  `generate_figures.py` (reads the CSV artifacts, renders the five figures in
  grayscale, imports only numpy + matplotlib). Plotting was removed from the
  compute modules: `mlb_core.py` (dropped `make_surface_figure`),
  `replicate_section3_4.py` (dropped the figure; added `--limiting-density`,
  which writes `limiting_density.csv` for Figure 1), and
  `size_power_cbar_comparison.py` (now compute-only; writes `power_comparison.csv`
  for Figure 3). `figs_ppp.py` was removed (its plots moved into
  `generate_figures.py`). This enforces a clean compute/figure boundary and lets
  a referee regenerate all figures from the shipped data without any simulation.
- Figure 5 (`fig_hl_forest`): the legend is now an opaque, on-top box so the
  bottom currency's (SEK) light-grey line no longer bleeds through it.

## [1.1.8] — 2026-07-12

Same reproducibility fix as v1.1.7, in the module it was found alongside
but not yet applied to: `boot_ppp_cbar.py`'s `--start-year` (main CLI) and
`--T` (the `--grid` single-cell sanity-check mode) both defaulted to the
pre-v1.1.2 window (1970 / T=55). Corrected to 1973 / T=52, matching the
paper's sample and every other module's default. `empirical_block()`'s own
function-level default (relevant only to direct, non-CLI callers) fixed the
same way.

### Fixed
- `boot_ppp_cbar.py`: `--start-year` default `1970` -> `1973`; `--grid`'s
  `--T` default `55` -> `52`; `empirical_block()`'s `start_year` keyword
  default `1970` -> `1973`. Verified: `--quick` now reports `[window]
  start_year=1973 -> calibration T=52` without needing `--start-year`
  passed explicitly. Archived on Zenodo (version DOI
  `10.5281/zenodo.21328011`).

## [1.1.7] — 2026-07-12

Reproducibility fix and archival completeness pass, found in a second
independent audit. Most consequential: `hl_median_unbiased.py --start-year`
defaulted to 1970, silently reproducing the wrong (pre-v1.1.2) window
(T=55) whenever the flag was omitted -- the exact bug class already fixed
in `pesaran_cd.py` (v1.1.2) and `ppp_ar_diagnostic.csv`'s AR-order selection
(v1.1.4), just in a third module. Also archives the full `cbar_surface.csv`
production surface (previously only a 3-cell placeholder shipped, despite
the Data statement's promise that "every simulation output cited in the
text" is included). Archived on Zenodo (version DOI
`10.5281/zenodo.21327947`).

### Fixed
- `hl_median_unbiased.py`: `--start-year` default `1970` -> `1973` (the
  paper's post-Bretton-Woods sample; the help text already said so, the
  default didn't agree). Usage header replaced with the two full production
  commands (`--boot wild`/`--boot recursive`, `--B 20000`), noting the
  defaults (`B=999`, `nsim=1500`) are a smoke test, not production. Console
  now echoes the effective configuration (window, T per currency, B, nsim,
  boot scheme) before the per-currency table, so a wrong window no longer
  passes silently. Typo "the the numba kernel" fixed.
- `mlb_core.py`: `surface_diagnostics`'s anchor comments were stale
  (0.89/~0.3/~0.15/~2e3); re-adjudicated against the production
  `cbar_surface.csv` (427 configs, 46 cells): R2_mT=0.911, lam_spread=0.241,
  R2_lambda_m1=0.163, cond=2162 -> comments now say 0.91/~0.24/~0.16/~2e3,
  with an explicit provenance note.
- `figs_ppp.py`: stale comment describing the includes-one-vs-alpha_ci_hi
  rule via a GBP example that no longer holds after the AR-order fix (v1.1.4
  moved GBP to p=1; its wild CI is now finite, not the unbounded case the
  comment described) -- rewritten to state the rule generally. Forest plot
  now also reserves the right margin explicitly via `subplots_adjust`
  rather than relying only on `tight_layout()` + `savefig`'s post-hoc
  `bbox_inches="tight"` for the "collapse" annotation's space.

### Added
- `cbar_surface.csv` now ships the **full** production surface (427 configs,
  46 `(m,T)` cells, `m=0..5`, `T=30..300`, both `sigma2_method`s) instead of
  a 3-cell placeholder. The previous placeholder is kept for reference as
  `cbar_applied_T52.csv` (no code path consumes it).
- `check_lam_spread.py`: reports the `(m,T)`-lookup `R^2`, within-cell
  `lambda`-spread, and the `lambda`-only surface's `R^2`/condition number at
  `m=1` (Section 4.3's justification for a lookup table over a `lambda`
  polynomial). Thin wrapper around `mlb_core.surface_diagnostics`.

## [1.1.6] — 2026-07-12

Cosmetic fix to `figs_ppp.py`: the "collapse" annotation on `fig_hl_forest.pdf`
was placed flush with the axes edge and could clip depending on renderer/font
metrics. Given an explicit 0.4 cm (11.34 pt) inset via `xytext`/`textcoords`
so it stays inside the axes regardless of the log-scale x-range. No data or
numerical output changes. `.gitignore` also gained `hl_results_wild.csv`
(the wild-bootstrap companion to `hl_results.csv`, both generated outputs
that were already meant to be excluded per the README but only one was
actually listed).

## [1.1.5] — 2026-07-12

Precision fix to the Section 6.3 dependence discussion, found in an
independent line-by-line audit of the v1.1.4 narrative rewrite. The feasible
power at the tangency is not homogeneous across the eight currencies (five
at `p=1` have power 0.30, three at `p=2` have power 0.31), so `dependence_bound.py`'s
generic three-point grid `{0.30, 0.325, 0.34}` -- a leftover of the pre-AR-order-fix
power range -- did not describe the actual applied sample even after v1.1.4
corrected the AR orders. Rewritten to use the true per-currency power
(heterogeneous, not `power^8`): independence `P(zero|H1)` moves from 0.046
to the exact 0.0552, and the dependence-adjusted figure from the stale
0.18-0.24 to 0.21-0.23. Archived on Zenodo (version DOI
`10.5281/zenodo.21321815`).

### Added
- `dependence_count_pmf.py`: the full probability mass function of the
  rejection count (not just `P(zero)`) under the same heterogeneous
  one-factor model, by exact enumeration of the `2^8` rejection patterns
  plus quadrature over the common factor. Backs a stronger manuscript claim:
  under the estimated dependence, zero rejections is the *modal* outcome
  (`P(0)=0.215`-`0.234 > P(1)=0.192`-`0.198 > ...`, monotone decreasing),
  whereas under independence the mode is two. Deterministic; selftest
  checks the pmf sums to one, recovers the independence (Poisson-binomial)
  limit as `rho -> 0`, and matches the closed-form product at `k=0`.

### Fixed
- `dependence_bound.py`: `POWERS` grid replaced by `CURRENCY_POWER`, the
  actual per-currency feasible power from Table pppsurface at each
  currency's own AR order; `prob_zero` now uses each currency's own `z_i`
  instead of a single value raised to the 8th power.

## [1.1.4] — 2026-07-12

Correctness fix, headline result affected: `ppp_ar_diagnostic.csv`'s AR-order
column (`k_bic_cq`, the only column any downstream script reads) was selected
by BIC on the 1970-2024 (T=55) window and was never re-run after the sample
was fixed to 1973-2024 (T=52) in v1.1.2 -- no script to regenerate it had ever
shipped. Added `select_ar_order.py` (reusing `hl_median_unbiased.py`'s own
`build_Z`/`detrend`/`adf_fit` kernels for methodological consistency) and
re-selected on the correct window: AUD, GBP, and NZD move from `p=4` to
`p=1` (decisively -- BIC is monotonically worse at every higher lag for
AUD/NZD, and still a clear call for GBP); CAD, CHF, JPY, NOK, SEK are
unchanged (`p=2,1,1,2,2`), which is itself a check that the correct-window
selection isn't just noise. No currency selects `p=4` any longer.

Re-running the empirical block and the half-life module with the corrected
diagnostic file changes the paper's headline persistence finding: the
half-life 95% CI collapse count drops from 5 of 8 currencies to 3 of 8 (AUD,
GBP, and NZD no longer collapse from unbounded to bounded, because their
constant-mean interval is now already bounded -- it was artificially
unbounded under the stale `p=4`, which over-fit AR coefficients to a short
effective sample). Median half-life (constant-mean) moves from 8.4 to 7.0
years; the level-break median is essentially unchanged (3.1 to 3.0). Unit-root
verdicts remain unanimous non-rejection for all 8 currencies; the calibration
surface (`tab:pppsurface`, `tab:cbar`, `tab:cv`) is unaffected, since it is
computed for the general `(m,T,p)` grid independent of which currencies use
which cell.

### Fixed
- `ppp_ar_diagnostic.csv`: regenerated on the correct 1973-2024 window
  (`--start 1973 --kmax 10`), replacing the stale 1970-2024 selection.
- `size_power_cbar_comparison.py`: `DEF` replication counts corrected
  `R_cv`/`R_table`/`R_curve` 6000/6000/2500 -> 10000/10000/10000, and the
  default (non-`--recalib`) calibrated c-bar literal corrected -8.7 -> -8.40,
  matching the actual production script (`figuras_v6.py::tab_power`, never
  shipped) that generated the published `tab:power`. The shipped script's old
  defaults reproduced neither the replication count nor the c-bar the paper
  reports; the predecessor `validacao_tab4_fig2.py` had the same defaults, so
  this was inherited at the v1.1.0 rename, not introduced by it.

### Added
- `select_ar_order.py`: BIC and general-to-specific AR-order selection on the
  constant-mean and level-break residuals of the PPP panel, common-sample
  comparison across `k=1..kmax`. Deterministic (OLS-based, no seeds);
  reproduces the shipped `ppp_ar_diagnostic.csv` exactly on re-run.

## [1.1.3] — 2026-07-11

Reproducibility fix: two seed constructions used Python's built-in `hash()` on
strings, which has been salted per process since Python 3.3 (`PYTHONHASHSEED`,
randomized by default and pinned nowhere in this package). The affected
outputs were therefore NOT bit-for-bit reproducible across runs, contradicting
the "fixed seeds"/"deterministic seeds" statements in the paper. The Monte
Carlo noise involved is small but larger than the naive binomial bound
suggests — the per-replication redraw of the AR(p) nuisance coefficients
fattens the null's lower tail, giving the 5% quantile at R=20,000 a
seed-to-seed scatter of ≈0.01–0.02 in MZt units (observed shifts across
seedings up to ±0.04) — and no verdict flips (the tightest margin, CAD, is
≈0.05). The determinism claim required the fix and one canonical rerun of
the empirical block; that rerun (2026-07-11) is now the canonical source of
the paper's Table 8 column (III) and Table 6 digits. Archived on Zenodo
(version DOI `10.5281/zenodo.21315201`).

### Verified
- Cross-platform bit-for-bit reproducibility of the configuration-seeded
  empirical block: the author's Windows run (numba 0.65.1) and an
  independent Linux run (numba 0.66.0, numpy 2.4.4) agree in EVERY field of
  `ppp_empirical_prelim.csv` for all 8 currencies.
- Identical configurations share identical critical values by construction
  (CHF = JPY at -2.0978; NOK = SEK at -2.1503); AUD's 4-decimal agreement
  with the previous currency-seeded run was checked at full precision
  (-2.4353295542 vs -2.4353332680): a 3.7e-6 coincidence of independent
  quantile draws, not a seeding defect.

### Fixed
- `boot_ppp_cbar.py`: the seed of the configuration-faithful critical values
  (`empirical_block`) now (i) uses `zlib.crc32` instead of the salted
  built-in `hash()`, and (ii) is keyed on the CONFIGURATION string
  `(T, m, break positions, p)` rather than on the currency label, so that
  identical configurations share the same simulated null and hence the same
  critical value by construction (CHF and JPY; NOK and SEK). Affects the
  Table 8 column (III) critical values (regenerate with `--empirical`;
  rerun is canonical).
- `mlb_core.py`: the bootstrap standard errors of the critical-value
  quantiles seeded with `hash(k)` on the statistic label; now
  `zlib.crc32(k.encode())`. Affects only the reported CV standard errors in
  the companion files, not the critical values themselves.

- `boot_ppp_cbar.py`: `--empirical` without `--quick`/`--full` ran with an
  empty in-memory surface, so the `(m,T,p)` lookup missed every key and
  silently fell back to `cbar = -7.0` for every currency. It now loads
  `<out>/calib/surface_ppp_boot.csv` from a prior `--full` run, and exits
  with an explicit message if none is found (no silent fallback). This makes
  the deterministic rerun of the empirical block cheap: the calibration
  surface itself is seeded by `config_seed` and was never affected by the
  `hash()` bug.

### Added
- `dependence_bound.py`: dependence-adjusted probability of zero rejections
  across the eight currencies under a one-factor equicorrelated Gaussian with
  the estimated mean pairwise correlation (Section 6.3 of the paper);
  deterministic quadrature, no seeds. Output: `dependence_bound.csv`
  (P(zero|H1) ≈ 0.18–0.24 at power 0.30–0.34 and ρ ∈ {0.37, 0.41}, against
  0.036–0.058 under independence).

### Unchanged
- `hl_median_unbiased.py` (Table 9) and `robustness.py` (the (m,T) surface)
  use fixed integer seeds throughout and were never affected.

## [1.1.2] — 2026-07-09

Bug fix: `pesaran_cd.py` computed the cross-sectional-dependence statistic over
the full 1970-2024 window (T=55), while the rest of the package (and the paper)
defines the sample as 1973-2024 (T=52, post-Bretton-Woods float; see
`boot_ppp_cbar.py --start-year`). On the correct window the numbers move from
CD=12.88/8.12 (levels/diff) to **CD=14.04/8.65**, signed mean correlation
0.328→0.368, and the count of negatively correlated pairs drops from 6 to 4 of
28 (3 involving the Swiss franc, down from 5). No other module or data file
affected; the conclusion (strong positive dependence) is unchanged and, if
anything, strengthened. Archived on Zenodo (version DOI
`10.5281/zenodo.21282961`).

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
