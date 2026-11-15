# Changelog

All notable changes to this replication package. The concept DOI
`10.5281/zenodo.21229773` always resolves to the latest version; each release has
its own version DOI (`CITATION.cff` lists them; that of 3.0.0 is added at release).

## [3.0.0]

The package now accompanies the paper *Calibrating Point-Optimal Unit Root
Tests for Level Breaks in Small Samples: Evidence from Real Exchange Rates*
(submitted for publication) and its online supplement. Versions 1.x and 2.x
accompany earlier drafts of the article and do not reproduce this paper.

### Reproducing the paper

- One script per section of the paper: `replicate_sec_2_3.py`, `replicate_sec_4.py`,
  `replicate_sec_5.py` and `replicate_sec_6.py` for Sections 2-6 of the article,
  `replicate_sup_S1.py` to `replicate_sup_S5.py` for Sections S1-S5 of the
  supplement, and `replicate_all.py` for every section with each step once. Each
  script lists the exhibits of its section and the steps of its two modes.
- The quick mode (the default; `--quick` names it) recomputes the exact results and
  the small simulations and compares them with `reference/`, byte for byte or within
  the tolerance each step states (1e-9 relative for most files), and shows the long simulations from `reference/`,
  drawing the figures and rendering the tables from it. `--full` prints a warning,
  the plan and the estimated time, and with `--yes` regenerates every simulation of
  the section from the recorded seeds into `output/` and compares it with
  `reference/`; the long regenerations resume where they stopped after an
  interruption.
- `mlb/` holds the whole code: the kernel and the shared functions, one module per
  computation, the three modules that build the data from the raw downloads, the
  steps (`mlb/steps.py`) and the runner (`mlb/runner.py`).
- `reference/` holds the results the paper prints, as computed for this version:
  `theory/`, `calibration/`, `serial_dependence/`, `application/` and `figures/`.
- The figures carry the paper's numbering: Figure 1 is the size of the nominal 5%
  test (`mlb/size_figure.py`) and Figure 2 the power of the over-detrending
  shortcuts (`mlb/overdetrend_figure.py`, drawn from
  `reference/calibration/overdetrend_power.json`); version 2 numbered them the
  other way around. Figure 3 is new: the nine real exchange rates with the break
  dates of design E and the regime means (`mlb/series_figure.py`, which also computes
  the facts of Section 6.1 and the half-power point of the constant alone at T = 52).

### New computations

- Sections 2-4 and S1-S4: Table 1 as the exact half-power points
  (`mlb/exact_surface_producer.py`); Table 2 from 20 blocks of 165,000 replications
  per cell (`mlb/critical_values_blocks.py`); the limit envelope
  (`mlb/limit_envelope.py`); Lemma 1 in exact rational arithmetic on 3,004 designs
  (`mlb/lemma1_bounds.py`); Proposition 1, the (m+1) regularity of the cell means and
  the strict decrease of the envelope (`mlb/propflat_exact.py`,
  `mlb/regularity_m_plus_1.py`, `mlb/envelope_monotonicity.py`); the comparison of a
  fixed c-bar = -7 with the half-power point (`mlb/rule_comparison.py`,
  `mlb/fixed_index.py`); the oracle quantile ladder and the MAIC lag-zero shares
  (`mlb/oracle_sigma2_ladder.py`, `mlb/maic_lag_zero.py`); the power comparison of
  Section 4.4 and the data of Figure 2 (`mlb/overdetrend_power_precise.py`,
  428,000,000 replications) and the trend-calibrated values of Carrion-i-Silvestre,
  Kim and Perron (2009) (`mlb/ckp_cbar_surface.py`); the summaries of the grid that
  the prose prints (`mlb/prose_numbers_grid.py`).
- Section 5 and S4: Tables S1 and S2 (`mlb/ar1_size_power.py`,
  `mlb/ar1_recalibration.py`).
- Section 6 and S5, the application, rebuilt: the panel of nine currencies against
  the US dollar, 1973-2024 (`mlb/build_ppp_panel.py`); the admission conditions
  C0-C6 over the BIS annual dollar series (`mlb/admissible_universe.py`,
  `mlb/c6_critical_values.py`, `mlb/gate_sizes.py`); the break dates of rule E from
  primary sources (`inputs/regime_events.csv`, `mlb/regime_events.py`); the crisis
  design S-JST (`mlb/build_jst_crises.py`, `mlb/sj_events.py`, `mlb/ppp_sj.py`); the
  statistics, the simulated nulls, the power and the count law
  (`mlb/ppp_application.py`); the half-lives (`mlb/ppp_half_lives.py`); the variance
  profile and Table S9 (`mlb/ppp_variance_scope.py`); the robust level-shift tests
  of Harvey, Leybourne and Taylor (2010) and Carrion-i-Silvestre and Gadea (2023)
  (`mlb/level_shifts.py`, a Python port of the authors' MATLAB functions, and
  `mlb/robust_level_shift.py`); Tables 3 and S3-S8 rendered from `reference/`
  (`mlb/ppp_tables.py`).
- Break dates. The date of the j-th break, floor(lambda_j T), is computed exactly
  (`mlb/break_dates.py`; the product is rounded to three decimals before the
  floor), where versions 2.x evaluated it in binary floating point: six of the 404
  calibration configurations of Table 1 (lambda = (0.15, 0.36, 0.57, 0.78) and
  (0.15, 0.36, 0.57, 0.85) at T = 100, 200 and 300) had the third break one period
  early. Table 1 is unchanged at the printed precision.

### Standalone use

- `python -m model_lb --csv FILE --date-col year --col q --breaks 1986,1993` applies
  the paper's Model LB test to a user's own series with known break dates: MZ_t with
  GLS detrending at c-bar = -7 and the MAIC lag, against the asymptotic critical
  value, a critical value simulated at the series' own configuration under its BIC
  sieve, and the same at the exact half-power point. The functions are in
  `model_lb/api.py`; `pip install -e .` (`pyproject.toml`) makes the package
  importable from any directory.

### Environment and files

- `requirements.txt` lists the exact versions that computed the results in
  `reference/` (numpy 2.3.5, scipy 1.16.3, numba 0.65.1, llvmlite 0.47.0,
  matplotlib 3.10.7, joblib 1.5.2, pandas 2.3.3; Python 3.12).
- Every text file has LF line endings.
- `inputs/DATA_TERMS.md` gives the sources and terms of the data, including the
  crisis years taken from the Jordà-Schularick-Taylor Macrohistory Database (the
  years only; the dataset is not redistributed), the OECD membership dates and the
  critical-value tables of the robust level-shift tests
  (`inputs/level_shift_cv/README.md`).

### Rows and columns that no exhibit prints

- Each result file ships whole, so some rows and columns are read by no table,
  figure or printed number: the band's impulse-response and unscaled arms; the
  i.i.d. arms of the null and power stages; the joint law of the N9 sample and the
  kappa = 0.5 rows; column III of `reference/application/crisis_design/sj_null.csv`;
  and the columns of the application's null and power results that no table reads.
  Their seeds stay in the registry (`mlb/app_seeds.py`), so that the seeds of every
  other item are the ones the results were computed with.

### Removed

The exhibits of the earlier drafts that this paper does not print, with the
inputs, results and code that served only them:

- the figure of the real exchange rate series of the earlier drafts (without the break
  dates and the regime means of the new Figure 3), Figure S1 (the limiting null
  density) and Figure S2 (the half-life intervals);
- the applied calibration surface of version 2 and the sensitivity arms of its
  supplement section (the common nuisance family, other seeds, the exact break
  fractions);
- the empirical modulus and displacement tables of the earlier supplement;
- the Monte Carlo calibration surface and its table (Table 1 is now computed
  exactly), and the version-2 results of the AR(1) tables, the PPP table and the
  half-lives, which the new computations replace;
- the hand-curated break dates of version 2 (replaced by `inputs/regime_events.csv`)
  and its AR-order diagnostic input (now a result of the application,
  `reference/application/main/ppp_ar_diagnostic.csv`);
- the command-line interface of version 2 and its section modules (replaced by the
  section scripts), the modules that nothing used, and the command-line
  experiments of version 2 that no section runs.

## Earlier versions

Versions before 3.0.0 accompany earlier drafts of the article. The entries below
describe those versions: the commands, modules and files they mention belong to them
and are not part of version 3.0.0.

## [2.0.1] — 2026-08-24

- Regenerated the half-life results of `tab:hl` at the size their documentation
  states (`nsim = 1,000,000`, `B = 19,999`, seed `20260704`). The results of 2.0.0
  had been computed with about `nsim = 10,000`, so the regeneration mode of version 2
  did not reproduce the printed values of `tab:hl` and Figure S2; after the
  regeneration it reproduced the shipped results, and the package matched the paper
  of that version.
- Figure S2 regenerated from these results.
- No code change and no change of classification: the three confidence intervals
  that collapse (JPY, NOK, SEK), whether each interval contains one, and the
  dependence of JPY's result on the bootstrap scheme are the same at both sizes.
  Only the point estimates and the interval limits move to their values at the
  larger size (the median half-life of the model without breaks, 7.2 years; the
  upper limit of JPY's interval in the model with breaks, 34.3 years; the half-life
  of each currency in the model without breaks).
- Added a provenance record for each of the two half-life results.

## [2.0.0] — 2026-08-23

- Replaced the content of the public package with a rewritten, self-contained tree:
  one Numba kernel, a single entry point with a default mode that read the shipped
  results and a regeneration mode, and five section modules.
- Ships the full finite-sample critical-value table of the M class (`P_T`,
  `MZ_alpha`, `MZ_t`, `MSB` and `MPT` at the 1%, 2.5%, 5% and 10% levels, c-bar = -7,
  20 blocks of 165,000 = 3,300,000 replications per cell).
- numba-only by design (no NumPy fallback ships); every printed table and figure
  reproduces from files inside the package.
