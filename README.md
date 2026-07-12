# Replication package

**Level Breaks and Finite-Sample GLS Detrending: The Point-Optimal Unit Root
Test and the Purchasing Power Parity Puzzle**

This package reproduces the paper section by section and provides a stand-alone
tool for applying the Model LB test to any user series. All code is Python 3.12,
deterministic given the seeds and replication counts stated below, and built
around a single numerical kernel so that every result — the calibration, the
robustness experiments, the empirical application, and a user's own test — runs
on the identical implementation and the identical long-run-variance convention.

---

## 1. Layout: one core, one driver per section, one tool, one diagnostic

| File | Kind | Reproduces |
|---|---|---|
| `mlb_core.py` | **Kernel library** (not run directly). The single implementation of the Model LB test and its finite-sample calibration: the `@njit` numerical kernels, the calibration driver `run_grid`/`aggregate`, the single-series interface `run_test`, and the validation gates (`python mlb_core.py --selftest`). Everything else imports from here. | — |
| `replicate_section3_4.py` | Driver | **Sections 3–4**: the calibration surface `c-bar(m,T)` with the critical values of the five M-class statistics, and Figure `cbar`. Writes `cbar_surface.csv` (the input to Sections 5 and 6). |
| `replicate_section5.py` | Driver | **Section 5**: robustness (`ar1`, `oracle`, `trimming`) and the power comparison (`power`) — Tables `ar1-sizepower`, `serial`, `trimming`, Table/Figure `power`. |
| `replicate_section6.py` | Driver | **Section 6**: the PPP application — admissibility `sweep`, AR(p) nuisance-family `boot`, median-unbiased `hl`, and the two `figures`. |
| `run_model_lb.py` | Tool | **Apply the test to your own series** (see §3 below). |
| `pesaran_cd.py` | Diagnostic | **Section 6 footnote**: the Pesaran (2004, 2015) cross-sectional-dependence (CD) test, run on the OLS residuals of the eight per-currency, strictly univariate Model LB fits (no panel model is estimated). Reports the CD statistic, its p-value, and mean pairwise `|rho_ij|`, plus a first-differenced robustness check against residual serial correlation. Quantifies the dependence induced by the common US-dollar numeraire; under exogenous level breaks each test's limiting law is invariant to that common component, so the diagnostic is reported but **not used for inference**. Run directly: `python pesaran_cd.py` (defaults to `ppp_panel.csv`/`exog_dates.csv`) or `python pesaran_cd.py --selftest` (3 gates: size, common-factor power, design). |
| `dependence_bound.py` | Diagnostic | **Section 6.3**: the dependence-adjusted probability of zero rejections across the eight currencies under a one-factor equicorrelated Gaussian at the estimated mean pairwise correlation (`pesaran_cd.py`'s $\bar\rho\in\{0.37,0.41\}$) and each currency's own feasible power at its applied AR order (0.30 for the five at $p=1$, 0.31 for the three at $p=2$; heterogeneous, not a single value); deterministic quadrature, no seeds. Run directly: `python dependence_bound.py` -> `dependence_bound.csv`. |
| `dependence_count_pmf.py` | Diagnostic | **Section 6.3**: the full probability mass function of the rejection count (not just $P(0)$) under the same heterogeneous one-factor model, by exact enumeration of the $2^8$ rejection patterns plus quadrature over the common factor. Shows zero rejections is the *modal* outcome under the estimated dependence, versus a mode of two under independence. Deterministic. Run directly: `python dependence_count_pmf.py`. |
| `select_ar_order.py` | Tool | Regenerates `ppp_ar_diagnostic.csv`: BIC and general-to-specific AR-order selection on the constant-mean and level-break residuals of `ppp_panel.csv`, common-sample comparison across `k=1..kmax`. Deterministic (OLS, no seeds); reuses `hl_median_unbiased.py`'s own `build_Z`/`detrend`/`adf_fit` kernels so the residual construction matches the rest of the package. Run directly: `python select_ar_order.py --start 1973 --kmax 10`. |
| `check_lam_spread.py` | Diagnostic | **Section 4.3**: the $(m,T)$-lookup $R^2$, within-cell $\lambda$-spread, and the $\lambda$-only CKP surface's $R^2$/condition number at $m=1$ that justify a lookup table over a $\lambda$ polynomial. Thin CLI wrapper around `mlb_core.surface_diagnostics` (single implementation, not a second copy). Run directly: `python check_lam_spread.py --csv cbar_surface.csv`. |

Supporting modules called by the drivers (not run directly): `mlb_kernel.py`
(pure-Python fallback for `mlb_core.py` when numba is unavailable), `robustness.py`
and `size_power_cbar_comparison.py` (Section 5), and `ppp_sweep_bis.py`,
`boot_ppp_cbar.py`, `hl_median_unbiased.py`, `figs_ppp.py`
(Section 6). The trend-break comparison curve in Section 5 uses the CKP
response surface, **inlined** in `size_power_cbar_comparison.py`; the package has no
dependency on the companion growth-empirics paper.

### Data and artifact files

| File | Category | Read by | Produced by | Required? |
|---|---|---|---|---|
| `ppp_panel.csv` | input (public data) | Section 6 (`boot`, `hl`, `figures`) | built from BIS + World Bank | **yes** |
| `exog_dates.csv` | input (public data) | Sections 6 (`sweep`, `boot`, `hl`, `figures`) | primary central-bank sources | **yes** |
| `ppp_ar_diagnostic.csv` | input (derived) | Section 6 (`boot`, `hl`) | AR(p) lag selection on the panel (see below) | **yes** for `hl`/`boot` |
| `cbar_surface.csv` | inter-section bridge | Sections 5–6, `mlb_core`, `run_model_lb` | **`replicate_section3_4.py --full`** | bundled: the full production surface (427 configs, 46 `(m,T)` cells, `m=0..5`, `T=30..300`, both `sigma2_method`s) — regenerate only to reproduce it from scratch |
| `cbar_applied_T52.csv` | reference (legacy) | none (kept for reference only) | superseded 3-cell `T=52` placeholder, retained under this name after `cbar_surface.csv` was replaced with the full surface | no |

Files the pipeline *generates* and that are therefore NOT shipped as inputs
(they are in `.gitignore`): `hl_results.csv` and `hl_results_wild.csv`
(written by `replicate_section6.py hl`, consumed by the `figures` stage).

**Provenance of `ppp_ar_diagnostic.csv`.** For each currency it records the
AR order `p` used under the null, selected by BIC and general-to-specific on the
post-break residual of `q_it` (max lag `k_max = 10`), together with the implied
`alpha` and scalar half-life for the constant-mean and level-break
specifications. It is supplied ready so the `hl`/`boot` stages need not re-run
the selection; it is regenerated by `select_ar_order.py` from `ppp_panel.csv`
and `exog_dates.csv` (only `k_bic_cq` is consumed downstream — the other
columns are diagnostic).

**Two Monte Carlo / bootstrap notes.** The package runs three distinct
resampling/simulation procedures — parametric Monte Carlo calibration (§3–4),
AR(p) nuisance-family calibration (§6 calibration), and wild bootstrap (§6 half-life). They
complement each other and are never interchangeable; see `MC_vs_BOOTSTRAP.md`
for the exact assumptions, interfaces, and which number each one produces.

---

## 2. Installation

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Only `numpy` and `numba` are needed for the core; `scipy` for the sweep,
`matplotlib` for the figures, `joblib` for parallel Monte Carlo. No
`pandas`/`openpyxl` — all I/O is via the `csv` standard library.

---

## 3. Apply the test to your own series

`run_model_lb.py` runs Model LB on one CSV series at break dates **you** supply
and justify as **exogenous** — a policy or regime event dated independently of
the data (the known-date regime of Perron, 1989). It is not an endogenous break
search. It uses the same kernel and the same long-run-variance convention
(OLS-detrended, Perron–Qu) that produce the paper's tables, so your numbers are
comparable to the published ones.

```bash
python run_model_lb.py --csv my_series.csv --date-col year --col log_rer \
    --breaks 1985,1998 --sigma2 maic --calib cbar_surface.csv
```

- `--breaks` accepts dates (matched against `--date-col`) or 1-based positions.
- `--sigma2 {const,maic}`: difference-based (baseline) or Ng-Perron MAIC
  (serial-correlation robust; the applied default).
- `--calib` reads `c-bar` and the tabulated MZt critical value by nearest
  `(m,T)`; omit it and `c-bar = -7` while all critical values are simulated on
  the fly in your exact break configuration.
- `--json out.json` writes the full result. Rejection is the LOWER tail.

---

## 4. Reproduce the paper (ordered pipeline)

Each driver has `--full` (paper numbers) and `--quick` (smoke test). Sections 5
and 6 read the calibration surface from Section 3–4.

```bash
# (1) Sections 3-4 -- calibration surface + Figure `cbar`  [hours at --full]
python replicate_section3_4.py --full --jobs -1
#     writes cbar_surface.csv (consumed by the next two)

# (2) Section 5 -- robustness + power comparison
python replicate_section5.py all --outdir section5_out
#     or individually:
python replicate_section5.py ar1 --recalibrate      # Table `ar1-sizepower`/`ar1-recal`
python replicate_section5.py oracle                 # Table `serial`
python replicate_section5.py trimming               # Table `trimming`
python replicate_section5.py power                  # Table/Figure `power`

# (3) Section 6 -- PPP application
python replicate_section6.py sweep --fetch          # admissibility funnel (see §5 below)
python replicate_section6.py boot                   # AR(p) nuisance-family calibration
python replicate_section6.py hl                     # half-lives + decomposition
python replicate_section6.py figures                # the two Section-6 figures
#     or the whole application:
python replicate_section6.py all
```

Validation gates (run these first to confirm the build):

```bash
python mlb_core.py --selftest            # size ~ 0.05; |dMZt| ~ 1e-16; power > size
python replicate_section5.py --selftest  # 7 pure-logic gates (no numba needed)
python hl_median_unbiased.py --selftest  # 8 gates (median function, coverage, ...)
python pesaran_cd.py --selftest          # 3 gates (size, common-factor power, design)
```

---

## 5. Data sources (public)

All data are public and free. The bundled CSVs were built from the sources
below; the sweep stage can rebuild the raw inputs directly.

**BIS — bilateral US-dollar exchange rates (WS_XRU, annual, period average).**
The admissibility sweep downloads the official bulk export with `--fetch`:
`python replicate_section6.py sweep --fetch` retrieves
`https://data.bis.org/static/bulk/WS_XRU_csv_flat.zip` (BIS Data Portal,
Exchange rates dataset) and caches it under `ppp_sweep/cache/`. Homepage:
https://data.bis.org (dataset WS_XRU, "US dollar exchange rates").

**World Bank — Consumer Price Index (FP.CPI.TOTL, 2010 = 100).** Fetched from
the World Bank Indicators API:
`https://api.worldbank.org/v2/country/all/indicator/FP.CPI.TOTL` (JSON), cached
under `ppp_sweep/cache/`. Landing page:
https://data.worldbank.org/indicator/FP.CPI.TOTL.

**Construction.** The real exchange rate is
`q_t = ln(CPI_US,t) − ln(CPI_i,t) − ln(XR_i,t)`, with `XR` in local currency per
US dollar. `ppp_panel.csv` is the resulting panel for the eight admissible
currencies over 1973–2024; `exog_dates.csv` lists the exogenous regime-break
dates and their primary sources. Accessed July 2026; the series are periodically
revised at source, so a fresh `--fetch` may differ in the last decimals from the
bundled snapshot without affecting any reject/fail-to-reject verdict.

---

## 6. Numerical reproducibility

- All Monte Carlo and bootstrap routines use fixed seeds (`config_seed` in
  `mlb_core.py`); results are bit-reproducible on a fixed platform for a given
  replication count.
- Published figures use the full counts (calibration `R` as set inside
  `mlb_core`; bootstrap `B = 9999`; on-the-fly critical values `R = 9999`). The
  `--quick` switches reduce these for testing and are not the published numbers.
- Cross-platform BLAS/threading and numba minor versions can move the last few
  decimals of a simulated critical value; this does not change any verdict at
  the counts above. For exact bit-reproduction, pin `requirements.txt` and run
  single-threaded (`NUMBA_NUM_THREADS=1`, `OMP_NUM_THREADS=1`).

---

## 7. Method, in one paragraph

Model LB is the no-trend restriction of the CKP (2009) multiple-break setup: a
constant plus known-date level dummies. Under GLS quasi-differencing a level
dummy becomes an impulse, orthogonal across break dates, so the point-optimal
test is asymptotically invariant to the number and location of the breaks and
the optimal detrending parameter keeps the ERS (1996) demeaned value `-7` for
every configuration. In finite samples the optimum departs from `-7` by an
`O((m+1)/T)` remainder, which the calibration prices; the test then delivers the
statistic and its critical value as the coupled output of one calibration. The
application dates the breaks exogenously from currency-regime events and reads
the resulting non-rejections as a correction to the measurement of persistence,
not as a verdict on purchasing power parity.

---

## 8. Citation

Please cite the paper (see `CITATION.cff` for the DOI once assigned on
acceptance) and this software archive:

> Gonçalves Silva, R. (2026). Replication package: Level Breaks and
> Finite-Sample GLS Detrending. Zenodo. https://doi.org/10.5281/zenodo.21229773

and the primary reference:

> Carrion-i-Silvestre, J.L., Kim, D., Perron, P. (2009). GLS-based unit root
> tests with multiple structural breaks under both the null and the alternative
> hypotheses. *Econometric Theory* 25(6), 1754–1792.
