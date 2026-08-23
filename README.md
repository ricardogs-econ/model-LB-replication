<h1 align="center">Level Breaks and Finite-Sample GLS Detrending</h1>

<p align="center">
  <em>The Point-Optimal Unit Root Test and the Purchasing Power Parity Puzzle</em>
</p>

<p align="center">
  Replication package for the <strong>Model&nbsp;LB</strong> test, its finite-sample
  calibration, and the PPP application.
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.21229773"><img alt="DOI" src="https://zenodo.org/badge/DOI/10.5281/zenodo.21229773.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg"></a>
  <img alt="Version 2.0.0" src="https://img.shields.io/badge/version-2.0.0-blue">
  <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Reproducible" src="https://img.shields.io/badge/build-reproducible-success">
  <img alt="Self-checked" src="https://img.shields.io/badge/tables%20%26%20figures-self--checked-success">
</p>

---

## What is this?

Standard unit-root tests **over-detrend** when a series has exogenous level
breaks (dated regime shifts), destroying power. This package implements
**Model&nbsp;LB** — the no-trend restriction of the Carrion-i-Silvestre, Kim &
Perron (2009) point-optimal test — together with the finite-sample calibration
surface `c̄(m,T)` that keeps it correctly sized at the short spans macro data
actually offer. It then applies the apparatus to the **purchasing power parity
puzzle** on eight admissible real exchange rates, 1973–2024.

Every number in the paper is reproducible from the code plus the shipped frozen
inputs, and every table and figure ships with a one-command self-check.

### Design in one line

```text
compute modules  ──write──▶  CSV artifacts  ──read──▶  figures ──▶ PDFs
(numba Monte Carlo)          (traceable data)          (grayscale, no simulation)
```

The numerical work lives in **one kernel** (`src/mlb/kernel_numba.py`). The
kernel is **numba-only**: there is no pure-Python fallback, because a fallback
could return a silently different calibration number on the MAIC `k≤0` `s²`
path — and a replication package must not ship that. `requirements.txt` pins
numba. Computation and plotting are strictly separated: compute modules only
write CSVs; the figure code only reads them. A referee can regenerate the
CSV-backed figures from the shipped data **without running any simulation**.

---

## Quick start

```bash
python -m pip install -r requirements.txt   # numpy, scipy, numba, matplotlib, joblib
python replicate.py                         # door table + every door in --quick
```

`python replicate.py` prints the door table and reproduces every printed object
from the shipped frozen inputs and golden artefacts — fast, no heavy Monte
Carlo. Add a door name to run just one; add `--full` to regenerate.

### Apply the test to your own series

`model_lb/model_lb_standalone.py` is a self-contained CLI: point it at a CSV,
name the series, and give the known break dates.

```bash
python model_lb/model_lb_standalone.py --csv myseries.csv --col rer --date-col year \
       --breaks 1985,1992 --sigma2 maic --calib inputs/cbar_surface.csv
```

| Flag | Meaning |
|---|---|
| `--csv` | CSV with the series, rows in time order (**required**) |
| `--col` | series column (default: last numeric column) |
| `--date-col` | optional date column, to match `--breaks` |
| `--breaks` | comma-separated exogenous break **dates** |
| `--sigma2` | long-run variance: `const` (difference-based) or `maic` |
| `--calib` | calibration surface (`inputs/cbar_surface.csv`) for the finite-sample `c̄` and critical value |
| `--alpha`, `--kmax`, `--nsim`, `--json` | level, MAIC lag ceiling, on-the-fly CV reps, JSON output |

---

## Repository layout

```
replicate.py              single entry point (door table + dispatch; --quick default, --full opt-in)
src/mlb/                  the numba kernel + facades + the single path resolver
  kernel_numba.py           @njit GLS-detrending, the five M-class statistics, calibration driver
  paths.py                  the single resolver (inputs AND outputs); no file finds its own path
  kernel_select.py          the one documented kernel binding (numba-only)
  design.py detrend.py dgp.py lrv.py resample.py statistics.py calibrate.py aggregate.py io.py
section_replications/     one door per theoretical layer (named by the manuscript's \label)
  critical_values.py        sec:calibration/cv/shortcuts: tab:cbar, tab:cv, tab:power, fig:overdetrend, fig:size
  robustness.py             sec:robustness: tab:ar1-sizepower, stab:recal  (ar1_sizepower_recal engine)
  bootstrap_empirical.py    sec:empirical: tab:dates, tab:hl, tab:ppp, tab:pppsurface, fig:rerseries
  supplement.py             appendix S: stab:objects, stab:disp, stab:disp/L_T, sfig:density, sfig:forest
  figures.py                fig1..figS2 (consolidated figure rendering)
model_lb/                 standalone: apply Model LB to your own series (model_lb_standalone.py)
inputs/                   frozen inputs a --quick run reads (never regenerated here)
  cbar_surface.csv          the calibration surface (produced_outside, regenerable_here=false)
  ppp_panel.csv exog_dates.csv ppp_ar_diagnostic.csv DATA_TERMS.md
  limiting_density.csv       Fig S1 limiting null law (frozen; renders sfig:density)
  boot_out/calib/            surface_ppp_boot.csv (tab:pppsurface source)
  boot_out/sensitivity/      §S5 common-nuisance / seed / lambda-exact arms (archived per the supplement)
  ppp_sweep/                 sweep_gates.csv -- series-by-series admissibility attrition (C1-C5)
  MANIFEST.json             per-file sha256, provenance_class, regenerable_here
  OBJECT_MAP.csv            number -> producer -> artefact -> digest (the reproduction map)
output/golden/            the promoted, frozen printed artefacts (.tex / .csv / .json)
tests/                    invariance/convention tests (theta, chunk, --jobs, PQ k=0)
```

Every path resolves through `src/mlb/paths.py`; no file computes its own path,
and no door takes a free `--outdir`. `--full` writes only under `output/`,
never into `inputs/` or `output/golden/`.

---

## Reproduce the paper

One entry point. `--quick` (default) reproduces each printed object from the
shipped frozen inputs and golden artefacts; `--full` regenerates the
regenerable-here objects (never the calibration surface, which is a frozen
external input).

```bash
python replicate.py                       # every door, --quick (fast)
python replicate.py cv                     # sec:calibration/cv/shortcuts
python replicate.py robustness --full      # regenerate tab:ar1-sizepower + stab:recal
python replicate.py ppp                    # sec:empirical
python replicate.py supplement             # appendix S
python replicate.py figures                # fig1..figS2
python tests/run_all.py                    # invariance/convention gates
```

### Exhibit map

| Exhibit (label) | Door | Frozen artefact |
|---|---|---|
| `tab:cbar` c̄ surface | `replicate.py cv` | `output/golden/printed_tables/tab_cbar_mT.tex` |
| `tab:cv` critical values (printed MZt 5%) | `replicate.py cv` | `output/golden/critical_values/tabcv_reconciliation.csv` |
| `tab:cv` full M-class percentiles (1/2.5/5/10% × P_T,MZα,MSB,MZt,MPT) | `replicate.py cv --full` | `output/golden/critical_values/tabcv_full_percentiles.csv` |
| `tab:power` size/power | `replicate.py cv` | `output/golden/critical_values/tabpower_runC.json` |
| `tab:ar1-sizepower` AR(1) size/power | `replicate.py robustness [--full]` | `output/golden/calibration_robustness/ar1_size_power.csv` |
| `stab:recal` recalibration | `replicate.py robustness [--full]` | `output/golden/calibration_robustness/ar1_recalibration.csv` |
| `tab:dates` break dates | *(input)* | `inputs/exog_dates.csv` |
| `tab:ppp` PPP verdicts | `replicate.py ppp` | `output/golden/bootstrap_empirical/ppp_empirical.csv` |
| `tab:hl` half-lives | `replicate.py ppp` | `output/golden/bootstrap_empirical/hl_results_wild.csv` |
| `tab:pppsurface` applied calibration | `replicate.py ppp` | `output/golden/bootstrap_empirical/tab_pppsurface.tex` |
| `stab:objects` / `stab:disp` modulus | `replicate.py supplement` | `output/golden/supplement_appendix/modulus_feasible_*.csv` |
| `stab:disp/L_T` displacement ladder | `replicate.py supplement` | `output/golden/supplement_appendix/g36_ladder.json` |
| `fig:overdetrend` (Figure 1) / `fig:size` (Figure 2) | `replicate.py cv --full` | regenerated (self-contained MC); data == `tab:power` |
| `fig:rerseries` / `sfig:density` / `sfig:forest` | `replicate.py figures` | regenerated from the shipped CSVs |

---

## Data sources

| Source | Series | Access |
|---|---|---|
| **BIS** | bilateral USD nominal exchange rates (`WS_XRU`, annual) | [data.bis.org](https://data.bis.org) (bulk `WS_XRU_csv_flat.zip`) |
| **World Bank** | Consumer Price Index (`FP.CPI.TOTL`, 2010 = 100) | World Bank Indicators API |
| **Central banks / G-5** | exogenous regime break dates (Plaza, ERM exits, floats) | primary sources, cited in `inputs/exog_dates.csv` |

The panel is rebuildable from these public sources; terms and the exact series
identifiers are in `inputs/DATA_TERMS.md`.

---

## Admissible universe: attrition cascade

The paper states the five admissibility conditions and the eight admitted
currencies (§6.1); the full attrition arithmetic lives here.

A currency `i` belongs to the admissible universe `U` iff it satisfies

| Condition | Statement |
|---|---|
| (C1) | Own currency, continuously observed 1973–2024 |
| (C2) | `T_i ≥ 45` |
| (C3) | `max_t π_it < 0.30` (no annual CPI inflation above 30%) |
| (C4) | `Var(Δq_it)` conditionally stable |
| (C5) | At least one exogenously datable currency-regime event (a known-date regime in the sense of Perron 1989) |

Attrition, applied to the complete BIS annual universe of **192 area–currency
series covering 147 distinct currencies**:

| Stage | Removed | Remaining |
|---|---|---|
| (C1)+(C2): own-currency/continuity triage (ISO-issuer rule removing currency unions, back-calculated euro-legacy series, and non-issuer users of shared currencies; hard-peg screen) and the span condition | 122 | **70** |
| (C3): inflation ceiling | 40 | **30** |
| (C4): conditional-variance gate | 5 | **25** |
| (C5): known-date condition — 14 managed floats against unannounced baskets or de-facto pegs with no exogenously datable regime event against the dollar, plus 3 crisis-identified near-misses | 17 | **8** |

Admitted: **AUD, CAD, CHF, GBP, JPY, NOK, NZD, SEK** against the US dollar,
annual, 1973–2024 (`T = 52`).

**Two operationalizations of (C4).** (1) *Declared gate (envelope form)* —
variance-ratio, robust Levene, and a CUSUM-of-squares statistic evaluated
against the envelope of the admitted set. (2) *Pre-registered absolute form*
— a two-sided `F` test for the variance ratio, Brown–Forsythe, and the
CUSUM-of-squares in the heteroskedasticity- and dependence-robust `κ₂` form
of Sansó et al. (2004), evaluated on a significance grid from 5% down to 1%.
Under the absolute form, **six of the eight** admitted currencies pass at
every level and **seven** pass at 1%; the single marginal case is the Swiss
franc.

**The three near-misses.**

| Currency | Why it clears the mechanical gates | Why it is excluded |
|---|---|---|
| Korean won | Passes, but narrowly: peak inflation 28.7% | 1997 event is a twin currency–banking crisis; the level shift arrives jointly with a variance shift the level-only apparatus does not model |
| Thai baht | Passes (C4) under the declared envelope | Same crisis-driven identification |
| South African rand | Passes (C4) under **both** operationalizations | Same crisis-driven identification |

All three are left to the variance-break extension discussed in the paper's
conclusion.

---

## Monte Carlo design

The finite-sample optimum `c̄(m,T)` and the critical values behind it are located by a
single tangency procedure applied identically to every cell. The design is anchored on the
demeaned ERS constant: the paper proves the tangency has the form
`c̄*(m,T,λ) = c̄_∞ + Δ_T` with `c̄_∞ = −7` and `Δ_T = O((m+1)/T)`, so the detrending target
is flat at −7 and the finite-sample table records only the `Δ_T` departure, largest at the
shortest spans (up to `(m+1)/T ≈ 0.10`).

**Grid.** `T ∈ {30, 45, 50, 60, 80, 100, 150, 200, 300}` and `m ∈ {0,…,5}`, with a minimum
sample size per `m` (`30,30,30,45,60,80` for `m=0,…,5`). Break fractions lie in `[ε,1−ε]`,
trimming `ε = 0.15`, minimum spacing `0.15`. This enumerates to **418 break configurations
for `m ≥ 1`**; the `m = 0` cell has no break locations and is a single refined search per
`T` (9 cells).

**Tangency search.** For each configuration and each candidate `c̄` on the grid
`{−20,−19.5,…,−3}` (step 0.5): (i) `R_cv` replications under the null (`c=0`) of
`y_t = Z_t'θ + u_t`, `u_t = (1+c/T)u_{t−1} + ε_t` (with `u_0 = 0`, i.e. `u_1 = ε_1`) give
the 5% critical value of the point-optimal statistic; (ii) `R_pow` replications under the
alternative at `c = c̄` give its rejection rate; (iii) the crossing of power 0.50 is located
by linear interpolation, with delta-method standard error `se(c̄*) = se(power)/|slope|`,
`se(power) = √(0.25/R_pow)`. The criterion is power at the single point `c = c̄`.

**Replication counts.** The exploratory surface uses `R_cv = 10,000`, `R_pow = 5,000` (code
defaults). The production surface the paper prints uses **`R = 200,000` for `m ≥ 1` and a
refined search at `R = 40,000` for `m = 0`**, both long-run-variance arms (a simple
difference-based estimator and the autoregressive MAIC estimator of Perron & Ng). Precision
is inherited by averaging over the break-fraction configurations within each `m ≥ 1` cell,
and by the refined search for `m = 0`; the table reports a per-cell delta-method standard
error. Treat the reported `se_cbar` as a lower bound on seed-to-seed uncertainty for
`m ≥ 1`, not its sampling standard error.

**The MAIC short-T degeneracy.** Under the MAIC estimator at the shortest sample sizes the
power curve can fail to reach 0.50 within the grid; across the twenty-three configurations
where this binds the search reports the grid value closest to 0.50 and flags the cell,
rather than reporting a false precision. The estimator, not the grid, is the binding
constraint there.

**Long-run variance.** Both estimators are applied to the OLS-detrended series, not the raw
series: the level dummies induce jumps in the raw first difference at the break dates. The
MAIC lag order is selected on the OLS regression up to `k_max = 12`; the spectral density
`s²` is then formed on the GLS-detrended residuals. (v1.3.1 formed `s²` on the OLS residuals
and conflated the two Ng–Perron error variances; both are corrected in v2.0.0.)

**Validation: the detrending value is flat at −7.** Fitting `c̄*(m,T)` in `1/T` for
`T ≥ 100`, weighted by the inverse squared standard error of the mean across break
configurations, returns an intercept `c̄_∞ ≈ −7` for every `m` alike, with a slope `a(m)`
increasing in `m`.

**Replication counts of the other Monte Carlo exhibits** (production values):

| Object | Replications |
|---|---|
| Finite-sample critical-value table (`tab:cv`, full M-class 1/2.5/5/10%, `c̄ = −7`) | `3,300,000` (K = 20 blocks × 165,000, CRN within a block. `cv --full` regenerates the whole table via the same block design; the printed values reproduce within twice the simulation standard error plus the two-decimal rounding half-width (the criterion the manuscript states)) |
| Size-corrected power table (`tab:power`) | `320,000,000` null + `80,000,000` per power entry |
| AR(1) size / power robustness (`tab:ar1-sizepower`) | `R_eval = 10,000` (i.i.d. anchor capped at 2,000) |
| Recalibration dispersion (`stab:recal`) | `4,000` |
| PPP verdicts and half-lives (`tab:ppp`, `tab:hl`) | sieve bootstrap `B = 19,999` per currency; half-life inversion at `nsim = 10⁶` |
| Sieve bootstrap surface / empirical critical values | `R_cv = 10,000`, `R_pow = 5,000` / `R_cv = 20,000` |
| Empirical modulus `M_T(h)` (`stab:objects`, `stab:disp`) | `200,000` |
| Displacement ladder `L_T` (`stab:disp/L_T`) | `1,000,000` |

Seeds are a pure function of the work item: each `(P,T,m,λ)` maps to a deterministic
`config_seed` from `seed_base = 20240601`; replication `r` draws from
`default_rng((seed0 + r) mod (2⁶³ − 1))`; the null stream is offset by `2·10⁶`. Parallelism,
chunking, and resume are invisible in the output; the test suite verifies bit-identity
across scheduling regimes.

---

## Reproducibility & integrity checks

- **Deterministic** given the seeds and replication counts stated in each
  module header and in `MC_vs_BOOTSTRAP.md`.
- **Invariance check.** The exact θ-invariance that the paper establishes for
  the GLS-detrended statistic (it does not depend on the break magnitudes at
  all) doubles as the single most useful integrity check on this
  implementation: because the detrended series — and hence every statistic —
  is identically independent of the break magnitudes, an implementation can
  be verified by generating the same innovation `u_t` and recomputing the
  statistic at `θ = 0` and at a large `θ`: the two must coincide to machine
  precision. A Monte Carlo verification across `m ∈ {0, 1, 2}` and `T` up
  to 400 confirms the invariance to **ten significant digits**. Run via
  `python tests/run_all.py`.
- Numba kernels are validated against `arch.DFGLS` to 10 decimals;
  bit-identity across scheduling regimes (`--jobs`, chunking) is checked by
  `tests/test_njobs_invariance.py` / `tests/test_chunk_invariance.py`.
- The analytic invariances (θ-invariance, the Perron–Qu `k=0` convention,
  chunk- and `--jobs`-invariance) anchor the kernel; the package ships one
  kernel by design, so there is no second-implementation equivalence test.
- Provenance and the per-version changelog are in `CHANGELOG.md`.

### Half-life bootstrap schemes (§6.4 footnote)

`hl_median_unbiased` (the `bootstrap_empirical` door) reports two interval
schemes for the median-unbiased (Andrews–Chen) half-life estimator, both at
`B = 19,999`, with the half-life inversion at `nsim = 10⁶`:

- **Grid-*t* bootstrap** (Hansen 1999), valid uniformly in `α` including the
  unit-root neighborhood (Mikusheva 2007), inverted on a grid refined to step
  **0.005** near the unit root.
- **Recursive residual bootstrap** — the design-consistent baseline under the
  conditional-homoskedasticity gate (C4).
- **Heteroskedasticity-robust wild bootstrap** — Rademacher multipliers in the
  fixed-design construction of Cavaliere and Taylor (2008). This is the scheme
  reported in the half-life table.

### Cross-sectional dependence (§6.3 footnote)

The Pesaran (2004, 2015) CD statistic on the eight univariate Model LB
residuals rejects cross-sectional independence in both levels (+14.04) and
first differences (+8.65); the drop indicates contemporaneous dependence
rather than a common trend. This dependence is priced into the §6.3
rejection-count argument with a one-factor equicorrelated Gaussian
approximation at mean pairwise correlations 0.37–0.41, raising the probability
of zero rejections under `H1`-throughout from **0.006** (independence) to
**0.09–0.10** (`supplement` door).

### External reference implementation

Carrion-i-Silvestre, Kim & Perron's publicly archived GAUSS library,
[`gauss-carrion-library`](https://github.com/aptech/gauss-carrion-library),
dispatches the slope-only and level-and-slope specifications to the same
response-surface procedure — corroborating on the implementation side the
cancellation result behind Model LB's no-trend restriction.

---

## Citation

If you use this software, please cite **both** the article and this archive
(see `CITATION.cff`):

> Gonçalves Silva, R. (2026). *Level Breaks and Finite-Sample GLS Detrending:
> The Point-Optimal Unit Root Test and the Purchasing Power Parity Puzzle.*
> Replication package archived on Zenodo — concept DOI
> [10.5281/zenodo.21229773](https://doi.org/10.5281/zenodo.21229773).

### Preprint versions

The article itself (not just this replication package) is also available as
a preprint, ahead of and independent from journal review:

- **SSRN**: [papers.ssrn.com/sol3/papers.cfm?abstract_id=7138278](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7138278)
- **MPRA** (Munich Personal RePEc Archive), paper no. 130117:
  [mpra.ub.uni-muenchen.de/130117/](https://mpra.ub.uni-muenchen.de/130117/)

---

## License

Code is released under the **MIT License** (`LICENSE`). The bundled public data
are redistributed under their original terms (BIS terms of use; World Bank CPI
under CC-BY-4.0); see the data-source notes above.

**Harvard Dataverse deposit.** The data-only Dataverse deposit ("Replication
Data for: ...") takes a single license/terms selection for the whole dataset.
Both upstream sources require attribution on redistribution — World Bank data
is explicitly CC-BY-4.0, and the BIS terms of permitted use require citing the
BIS as the source — so **CC0** is not a correct choice for that deposit's
terms; set the Dataverse dataset's terms to **CC BY 4.0** to match.
