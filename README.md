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

Every number in the paper is reproducible from the code plus two raw data
files, and every table and figure ships with a one-command self-check.

### Design in one line

```text
compute modules  ──write──▶  CSV artifacts  ──read──▶  generate_figures.py ──▶ PDFs
(numba Monte Carlo)          (traceable data)          (color, no simulation)
```

The numerical work lives in **one kernel** (`mlb_core.py`, with a pure-Python
fallback). Computation and plotting are strictly separated: compute modules
only write CSVs; `generate_figures.py` only reads them. A referee can
regenerate every figure from the shipped data **without running any
simulation**.

---

## Quick start

```bash
python -m pip install -r requirements.txt   # numpy, scipy, numba, matplotlib, joblib
python mlb_core.py --selftest               # sanity gates (kernel vs arch.DFGLS, invariance)
```

### Apply the test to your own series

`run_model_lb.py` is a self-contained CLI: point it at a CSV, name the series,
and give the known break dates.

```bash
python run_model_lb.py --csv myseries.csv --col rer --date-col year \
       --breaks 1985,1992 --sigma2 maic --calib cbar_surface.csv
```

| Flag | Meaning |
|---|---|
| `--csv` | CSV with the series, rows in time order (**required**) |
| `--col` | series column (default: last numeric column) |
| `--date-col` | optional date column, to match `--breaks` |
| `--breaks` | comma-separated exogenous break **dates** |
| `--sigma2` | long-run variance: `const` (difference-based) or `maic` |
| `--calib` | calibration surface (`cbar_surface.csv`) for the finite-sample `c̄` and critical value |
| `--alpha`, `--kmax`, `--nsim`, `--json` | level, MAIC lag ceiling, on-the-fly CV reps, JSON output |

---

## Repository layout

### Core kernel — imported, not run directly

| File | Role |
|---|---|
| `mlb_core.py` | **The kernel.** `@njit` GLS-detrending and the five M-class statistics, the calibration driver (`run_grid` / `aggregate`), the single-series `run_test`, and the validation gates. Aggregate cached checkpoints with `--post`. |
| `mlb_kernel.py` | Pure-Python fallback for `mlb_core` when numba is unavailable (bit-identical, ~45× slower). |

### Section drivers — one per paper section

| File | Reproduces |
|---|---|
| `replicate_section3_4.py` | **§3–4** — the calibration surface `c̄(m,T)` and the five critical values → `cbar_surface.csv`. `--limiting-density` writes the Figure-1 null-law data. Compute-only. |
| `replicate_section5.py` | **§5** — robustness (`ar1`, `oracle`, `trimming`) and the power comparison (`power`). |
| `replicate_section6.py` | **§6** — the PPP application: admissibility `sweep`, AR(p) `boot`, median-unbiased `hl`. |

### Experiment / compute modules

| File | Produces |
|---|---|
| `robustness.py` | §5 experiments: LRV-estimator choice, AR(1) size/power, recalibration, trimming, cross-validation. |
| `size_power_cbar_comparison.py` | §5 power table (`tab_power.tex`) and the Figure-3 curve data (`power_comparison.csv`). |
| `boot_ppp_cbar.py` | §6 applied calibration and the per-currency empirical block (Tables 7–8). See the nuisance note below. |
| `hl_median_unbiased.py` | §6 median-unbiased (Andrews–Chen) half-lives with grid-t / wild bootstrap CIs (Table 9). |
| `ppp_sweep_bis.py` | Exhaustive BIS-universe admissibility sweep (the funnel / gate table). |

> **Nuisance treatments in `boot_ppp_cbar.py` (v1.2.0).** The **surface** is
> calibrated under an AR(p) coefficient *family* — first PAC fixed at
> `--pac1` (default **0.27**, the median first-lag ADF coefficient of the eight
> residual series; provenance in `pac_diagnostic.py`), higher PACs redrawn per
> replication. Each currency's **empirical critical value** is *sieve-own* —
> its fitted ADF φ̂ at its BIC order, held fixed across replications — reported
> with empirical p-values. `--common-nuisance` reproduces the family design,
> `--bp` calibrates an exact break configuration, and `--seed-base` quantifies
> the Monte Carlo resolution of the critical value.

### Figures — one entry point

| File | Role |
|---|---|
| `generate_figures.py` | **The single figure script.** Reads the CSV artifacts and renders all five figures (`Figure_1.pdf`…`Figure_5.pdf`) with a colorblind-safe palette (Okabe–Ito, line-style redundancy retained). Run `python generate_figures.py` (or `--only fig2`). No simulation. |

### Tools & diagnostics

| File | Role |
|---|---|
| `run_model_lb.py` | Stand-alone: apply Model&nbsp;LB to any series (see Quick start). |
| `select_ar_order.py` | Regenerates `ppp_ar_diagnostic.csv` (BIC + general-to-specific AR order, 1973 window). |
| `pac_diagnostic.py` | Provenance of the nuisance persistence: per-currency PACF of Δ(LB residual) and first-lag ADF coefficients → `ppp_pac_diagnostic.csv` (the median, 0.27, is the canonical `--pac1`). |
| `pesaran_cd.py` | Pesaran (2004, 2015) cross-sectional-dependence test on the eight residual series (§6 footnote). |
| `dependence_bound.py` | §6.3 dependence-adjusted probability of zero rejections (one-factor Gaussian). |
| `dependence_count_pmf.py` | §6.3 full rejection-count PMF by exact enumeration. |
| `check_lam_spread.py` | §4.3 lookup-vs-λ-polynomial diagnostic (thin wrapper over `mlb_core.surface_diagnostics`). |

### Self-checks

| File | Role |
|---|---|
| `reconcile_tables.py` | Reconciles the manuscript's Tables 1/2/6 (always) and 4/5/9 (after their drivers run) against the artifacts. |
| `reconcile_boot.py` | Reconciles Tables 7–8 against `boot_out/`. |

### Data & shipped artifacts

| File | Kind |
|---|---|
| `ppp_panel.csv` | input — 8 real exchange rates, 1973–2024 (BIS × World Bank) |
| `exog_dates.csv` | input — exogenous currency-regime break dates and sources |
| `ppp_ar_diagnostic.csv` | derived — AR(p) order per currency |
| `cbar_surface.csv` | bridge — the full production surface (46 `(m,T)` cells, `m = 0…5`, `T = 30…300`) |
| `ppp_pac_diagnostic.csv` | derived — per-currency nuisance persistence (provenance of `pac1 = 0.27`) |
| `boot_out/calib/surface_ppp_boot.csv` | output — the applied calibration (Table 8; `pac1 = 0.27` family) |
| `boot_out/empirical/ppp_empirical.csv` | output — per-currency verdicts with p-values (Table 7; sieve-own CVs) |
| `boot_out/sensitivity/` | output — nuisance / seed / λ-exact sensitivities behind the Table-7 boundary note (see its `README.md`) |

Figure PDFs and the `hl_results*.csv` / `power_comparison.csv` /
`limiting_density.csv` files are **generated on demand** (git-ignored).

---

## Reproduce the paper

Each driver has `--full` (paper numbers) and `--quick` (smoke test). Sections
5–6 and the figures read the calibration surface from §3–4.

```bash
# (1) §3–4  calibration surface                             [hours at --full]
python replicate_section3_4.py --full --jobs -1             # -> cbar_surface.csv
python select_ar_order.py --start 1973 --kmax 10            # -> ppp_ar_diagnostic.csv

# (2) §5   robustness + power
python replicate_section5.py all --outdir section5_out
python size_power_cbar_comparison.py                        # -> tab_power.tex, power_comparison.csv

# (3) §6   PPP application
python pac_diagnostic.py                                    # -> ppp_pac_diagnostic.csv (pac1 provenance)
python boot_ppp_cbar.py --full --empirical                  # -> boot_out/ (Tables 7-8; sieve-own CVs + p-values)
python hl_median_unbiased.py --B 20000 --boot recursive --out hl_results.csv
python hl_median_unbiased.py --B 20000 --boot wild      --out hl_results_wild.csv

# (4) figures
python replicate_section3_4.py --limiting-density           # -> limiting_density.csv (Fig 1 data)
python generate_figures.py                                  # -> Figure_1.pdf … Figure_5.pdf (color)

# (5) validate
python reconcile_tables.py
python reconcile_boot.py --boot-out boot_out
```

> **Already have the calibration checkpoints?** Skip the multi-hour surface:
> `python mlb_core.py --post --outdir <checkpoints_dir>` re-aggregates them into
> `cbar_surface.csv` in seconds.

### Exhibit map

| Exhibit | Command | Artifact |
|---|---|---|
| **Table 1–2** c̄ surface & CVs | `replicate_section3_4.py --full` | `cbar_surface.csv` |
| **Table 3** size / power | `size_power_cbar_comparison.py` | `tab_power.tex` |
| **Table 4–5** AR(1) size/power, recalibration | `replicate_section5.py ar1 --recalibrate` | `robustness_out/` |
| **Table 6** break dates | *(input)* | `exog_dates.csv` |
| **Table 7–8** PPP verdicts & applied calibration | `boot_ppp_cbar.py --full --empirical` | `boot_out/` |
| **Table 9** half-lives | `hl_median_unbiased.py --boot wild` | `hl_results_wild.csv` |
| **Fig 1** limiting null law | `replicate_section3_4.py --limiting-density` → `generate_figures.py --only fig1` | `Figure_1.pdf` |
| **Fig 2** the c̄(m,T) surface | `generate_figures.py --only fig2` | `Figure_2.pdf` |
| **Fig 3** power curves | `size_power_cbar_comparison.py` → `generate_figures.py --only fig3` | `Figure_3.pdf` |
| **Fig 4** real exchange rates | `generate_figures.py --only fig4` | `Figure_4.pdf` |
| **Fig 5** half-life forest | `hl_median_unbiased.py …` → `generate_figures.py --only fig5` | `Figure_5.pdf` |

---

## Data sources

| Source | Series | Access |
|---|---|---|
| **BIS** | bilateral USD nominal exchange rates (`WS_XRU`, annual) | [data.bis.org](https://data.bis.org) (bulk `WS_XRU_csv_flat.zip`) |
| **World Bank** | Consumer Price Index (`FP.CPI.TOTL`, 2010 = 100) | World Bank Indicators API |
| **Central banks / G-5** | exogenous regime break dates (Plaza, ERM exits, floats) | primary sources, cited in `exog_dates.csv` |

`python replicate_section6.py sweep --fetch` rebuilds the panel from these
public sources.

---

## Monte Carlo design

The calibration surface `c̄(m,T)` (Table 1) and every critical value behind
it are located by the tangency procedure implemented in
`replicate_section3_4.py`. The paper (§4) states only the essential design
parameters and points here for the full procedural detail.

**Grid.** Sample sizes `T ∈ {30, 45, 50, 60, 80, 100, 150, 200, 300}` — the
upper end matching Carrion-i-Silvestre, Kim & Perron (2009); the lower end
covering the short cross-country panels the application motivates — and
break counts `m ∈ {0, …, 5}`, with a minimum sample size imposed per `m` so
every regime contains enough observations. Break fractions sit on a grid in
`[ε, 1−ε]` with trimming `ε = 0.15` and a minimum spacing of `0.15` between
breaks.

**Tangency search.** For each configuration and each candidate `c̄` on the
evaluation grid `{−20, −19.5, …, −3}` (step 0.5): (i) `R_cv = 10,000`
replications under the null (`c = 0`) of the local-to-unity DGP
`y_t = Z_t'θ + u_t`, `u_t = (1 + c/T) u_{t−1} + ε_t`, deliver the 5% critical
value of the point-optimal (`PT`) statistic, which rejects for small values;
(ii) `R_pow = 5,000` replications under the local alternative *at* `c = c̄`
deliver its rejection rate against that critical value; (iii) the
`(c̄, power)` pairs are collected across the grid and the crossing of power
0.50 is located by linear interpolation, reported with the delta-method
standard error `se(c̄*) = se(power)/|slope|` (slope taken in the bracketing
interval, `se(power) = √(0.5·0.5/R_pow)`). The criterion is power at the
*single point* `c = c̄`, not average power over a grid of alternatives —
averaging dilutes the objective with near-null alternatives at which no test
has power and drives the selected `c̄` to the grid boundary.

Where the curve never crosses 0.50 within the grid — an issue only for the
autoregressive-MAIC long-run-variance estimator at the shortest sample
sizes — the search falls back to the grid value closest to 0.50 and flags
the cell explicitly rather than reporting a false precision. In the
production surface this fallback binds in 23 cells, all under MAIC at
`T = 30` (1, 7, and 15 cells at `m = 0, 1, 2`): located values range from
−13.5 to −20 with only four at the grid edge, and the power attained never
exceeds 0.48 — the estimator, not the grid, is the binding constraint, so any
edge truncation understates the degeneracy the flags record.

**Seed averaging for the `m = 0` cells.** Each `m ≥ 1` cell of the surface
averages the interpolated tangency over the break-location configurations
within the cell, so seed-level simulation noise is averaged away as a
by-product. The `m = 0` cell has no break locations: a single tangency
search is one draw of the located crossing, whose seed-to-seed standard
deviation we measure at approximately 0.07 — about 2.5 times the
delta-method standard error of a single refined search, which captures only
the binomial noise of the power estimates around a fixed crossing, not the
seed-to-seed scatter of the crossing's location. Left unaveraged, this
scatter is large enough to displace individual `m = 0` cells by several
tenths and to produce spurious non-monotonicities in `T`. The production
design therefore repeats the entire `m = 0` calibration over `K = 20`
independent pseudo-random streams — the seed offsets are multiples of
`10^9`, exceeding the largest internal offset of a single calibration so
that no two searches share a seed integer, each integer expanded through
NumPy's `SeedSequence` entropy hashing before it initializes the generator,
so the resulting streams are statistically independent — and reports the
across-seed mean; the standard error attached to each `m = 0` cell is the
across-seed standard deviation divided by `√K` (≈0.04 at `T = 30`, shrinking
with `T`), which dominates and replaces the within-search delta-method
figure. `python mlb_core.py --post` reproduces the entire surface, including
the `m = 0` averaging and its aggregation, in a single deterministic run.

**Evaluation grid for the tangency.** The power curve from which `c̄*` is
read — by the interpolation rule and delta-method standard error above — is
traced on the grid `{−20, −19.5, …, −3}` in steps of 0.5. The grid serves
only to bracket the crossing of 0.50 densely; `c̄*` itself is the
interpolated crossing, not a grid point.

**Extension grid.** A targeted extension of the calibration grid to
`T ∈ {400, 600}` for `m ∈ {4, 5}` — 22 additional configurations under the
same seed policy and checkpointing — supports the intercept discussion in
§4.2 of the paper (the `m = 4, 5` intercepts moving from −7.33 and −7.38 to
−7.21 and −7.24). Its output is archived separately and does not enter the
baseline surface.

**Long-run variance.** Computed by two methods, both applied to the
OLS-detrended series `y_t − Z_t'θ̂_OLS` rather than the raw series: a simple
estimator that takes the sample variance of the first difference of the
detrended series (consistent under the i.i.d. null, and equivalent to the
autoregressive estimator with zero lags), and the autoregressive
spectral-density estimator of Perron & Ng (1998) with MAIC lag selection
(Ng & Perron 2001) up to `k_max = 12`. Detrending *before* estimating `s²`
is essential: the level dummies induce jumps in the raw first difference at
the break dates, so estimating `s²` from the undetrended series would
contaminate it with the deterministic component. The paper reports the
sensitivity of `c̄` and the critical values to the two methods. Full design
parameters, seeds, and the raw per-replication vectors are archived to
permit exact reproduction.

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
  precision. A nonzero discrepancy signals that the deterministic component
  is not being removed exactly — the single most consequential coding error
  possible here. We use this check rather than a sample variance of the
  innovation, which is `O_p(1/T)` even under correct code and would raise
  false alarms. A Monte Carlo verification across `m ∈ {0, 1, 2}` and `T` up
  to 400, with `θ` ranging from zero through the large-shift class
  `T^{1/2+η}`, confirms the invariance to **ten significant digits**; the
  residual discrepancy in the most extreme cell is at the level of
  floating-point cancellation, not of the invariance itself. Run via
  `python mlb_core.py --selftest`.
- Numba kernels are validated against `arch.DFGLS` to 10 decimals;
  cross-platform bit-parity (Windows / Linux) is checked.
- **No pandas** — all I/O uses the CSV standard library.
- Provenance and the per-version changelog are in `CHANGELOG.md`; the
  dependency / data flow is in `DEPENDENCY_GRAPH.md`.

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
