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
(numba Monte Carlo)          (traceable data)          (grayscale, no simulation)
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
| `generate_figures.py` | **The single figure script.** Reads the CSV artifacts and renders all five figures in grayscale. Run `python generate_figures.py` (or `--only fig2`). No simulation. |

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
python generate_figures.py                                  # -> all 5 figure PDFs (grayscale)

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
| **Fig 1** limiting null law | `replicate_section3_4.py --limiting-density` → `generate_figures.py --only fig1` | `limiting_density.pdf` |
| **Fig 2** the c̄(m,T) surface | `generate_figures.py --only fig2` | `cbar_surface.pdf` |
| **Fig 3** power curves | `size_power_cbar_comparison.py` → `generate_figures.py --only fig3` | `fig_power.pdf` |
| **Fig 4** real exchange rates | `generate_figures.py --only fig4` | `fig_rer_series.pdf` |
| **Fig 5** half-life forest | `hl_median_unbiased.py …` → `generate_figures.py --only fig5` | `fig_hl_forest.pdf` |

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

## Reproducibility

- **Deterministic** given the seeds and replication counts stated in each
  module header and in `MC_vs_BOOTSTRAP.md`.
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

---

## License

Code is released under the **MIT License** (`LICENSE`). The bundled public data
are redistributed under their original terms (BIS terms of use; World Bank CPI
under CC-BY-4.0); see the data-source notes above.
