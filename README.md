# Calibrating Point-Optimal Unit Root Tests for Level Breaks in Small Samples: Evidence from Real Exchange Rates

**Replication package, version 3.0.0**

This package accompanies the article *Calibrating Point-Optimal Unit Root Tests for Level Breaks
in Small Samples: Evidence from Real Exchange Rates* (Ricardo Gonçalves Silva; submitted for
publication) and its online supplement. It holds the code, the data and the result files that
reproduce the tables, the figures and the numbers of the two documents, one script per section,
and a command that applies the article's test to your own series.

- Version 3.0.0 accompanies this article. Versions 1.x and 2.x accompany earlier drafts of the
  article and do not reproduce this paper.
- The concept DOI [10.5281/zenodo.21229773](https://doi.org/10.5281/zenodo.21229773) always
  resolves to the latest version. The version DOI of 3.0.0 is added at release.
- To cite the article and this archive, see [Citation](#9-license-citation-contact-and-changelog)
  and `CITATION.cff`.

## Checking the paper in ten minutes

```bash
python -m pip install -r requirements.txt     # Python 3.12
python replicate_all.py                       # every section, quick mode: about 10 minutes
```

The run closes with a table of the steps and the line `30 step(s), 0 failed; ... PASS` (exit
status 0), then one line on the full mode. Each step either computed its result again and found
it equal to the result file of `reference/` that the paper prints (byte for byte, or within the
tolerance the step prints on its `how:` line: 1e-9 relative for most files, 1e-6 for the
half-power points of Table 1, 1e-7 (1 + |b|) for those of Proposition 1), or drew the figure or
rendered the table from that file. Nothing in the quick mode is a long simulation; the long
simulations are read from `reference/`, and `--full` regenerates them from their seeds (hours;
[section 2](#2-reproduce-the-paper)). One section alone: `python replicate_sec_4.py`, ...,
`python replicate_sup_S5.py`.

Where each exhibit is (the files the run writes are under `output/`):

| Exhibit | Script (step) | What the run writes | Compared with, or drawn from |
|---|---|---|---|
| Table 1 | `replicate_sec_4.py` (`exact_surface`) | `output/replicate/steps/exact_surface/tab_cbar_mT_exact.tex` | `reference/calibration/cbar_surface_exact/cbar_surface_exact.csv`, recomputed exactly |
| Table 2 | `replicate_sec_4.py` (`tabcv_from_reference`) | `output/critical_values/critical_values_blocks/tabcv_full_percentiles.csv` | `reference/calibration/tabcv_full_percentiles.csv`, rebuilt from the 20 blocks of `reference/calibration/critical_values_blocks/critical_values_blocks.json` |
| Table 3 | `replicate_sec_6.py` (`app_data`, `ppp_tables`) | `output/bootstrap_empirical/tables/tab_ppp.tex` | `reference/application/main/ppp_statistics.csv` (the statistics, recomputed on the data) and `ppp_null.csv` (the simulated nulls) |
| Figure 1 | `replicate_sec_4.py` (`figure_size`) | `output/replicate/steps/figure_size/Figure_1.pdf` | `reference/figures/Figure_1_data.csv` |
| Figure 2 | `replicate_sec_4.py` (`figure_overdetrend`) | `output/replicate/steps/figure_overdetrend/Figure_2.pdf` | `reference/calibration/overdetrend_power.json` and `reference/figures/Figure_2_data.csv` |
| Figure 3 | `replicate_sec_6.py` (`figure_series`) | `output/replicate/steps/figure_series/Figure_3.pdf` | recomputed from `inputs/ppp_panel.csv`; `reference/figures/Figure_3_data.csv` |
| Table S1 | `replicate_sup_S4.py` (`table_s1_reference`) | `output/replicate/steps/table_s1/stab_ar1.tex` | `reference/serial_dependence/ar1_size_power/ar1_size_power.csv` |
| Table S2 | `replicate_sup_S4.py` (`ar1_recalibration`) | `output/robustness/ar1_recalibration/stab_recal.tex` | `reference/serial_dependence/ar1_recalibration/ar1_recalibration_envelope.csv`, recomputed exactly |
| Tables S3-S8 | `replicate_sup_S5.py` (`ppp_tables`) | `output/bootstrap_empirical/tables/stab_app_*.tex` | `reference/application/`, and `inputs/ppp_sweep/` for Table S3 |
| Table S9 | `replicate_sup_S5.py` (`variance_scope`) | `output/replicate/steps/variance_scope/stab_app_var.tex` | `reference/application/variance/ppp_variance_scope.csv`, recomputed (R = 200,000 per cell) |

Besides its tables and figures, a run writes next to each result a small run record
(`<name>.manifest.json`: the parameters, the seeds and the sha256 of the files it read and wrote);
nothing compares it. The numbers that the prose prints are read from the same result files; each section script lists
its exhibits and the files they come from when it starts, and [section 3](#3-monte-carlo-design-and-simulation-error)
gives the seeds and the Monte Carlo error of every simulated number.

**Contents.**
[1 Requirements](#1-requirements-and-installation) ·
[2 Reproduce the paper](#2-reproduce-the-paper) ·
[3 Monte Carlo design and simulation error](#3-monte-carlo-design-and-simulation-error) ·
[4 Data](#4-data) ·
[5 Robust level-shift tests](#5-the-robust-level-shift-tests) ·
[6 Your own series](#6-apply-the-test-to-your-own-series) ·
[7 Layout](#7-repository-layout) ·
[8 Limits](#8-what-the-package-does-not-contain) ·
[9 License, citation](#9-license-citation-contact-and-changelog)

---

## 1. Requirements and installation

Python 3.12. The results in `reference/` were computed with Python 3.12.4 on Windows.

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` lists seven packages at the exact versions that computed the results in
`reference/`:

| Package | Version | Used for |
|---|---|---|
| `numpy` | 2.3.5 | everything |
| `scipy` | 1.16.3 | exact laws (Imhof, the limit envelope), the application's tests |
| `numba` | 0.65.1 | the compiled kernel (`mlb/kernel_numba.py`); there is no pure-Python fallback |
| `llvmlite` | 0.47.0 | numba's compiler backend, the version numba 0.65.1 requires |
| `matplotlib` | 3.10.7 | Figures 1, 2 and 3 |
| `joblib` | 1.5.2 | the worker processes (`--jobs`) |
| `pandas` | 2.3.3 | the reader of the crisis database (`mlb/build_jst_crises.py`), `mlb/prose_numbers_grid.py` |

With these versions the comparisons reproduce `reference/` byte for byte wherever they can.
Other versions of numpy, and of the LAPACK and BLAS builds it ships, move the last digits of
least squares and eigensolvers; such differences stay within the tolerances that each comparison
states (1e-9 relative for most files, 1e-6 for the half-power points of Table 1, whose roots are
found to xtol = 1e-6, and 1e-7 (1 + |b|) for those of Proposition 1).

The computations of this version were also run on Linux (x86_64, Python 3.12.3, the requirements
above, 2 processors): every section passed in 8.1 min, and the largest relative difference from
the result files was 2.0e-12. Runs on macOS have not been tested.

- No `PYTHONPATH` is needed, and the current directory does not matter: every script resolves its
  paths from the package root.
- The first run compiles the numba kernels and caches them; the compilation costs about 20 to
  45 s of CPU once.
- The section scripts write under `output/` only; they never overwrite `reference/` or `inputs/`.
  `.gitignore` excludes `output/`. (The data-construction modules of [section 4](#4-data) can
  rewrite a shipped input when asked to: `mlb/build_ppp_panel.py` writes the file named by
  `--out`, whose default is `inputs/ppp_panel.csv`, and `mlb/build_jst_crises.py --write` and
  `mlb/sj_events.py --write` rewrite `inputs/jst_crises.csv` and `inputs/sj_dates.csv`.)

---

## 2. Reproduce the paper

One script per section of the paper, at the root of the package. Each has two modes.

**Quick mode** (the default: no option, or `--quick`). It takes minutes. It recomputes every
exact result of the section (Table 1, the limit envelope, Lemma 1, Proposition 1, Table S2...) and
the small simulations (the oracle cells, the MAIC cells, the half-lives, Table S9...) with the
recorded seeds, and compares each with `reference/`. It shows the long simulations from
`reference/`: it draws the figures and renders the tables from the result files, and rebuilds
Table 2 from its 20 blocks. It is enough to check the paper.

**Full mode** (`--full`). It regenerates every simulation of the section from the recorded seeds
into `output/` and compares each regenerated file with `reference/`. Some sections take hours.
Without `--yes` the script prints a warning, the plan and the estimated time of each step, runs
nothing and exits with status 2; `--full --yes` starts. The long steps keep resumable state
(`output/replicate/steps/`, and `output/critical_values/replicate_full/` for Table 2), so an
interrupted run resumes where it stopped; remove those folders to start again from nothing.

```bash
python replicate_sec_4.py                # Section 4, quick mode
python replicate_sec_4.py --full         # the warning and the plan; nothing runs
python replicate_sec_4.py --full --yes   # Section 4, full mode
python replicate_all.py                  # every section, each step once
python replicate_all.py --jobs 4         # 4 worker processes (default: min(8, processors))
python replicate_sec_6.py --help         # the options
```

| Script | Reproduces | Quick (est.) | Full (est.) |
|---|---|---|---|
| `replicate_sec_2_3.py` | Sections 2-3: the model, the half-power point and the theory | 81 s | 81 s |
| `replicate_sec_4.py` | Section 4: calibration, the detrending rule and critical values | 3.4 min | 2.9 h |
| `replicate_sec_5.py` | Section 5: serially dependent innovations | 15 s | 10.3 min |
| `replicate_sec_6.py` | Section 6: the purchasing power parity puzzle | 2.4 min | 39.5 min |
| `replicate_sup_S1.py` | Supplement S1: notation, conventions and the limit envelope | 77 s | 77 s |
| `replicate_sup_S2.py` | Supplement S2: invariance and the proof of Lemma 1 | 3 s | 3 s |
| `replicate_sup_S3.py` | Supplement S3: the proofs of Theorem 1 and Proposition 1 | 1.7 min | 1.7 min |
| `replicate_sup_S4.py` | Supplement S4: numerical material and Monte Carlo errors | 2.5 min | 2.6 h |
| `replicate_sup_S5.py` | Supplement S5: the application | 4.5 min | 43.6 min |
| `replicate_all.py` | every section, each step once | 9.5 min | 3.7 h |

The times are estimates in wall time at 8 worker processes, from the steps' times in
`mlb/steps.py`; each script prints them. A quick-mode time was measured on a Windows machine with
12 logical processors; a full-mode time is the CPU time of the regeneration divided by 8, or the
wall time measured on that machine where the two differed by more than 30%. The time grows
roughly in proportion as the number of worker processes falls. Sections share steps (Table 1's
surface, the stages of the application), so the nine scripts run one after the other take
17.4 min in the quick mode and 7.2 h in the full mode, against 9.5 min and 3.7 h for
`replicate_all.py`, which runs each step once.

**Measured runs.** On a clean copy of the package, on a Windows machine with 12 logical processors
(Python 3.12.4, the requirements above, 8 worker processes, nothing else running):
`python replicate_all.py` with an empty numba cache ended with exit status 0 after 586 s (9.8 min),
all 30 steps passed. Then, with the kernels compiled, the nine scripts one after the other: 67 s,
222 s, 16 s, 153 s, 65 s, 3 s, 95 s, 162 s and 291 s (Sections 2-3, 4, 5, 6, S1, S2, S3, S4, S5;
17.9 min in all), each with exit status 0. `python replicate_sec_5.py --full --yes` took 9.3 min
(estimated 10.3 min) and `python replicate_sup_S2.py --full --yes` 3 s; both passed. A single
step varies by about a fifth from one run to the next.

**What a run prints.** Each script first lists the exhibits of its section and what each mode
does with them. Then, for each step, its command, its result and its time:

- `recomputed: equal to reference/ (...)`: the step computed the result again and found it equal
  to the reference file, byte-identical or within 1e-9 relative (with the largest relative
  difference), or as the module's own comparison states (the `how:` line);
- `shown from reference/ (--full regenerates it); ...`: the step drew a figure or rendered a table
  from `reference/`, or rebuilt a table from other reference files, and checked what it drew;
- `DIFFERENT or FAILED: ...`: a difference, with the file and the first value that differs.

A step writes its tables (`.tex`) and figures (`.pdf`) under `output/` and prints their paths
(`wrote:`); its console goes to `output/replicate/logs/<step>.log`. At the end a table gives each
step's result and time, and the exit status is 0 only if every step passed (1 otherwise, 2 on a
usage error or a `--full` run without `--yes`).

### The steps of each section

Each script lists its steps in the tuples `QUICK` and `FULL`; `mlb/steps.py` defines each step
once (its command, the reference files it is compared with, how it decides, its time). The quick
mode of each section:

| Script | Recomputed in the quick mode | Shown from `reference/` in the quick mode (each such step names the `--full` step and script that regenerate it) |
|---|---|---|
| `replicate_sec_2_3.py` | the limit envelope; Lemma 1 | the strict decrease of the envelope and Table 2 rebuilt from its blocks (regenerated by `replicate_sec_4.py --full`) |
| `replicate_sec_4.py` | Table 1; Proposition 1; the (m+1) regularity; the surface summaries; the fixed index; 8 of the 12 oracle cells; the CKP (2009) surface; the agreement of the rule comparison's engine with the kernel | Figures 1 and 2; Table 2; the power comparison of Sections 4.2 and 4.4; the strict decrease of the envelope; the grid summaries of the prose |
| `replicate_sec_5.py` | the two MAIC cells that bound the printed range; Table S2 | Table S1; Table 2 rebuilt from its blocks (the critical values Section 5 uses); the grid summaries |
| `replicate_sec_6.py` | Figure 3 (the nine series, the break dates of design E, the regime means) and the facts of Section 6.1; the event dates; the seed registry; c-bar_1/2 of each design; the statistics on the data; the half-lives; the robust tests | the simulated nulls, the power, the count and the sets; Table 3 rendered |
| `replicate_sup_S1.py` | the limit envelope | - |
| `replicate_sup_S2.py` | Lemma 1 | - |
| `replicate_sup_S3.py` | the limit envelope; Proposition 1 | - |
| `replicate_sup_S4.py` | Table 1; the surface summaries; the CKP (2009) surface; Table S2; the fixed index | Table 2; Table S1; the grid summaries |
| `replicate_sup_S5.py` | as Section 6 without Figure 3 and the facts of Section 6.1, and Table S9 (R = 200,000 per cell, B = 19,999) | as Section 6; Tables S3-S8 rendered |

### Full mode: the long steps

One step takes more than 20 minutes:

| Step | What it regenerates | Estimated time |
|---|---|---|
| `full_overdetrend` | the power of the over-detrending shortcuts: 428,000,000 replications (the data of Figure 2 and of Section 4.4) | 2.3 h |

`full_overdetrend` was never run whole on this version: it was run at one twentieth of its size,
with the same seeds and structure, and its critical values, powers and losses agree with
`reference/calibration/overdetrend_power.json` within sampling error; its time is its CPU time
divided by 8. The steps between 10 and 20 minutes are the application's joint stage (12.6 min),
its power stage (10.5 min) and the 20 blocks of Table 2 (10.0 min).

**Table S1's agreement checks.** Before it writes Table S1, `mlb/ar1_size_power.py` checks that
its `const` arm reproduces Table 2's calibration in the nine cells T in {30, 60, 100},
m in {0, 1, 2} at rho = 0 and c = 0: the 5% critical value from its 200,000 draws within 2
standard errors of Table 2 (`cv_agreement`), and the size at Table 2's critical value within 2
standard errors of 0.05 (`size_agreement`). In the run that computed `reference/`, one cell of
the first check (T = 60, m = 0, z = +2.30) and two of the second ((60, 0), z = -2.34; (60, 1),
z = -2.22) are beyond 2 standard errors. These are 18 two-sided tests with no allowance for
multiplicity; the joint tests over the nine cells do not reject at 5% (sums of z^2 12.74 and
13.51, p = 0.17 and 0.14), and the Bonferroni threshold for nine tests at 5% is 2.77. The table
is computed anyway (the option `--proceed-after-agreement-failure`). In the full mode the step
`full_ar1_size_power` prints these results as information (the lines starting
`[ar1_size_power]`); the step is decided by the comparison of the regenerated files with
`reference/`.

---

## 3. Monte Carlo design and simulation error

### What is exact and what is simulated

- Table 1 (the half-power points of the test) is computed with no simulation: the Gaussian
  envelope statistic is a quadratic form in the innovations, and its distribution is inverted
  numerically (Imhof). `reference/calibration/cbar_surface_exact/cbar_surface_exact.csv` has one
  row per configuration: 404 configurations of break fractions and sample sizes, of which 396
  have one to five breaks and 8 have none. Its entries carry no simulation error. The limit
  envelope (`mlb/limit_envelope.py`), Lemma 1 on 3,004 designs (`mlb/lemma1_bounds.py`, exact
  rational arithmetic) and Table S2 (`mlb/ar1_recalibration.py`) are exact as well.
- The critical values (Table 2), Figure 1, the power comparison of Figure 2 and Section 4.4, the
  rule comparison of Section 4.2, the oracle ladder of Section 4.3, Table S1 and the application
  are simulated.

### Grid

Sample sizes T in {45, 50, 60, 80, 100, 150, 200, 300} and numbers of breaks m in {0, ..., 5}; the
smallest T is 45 for m up to 3, 60 for m = 4 and 80 for m = 5. For m = 1, ..., 5 the break
fractions are points of an equispaced lattice on [0.15, 0.85] with 7, 7, 8, 11 and 14 nodes, each
node rounded to three decimals, with consecutive breaks at least 1, 2, 2, 3 and 3 lattice steps
apart. This gives 7, 15, 20, 5 and 6 configurations of break fractions, and minimum spacings of
0.23 (m = 2), 0.20, 0.21 and 0.16 (m = 5) (Supplement S4.1). The break dates are
tau_j = floor(lambda_j T), clamped to [1, T - 1] and computed exactly: in binary floating point
lambda_j T can fall just below an integer that the exact product equals (0.57 x 100 evaluates to
56.99999999999999), so `mlb/break_dates.py` rounds the product to three decimals before the
floor, which is exact for every fraction the package uses. The detrending parameter is
c-bar = -7 in every configuration; the statistic is MZ_t unless a table says otherwise, with the
difference-based long-run variance (`const`) or the autoregressive one (`maic`, Perron and Ng,
with the lag chosen by the MAIC up to k_max = 12). The GLS detrending does not depend on the
break magnitudes (Supplement S2.1); the kernel solves its normal equations with a ridge of 1e-10
on the diagonal, so this invariance holds to about nine significant digits.

### Replications, blocks and seeds

Replication r of a work item draws from `numpy.random.default_rng(seed0 + r)`. The seed `seed0` is
a pure function of the work item, so the number of worker processes, the chunking and a resumed
run do not change any number.

| Object | Replications | Seeds | Result file |
|---|---|---|---|
| Table 2 (`tab:cv`), 46 cells computed, 43 printed | K = 20 blocks of 165,000 = 3,300,000 per cell, one stream per block shared by every cell (common random numbers within a block) | block b, replication r: 22,240,601 + b x 10^9 + r | `reference/calibration/critical_values_blocks/critical_values_blocks.json` |
| Figure 1 (`fig:size`) | 200,000 per cell, 18 cells | 20,260,720 + k x 10^6 for the cell index k | `reference/figures/Figure_1_data.csv` |
| Figure 2 (`fig:overdetrend`) and Section 4.4 | 320,000,000 under the null; 80,000,000 at c = -10; 2,000,000 at each of the other 14 points; 428,000,000 in all | 520,260,724 at stride 10^9 | `reference/calibration/overdetrend_power.json` |
| Section 4.2, rule comparison | 200,000 on each of 78 configurations; alternatives c in {-5, -10, -15, -20} | 20,260,925 + 1,000,003 T + (crc32 of the configuration's id mod 10^6); the alternatives at stride 10^9 | `reference/calibration/rule_comparison/rule_levels.csv` |
| Section 4.3, oracle ladder at T = 300, 1,200 and 9,600 | 200,000 per cell, blocks of 2,000 | 54,020,802 + 1000 T + m | `reference/calibration/oracle_sigma2_ladder/oracle_sigma2_ladder.csv` |
| Table S1 (`stab:ar1`) | 200,000 per entry; the `maic` critical value from 1,000,000 separate i.i.d. null replications | 20,260,926; cells at stride 10^8 | `reference/serial_dependence/ar1_size_power/ar1_size_power.csv` |
| Application, columns II and III of Table 3 | 200,000 null replications per configuration, 33 configurations | the registry below | `reference/application/main/ppp_null.csv` |
| Application, half-lives (Tables S7, S8) | B = 19,999 per node of the grid alpha = k/200, per bootstrap scheme | the registry below | `reference/application/main/hl_cells.csv` |
| Admission condition C6 (Table S3) | 100,000 null draws per currency to calibrate and 100,000 fresh draws to validate | the registry below | `reference/application/admission/c6_critical_values.csv` |

**The seed registry of the application.** `mlb/app_seeds.py` declares 1,793 work items, written to
`reference/application/main/app_seeds.csv`. The seed of item w is 10^16 + 10^9 x (its index), and
no item draws more than 10^9 replications, so the intervals of two items are disjoint; the
registry asserts this, and that every seed lies above the seeds of Table 2, which the application
reads. An item added at the end leaves the seeds of every item before it unchanged. Where two arms
of the application are compared (the sieve null against the i.i.d. null, the start u_0 = 0
against the stationary start), they share their draws; the rules are in `mlb/app_seeds.py`.

### Simulation-error accounting

The article puts the Monte Carlo errors in Supplement S4 (Numerical material and Monte Carlo
errors) and in the notes of its tables. The rules, as the supplement states them, and the file
that holds each:

- **Rejection frequencies against a critical value fixed in advance** carry the binomial standard
  error sqrt(p (1 - p) / R). Examples: the `size_I_se` column of
  `reference/application/main/ppp_null.csv`, and the `const` arm of Table S1, whose critical
  values come from Table 2.
- **Critical values, as quantiles,** carry the standard deviation of the estimates across
  independent blocks of the simulation: 20 blocks for Table 2 (the 20 block quantiles are in
  `critical_values_blocks.json`), and the larger of the 8- and 20-block values for the critical
  values of the application (`cv_II_se8`, `cv_II_se20` and their larger, `cv_II_se`, in
  `ppp_null.csv`).
- **The quantile at T = 9,600** carries the asymptotic standard error sqrt(p (1 - p) / R) / f,
  with f the density estimated from the quantile spacing (columns `q_lo`, `q_hi` and `se_oracle`
  of `reference/calibration/oracle_sigma2_ladder/oracle_sigma2_ladder.csv`).
- **A power confronted with a critical value estimated from the same simulation:** the
  replications are split into B in {8, 20} independent blocks, the critical value and the power
  are recomputed within each block, and the larger block standard error is reported. In Table S1
  the reported `se` is the larger of that and the binomial standard error;
  `reference/serial_dependence/ar1_size_power/ar1_size_power.csv` keeps the block standard error
  in `se_block` and names the one reported in `se_kind`: of its 48 entries, 24 use the binomial
  error against a fixed critical value, 19 the block error, and 5 the binomial error because it
  exceeds the block error. The largest `se` of the table is 0.0022.
- **The powers of Section 4.4** add to the binomial variance the variance of the critical value
  carried through the density of the statistic (`se_cv` and `f_null_at_cv` in
  `reference/calibration/overdetrend_power.json`).

Each result file ships whole, so some rows and columns of `reference/` are read by no table,
figure or printed number (the band's impulse-response and unscaled arms, the i.i.d. arms of the
null and power stages, the joint law of the N9 sample and the kappa = 0.5 rows, column III of
`reference/application/crisis_design/sj_null.csv`, and the columns of the application's null and
power results that no table reads); their seeds stay in the registry, so that the seeds of every
other item are the ones the results were computed with.

---

## 4. Data

No raw source file is redistributed. The files under `inputs/` are derived from public sources or
compiled from published records; `inputs/DATA_TERMS.md` gives their sources, the download dates,
their terms and the references to cite.

| Source | What is used | Raw file and date | Derived input |
|---|---|---|---|
| Bank for International Settlements, bilateral US dollar exchange rates (`WS_XRU`, annual, period average) | the nominal exchange rates | the bulk export `WS_XRU_csv_flat.zip` from `data.bis.org`, a data file dated 2026-08-12 | `inputs/ppp_panel.csv`; the admission screen in `inputs/ppp_sweep/` |
| World Bank, World Development Indicators, `FP.CPI.TOTL` (consumer prices, 2010 = 100) | the consumer price indexes | the indicators API payload, last updated 2026-07-13 | the same |
| OECD list of members and United Nations Treaty Collection (UNTS vol. 888, p. 179) | the members by 31 December 1973 (admission condition C0) and the date the Convention entered into force for each | public records, cited in each row | `inputs/oecd_1973.csv` |
| Central bank and government publications, G-5 and G-7 communiqués | the dates of the regime events of rule E | cited in each row | `inputs/regime_events.csv` |
| Jordà, Schularick and Taylor, Macrohistory Database, release 6 | only the first year of each systemic banking crisis (`crisisJST`) of the issuers of eight currencies | `JSTdatasetR6.dta`, downloaded 2026-10-02; not redistributed | `inputs/jst_crises.csv` (12 rows) |

Later downloads of the BIS and World Bank files may differ from the ones used.

**Terms.** BIS statistics: no restriction on use provided the BIS is cited as the source. World
Bank WDI: Creative Commons Attribution 4.0; the series is used with attribution and has been
transformed (real exchange rates, annual alignment, common-sample trimming). The JST database is
distributed under CC BY-NC-SA 4.0: the dataset is not redistributed, and the crisis years are
reproduced with attribution to the database and, to the extent that its license applies to them,
under its terms (non-commercial use, share-alike), not under the license of the code. Please cite
Jordà, Schularick and Taylor (2017), *Macrofinancial history and the new business cycle facts*,
NBER Macroeconomics Annual 31, 213-263. The critical-value tables of the robust level-shift tests
are the authors' ([section 5](#5-the-robust-level-shift-tests)).

**Building the inputs from the raw downloads (data construction).** Neither mode builds the
inputs: both read `inputs/` as shipped. Three modules of `mlb/` build them from the raw files, for
a reader who wants to redo Sections S5.1 and S5.2 from the sources. Download the raw files into a
folder `<dir>` (or let `--fetch` do it for the two files of the panel) and run:

```bash
python mlb/build_ppp_panel.py --cache <dir> --currencies AUD,CAD,CHF,DKK,GBP,JPY,NOK,NZD,SEK --fetch --out my_ppp_panel.csv
python mlb/admissible_universe.py --cache <dir> --check
python mlb/build_jst_crises.py --cache <dir> --check
```

The first reads (or, with `--fetch`, downloads into `<dir>`) the BIS bulk file
`WS_XRU_csv_flat.zip` and the World Bank payload `wb_cpi.json`, and computes
q = ln X + ln P_US - ln P; the panel is written to the file named by `--out`, so the shipped one
is not overwritten. The second applies the admission conditions C0 to C6 to the BIS annual dollar
series, writes the three files of `inputs/ppp_sweep/` to its `--out-dir` (by default
`output/replicate/steps/admissible_universe`; it refuses `inputs/ppp_sweep/`) and, with
`--check`, reports which of them differ from the shipped ones. The third needs
`JSTdatasetR6.dta` from the database's page in `<dir>`; it regenerates the crisis years and
compares them with the shipped ones. Because later downloads may differ, a rebuilt panel can
differ from the shipped one.

---

## 5. The robust level-shift tests

Supplement S5.6 compares the Model LB test with two robust procedures for multiple level shifts of
unknown date:

- Harvey, D.I., Leybourne, S.J. and Taylor, A.M.R. (2010). Robust methods for detecting multiple
  level breaks in autocorrelated time series. *Journal of Econometrics* 157, 342-358.
- Carrion-i-Silvestre, J.L. and Gadea, M.D. (2023). Testing for multiple level shifts with an
  integrated or stationary noise component. *Journal of Applied Econometrics* 38(6), 801-819.

`mlb/level_shifts.py` is a Python port of the authors' MATLAB functions, one function per file;
it is this package's own code, released under the license of the package, and it was checked
against values that the authors' MATLAB functions produced on the series of
`inputs/ppp_panel.csv`. `mlb/robust_level_shift.py` runs it for Section S5.6; its results are in
`reference/application/robust_tests/`. The MATLAB code is not shipped.

The eight tables of critical values and kappa constants in `inputs/level_shift_cv/` were computed by
Carrion-i-Silvestre and Gadea and are copied byte for byte from their replication files (the
archive `jae_empirical_web.zip`, distributed from M. Dolores Gadea's web page, folder
`functions_empirical`). The critical values of Harvey, Leybourne and Taylor (2010) for the windows
0.10 and 0.15, and the response-surface coefficients of the sequential test for the trimming 0.15,
are written in the source of the port, as they are in the authors' MATLAB code.
`inputs/level_shift_cv/README.md` gives the provenance and sha256 of each table. No license
accompanies the authors' files; their readme asks users to cite the authors and their web page.
The tables are distributed here with that attribution; they are the authors' work and are not
covered by the license of this package's code.

**If you use these tables or the tests, please cite the two papers above.**

---

## 6. Apply the test to your own series

`model_lb` runs the test of the article on any series, at break dates you supply: the three steps
of Section 5 and the columns I, II and III of Table 3. The break dates are inputs, known and
exogenous; they are not estimated.

1. MZ_t with GLS detrending on (1, DU_1, ..., DU_m) at c-bar = -7, with s^2_AR and the lag chosen
   by the MAIC (k_max = 12). Column I compares it with -1.98.
2. The autoregressive sieve of the null: the ADF regression of the first differences of the OLS
   residual of y on (1, DU) on a constant, the lagged residual and p lagged differences, with p
   chosen by the BIC (k_max = 10) on a common sample.
3. Column II: the null simulated at the series' own sample length and break dates under the
   estimated sieve, 200,000 replications by default; the 5% critical value, the p-value (the share
   of null draws at or below the statistic) and the true size of -1.98, each with its Monte Carlo
   standard error. Column III: the same with the detrending at the exact half-power point
   c-bar_1/2 of the configuration (Section 4).

### Install

Run once, in the repository root:

```bash
python -m pip install "pip>=21.3" "setuptools>=64"
python -m pip install -r requirements.txt
python -m pip install -e .
```

The last command makes `python -m model_lb` work from any directory. An editable install of a
project described by `pyproject.toml` needs pip 21.3 or later and setuptools 64 or later; the first
command upgrades older versions and leaves newer ones as they are. The editable install adds no
further dependency (the list is `requirements.txt`) and copies no file: it points the environment at
the repository, where the package reads its inputs, and pip leaves a
`model_lb_replication.egg-info/` folder there (ignored by `.gitignore`, like `build/`). It makes the
packages `model_lb` and `mlb` importable in that environment, so a virtual environment of its own
is advisable.

Plain `pip install .` copies the two packages into the environment's `site-packages` and leaves a
`build/` folder in the repository (also ignored). `python -m model_lb` and the functions below then
work from the copy, but the copy does not follow later edits of the repository, and the section
scripts, which look for `inputs/` and `reference/` next to `mlb/`, must be run from the repository.
To switch: `python -m pip uninstall model-lb-replication`, then `python -m pip install -e .`.
Without any install, `python -m model_lb` works from the repository root.

### Example

The pound's real exchange rate of the article, read from the shipped panel, with its two breaks of
design E (1986 and 1993):

```bash
python -m model_lb --csv inputs/ppp_panel.csv --where currency=GBP --date-col year --col q --breaks 1986,1993
```

The output of a run of this command (default seeds, one worker):

```text
Model LB unit-root test: MZ_t, GLS detrending on (1, DU_1, ..., DU_m) at c-bar = -7
  series     : column 'q' of inputs/ppp_panel.csv where currency=GBP; T = 52, 1973 to 2024
  breaks     : 1986, 1993 (first date of each new regime); break_pos 13, 20; m = 2
  statistic  : MZ_t = -1.1012; MAIC lag k = 6 (k_max 12)
  sieve      : BIC order p_hat = 1 (k_max 10); phi_hat = 0.3849 (se 0.1388)
               sum phi_hat = 0.3849; largest |root| 0.3849: stationary
  (I)   asymptotic critical value -1.98: do not reject at 5%
        true size of -1.98 under the sieve null 0.0400 (se 0.0004)
  (II)  critical value -1.8999 (se 0.0045); p-value 0.3588 (se 0.0011): do not reject at 5%
  (III) c-bar_1/2 = -9.3016: MZ_t = -1.1144; critical value -2.0139 (se 0.0040);
        p-value 0.4673 (se 0.0011): do not reject at 5%
  null draws : R = 200,000 per column; seed (II) 20260701, seed (III) 1020260701; degenerate draws 0 and 0
```

Two progress lines go to stderr. With the seeds that the application registered for the pound, the
same command reproduces its row of Table 3 to the last digit
(`reference/application/main/ppp_null.csv`, row GBP; Table 3 prints -1.10, -1.90, 0.36, 4.1% and
0.47):

```bash
python -m model_lb --csv inputs/ppp_panel.csv --where currency=GBP --date-col year --col q --breaks 1986,1993 --seed 10000051000000000 --seed-iii 10000084000000000
```

Its columns I to III:

```text
  (I)   asymptotic critical value -1.98: do not reject at 5%
        true size of -1.98 under the sieve null 0.0405 (se 0.0004)
  (II)  critical value -1.9040 (se 0.0042); p-value 0.3600 (se 0.0011): do not reject at 5%
  (III) c-bar_1/2 = -9.3016: MZ_t = -1.1144; critical value -2.0079 (se 0.0035);
        p-value 0.4675 (se 0.0011): do not reject at 5%
  null draws : R = 200,000 per column; seed (II) 10000051000000000, seed (III) 10000084000000000; degenerate draws 0 and 0
```

The two runs differ by Monte Carlo error only: -1.8999 against -1.9040 is 0.7 combined standard
errors, and 0.3588 against 0.3600 is 0.8.

### Options

| Option | Meaning |
|---|---|
| `--csv FILE` | CSV with the series, one row per period, in time order (required) |
| `--col NAME` | the series column (default: the last numeric column) |
| `--date-col NAME` | the date column; `--breaks` gives values of it |
| `--where COL=VALUE` | keep the rows whose `COL` equals `VALUE` (repeatable): one series of a long panel |
| `--breaks LIST` | break dates: the first period of each new regime (without `--date-col`: their 1-based observation numbers); none = no break |
| `--break-pos LIST` | instead, the number of observations before each break |
| `--R N` | replications per simulated null (default 200,000) |
| `--seed S`, `--seed-iii S` | replication r of column II draws from `numpy.random.default_rng(S + r)` (default 20260701); column III from its own seed (default S + 10^9) |
| `--no-iii` | skip column III and the half-power point |
| `--jobs N` | worker processes; the draws, hence every number, do not depend on N |
| `--json FILE` | every number at full precision, as JSON |

c-bar = -7, the MAIC (k_max = 12) and the BIC sieve (k_max = 10) are the article's and are not
options. The critical values are at 5%; the p-value serves any level. A series whose estimated sieve
is not stationary is not simulated: its statistic is printed and column II says so, as in the
application.

### What the command checks

- **One series, in time order.** The dates of `--date-col` must be unique (a file that holds several
  series needs `--where`; the message suggests a column) and, when they are numbers such as years,
  quarters (`1990Q1`), months (`1990M01`, `1990-01`) or ISO dates, increasing.
- **Degenerate series.** A constant series, or one constant within every regime (it moves only at
  the breaks), is refused: the statistic is not defined.
- **Scale and the band [1e-3, 1e3].** MZ_t, the p-values and the decisions do not depend on the
  units of the series, but the numerical kernel has absolute floors (s^2_AR is at least 1e-10, the
  lag regressions carry a ridge of 1e-12) and the null is simulated at unit scale. Let s be the
  root mean square of the first differences within regimes. Multiplying 309 series (the nine
  currencies at their design-E breaks and 300 simulated random walks and AR(1) series) by powers of
  ten, the floors moved MZ_t by at most about 1.2e-12 / s^2 (1.2e-6 at s = 1e-3), with no lag or
  order changed; below s of about 1e-5 the statistic is lost (the pound times 1e-4, s = 7.5e-6,
  gives MZ_t = -2.73 instead of -1.10), and above s = 1 nothing moved (to s = 1e12). The command
  therefore uses the series as given when s lies in [1e-3, 1e3]; otherwise it divides the series by
  s before computing and prints a `scale` line (the rescaling report), and the pound times 1e-4
  then gives the pound's numbers.
- **Sample length.** T >= 24 is computable; below T = 45, where the grids of the article start, a
  warning is printed. On 500 Gaussian random walks per cell (m = 0, 1, 2): at T = 24 the BIC order
  is at its bound 10 in 51% to 56% of the series and the sieve is non-stationary in 69% to 73%
  (column II is then not simulated); at T = 30, in 1% to 6% and 7% to 16%; at T = 45 and above, in
  none and at most 1.2%. A warning is also printed whenever the BIC order reaches k_max = 10 or the
  MAIC lag reaches 12.
- **Half-power point.** The command finds c-bar_1/2 in the bracket (-25, -1), then (-40, -1), as the
  application does; the article defines it on K = [-20, -3]. A root outside K is reported with a
  warning. When no root is found (some designs whose regimes have very few observations make the
  exact computation fail), column III is reported as not computed and columns I and II are kept.

### Time

The first run compiles the numerical kernels (numba; cached afterwards). Measured on the example
above, on a Windows machine with 12 logical processors and one worker:
44.4 s with an empty numba cache (cold) and 36.5 s with the kernels already compiled (warm), with
the same output.
In two sessions on a machine that was shared with other jobs (17 to 34 other Python processes) the
same example took 168 to 191 s cold and 123 to 145 s warm on one worker, and 50 to 61 s warm with
`--jobs 4`; with `--R 20000 --no-iii`, 35 to 82 s cold and 10 to 18 s warm.

### From Python

`model_lb` exposes the same steps as functions (`model_lb/api.py` documents each): `statistic`,
`sieve`, `null_distribution`, `summarize_null`, `column_ii`, `half_power_point`, `column_iii`,
`model_lb_test`, `prepare` (the checks and the scale above) and `design_positions`, and, for the supplement's
application, `half_life` (the grid-t median-unbiased half-life, recursive or wild bootstrap),
`holm` (Holm's adjustment across series) and `variance_profile_size` (the sizes of columns I and II
under a variance that changes at the regime dates, and the wild sieve bootstrap). Every result
carries the keys `scale` and `warnings`.

```python
import csv

import model_lb as mlb

rows = csv.DictReader(open("inputs/ppp_panel.csv", encoding="utf-8"))
q = [float(r["q"]) for r in rows if r["currency"] == "GBP"]
bp = (13, 20)  # 1986 and 1993: 13 and 20 observations before the breaks

hl = mlb.half_life(q, bp, scheme="rec")  # B = 19,999 per grid node
print("alpha_MU", round(hl["alpha_mu"], 3), "half-life", round(hl["hl_mu"], 1),
      "95% set", round(hl["hl_lo"], 1), "to", hl["hl_hi"])

v = mlb.variance_profile_size(q, bp, float_breaks=(20,), first_period=1973,
                              jobs=4)
print("variance dates", v["variance_dates"], "ratio", round(v["variance_ratio"], 2),
      "size (I)", v["profile"]["size_I"], "size (II)", v["profile"]["size_IIown"])

p = {"AUD": 0.017165, "CAD": 0.37064, "DKK": 0.22123, "GBP": 0.36005,
     "JPY": 0.7712, "NOK": 0.555415, "NZD": 0.540355, "SEK": 0.40155}
print("Holm", {c: round(adj, 4) for c, adj in mlb.holm(p).items()})
```

Output of this code:

```text
alpha_MU 0.865 half-life 4.8 95% set 1.7 to inf
variance dates (1993,) ratio 1.76 size (I) 0.04263 size (II) 0.053095
Holm {'AUD': 0.1373, 'DKK': 1.0, 'GBP': 1.0, 'CAD': 1.0, 'SEK': 1.0, 'NZD': 1.0, 'NOK': 1.0, 'JPY': 1.0}
```

The p-values are column II of the eight admitted currencies
(`reference/application/main/ppp_null.csv`); Holm's 0.1373 for the Australian dollar is the 0.14
of Section 6. With the registered seeds, `half_life` and `variance_profile_size` return the
application's half-lives (`reference/application/main/hl_cells.csv`) and variance scope
(`reference/application/variance/ppp_variance_scope.csv`) to the last digit.

### Limitations of the command

- A long panel read without `--date-col` is accepted silently: nothing tells the command that the
  file holds several series. Use `--date-col` and `--where`.
- A large constant offset in the series is not handled specially: the scale rule looks at the
  first differences within regimes only.

---

## 7. Repository layout

```text
replicate_sec_2_3.py ... replicate_sec_6.py   one script per section of the article (quick and full modes)
replicate_sup_S1.py ... replicate_sup_S5.py   one script per section of the supplement
replicate_all.py               every section, each step once
requirements.txt               the exact versions of the dependencies
pyproject.toml                 makes `python -m model_lb` importable from any directory (pip install -e .)
CITATION.cff, .zenodo.json     how to cite; the archive's metadata
CHANGELOG.md, LICENSE          the changes of each version; the MIT license of the code
.gitattributes, .gitignore     LF line endings; output/ is not tracked
README.md                      this file
mlb/                           48 files: the code
  kernel_numba.py, envelope_exact.py, envelope_limit.py, break_dates.py, level_shifts.py,
  mzt_exact_linear.py, ppp_sieve.py, ppp_ar_null.py, pesaran_cd.py, app_seeds.py,
  regime_events.py, sj_events.py, _figstyle.py, paths.py
                               the kernel and the shared functions
  28 modules                   one per computation (Table 1, Table 2, the figures, ... Table S9)
  build_ppp_panel.py, admissible_universe.py, build_jst_crises.py
                               data construction from the raw downloads (section 4)
  steps.py, runner.py, __init__.py
                               the steps of the section scripts and the runner
model_lb/                      3 files: `python -m model_lb` and model_lb/api.py (your own series)
inputs/                        the data: the panel (ppp_panel.csv), the regime events, the crisis
                               years and dates, the OECD members, DATA_TERMS.md
inputs/ppp_sweep/              3 files: the admission screen over the BIS annual series (C0-C6)
inputs/level_shift_cv/         9 files: the eight critical-value tables of the robust tests and
                               their README
reference/                     62 files: the results the paper prints, as computed for this
                               version; every comparison is made with them
  theory/                      the limit envelope and Lemma 1
  calibration/                 Tables 1-2, the data of Figure 2 and the Section 4 numerics
  serial_dependence/           Tables S1-S2
  application/                 the application: main/, crisis_design/, variance/,
                               robust_tests/, tables/, admission/
  figures/                     the data of Figures 1, 2 and 3
```

---

## 8. What the package does not contain

- **Heavy simulations in the quick mode.** The quick mode does not rerun the long simulations
  (Table 2, the power comparison, the application's null, power and joint stages); it shows them
  from `reference/`. Their regeneration is the full mode.
- **A full-size run of the over-detrending power.** The data of Figure 2 (`full_overdetrend`,
  estimated 2.3 h, from its CPU time) was regenerated only at one twentieth of its size. The other
  full-mode times are estimates or measurements on one machine and scale with the number of
  processors.
- **The raw data** of the BIS and the World Bank, and the JST database (section 4). Neither mode
  rebuilds the shipped inputs from raw data; the data-construction modules do, from your
  downloads.
- **The MATLAB code** of the robust level-shift tests, of which `mlb/level_shifts.py` is a port
  (section 5).
- **Earlier drafts' exhibits.** Figures and tables that this article does not print are not in the
  package; `CHANGELOG.md` lists what version 3.0.0 withdrew from version 2.
- **Other platforms.** Runs on macOS have not been tested for this release.

---

## 9. License, citation, contact and changelog

**License.** The code is released under the MIT license (`LICENSE`). The data keep their own terms
([section 4](#4-data), `inputs/DATA_TERMS.md`); the critical-value tables of
`inputs/level_shift_cv/` are the authors' work and are not covered by the license of the code.

**Citation.** If you use this software, please cite both the article and this archive (see
`CITATION.cff` and `.zenodo.json`):

> Gonçalves Silva, R. (2026). *Calibrating Point-Optimal Unit Root Tests for Level Breaks in Small
> Samples: Evidence from Real Exchange Rates.* Submitted for publication. Replication package,
> version 3.0.0, archived on Zenodo; concept DOI
> [10.5281/zenodo.21229773](https://doi.org/10.5281/zenodo.21229773). The version DOI is added at
> release.

**Contact.** Ricardo Gonçalves Silva, ricardogs@alumni.usp.br. Questions and reports of problems: the
issue tracker of https://github.com/ricardogs-econ/model-LB-replication, or by e-mail.

**Changelog.** `CHANGELOG.md` lists the changes of version 3.0.0 and of the 2.x releases (the
1.x releases are listed with their DOIs in `CITATION.cff`). Version 3.0.0 is the first to
accompany this article; the entries for versions before 3.0.0 describe earlier drafts.
