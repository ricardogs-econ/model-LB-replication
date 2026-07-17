# Monte Carlo and bootstrap in this package: three resampling procedures plus a robustness stage

A recurring source of confusion is that the words "Monte Carlo" and "bootstrap"
are used loosely, as if the package ran one simulation. It runs **three**
distinct resampling/simulation procedures, at three stages, on three different
data-generating assumptions — plus a **robustness stage (Section 5)** that
re-uses the first procedure's machinery under deliberately misspecified nulls.
They are complementary — each answers a question the others cannot — and they are
never interchangeable. This note fixes the vocabulary and the interfaces so that
a replicator knows exactly which procedure produces which number.

The unifying object is the point-optimal statistic and its finite-sample
critical value. Everything below is about how the **null distribution** is
obtained in each stage.

---

## Procedure 1 — Monte Carlo CALIBRATION (Sections 3–4)

**Question.** What is the finite-sample optimal detrending parameter
`c-bar*(m,T)` and the critical values of the five M-class statistics, in the
exact Model LB break configuration?

**Assumed DGP (the null).** A parametric, fully specified null: a driftless
Gaussian random walk with the level-break design imposed,
`y_t = Z_t'θ + u_t`, `u_t = u_{t-1} + ε_t`, `ε_t ~ i.i.d. N(0,1)`. Nothing is
resampled from data — pseudo-random innovations are drawn from the assumed law.
This is **Monte Carlo simulation**, not bootstrap: the null is known and
simulated, not estimated from a sample and resampled.

**What it produces.** `cbar_surface.csv`: for each `(m,T)` cell, the tangency
`c-bar*` (the c-bar at which local power crosses 0.50) and the lower-percentile
critical values of MZa, MSB, MZt, PT, MPT.

**Driver / control.** `replicate_section3_4.py`; replication count `R` inside
`mlb_core.DEFAULTS`; seeds via `config_seed(P,T,m,λ)` (disjoint by construction).

**Runs.** `python replicate_section3_4.py --full`

---

## Robustness of the calibration (Section 5) — Monte Carlo under a *misspecified* null

Section 5 does not introduce a new resampling scheme. It re-runs the machinery
of Procedure 1 — parametric Monte Carlo of the null — but deliberately changes
the null's short-run structure, to measure how far the i.i.d.-calibrated
`c-bar*(m,T)` and its critical values degrade when the maintained assumptions
fail. It answers a different question ("is the Section 3–4 calibration robust?")
with the same tool, so it belongs here rather than among the bootstraps.

Four experiments, all driven by `replicate_section5.py`:

- **`ar1` — AR(1) size/power and recalibration.** The null innovations are drawn
  from an AR(1), `ε_t = ρ ε_{t-1} + e_t`, instead of i.i.d. Two readings: (i) the
  empirical size/power of the i.i.d.-calibrated test *applied* under dependence
  (how much distortion the practitioner incurs); (ii) with `--recalibrate`, the
  `c-bar*` *re-found* under the AR(1) null, isolating whether the distortion comes
  from the variance-estimation channel or from displacement of `c-bar`. The
  long-run-variance estimator matters here: the difference-based `s²` is
  inconsistent under serial correlation (`plim = σ²/(1−ρ²)` against
  `ω² = σ²/(1−ρ)²`, an inflation factor `(1+ρ)/(1−ρ)`), whereas autoregressive
  MAIC remains consistent but degrades at very short `T`.
- **`oracle` — the infeasible benchmark.** Critical values computed with the
  *true* long-run variance substituted for its estimate, giving the power ceiling
  the feasible test aspires to; the feasible–oracle gap quantifies the price of
  estimating `ω²`.
- **`trimming` — break-fraction sensitivity.** Re-computes the surface under
  trimming `ε ∈ {0.10, 0.15, 0.20}`; since `c-bar` depends only weakly on the
  break locations `λ`, each cell should be near-invariant (within one grid step /
  Monte Carlo error) — a direct check of the location-invariance the theory
  predicts.
- **`power` — c-bar specification comparison.** Size and power of MZt under three
  `c-bar` specifications (calibrated Model 1, linear break-count scaling,
  trend-break surface) and three DGPs; produces `power_comparison.csv`,
  `tab_power.tex`, and the power-curve figure. Column (ii) ≈ column (i) is the
  empirical confirmation of the exact invariance of GLS detrending to break
  magnitudes.

**Reads.** the Section 3–4 surface `cbar_surface.csv` (the i.i.d. anchor whose
robustness is being probed). **Runs.** `python replicate_section5.py all`
(or `ar1 --recalibrate`, `oracle`, `trimming`, `power`).

**Why it is not a bootstrap.** Nothing is resampled from data. The null is
parametric and *assumed*, exactly as in Procedure 1 — only the assumed law is
varied (AR(1), true-`ω²` oracle, alternative trimming) to stress-test the
calibration. It is Monte Carlo, second-order: a simulation *about* a simulation's
robustness.

---

## Procedure 2 — Sieve-AR(p) BOOTSTRAP (Section 6, calibration for the application)

**Question.** For the eight admissible currencies, whose real exchange rates are
serially dependent, what is the applied pair `(c-bar*(m,T,p), cv(m,T,p))` once an
AR(p) short-run structure is allowed under the null?

**Estimated DGP (the null).** Here the null is **not** a clean random walk: the
short-run dynamics are estimated from the data by a sieve autoregression of
order `p` (selected per currency), and the null series are generated by
resampling/redrawing the sieve innovations with the fitted AR(p) filter and the
unit root imposed. This is a **model-based (sieve) bootstrap**: the persistence
structure is taken from the data, not assumed. It nests Procedure 1 as the
special case `p = 0`, which is precisely why the two are easy to conflate — but
`p > 0` is estimated, and that is the whole point of Section 6.

**What it produces.** `boot_out/calib/surface_ppp_boot.csv`: per
`(m,T,p,method)`, the calibrated `c-bar*`, its standard error, feasible power,
and the 5% critical values. This is the calibration the empirical verdicts in
Section 6 use.

**Two nuisance treatments, two roles (v1.2.0).** The *surface* is calibrated
under a coefficient FAMILY: first PAC fixed at 0.27 (the median first-lag ADF
coefficient of the eight residual series; provenance in
`ppp_pac_diagnostic.csv`), higher PACs redrawn uniformly per replication —
robust over a neighborhood, tabulatable. The *empirical critical value* each
currency confronts is sieve-own: the currency's fitted ADF phi at its BIC
order, held fixed across replications — the "per currency" estimation this
note describes is realised literally in the cv. The common-family cvs are
archived as a sensitivity in `boot_out/sensitivity/`.

**Reads.** the Section 3–4 surface `cbar_surface.csv` (as the `p = 0` anchor and
lookup fallback).

**Driver / control.** `replicate_section6.py boot`
(wraps `boot_ppp_cbar.py`); replications `R_cv = 10,000` / `R_pow = 5,000`
(surface) and `R_cv_emp = 20,000` (empirical critical values and p-values).

**Runs.** `python replicate_section6.py boot`

---

## Procedure 3 — WILD BOOTSTRAP (Section 6, half-life inference)

**Question.** What is a confidence interval for the half-life of PPP deviations
that is valid under the heteroskedasticity the real exchange rates exhibit
(non-constant innovation variance across the float)?

**Estimated DGP with sign-randomized innovations (the null/resampling scheme).**
The residuals of the fitted AR are multiplied by external random signs (a wild
scheme, e.g. Rademacher), preserving conditional heteroskedasticity that a
recursive residual resample would destroy. This is the **wild bootstrap**, used
only for the half-life grid-t intervals — a different object (a persistence
functional) from the unit-root statistics of Procedures 1–2, and a different
resampling scheme (sign-randomization, not innovation redraw).

**What it produces.** the wild grid-t confidence intervals in the half-life
table and the forest plot; the recursive scheme is reported alongside as the
scheme-invariance check.

**Driver / control.** `replicate_section6.py hl`
(wraps `hl_median_unbiased.py`); the `--selftest` gates G7/G8 verify that the
wild and recursive schemes agree under homoskedasticity and that the wild scheme
attains at least nominal coverage under heteroskedasticity.

**Runs.** `python replicate_section6.py hl`

---

## A fourth thing that is NOT a fourth procedure: on-the-fly critical values

`run_model_lb.py` (the stand-alone tool) and the empirical tests simulate the
null **in the user's exact break configuration** to obtain critical values when
no calibration cell matches. This is Procedure 1's machinery (Monte Carlo of the
parametric null) invoked pointwise, not a new bootstrap. It is labelled
"on-the-fly null simulation, R=…" in the report precisely to avoid calling it a
bootstrap.

---

## One-line summary

| # | Name | Stage | Null | Resampling | Driver | Count |
|---|---|---|---|---|---|---|
| 1 | Monte Carlo calibration | §3–4 | parametric RW (assumed) | draw N(0,1) innovations | `replicate_section3_4.py` | `R` |
| — | Robustness (MC, misspecified null) | §5 | AR(1) / true-ω² oracle / alt. trimming (assumed) | draw from the varied null | `replicate_section5.py` | `R` |
| 2 | Sieve-AR(p) bootstrap | §6 calib | AR(p)+unit root (estimated) | redraw sieve innovations | `replicate_section6.py boot` | `B=9999` |
| 3 | Wild bootstrap | §6 half-life | AR + heteroskedastic (estimated) | sign-randomize residuals | `replicate_section6.py hl` | `B=9999` |

**They complement, they do not substitute.** Procedure 1 prices the
finite-sample distortion of the *test* under the clean null; Procedure 2 carries
that calibration into a *serially dependent* null for the application; Procedure
3 does *not* test the unit root at all — it builds a heteroskedasticity-robust
interval for a *persistence functional*. Reporting any one of them in place of
another would be a category error.
