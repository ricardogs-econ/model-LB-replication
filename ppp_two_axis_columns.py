#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ppp_two_axis_columns.py
=======================

Decomposition of the two axes inside the PPP application: generates the
additional column of Table `tab:ppp` that pairs the theorem's detrending
value, cbar = -7, with the FINITE-SAMPLE critical value simulated at each
currency's own configuration.

Motivation
----------
The published Table `tab:ppp` contrasts two coupled pairs:

    (II)  cbar = -7                  x  asymptotic cv  = -1.98
    (III) cbar = cbar*(2,52,p)       x  finite-sample cv, simulated
                                        at the currency's configuration

That contrast does not identify each axis's contribution: between (II) and
(III), the detrending parameter (POWER axis) and the critical value (SIZE
axis) change simultaneously. The new column closes the square of the design:

    (N)   cbar = -7                  x  finite-sample cv, simulated
                                        at the currency's configuration

so that

    (II) -> (N)   : IDENTICAL statistic (same cbar), only the cv changes
                    ==> pure SIZE axis
    (N)  -> (III) : both with a finite-sample cv, only cbar changes
                    ==> pure DETRENDING/POWER axis

Common random numbers
----------------------
For the (N) -> (III) difference to be attributable to cbar rather than to
Monte Carlo noise, the seed is derived ONLY from the configuration
(m, dates, p, T) and NOT from cbar: the same innovation paths feed every
value of cbar for a given currency. This is the common-random-numbers
analogue and sharply reduces the variance of the DIFFERENCE between columns.

Maintained hypotheses (stated explicitly)
------------------------------------------
H1. EXOGENOUS, known break dates (Table `tab:dates`), Perron's (1989)
    condition. No endogenous search is performed, here or in the article.
H2. Exact invariance to break magnitudes (Lemma 1): under H0 the law of the
    statistic does not depend on theta, so the simulation fixes theta = 0.
    The `test_exact_invariance` self-test verifies this numerically.
H3. Under H0 the process is y_t = Z_t'theta + u_t, u_t = u_{t-1} + v_t, with
    v_t following the AR(p) estimated on that currency's own post-break
    residuals. p is selected by BIC. This is the null DGP of column (III) in
    the article and is reused unchanged by the new column.

    IMPORTANT (v1.3.0): the article's own p is NOT selected by this file. It
    is `k_bic_cq` of `ppp_ar_diagnostic.csv`, produced by `select_ar_order.py`
    -- an ADF-style regression that KEEPS the level term,
        du_t = (alpha-1) u_{t-1} + sum_{j=1..k-1} c_j du_{t-j} + e_t,
    with k counting the level regressor plus the lagged differences, common
    sample t=kmax..T-1, kmax=10. `boot_ppp_cbar.py` (the production
    calibration driver) reads that same column as `p_hat`; see its provision
    at line ~408 of that file. This file's own BIC search, `select_p_by_bic`
    below, fits a DIFFERENT model: a pure AR(p) on the FIRST DIFFERENCE of
    the residual, with NO level term,
        de_t = sum_{j=1..p} phi_j de_{t-j} + eps_t,
    p = 0..PMAX_BIC(=6), common sample fixed at row `pmax`. These are two
    distinct order-selection procedures answering two different questions
    (unit-root augmentation-lag choice vs. short-run-innovation AR order),
    and there is no reason for their argmin to coincide; empirically it does
    not, for most of the eight currencies. By DEFAULT this file therefore
    reads p from `ppp_ar_diagnostic.csv` (matching the article and
    `boot_ppp_cbar.py`) and uses `select_p_by_bic`'s AR(p) MACHINERY only to
    FIT phi/sigma AT that externally supplied order -- order selection and
    coefficient fitting are separate steps (`fit_ar_at_order`). Pass
    `--recompute-p` to instead select p independently, in which case the
    file prints both values and flags any disagreement; this is a robustness
    check on the AR(p) fit, not a replication of the published p.
H4. Long-run variance via s2_AR (Perron-Ng) with lag order by MAIC
    (Ng-Perron 2001), the auxiliary regression run on the OLS-demeaned
    series (Perron-Qu 2007 modification), kmax = 12. Identical to the
    article.
H5. Initialization u_0 = 0 (ERS convention). No burn-in is used on the I(1)
    component -- that would alter the law under H0. Burn-in applies only to
    the stationary AR(p) recursion of v_t.

Mandatory validation before any new result
--------------------------------------------
The `--validate` mode recomputes the already-published columns (II) and
(III) and compares them against the values in Table `tab:ppp`. The new
column should only be carried into the article if this reproduction passes:
a pipeline that fails to reproduce the published values does not license any
inference about the new column. `--validate` requires `ppp_ar_diagnostic.csv`
(default path, override with `--ar-diagnostic`) alongside `--data`, since p
is read from it by default (see H3 above); `--validate --recompute-p` is a
strictly harder, and currently failing, bar -- it additionally requires the
independent AR(p) search to land on the published p, which it does not.

Usage
-----
    # 1) validation (mandatory)
    python ppp_two_axis_columns.py --data path/to/rer.csv --validate

    # 2) new column + LaTeX table + raw results
    python ppp_two_axis_columns.py --data path/to/rer.csv \
        --nrep 50000 --latex tab_ppp_three_columns.tex \
        --csv-out ppp_two_axis_results.csv

    # 3) self-tests of the econometric core (no data required)
    python ppp_two_axis_columns.py --selftest

    # 4) robustness of the innovation resampling scheme
    python ppp_two_axis_columns.py --data path/to/rer.csv --innov wild

    # 5) independent p (diagnostic only; expected to disagree, see H3)
    python ppp_two_axis_columns.py --data path/to/rer.csv --recompute-p

Data format
-----------
Long (tidy) CSV with, at minimum, the columns

    year,currency,q

where `q` is the LOG of the bilateral real exchange rate against the dollar,
q_t = log(S_t) + log(P*_t) - log(P_t), already constructed (BIS + World Bank
CPI, 2010=100), 1973-2024. If the replication package stores the nominal
rate and the two CPIs separately, use `--build-rer` with the `s`, `cpi_us`,
`cpi_for` columns.

Author: replication pipeline for the article "Level Breaks and Finite-Sample
GLS Detrending" (Silva). This file imports no OTHER MODULE from the
replication package (numpy/pandas only), so the statistics (MZt, cv,
p-value) it recomputes are an independent test, not a tautology. It DOES
read `ppp_ar_diagnostic.csv` as a data input by default (v1.3.0, see H3) --
that is a data dependency, not a code import, and is exactly what
`boot_ppp_cbar.py` also does for `p_hat`; `--recompute-p` removes even that
data dependency, at the cost of no longer matching the published p (see the
"Mandatory validation" note above).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # pandas is only needed at the I/O layer
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None


# ---------------------------------------------------------------------------
# 1. Configuration: everything that comes from the article, in one auditable
#    place
# ---------------------------------------------------------------------------

CBAR_ASYMPTOTIC: float = -7.0          # the theorem's value (ERS demeaned)
CV_ASYMPTOTIC: float = -1.98           # asymptotic cv, demeaned MZ_t case
KMAX: int = 12                         # Schwert-plus, as in the article
PMAX_BIC: int = 6                      # maximum AR order in the BIC selection
SAMPLE_START, SAMPLE_END = 1973, 2024  # post-Bretton Woods, annual

# cbar*(2, 52, p) from Table `tab:pppsurface` -- anchored at m = 2.
# p = 0 is the iid case (tangency of the oracle envelope); p = 1, 2 are the
# fitted short-run AR orders. Orders p > 2 are not tabulated: the script
# aborts with an explicit message rather than extrapolating the surface.
CBAR_STAR_BY_P: Dict[int, float] = {0: -12.79, 1: -10.58, 2: -11.10}

# Table `tab:dates`: exogenous break dates (year of the regime event)
BREAK_DATES: Dict[str, Tuple[int, ...]] = {
    "AUD": (1983,),               # 12 Dec 1983 float (RBA)
    "CAD": (1985,),               # Plaza
    "CHF": (1985,),               # Plaza
    "GBP": (1985, 1992),          # Plaza; ERM exit, 16 Sep 1992
    "JPY": (1985,),               # Plaza
    "NOK": (1985, 1992),          # Plaza; krone float, Dec 1992
    "NZD": (1985,),               # 4 Mar 1985 float + Plaza, coincident
    "SEK": (1985, 1992),          # Plaza; krona float, 19 Nov 1992
}

CURRENCIES: Tuple[str, ...] = ("AUD", "CAD", "CHF", "GBP", "JPY", "NOK", "NZD", "SEK")

# Values published in Table `tab:ppp`, for the --validate mode.
# (mzt_II, mzt_III, cv_III, pval_III, p_ar)
PUBLISHED: Dict[str, Dict[str, float]] = {
    "AUD": {"mzt_II": -0.38, "mzt_III": -0.97, "cv_III": -2.07, "pval_III": 0.575, "p": 1},
    "CAD": {"mzt_II": -1.89, "mzt_III": -2.08, "cv_III": -2.07, "pval_III": 0.049, "p": 2},
    "CHF": {"mzt_II": +2.27, "mzt_III": +1.68, "cv_III": -2.14, "pval_III": 0.998, "p": 1},
    "GBP": {"mzt_II": -0.33, "mzt_III": -0.78, "cv_III": -2.12, "pval_III": 0.800, "p": 1},
    "JPY": {"mzt_II": +0.17, "mzt_III": -0.22, "cv_III": -2.19, "pval_III": 0.884, "p": 1},
    "NOK": {"mzt_II": -0.77, "mzt_III": -0.99, "cv_III": -2.16, "pval_III": 0.726, "p": 2},
    "NZD": {"mzt_II": -0.33, "mzt_III": -0.82, "cv_III": -2.03, "pval_III": 0.624, "p": 1},
    "SEK": {"mzt_II": -0.95, "mzt_III": -1.44, "cv_III": -2.10, "pval_III": 0.378, "p": 2},
}

# Validation tolerances. The statistic is deterministic given the data: tight
# tolerance. The cv is simulated: tolerance reflects Monte Carlo error.
TOL_STAT: float = 0.02      # rounding to 2 decimals in the published table
TOL_CV: float = 0.04        # ~2 MC standard errors at NREP = 50k
TOL_PVAL: float = 0.015


# ---------------------------------------------------------------------------
# 2. Econometric core
# ---------------------------------------------------------------------------

def level_break_design(n_obs: int, break_idx: Sequence[int]) -> np.ndarray:
    """Model LB's Z matrix: constant + m level dummies DU_j.

    DU_{j,t} = 1{t > tau_j}, with `break_idx` in 0-based indexing of the
    observation vector (tau_j = the position of the LAST observation of the
    previous regime).

    Raises
    ------
    ValueError
        If the dates are not strictly increasing, or if there is not at
        least one observation after the last break (the T > tau_m condition
        of Lemma 1, without which Z loses column rank).
    """
    idx = list(break_idx)
    if any(b <= 0 or b >= n_obs - 1 for b in idx):
        raise ValueError(f"break dates outside the usable range: {idx} (T={n_obs})")
    if any(b2 <= b1 for b1, b2 in zip(idx, idx[1:])):
        raise ValueError(f"break dates not strictly increasing: {idx}")

    Z = np.ones((n_obs, 1 + len(idx)))
    t = np.arange(n_obs)
    for j, tau in enumerate(idx, start=1):
        Z[:, j] = (t > tau).astype(float)
    if np.linalg.matrix_rank(Z) < Z.shape[1]:  # belt and suspenders
        raise ValueError("design matrix Z does not have full column rank")
    return Z


def quasi_difference(x: np.ndarray, cbar: float, n_obs: int) -> np.ndarray:
    """Quasi-difference at abar = 1 + cbar/T, with the ERS convention on obs. 1.

    x^abar_1 = x_1 ;  x^abar_t = x_t - abar * x_{t-1},  t >= 2.
    Accepts a vector (T,) or a matrix (T, k).
    """
    abar = 1.0 + cbar / n_obs
    xa = np.asarray(x, dtype=float)          # preserves the (T,) or (T, k) shape
    if xa.shape[0] != n_obs:
        raise ValueError(f"first dimension {xa.shape[0]} != T={n_obs}")
    xq = np.empty_like(xa)
    xq[0] = xa[0]
    xq[1:] = xa[1:] - abar * xa[:-1]
    return xq


def gls_detrend(y: np.ndarray, Z: np.ndarray, cbar: float) -> np.ndarray:
    """GLS-detrended series ytil_t = y_t - Z_t' thetahat(cbar).

    thetahat solves least squares on the quasi-differenced data; by
    construction (Lemma 1) the result is exactly invariant to theta.
    """
    n_obs = len(y)
    yq = quasi_difference(y, cbar, n_obs)
    Zq = quasi_difference(Z, cbar, n_obs)
    theta, *_ = np.linalg.lstsq(Zq, yq, rcond=None)
    return y - Z @ theta


def ols_detrend(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """OLS-demeaned series: the input to the Perron-Qu (2007) modification."""
    theta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ theta


def _adf_style_design(ytil: np.ndarray, k: int, kmax: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Regressors of the deterministic-free ADF-style regression, common sample.

    dytil_t = b_0 ytil_{t-1} + sum_{j=1..k} b_j dytil_{t-j} + e_t,
    evaluated at t = kmax+1, ..., T-1 (0-based), so that ALL k share the
    same effective sample -- required by MAIC.
    """
    dy = np.diff(ytil)                      # length T-1, dy[i] = ytil[i+1]-ytil[i]
    start = kmax                            # first usable row
    yy = dy[start:]
    cols = [ytil[start:-1]]                 # ytil_{t-1} aligned with dy[start:]
    for j in range(1, k + 1):
        cols.append(dy[start - j:len(dy) - j])
    return np.column_stack(cols), yy


def maic_lag_and_lrv(ytil_ols: np.ndarray, kmax: int = KMAX
                     ) -> Tuple[int, float]:
    """Lag selection by MAIC and the long-run variance s2_AR.

    MAIC(k) = ln(sigma2_k) + 2 (tau_T(k) + k) / (T - kmax),
    tau_T(k)  = sigma2_k^{-1} b0hat^2 sum_{t=kmax+1}^{T} ytil_{t-1}^2,
    s2_AR     = sigma2_ehat / (1 - sum_j bjhat)^2.

    Follows Ng-Perron (2001) with the Perron-Qu (2007) modification: the
    auxiliary regression is run on the OLS-demeaned series, although the
    statistics are computed on the GLS-detrended series.

    Returns
    -------
    (khat, s2_AR)
    """
    n_eff_den = len(ytil_ols) - kmax
    if n_eff_den <= kmax + 2:
        raise ValueError(f"sample too short for kmax={kmax} (T={len(ytil_ols)})")

    best = (np.inf, 0, np.nan)
    for k in range(0, kmax + 1):
        X, yy = _adf_style_design(ytil_ols, k, kmax)
        if X.shape[0] <= X.shape[1] + 1:
            continue
        beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ beta
        sigma2 = float(resid @ resid) / n_eff_den
        if not np.isfinite(sigma2) or sigma2 <= 0:
            continue
        b0 = float(beta[0])
        lagsum = float(np.sum(X[:, 0] ** 2))          # sum ytil_{t-1}^2, common sample
        tau = (b0 ** 2) * lagsum / sigma2
        maic = np.log(sigma2) + 2.0 * (tau + k) / n_eff_den
        if maic < best[0]:
            denom = 1.0 - float(np.sum(beta[1:])) if k > 0 else 1.0
            # Guard against a near-unit root in the short-run polynomial:
            # s2_AR would explode and the statistic would become meaningless.
            # This is a boundary case, flagged rather than silenced.
            if abs(denom) < 1e-6:
                continue
            best = (maic, k, sigma2 / denom ** 2)

    if not np.isfinite(best[2]):
        raise RuntimeError("MAIC found no admissible lag order")
    return best[1], best[2]


def m_statistics(ytil: np.ndarray, s2: float, cbar: float) -> Dict[str, float]:
    """M-class statistics on the GLS-detrended series.

    MZ_alpha = (T^{-1} ytil_T^2 - s2) / (2 T^{-2} sum ytil_{t-1}^2)
    MSB      = (s2^{-1} T^{-2} sum ytil_{t-1}^2)^{1/2}
    MZ_t     = MZ_alpha * MSB
    MPT      = (cbar^2 T^{-2} sum ytil_{t-1}^2 - cbar T^{-1} ytil_T^2) / s2
               (the DEMEANED form of Ng-Perron eq. 9, p = 0: Model LB has no
                trend, so the (1 - cbar) form of eq. 10 is not used)
    """
    n_obs = len(ytil)
    ssq = float(np.sum(ytil[:-1] ** 2)) / n_obs ** 2      # T^{-2} sum ytil_{t-1}^2
    endterm = float(ytil[-1] ** 2) / n_obs                # T^{-1} ytil_T^2
    mza = (endterm - s2) / (2.0 * ssq)
    msb = np.sqrt(ssq / s2)
    return {
        "MZa": mza,
        "MSB": msb,
        "MZt": mza * msb,
        "MPT": (cbar ** 2 * ssq - cbar * endterm) / s2,
    }


def mzt_observed(y: np.ndarray, Z: np.ndarray, cbar: float,
                 kmax: int = KMAX) -> Tuple[float, int, float]:
    """MZ_t observed under the article's full pipeline.

    Returns
    -------
    (MZ_t, khat_MAIC, s2_AR)
    """
    ytil_gls = gls_detrend(y, Z, cbar)
    ytil_ols = ols_detrend(y, Z)
    khat, s2 = maic_lag_and_lrv(ytil_ols, kmax)
    return m_statistics(ytil_gls, s2, cbar)["MZt"], khat, s2


# ---------------------------------------------------------------------------
# 3. Estimated short-run dynamics and the null DGP
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShortRun:
    """AR(p) fitted to the first difference of the post-break residual, p by BIC."""
    p: int
    phi: np.ndarray          # AR coefficients, length p
    sigma: float             # innovation standard deviation
    resid: np.ndarray        # standardized residuals (for the bootstrap)


def _short_run_design(de: np.ndarray, start: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
    """Regressors of the AR(p)-on-differences design, common sample `start`.

    Shared by `select_p_by_bic` and `fit_ar_at_order` so the two can never
    silently drift apart: the order-selection loop and the final fit at the
    selected (or externally supplied) order use the identical design.
    """
    n_total = len(de)
    yy = de[start:]
    if p == 0:
        X = np.zeros((len(yy), 0))
    else:
        X = np.column_stack([de[start - j - 1:n_total - j - 1] for j in range(p)])
    return X, yy


def select_p_by_bic(y: np.ndarray, Z: np.ndarray, pmax: int = PMAX_BIC) -> int:
    """Select p for an AR(p) on Delta e_t, e_t = the OLS residual of y on Z, by BIC.

    Under H0 the residual is I(1), so the AR is fitted to the FIRST
    DIFFERENCE, which is I(0) and for which BIC is the appropriate criterion.

    All candidate orders p = 0, ..., pmax are compared on the SAME effective
    sample (rows kept fixed at `pmax`), exactly as `_adf_style_design` already
    does for the MAIC lag search above. BIC values computed on samples of
    different size are not comparable; letting the row count shrink with p
    biases the comparison towards smaller p and can select an order outside
    the tabulated calibration surface (`CBAR_STAR_BY_P`).

    NOTE (v1.3.0): this is an AR(p) fitted directly to Delta e_t, with NO
    level term. It is NOT the same regression as `select_ar_order.py`'s
    `bic_select` (an ADF-style design that keeps u_{t-1}), which is what
    actually produced the article's p (`ppp_ar_diagnostic.csv`'s `k_bic_cq`,
    also read by `boot_ppp_cbar.py`). The two need not, and empirically do
    not, agree -- see the module docstring's H3. This function exists as an
    independent robustness check (`--recompute-p`), not as the default
    source of p; see `fit_ar_at_order` for fitting at an externally chosen
    order, and `main()` for how the two are wired together.
    """
    e = ols_detrend(y, Z)
    de = np.diff(e)
    n_total = len(de)
    if n_total - pmax <= pmax + 2:
        raise ValueError(f"sample too short for pmax={pmax} (T={len(y)})")
    start = pmax                            # common first usable row, all p
    nn = n_total - start
    best_bic, best_p = np.inf, 0
    for p in range(0, pmax + 1):
        X, yy = _short_run_design(de, start, p)
        beta = (np.linalg.lstsq(X, yy, rcond=None)[0] if p > 0 else np.zeros(0))
        resid = yy - (X @ beta if p > 0 else 0.0)
        sigma2 = float(resid @ resid) / nn
        if sigma2 <= 0 or not np.isfinite(sigma2):
            continue
        bic = np.log(sigma2) + p * np.log(nn) / nn
        if bic < best_bic:
            best_bic, best_p = bic, p
    return best_p


def fit_ar_at_order(y: np.ndarray, Z: np.ndarray, p: int, pmax: int = PMAX_BIC) -> ShortRun:
    """Fit an AR(p) to Delta e_t at a GIVEN, externally chosen order p.

    Order selection and coefficient fitting are deliberately separate steps
    (v1.3.0): `p` may come from `select_p_by_bic` (independent recomputation)
    or, by default in `main()`, from `ppp_ar_diagnostic.csv`'s `k_bic_cq`
    (the order that actually produced the published table). `pmax` only sets
    the common-sample trim point `start`, so a fit at p <= pmax uses exactly
    the sample `select_p_by_bic` would have compared it on; p > pmax is
    accepted (the trim point becomes p itself) but is outside the tabulated
    calibration surface (`CBAR_STAR_BY_P`) and `main()` will refuse it there.
    """
    if p < 0:
        raise ValueError(f"AR order must be >= 0, got p={p}")
    e = ols_detrend(y, Z)
    de = np.diff(e)
    start = max(pmax, p)
    n_eff = len(de) - start
    if n_eff <= p + 2:
        raise ValueError(f"sample too short for p={p} (T={len(y)})")
    X, yy = _short_run_design(de, start, p)
    beta = (np.linalg.lstsq(X, yy, rcond=None)[0] if p > 0 else np.zeros(0))
    resid = yy - (X @ beta if p > 0 else 0.0)
    sigma2 = float(resid @ resid) / n_eff
    if sigma2 <= 0 or not np.isfinite(sigma2):
        raise RuntimeError(f"short-run AR fit failed at p={p} (sigma2={sigma2})")
    sigma = float(np.sqrt(sigma2))
    return ShortRun(p=p, phi=np.asarray(beta, dtype=float), sigma=sigma,
                    resid=np.asarray(resid / sigma, dtype=float))


def fit_short_run(y: np.ndarray, Z: np.ndarray, pmax: int = PMAX_BIC) -> ShortRun:
    """Select p by BIC and fit the AR(p) at that order, in one call.

    Convenience composition of `select_p_by_bic` + `fit_ar_at_order`, kept
    for the self-tests and for `--recompute-p`. NOT the default source of p
    in `main()` as of v1.3.0 -- see the module docstring's H3.
    """
    p = select_p_by_bic(y, Z, pmax)
    return fit_ar_at_order(y, Z, p, pmax)


def config_seed(currency: str, break_idx: Sequence[int], p: int, n_obs: int) -> int:
    """Deterministic seed derived from the CONFIGURATION, not the currency label or cbar.

    Two intended consequences:
    (i) the critical value is a property of the configuration (m, dates, p,
        T), as the article states -- currencies with an identical
        configuration receive the same draw;
    (ii) COMMON random numbers across values of cbar, so that the
         (N) -> (III) difference carries no Monte Carlo noise.

    Note: `currency` deliberately does NOT enter the hash. It is present
    only as a documentary argument, to keep this choice visible at the call
    site.
    """
    key = f"m={len(break_idx)};tau={tuple(break_idx)};p={p};T={n_obs}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def draw_innovations(nrep: int, n_obs: int, sr: ShortRun, rng: np.random.Generator,
                     scheme: str = "gaussian", burn: int = 100) -> np.ndarray:
    """Raw shocks (nrep, n_obs + burn) under the chosen scheme.

    scheme:
      'gaussian' -- N(0,1), the article's default design;
      'resample' -- iid resampling of the standardized residuals (bootstrap);
      'wild'     -- Rademacher wild bootstrap on the standardized residuals,
                    robust to unmodeled heteroskedasticity.
    """
    total = n_obs + burn
    if scheme == "gaussian":
        return rng.standard_normal((nrep, total))
    if scheme == "resample":
        return rng.choice(sr.resid, size=(nrep, total), replace=True)
    if scheme == "wild":
        base = rng.choice(sr.resid, size=(nrep, total), replace=True)
        return base * rng.choice(np.array([-1.0, 1.0]), size=(nrep, total))
    raise ValueError(f"unknown innovation scheme: {scheme}")


def simulate_null_paths(eps: np.ndarray, sr: ShortRun, n_obs: int,
                        burn: int = 100) -> np.ndarray:
    """Paths y under H0: y_t = u_t, u_t = u_{t-1} + v_t, v_t ~ fitted AR(p).

    theta = 0 by H2 (exact invariance). u_0 = 0 by H5: burn-in applies ONLY
    to the stationary recursion of v_t; the I(1) partial sum starts at zero,
    as required by the ERS convention the theorem uses.
    """
    nrep = eps.shape[0]
    p = sr.p
    v = np.empty((nrep, n_obs + burn))
    scaled = sr.sigma * eps
    if p == 0:
        v = scaled
    else:
        v[:, :p] = scaled[:, :p]
        phi = sr.phi
        for t in range(p, n_obs + burn):
            v[:, t] = scaled[:, t] + v[:, t - p:t][:, ::-1] @ phi
    v_eff = v[:, burn:]                       # discard the AR transient
    return np.cumsum(v_eff, axis=1)           # u_0 = 0, I(1) with no burn-in


def null_distribution(y_paths: np.ndarray, Z: np.ndarray, cbar: float,
                      kmax: int = KMAX) -> np.ndarray:
    """MZ_t on each null path. Numerical failures become NaN and are counted."""
    nrep = y_paths.shape[0]
    out = np.full(nrep, np.nan)
    for i in range(nrep):
        try:
            out[i] = mzt_observed(y_paths[i], Z, cbar, kmax)[0]
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
    return out


# ---------------------------------------------------------------------------
# 4. Per-currency driver
# ---------------------------------------------------------------------------

@dataclass
class CurrencyResult:
    currency: str
    n_obs: int
    m: int
    p: int
    khat: int
    mzt_by_cbar: Dict[float, float] = field(default_factory=dict)
    cv_by_cbar: Dict[float, float] = field(default_factory=dict)
    cv_se_by_cbar: Dict[float, float] = field(default_factory=dict)
    pval_by_cbar: Dict[float, float] = field(default_factory=dict)
    n_valid_by_cbar: Dict[float, int] = field(default_factory=dict)


def quantile_se(sample: np.ndarray, q: float, n_boot: int = 400,
                seed: int = 0) -> float:
    """Bootstrap standard error of a quantile.

    Not decorative: the CAD boundary decision has a margin of 0.005 against
    its own cv, and without this number the binary reading of that cell is
    not defensible.
    """
    rng = np.random.default_rng(seed)
    n = len(sample)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        draws[b] = np.quantile(rng.choice(sample, size=n, replace=True), q)
    return float(np.std(draws, ddof=1))


def prepare_currency(currency: str, years: np.ndarray, q: np.ndarray
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """Trim the sample, build Z, and return (years, y, Z, break indices).

    The SINGLE point where the effective sample is constructed: both
    `run_currency` and the CLI driver call this function, so the p-selection
    probe and the estimation use exactly the same T and the same Z (a
    divergence here would silently swap out column (III)'s calibration).
    """
    mask = (years >= SAMPLE_START) & (years <= SAMPLE_END) & np.isfinite(q)
    yy, yrs = q[mask], years[mask]
    n_obs = len(yy)
    if n_obs < 45:
        raise ValueError(f"{currency}: T={n_obs} violates condition C2 (T >= 45)")

    # break dates -> 0-based indices (tau = last observation of the previous regime)
    idx: List[int] = []
    for yr in BREAK_DATES[currency]:
        pos = int(np.searchsorted(yrs, yr))
        if not (0 < pos < n_obs - 1):
            raise ValueError(f"{currency}: break {yr} falls outside the usable sample")
        idx.append(pos - 1)
    return yrs, yy, level_break_design(n_obs, idx), idx


def run_currency(currency: str, years: np.ndarray, q: np.ndarray,
                 cbars: Sequence[float], nrep: int, sr: ShortRun, kmax: int = KMAX,
                 scheme: str = "gaussian", compute_se: bool = True,
                 verbose: bool = True) -> CurrencyResult:
    """Run all three calibrations for one currency, with common random numbers.

    `sr` (the fitted short-run AR object) is supplied by the caller rather
    than recomputed here (v1.3.0): `main()` fits it once -- from
    `ppp_ar_diagnostic.csv` by default, or via `select_p_by_bic` under
    `--recompute-p` -- and both the cbar*(p) lookup and the null-DGP
    simulation must use that SAME p. Recomputing independently in each place
    risked exactly the silent divergence this refactor closes: two calls to
    a p-selector that no longer agree by construction once p can come from
    an external source.
    """
    _, yy, Z, idx = prepare_currency(currency, years, q)
    n_obs = len(yy)

    res = CurrencyResult(currency=currency, n_obs=n_obs, m=len(idx), p=sr.p, khat=-1)

    # --- random numbers COMMON to every cbar for this currency ---
    seed = config_seed(currency, idx, sr.p, n_obs)
    rng = np.random.default_rng(seed)
    eps = draw_innovations(nrep, n_obs, sr, rng, scheme=scheme)
    paths = simulate_null_paths(eps, sr, n_obs)

    for cbar in cbars:
        stat, khat, _ = mzt_observed(yy, Z, cbar, kmax)
        if cbar == CBAR_ASYMPTOTIC:
            res.khat = khat
        null = null_distribution(paths, Z, cbar, kmax)
        good = null[np.isfinite(null)]
        res.n_valid_by_cbar[cbar] = len(good)
        if len(good) < 0.95 * nrep:
            print(f"  [warn] {currency} cbar={cbar}: "
                  f"{nrep - len(good)} replications discarded", file=sys.stderr)
        cv = float(np.quantile(good, 0.05))
        res.mzt_by_cbar[cbar] = stat
        res.cv_by_cbar[cbar] = cv
        res.pval_by_cbar[cbar] = float(np.mean(good <= stat))
        res.cv_se_by_cbar[cbar] = (quantile_se(good, 0.05, seed=seed)
                                   if compute_se else float("nan"))
        if verbose:
            print(f"  {currency}  cbar={cbar:7.2f}  MZt={stat:7.3f}  "
                  f"cv={cv:7.3f} (se {res.cv_se_by_cbar[cbar]:.3f})  "
                  f"p={res.pval_by_cbar[cbar]:.3f}")
    return res


# ---------------------------------------------------------------------------
# 5. I/O
# ---------------------------------------------------------------------------

def load_panel(path: str, build_rer: bool = False) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Read the long CSV and return {currency: (years, log real rate)}."""
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required for --data")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "currency" not in df or "year" not in df:
        raise ValueError("the CSV needs the 'year' and 'currency' columns")
    if build_rer:
        need = {"s", "cpi_us", "cpi_for"}
        if not need.issubset(df.columns):
            raise ValueError(f"--build-rer requires the columns {sorted(need)}")
        # q = log S + log P* - log P  (S in units of foreign currency per USD;
        # flip the sign of log S if your convention is the opposite one)
        df["q"] = np.log(df["s"]) + np.log(df["cpi_for"]) - np.log(df["cpi_us"])
    if "q" not in df:
        raise ValueError("the CSV has no 'q' column (use --build-rer to construct it)")

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for cur, g in df.groupby("currency"):
        cur = str(cur).upper().strip()
        if cur not in BREAK_DATES:
            continue
        g = g.sort_values("year")
        out[cur] = (g["year"].to_numpy(dtype=int), g["q"].to_numpy(dtype=float))
    missing = set(CURRENCIES) - set(out)
    if missing:
        raise ValueError(f"currencies missing from the CSV: {sorted(missing)}")
    return out


def load_ar_diagnostic(path: str) -> Dict[str, int]:
    """Read `ppp_ar_diagnostic.csv` and return {currency: k_bic_cq}.

    `k_bic_cq` (produced by `select_ar_order.py`, "cq" = with breaks) is the
    AR order the article and `boot_ppp_cbar.py` actually use; see the module
    docstring's H3 for why it is not the same quantity `select_p_by_bic`
    below would choose. Uses csv/pandas interchangeably with `load_panel`'s
    reader (pandas is already a hard requirement of this file via --data).
    """
    if pd is None:  # pragma: no cover
        raise ImportError("pandas is required to read --ar-diagnostic")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"currency", "k_bic_cq"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path}: expected columns {sorted(need)}, "
                         f"got {list(df.columns)}")
    return {str(r["currency"]).upper().strip(): int(r["k_bic_cq"])
            for _, r in df.iterrows()}


def emit_latex(results: Dict[str, CurrencyResult], path: str) -> None:
    """Write Table `tab:ppp` with the three columns."""
    head = r"""\begin{table}[t!]
\centering
\caption{$\mathrm{MZ}_t$ for the real exchange rate against the US dollar
under three calibrations of the Model~LB test, at the exogenous break dates
of Table~\ref{tab:dates}; each specification's own $5\%$ critical value in
parentheses (rejection --- small values --- is evidence \emph{for} PPP).
Column~(N) isolates the two axes: it shares the statistic of column~(II)
(same $\bar c=-7$) and the critical-value construction of column~(III)
(finite-sample, at the currency's own configuration), so (II)$\to$(N) is the
size axis alone and (N)$\to$(III) the detrending axis alone.}
\label{tab:ppp}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc rr rr rrr}
\toprule
& & & & \multicolumn{2}{c}{(II) asymptotic $\mu$}
& \multicolumn{2}{c}{(N) $\bar c=-7$, finite-sample cv}
& \multicolumn{3}{c}{(III) finite-sample pair} \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-11}
Currency & $T$ & $m$ & $p$ & $\mathrm{MZ}_t$ (cv) & PPP? &
$\mathrm{MZ}_t$ (cv) & $p$-val &
$\mathrm{MZ}_t$ (cv) & $p$-val & PPP? \\
\midrule
"""
    rows: List[str] = []
    for cur in CURRENCIES:
        r = results[cur]
        cstar = CBAR_STAR_BY_P[r.p]
        s7 = r.mzt_by_cbar[CBAR_ASYMPTOTIC]
        cvN = r.cv_by_cbar[CBAR_ASYMPTOTIC]
        pN = r.pval_by_cbar[CBAR_ASYMPTOTIC]
        s3 = r.mzt_by_cbar[cstar]
        cv3 = r.cv_by_cbar[cstar]
        p3 = r.pval_by_cbar[cstar]
        verdict = "yes" if s3 <= cv3 else "no"

        def fmt(x: float) -> str:
            return (r"\phantom{-}" + f"{x:.2f}") if x >= 0 else f"{x:.2f}"

        rows.append(
            f"{cur} & {r.n_obs} & {r.m} & {r.p} & "
            f"${fmt(s7)}\\;({CV_ASYMPTOTIC:.2f})$ & "
            f"{'yes' if s7 <= CV_ASYMPTOTIC else 'no'} & "
            f"${fmt(s7)}\\;({cvN:.2f})$ & ${pN:.3f}$ & "
            f"${fmt(s3)}\\;({cv3:.2f})$ & ${p3:.3f}$ & {verdict} \\\\"
        )
    tail = r"""
\bottomrule
\end{tabular}
\par\medskip
\footnotesize $p$ by BIC on the post-break residual; long-run variance by
MAIC. Column~(N) pairs the theorem's detrending value $\bar c=-7$ with the
critical value simulated at the currency's own configuration; column~(III)
pairs $\bar c^{*}(2,52,p)$ of Table~\ref{tab:pppsurface} with the critical
value simulated at that $\bar c$. Columns (N) and (III) use common random
numbers, so their difference is attributable to $\bar c$ and not to
simulation error. Data: BIS bilateral nominal USD rates and World Bank CPI
(2010$=$100), 1973--2024.
\end{table}
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(head + "\n".join(rows) + tail)
    print(f"[ok] LaTeX table written to {path}")


# ---------------------------------------------------------------------------
# 6. Validation and self-tests
# ---------------------------------------------------------------------------

def validate(results: Dict[str, CurrencyResult]) -> bool:
    """Check columns (II) and (III) against the published values."""
    ok = True
    print("\n--- validation against the published Table tab:ppp ---")
    print(f"{'currency':6} {'item':10} {'got':>9} {'published':>10} {'diff':>8}  ")
    for cur in CURRENCIES:
        r, pub = results[cur], PUBLISHED[cur]
        cstar = CBAR_STAR_BY_P[r.p]
        checks = [
            ("MZt (II)", r.mzt_by_cbar[CBAR_ASYMPTOTIC], pub["mzt_II"], TOL_STAT),
            ("MZt (III)", r.mzt_by_cbar[cstar], pub["mzt_III"], TOL_STAT),
            ("cv (III)", r.cv_by_cbar[cstar], pub["cv_III"], TOL_CV),
            ("pval (III)", r.pval_by_cbar[cstar], pub["pval_III"], TOL_PVAL),
        ]
        if r.p != int(pub["p"]):
            print(f"{cur:6} {'p (BIC)':10} {r.p:>9} {int(pub['p']):>10}  MISMATCH")
            ok = False
        for name, got, want, tol in checks:
            diff = got - want
            flag = "" if abs(diff) <= tol else "  <-- OUT OF TOLERANCE"
            if flag:
                ok = False
            print(f"{cur:6} {name:10} {got:9.3f} {want:10.3f} {diff:8.3f}{flag}")
    print("\n[OK] reproduction accepted" if ok else
          "\n[FAIL] reproduction rejected -- DO NOT use the new column")
    return ok


def _test_quasi_difference_of_dummy() -> None:
    """eq:qd-step: the quasi-difference of DU is an impulse + an O(cbar/T) tail."""
    T, tau, cbar = 60, 25, -7.0
    du = (np.arange(T) > tau).astype(float)
    got = quasi_difference(du, cbar, T).ravel()
    want = np.zeros(T)
    want[tau + 1] = 1.0
    want[tau + 2:] = -cbar / T
    want[0] = du[0]
    assert np.allclose(got, want), "quasi-difference of the level dummy is wrong"


def _test_gram_expansion() -> None:
    """eq:Gram-qd: sum z z' = I + B/T + O(T^-2), B with the entries of Lemma 2."""
    T, cbar = 4000, -7.0
    lams = (0.3, 0.7)
    idx = [int(l * T) - 1 for l in lams]
    Z = level_break_design(T, idx)
    Zq = quasi_difference(Z, cbar, T)
    G = Zq.T @ Zq
    lam = [0.0] + [(i + 1) / T for i in idx]
    k = len(lam)
    B = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            B[i, j] = (cbar ** 2 * (1 - lam[i]) if i == j
                       else -cbar + cbar ** 2 * (1 - lam[max(i, j)]))
    err = np.max(np.abs(G - (np.eye(k) + B / T)))
    assert err < 5e-2, f"Gram expansion failed: max error {err:.4f}"


def _test_exact_invariance() -> None:
    """Lemma 1: the statistic does not depend on theta, whatever its magnitude."""
    rng = np.random.default_rng(7)
    T = 52
    Z = level_break_design(T, [12, 30])
    u = np.cumsum(rng.standard_normal(T))
    for theta in ([0, 0, 0], [3.1, -2.7, 5.0], [0, 40.0, -55.0]):
        y = Z @ np.asarray(theta, float) + u
        s, _, _ = mzt_observed(y, Z, -7.0)
        if theta == [0, 0, 0]:
            base = s
        assert abs(s - base) < 1e-8, f"exact invariance violated at theta={theta}"


def _test_asymptotic_cv(nrep: int = 4000) -> None:
    """Corollary: as T grows the cv of MZ_t tends to the demeaned value -1.98."""
    T = 1500
    Z = level_break_design(T, [400, 900])
    rng = np.random.default_rng(11)
    stats = np.empty(nrep)
    for i in range(nrep):
        y = np.cumsum(rng.standard_normal(T))
        stats[i] = mzt_observed(y, Z, -7.0, kmax=4)[0]
    cv = np.quantile(stats[np.isfinite(stats)], 0.05)
    assert abs(cv - CV_ASYMPTOTIC) < 0.12, (
        f"asymptotic cv = {cv:.3f}, expected ~{CV_ASYMPTOTIC}")


def _test_order_selection_fit_split() -> None:
    """v1.3.0 refactor: fit_short_run == select_p_by_bic + fit_ar_at_order,
    and fit_ar_at_order at an EXTERNALLY supplied p reproduces the same fit
    select_p_by_bic's own internal loop would have produced at that p --
    the split must not silently change either function's numbers."""
    rng = np.random.default_rng(23)
    T = 80
    Z = level_break_design(T, [20, 55])
    y = np.cumsum(rng.standard_normal(T))

    combo = fit_short_run(y, Z)
    p_sel = select_p_by_bic(y, Z)
    direct = fit_ar_at_order(y, Z, p_sel)
    assert combo.p == p_sel == direct.p, "order mismatch across the split"
    assert np.allclose(combo.phi, direct.phi), "phi mismatch across the split"
    assert abs(combo.sigma - direct.sigma) < 1e-10, "sigma mismatch across the split"

    # An order NOT chosen by BIC must still be fittable (run_currency needs
    # this whenever --ar-diagnostic's k_bic_cq differs from the independent
    # BIC argmin, which is the expected, documented case -- see H3).
    p_other = 0 if p_sel != 0 else 1
    alt = fit_ar_at_order(y, Z, p_other)
    assert alt.p == p_other, "fit_ar_at_order did not honor the requested order"


def selftest() -> None:
    for fn in (_test_quasi_difference_of_dummy, _test_gram_expansion,
               _test_exact_invariance, _test_asymptotic_cv,
               _test_order_selection_fit_split):
        fn()
        print(f"[ok] {fn.__name__}")
    print("\nall self-tests passed")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="long CSV with year,currency,q")
    ap.add_argument("--build-rer", action="store_true",
                    help="build q from s, cpi_us, cpi_for")
    ap.add_argument("--ar-diagnostic", default="ppp_ar_diagnostic.csv",
                    help="source of p (k_bic_cq column); default source as "
                         "of v1.3.0 -- see the module docstring's H3")
    ap.add_argument("--recompute-p", action="store_true",
                    help="ignore --ar-diagnostic; select p independently via "
                         "select_p_by_bic (a DIFFERENT regression -- expected "
                         "to disagree with the published p; diagnostic only)")
    ap.add_argument("--nrep", type=int, default=50_000)
    ap.add_argument("--kmax", type=int, default=KMAX)
    ap.add_argument("--innov", choices=("gaussian", "resample", "wild"),
                    default="gaussian")
    ap.add_argument("--validate", action="store_true",
                    help="recompute the published columns (II) and (III)")
    ap.add_argument("--latex", help="output path for the LaTeX table")
    ap.add_argument("--csv-out", help="output path for the results CSV")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        selftest()
        return 0
    if not args.data:
        ap.error("--data is required (or use --selftest)")

    panel = load_panel(args.data, build_rer=args.build_rer)

    ar_diag: Optional[Dict[str, int]] = None
    if not args.recompute_p:
        try:
            ar_diag = load_ar_diagnostic(args.ar_diagnostic)
        except (OSError, ValueError) as exc:
            ap.error(f"--ar-diagnostic '{args.ar_diagnostic}' unavailable "
                     f"({exc}); pass a valid path or use --recompute-p to "
                     f"select p independently instead (diagnostic only, see "
                     f"the module docstring's H3)")

    print(f"[info] {len(panel)} currencies loaded; nrep={args.nrep}; "
          f"innovations={args.innov}; "
          f"p source={'independent BIC (--recompute-p)' if ar_diag is None else args.ar_diagnostic}")

    results: Dict[str, CurrencyResult] = {}
    for cur in CURRENCIES:
        years, q = panel[cur]
        # p probe: determines which surface cbar* will be used in column
        # (III), and fits the ShortRun object `run_currency` simulates the
        # null DGP with. Uses the SAME sample/Z as run_currency (see
        # prepare_currency) -- computed ONCE here and passed down, so the
        # cbar*(p) lookup and the null simulation can never see two
        # different p's for the same currency (see run_currency's docstring).
        _, yy_probe, Z_probe, _ = prepare_currency(cur, years, q)
        if ar_diag is not None:
            if cur not in ar_diag:
                ap.error(f"{args.ar_diagnostic}: currency {cur} not found")
            p = ar_diag[cur]
            sr_probe = fit_ar_at_order(yy_probe, Z_probe, p)
        else:
            sr_probe = fit_short_run(yy_probe, Z_probe)
        if sr_probe.p not in CBAR_STAR_BY_P:
            raise ValueError(
                f"{cur}: p={sr_probe.p} (source: "
                f"{'independent BIC' if ar_diag is None else args.ar_diagnostic}) "
                f"is outside the tabulated surface {sorted(CBAR_STAR_BY_P)}; "
                f"extend CBAR_STAR_BY_P")
        if ar_diag is not None:
            p_indep = select_p_by_bic(yy_probe, Z_probe)
            if p_indep != sr_probe.p:
                print(f"  [note] {cur}: p={sr_probe.p} from "
                      f"{args.ar_diagnostic} (k_bic_cq, ADF-with-level "
                      f"design); independent AR(p)-on-differences BIC would "
                      f"pick p={p_indep} instead -- expected, see the module "
                      f"docstring's H3, not treated as an error")
        cbars = [CBAR_ASYMPTOTIC, CBAR_STAR_BY_P[sr_probe.p]]
        results[cur] = run_currency(cur, years, q, cbars, args.nrep, sr_probe,
                                    kmax=args.kmax, scheme=args.innov)

    if args.validate and not validate(results):
        return 1
    if args.latex:
        emit_latex(results, args.latex)
    if args.csv_out and pd is not None:
        recs = []
        for cur, r in results.items():
            cstar = CBAR_STAR_BY_P[r.p]
            recs.append({
                "currency": cur, "T": r.n_obs, "m": r.m, "p": r.p, "k_maic": r.khat,
                "n_valid_cbar7": r.n_valid_by_cbar[CBAR_ASYMPTOTIC],
                "n_valid_cbar_star": r.n_valid_by_cbar[cstar],
                "mzt_cbar7": r.mzt_by_cbar[CBAR_ASYMPTOTIC],
                "cv_asym": CV_ASYMPTOTIC,
                "cv_fs_at_cbar7": r.cv_by_cbar[CBAR_ASYMPTOTIC],
                "cv_fs_at_cbar7_se": r.cv_se_by_cbar[CBAR_ASYMPTOTIC],
                "pval_fs_at_cbar7": r.pval_by_cbar[CBAR_ASYMPTOTIC],
                "cbar_star": cstar,
                "mzt_cbar_star": r.mzt_by_cbar[cstar],
                "cv_fs_at_cbar_star": r.cv_by_cbar[cstar],
                "cv_fs_at_cbar_star_se": r.cv_se_by_cbar[cstar],
                "pval_fs_at_cbar_star": r.pval_by_cbar[cstar],
            })
        pd.DataFrame(recs).to_csv(args.csv_out, index=False)
        print(f"[ok] results written to {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
