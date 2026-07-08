#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pesaran_cd.py -- Pesaran cross-sectional dependence (CD) test on the PPP panel.

Quantifies the cross-sectional dependence induced by the common (US-dollar)
numeraire, from the residuals of the STRICTLY UNIVARIATE Model LB fits -- no
panel model is estimated. This is the diagnostic referenced in the footnote of
Section (PPP verdicts): the dependence is measurable from the residuals already
in hand, it is reported here, and inference is NOT built on it (under exogenous
level breaks each statistic's limiting law is invariant to the common level
component, so the dependence bears only on the aggregate count).

--------------------------------------------------------------------------------
THE STATISTIC (Pesaran 2004, 2015)
--------------------------------------------------------------------------------
For N series with pairwise residual correlations rho_hat_ij over the T_ij
overlapping periods,

    CD = sqrt( 2 / (N(N-1)) ) * sum_{i<j} sqrt(T_ij) * rho_hat_ij   ~  N(0,1)

under H0: cross-sectional independence (E[rho_ij] = 0 for all i != j). Rejection
(|CD| large) indicates cross-sectional dependence. The statistic is exactly
standard-normal in T for fixed N and robust to the ordering of the units; it does
NOT require T > N and is valid for the small-N, moderate-T panel here (N=8, T~55).

We also report:
  * the average absolute pairwise correlation, mean |rho_ij|  (effect size), and
  * the CD applied to FIRST DIFFERENCES of the residuals, as a robustness check
    against any residual serial correlation inflating the levels statistic.

--------------------------------------------------------------------------------
RESIDUALS
--------------------------------------------------------------------------------
For each currency i the residual is u_hat_it = q_it - Z_it' theta_hat_i, where
Z_it = (1, DU_{i1,t}, ..., DU_{i m_i, t}) is the Model LB design (constant + level
dummies at the EXOGENOUS break years from exog_dates.csv; NO trend), theta_hat_i
by OLS. This is the same deterministic specification tested in the paper; the CD
is computed on its residuals so that the dependence measured is that which
survives the univariate mean model -- exactly the object a referee would ask about.

--------------------------------------------------------------------------------
INPUTS  (defaults match the replication package layout)
--------------------------------------------------------------------------------
    ppp_panel.csv   long format: columns currency, year, q  (log real rate)
    exog_dates.csv  long format: columns currency, break_year

--------------------------------------------------------------------------------
USAGE
--------------------------------------------------------------------------------
    python pesaran_cd.py                      # defaults
    python pesaran_cd.py --panel ppp_panel.csv --dates exog_dates.csv --col q
    python pesaran_cd.py --selftest           # internal checks, no data needed
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

import numpy as np


# ============================================================================
# Model LB design and residuals
# ============================================================================
def build_design(years: np.ndarray, break_years) -> np.ndarray:
    """Model LB design matrix Z = [1, DU_1, ..., DU_m], with
    DU_j,t = 1{year_t >= break_year_j}. No trend (Model LB). Level dummies only."""
    T = len(years)
    cols = [np.ones(T)]
    for by in sorted(break_years):
        du = (years >= by).astype(float)
        # guard: a dummy that is all-zeros or all-ones is collinear/empty -> skip
        if 0.0 < du.mean() < 1.0:
            cols.append(du)
    return np.column_stack(cols)


def ols_residuals(y: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """OLS residuals u_hat = y - Z (Z'Z)^{-1} Z' y, via least squares (stable)."""
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    return y - Z @ beta


# ============================================================================
# Pesaran CD
# ============================================================================
def pairwise_corr(U: dict) -> tuple:
    """Pairwise correlations of the residual series in dict U {name: array}.
    Series are aligned by position (the panel is balanced here). Returns
    (names, rho matrix, T_ij matrix) using only overlapping non-NaN periods."""
    names = sorted(U)
    N = len(names)
    rho = np.full((N, N), np.nan)
    Tij = np.zeros((N, N), dtype=int)
    for a in range(N):
        for b in range(a + 1, N):
            ua, ub = U[names[a]], U[names[b]]
            m = np.isfinite(ua) & np.isfinite(ub)
            t = int(m.sum())
            if t >= 3:
                xa, xb = ua[m] - ua[m].mean(), ub[m] - ub[m].mean()
                denom = np.sqrt((xa @ xa) * (xb @ xb))
                r = float(xa @ xb / denom) if denom > 0 else np.nan
                rho[a, b] = rho[b, a] = r
                Tij[a, b] = Tij[b, a] = t
    return names, rho, Tij


def cd_statistic(rho: np.ndarray, Tij: np.ndarray) -> dict:
    """Pesaran CD = sqrt(2/(N(N-1))) * sum_{i<j} sqrt(T_ij) rho_ij ~ N(0,1).
    Returns the statistic, two-sided p-value, and the average |rho|."""
    N = rho.shape[0]
    iu = np.triu_indices(N, k=1)
    r = rho[iu]
    t = Tij[iu].astype(float)
    ok = np.isfinite(r)
    r, t = r[ok], t[ok]
    npair = len(r)
    cd = np.sqrt(2.0 / (N * (N - 1))) * np.sum(np.sqrt(t) * r)
    # two-sided normal p-value
    from math import erfc, sqrt
    pval = erfc(abs(cd) / sqrt(2.0))
    return dict(CD=float(cd), pvalue=float(pval), n_pairs=int(npair),
                mean_abs_rho=float(np.mean(np.abs(r))),
                mean_rho=float(np.mean(r)))


# ============================================================================
# Data loading
# ============================================================================
def load_panel(path: str, col: str):
    """Load {currency: (years, series)} from a long-format CSV with columns
    currency, year, <col>. Series sorted by year."""
    import csv
    rows = defaultdict(list)
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        if col not in rd.fieldnames:
            raise SystemExit(f"Column '{col}' not in {path}; have {rd.fieldnames}")
        for r in rd:
            rows[r["currency"]].append((int(r["year"]), float(r[col])))
    out = {}
    for cur, pairs in rows.items():
        pairs.sort()
        yrs = np.array([p[0] for p in pairs])
        val = np.array([p[1] for p in pairs])
        out[cur] = (yrs, val)
    return out


def load_dates(path: str):
    """Load {currency: [break_year, ...]} from a long-format CSV with columns
    currency, break_year."""
    import csv
    d = defaultdict(list)
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                d[r["currency"]].append(int(r["break_year"]))
    except FileNotFoundError:
        print(f"[warn] {path} not found; using constant-mean residuals (no breaks).")
    return d


# ============================================================================
# Driver
# ============================================================================
def run(panel_path: str, dates_path: str, col: str, difference: bool = False):
    panel = load_panel(panel_path, col)
    dates = load_dates(dates_path)

    U = {}
    info = []
    for cur, (yrs, q) in sorted(panel.items()):
        Z = build_design(yrs, dates.get(cur, []))
        u = ols_residuals(q, Z)
        if difference:
            u = np.diff(u)
        U[cur] = u
        info.append((cur, Z.shape[1] - 1, len(q)))

    names, rho, Tij = pairwise_corr(U)
    res = cd_statistic(rho, Tij)

    tag = " (first differences)" if difference else ""
    print("=" * 60)
    print(f"  Pesaran CD test on Model LB residuals{tag}")
    print("=" * 60)
    print(f"  N currencies : {len(names)}")
    print(f"  break dummies: " + ", ".join(f"{c}:{m}" for c, m, _ in info))
    print(f"  pairs        : {res['n_pairs']}")
    print("-" * 60)
    print(f"  CD statistic : {res['CD']:+.4f}")
    print(f"  p-value      : {res['pvalue']:.4f}   (H0: cross-sectional independence)")
    print(f"  mean rho_ij  : {res['mean_rho']:+.4f}")
    print(f"  mean |rho_ij|: {res['mean_abs_rho']:.4f}")
    print("-" * 60)
    verdict = ("reject H0: cross-sectional dependence present"
               if res["pvalue"] < 0.05 else
               "fail to reject H0: no evidence of dependence")
    print(f"  verdict (5%) : {verdict}")
    print("=" * 60)
    print("  Note: the sign of mean rho_ij indicates the direction of the common")
    print("  (numeraire) effect; positive dependence raises the probability of a")
    print("  common non-rejection across the eight statistics, so the 0.046")
    print("  independence bound on P(zero rejections) is a LOWER bound.")
    return res


# ============================================================================
# Self-test
# ============================================================================
def _selftest():
    print("SELF-TEST pesaran_cd")
    ok = True
    rng = np.random.default_rng(0)
    N, T = 8, 55

    # (1) independent series -> CD ~ N(0,1), should not reject on average
    rej = 0
    for _ in range(200):
        U = {f"c{i}": rng.standard_normal(T) for i in range(N)}
        _, rho, Tij = pairwise_corr(U)
        if cd_statistic(rho, Tij)["pvalue"] < 0.05:
            rej += 1
    size = rej / 200
    c1 = 0.01 <= size <= 0.12
    print(f"  (1) size under independence ~0.05: {size:.3f}  [{'OK' if c1 else 'CHECK'}]")
    ok &= c1

    # (2) common factor -> strong positive dependence, CD rejects
    rej = 0
    for _ in range(100):
        f = rng.standard_normal(T)                      # common factor
        U = {f"c{i}": f + rng.standard_normal(T) for i in range(N)}
        _, rho, Tij = pairwise_corr(U)
        r = cd_statistic(rho, Tij)
        if r["pvalue"] < 0.05 and r["mean_rho"] > 0:
            rej += 1
    power = rej / 100
    c2 = power > 0.90
    print(f"  (2) power under common factor: {power:.2f}  [{'OK' if c2 else 'FAIL'}]")
    ok &= c2

    # (3) design matrix: DU is a step, no trend, drops degenerate dummies
    yrs = np.arange(1970, 1980)
    Z = build_design(yrs, [1975, 1900])   # 1900 is before sample -> all-ones -> dropped
    c3 = Z.shape == (10, 2) and Z[0, 1] == 0 and Z[5, 1] == 1
    print(f"  (3) build_design DU step, drops degenerate: [{'OK' if c3 else 'FAIL'}]")
    ok &= c3

    print("SELF-TEST:", "ALL PASSED" if ok else "FAILURES")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description="Pesaran CD test on PPP Model LB residuals.")
    ap.add_argument("--panel", default="ppp_panel.csv")
    ap.add_argument("--dates", default="exog_dates.csv")
    ap.add_argument("--col", default="q", help="residual target column (log real rate)")
    ap.add_argument("--difference", action="store_true",
                    help="CD on first differences of residuals (serial-correlation robustness)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    run(args.panel, args.dates, args.col, difference=False)
    print()
    run(args.panel, args.dates, args.col, difference=True)   # robustness
    return 0


if __name__ == "__main__":
    sys.exit(main())
