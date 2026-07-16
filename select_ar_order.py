#!/usr/bin/env python3
"""
select_ar_order.py -- BIC and general-to-specific AR-order selection on the
post-break (level-break) and constant-mean residuals of the PPP panel,
reproducing the methodology documented in the replication package's README
("Provenance of ppp_ar_diagnostic.csv") on the CORRECT 1973-2024 (T=52)
window -- the original ppp_ar_diagnostic.csv was computed on 1970-2024 (T=55)
and never re-run after the window was fixed to 1973 in v1.1.2.

Reuses hl_median_unbiased.py's own build_Z/detrend/adf_fit kernels so the
residual construction and AR regression form are IDENTICAL to the rest of
the package (constant + exogenous level dummies via searchsorted on break
years; ADF-form du_t = (alpha-1) u_{t-1} + sum c_j du_{t-j} + e_t, no
deterministic terms since u is already detrended).

BIC: common-sample OLS BIC = T_eff*ln(RSS/T_eff) + k*ln(T_eff), all
candidate orders k=1..kmax fit on the SAME trimmed sample (t=kmax..T-1) so
they are comparable.

General-to-specific: start at k=kmax, drop the highest lagged-difference
term while its t-stat is insignificant at 5% (|t|<1.96), stop at the first
significant one or at k=1.

Usage:
    python select_ar_order.py --start 1973 --kmax 10
        # writes ppp_ar_diagnostic.csv (the file hl_median_unbiased.py and
        #  boot_ppp_cbar.py read); pass --out to write elsewhere.
"""
from __future__ import annotations
import argparse, csv, sys
from collections import defaultdict
import numpy as np


def build_Z(years, break_years, with_breaks=True):
    T = len(years)
    cols = [np.ones(T)]
    if with_breaks:
        for b in break_years:
            if years[0] < b <= years[-1]:
                cols.append((years >= b).astype(float))
    return np.column_stack(cols)


def detrend(q, Z):
    beta, *_ = np.linalg.lstsq(Z, q, rcond=None)
    return q - Z @ beta


def _design(u, k, kmax):
    """Common-sample ADF design at lag order k, trimmed to t=kmax..T-1 so
    every k in 1..kmax is fit on the identical sample (fair BIC/GS comparison)."""
    n = len(u)
    du = np.diff(u)
    T_eff = n - kmax
    X = np.empty((T_eff, k))
    y = np.empty(T_eff)
    for row, t in enumerate(range(kmax, n)):
        y[row] = du[t - 1]
        X[row, 0] = u[t - 1]
        for j in range(1, k):
            X[row, j] = du[t - 1 - j]
    return X, y


def bic_select(u, kmax):
    n = len(u)
    T_eff = n - kmax
    best_bic, best_k = np.inf, 1
    for k in range(1, kmax + 1):
        X, y = _design(u, k, kmax)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        rss = float(resid @ resid)
        bic = T_eff * np.log(rss / T_eff) + k * np.log(T_eff)
        if bic < best_bic:
            best_bic, best_k = bic, k
    return best_k


def gs_select(u, kmax, tcrit=1.96):
    n = len(u)
    T_eff = n - kmax
    for k in range(kmax, 1, -1):
        X, y = _design(u, k, kmax)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = T_eff - k
        s2 = float(resid @ resid) / dof
        XtXinv = np.linalg.inv(X.T @ X)
        se_last = np.sqrt(s2 * XtXinv[k - 1, k - 1])
        tstat = beta[k - 1] / se_last
        if abs(tstat) > tcrit:
            return k
    return 1


def adf_alpha_hl(u, p):
    """Raw OLS ADF alpha (diagnostic only, not the paper's reported
    median-unbiased figure) and its implied scalar half-life."""
    n = len(u)
    du = np.diff(u)
    T_eff = n - p
    X = np.empty((T_eff, p)); y = np.empty(T_eff)
    for row, t in enumerate(range(p, n)):
        y[row] = du[t - 1]
        X[row, 0] = u[t - 1]
        for j in range(1, p):
            X[row, j] = du[t - 1 - j]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha = 1.0 + beta[0]
    hl = np.log(0.5) / np.log(alpha) if 0 < alpha < 1 else float("inf")
    return alpha, hl


def load_panel(path, start_year):
    rows = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            yr = int(r["year"])
            if yr < start_year:
                continue
            rows[r["currency"]].append((yr, float(r["q"])))
    out = {}
    for cur, pairs in rows.items():
        pairs.sort()
        out[cur] = (np.array([p[0] for p in pairs]),
                    np.array([p[1] for p in pairs]))
    return out


def load_dates(path):
    d = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            d[r["currency"]].append(int(r["break_year"]))
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="ppp_panel.csv")
    ap.add_argument("--dates", default="exog_dates.csv")
    ap.add_argument("--start", type=int, default=1973)
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--out", default="ppp_ar_diagnostic.csv")
    args = ap.parse_args(argv)

    panel = load_panel(args.panel, args.start)
    dates = load_dates(args.dates)

    rows = []
    for cur, (years, q) in sorted(panel.items()):
        T = len(years)
        by = dates.get(cur, [])
        m = sum(1 for b in by if years[0] < b <= years[-1])

        Z_sq = build_Z(years, by, with_breaks=False)
        u_sq = detrend(q, Z_sq)
        k_bic_sq = bic_select(u_sq, args.kmax)
        k_gs_sq = gs_select(u_sq, args.kmax)
        alpha_sq, hl_sq = adf_alpha_hl(u_sq, k_bic_sq)

        Z_cq = build_Z(years, by, with_breaks=True)
        u_cq = detrend(q, Z_cq)
        k_bic_cq = bic_select(u_cq, args.kmax)
        k_gs_cq = gs_select(u_cq, args.kmax)
        alpha_cq, hl_cq = adf_alpha_hl(u_cq, k_bic_cq)

        rows.append(dict(currency=cur, m=m, T=T, kmax=args.kmax,
                          k_bic_sq=k_bic_sq, k_gs_sq=k_gs_sq,
                          k_bic_cq=k_bic_cq, k_gs_cq=k_gs_cq,
                          alpha_sq=alpha_sq, alpha_cq=alpha_cq,
                          hl_sq=hl_sq, hl_cq=hl_cq))
        print(f"{cur}: T={T} m={m} | sq: BIC={k_bic_sq} GS={k_gs_sq} "
              f"alpha={alpha_sq:.4f} hl={hl_sq:.2f} | "
              f"cq: BIC={k_bic_cq} GS={k_gs_cq} alpha={alpha_cq:.4f} hl={hl_cq:.2f}")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"-> {args.out}")


