#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pac_diagnostic.py -- provenance for the AR(p) nuisance persistence (pac1).
==========================================================================
Computes, per admissible currency, the empirical counterparts of the first
partial autocorrelation of the stationary nuisance w_t in the calibration
DGP  u_t = rho u_{t-1} + w_t  (boot_ppp_cbar.py):

  * PACF(1..3) of Delta(u_hat), where u_hat is the OLS residual of q on
    [1, DU_1, ..., DU_m] at the exogenous break dates (under the null,
    Delta u = w exactly, so PACF(k) of Delta u_hat estimates the PACs of w);
  * the first-lag coefficients gamma_1..gamma_k of the ADF regression
    Delta u_t = b u_{t-1} + sum_j gamma_j Delta u_{t-j} + e_t at the BIC
    order k = k_bic_cq of ppp_ar_diagnostic.csv (the regression the order
    selection actually fit).

Output: ppp_pac_diagnostic.csv (per currency) + a summary line. The v1.2.0
canonical pac1 = 0.27 is the MEDIAN of gamma_1 across the eight currencies
(mean 0.23; range -0.01..0.39). The pre-1.2.0 value 0.4 exceeded the entire
empirical range (legacy of an earlier 4-lag diagnostic).

Usage:  python pac_diagnostic.py [--start-year 1973] [--out ppp_pac_diagnostic.csv]
Inputs: ppp_panel.csv, exog_dates.csv, ppp_ar_diagnostic.csv (script dir/cwd).
Pure numpy; no numba required.
"""
import argparse, csv, os
import numpy as np

def pacf_ld(x, nlags):
    """PACF via Levinson-Durbin on sample autocorrelations."""
    x = np.asarray(x, float) - np.mean(x)
    acf = np.array([1.0] + [np.dot(x[:-k], x[k:]) / np.dot(x, x)
                            for k in range(1, nlags + 1)])
    pacf = np.zeros(nlags + 1); pacf[0] = 1.0
    phi_prev = np.zeros(nlags + 1); phi_curr = np.zeros(nlags + 1)
    phi_prev[1] = acf[1]; pacf[1] = acf[1]
    for k in range(2, nlags + 1):
        num = acf[k] - np.dot(phi_prev[1:k], acf[1:k][::-1])
        den = 1.0 - np.dot(phi_prev[1:k], acf[1:k])
        a = num / den
        phi_curr[1:k] = phi_prev[1:k] - a * phi_prev[1:k][::-1]
        phi_curr[k] = a; pacf[k] = a
        phi_prev = phi_curr.copy()
    return pacf[1:]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=1973)
    ap.add_argument("--out", default="ppp_pac_diagnostic.csv")
    args = ap.parse_args()
    base = os.path.dirname(os.path.abspath(__file__))

    def find(name):
        for d in (os.getcwd(), base):
            f = os.path.join(d, name)
            if os.path.exists(f):
                return f
        raise SystemExit(f"[pac_diagnostic] input not found: {name}")

    panel = {}
    for r in csv.DictReader(open(find("ppp_panel.csv"))):
        panel.setdefault(r["currency"], []).append((int(r["year"]), float(r["q"])))
    ed = {}
    for r in csv.DictReader(open(find("exog_dates.csv"))):
        ed.setdefault(r["currency"], []).append(int(r["break_year"]))
    pmap = {}
    for r in csv.DictReader(open(find("ppp_ar_diagnostic.csv"))):
        pmap[r["currency"]] = int(float(r["k_bic_cq"]))

    rows_out, p1s, g1s = [], [], []
    for cur in sorted(panel):
        obs = sorted(o for o in panel[cur] if o[0] >= args.start_year)
        yrs = np.array([o[0] for o in obs]); q = np.array([o[1] for o in obs])
        T = len(q)
        bp = [int(np.searchsorted(yrs, b)) for b in ed.get(cur, [])
              if yrs[0] < b <= yrs[-1]]
        Z = np.ones((T, 1))
        for b in bp:
            du = np.zeros(T); du[b:] = 1.0
            Z = np.column_stack([Z, du])
        beta, *_ = np.linalg.lstsq(Z, q, rcond=None)
        u = q - Z @ beta
        du_ = np.diff(u)
        pac = pacf_ld(du_, 3)
        k = pmap.get(cur, 1)
        yv = du_[k:]
        X = [u[k:-1]] + [du_[k - j:-j] for j in range(1, k + 1)]
        X = np.column_stack([np.ones(len(yv))] + X)
        bcoef, *_ = np.linalg.lstsq(X, yv, rcond=None)
        g1 = float(bcoef[2]) if len(bcoef) > 2 else np.nan
        p1s.append(pac[0]); g1s.append(g1)
        rows_out.append(dict(currency=cur, T=T, m=len(bp), k_bic=k,
                             pacf1_du=round(float(pac[0]), 4),
                             pacf2_du=round(float(pac[1]), 4),
                             pacf3_du=round(float(pac[2]), 4),
                             gamma1_adf=round(g1, 4)))

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    for r in rows_out:
        print(r)
    print(f"[summary] PACF1(dU): mean={np.mean(p1s):.3f} "
          f"median={np.median(p1s):.3f} range=[{min(p1s):.3f},{max(p1s):.3f}]")
    print(f"[summary] gamma1(ADF): mean={np.nanmean(g1s):.3f} "
          f"median={np.nanmedian(g1s):.3f} "
          f"range=[{np.nanmin(g1s):.3f},{np.nanmax(g1s):.3f}]")
    print(f"[summary] canonical pac1 (v1.2.0) = median gamma1 rounded = "
          f"{round(float(np.nanmedian(g1s)), 2)}")
    print(f"[out] {args.out}")

if __name__ == "__main__":
    main()
