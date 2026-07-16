# -*- coding: utf-8 -*-
"""
reconcile_boot.py -- validate the PPP boot output against manuscript Tables 7 & 8.

Run AFTER, on the numba machine:
    python boot_ppp_cbar.py --full --empirical      # writes boot_out/...
then:
    python reconcile_boot.py --boot-out boot_out

It reconciles, digit by digit:
  * Table 8 (tab:pppsurface)  <- calib/surface_ppp_boot.csv
        rows (m=2, T=52, p, method): cbar_star, cbar_star_se,
        cv5_mzt_at_star, power_feasible_at_star
  * Table 7 (tab:ppp)         <- empirical/ppp_empirical[_prelim].csv
        per currency: MZt_asym (II), MZt_cal + cv_configfaithful (III)

NaN in any looked-up cell is forced to FAIL (never a silent pass). Tolerances
are loose enough for cross-run/cross-platform MC noise, tight enough to catch a
real drift: cbar +/-0.05, se +/-0.03, cv +/-0.03, power +/-0.03, MZt +/-0.03.
"""
from __future__ import annotations
import argparse, math, os, sys
import pandas as pd

# ---- manuscript v70 literals ----------------------------------------------
# Table 8: label -> (p, method, cbar*, se, cv, power_feas)
TAB8 = {
    "iid, MAIC":              (0, "maic",  -12.79, 0.17, -2.46, 0.58),
    "iid, difference-based":  (0, "const", -12.79, 0.17, -2.86, 0.70),
    "AR(1), MAIC":            (1, "maic",   -9.85, 0.12, -2.11, 0.30),
    "AR(2), MAIC":            (2, "maic",  -10.37, 0.17, -2.15, 0.31),
}
# Table 7: currency -> (MZt_asym(II), MZt_cal(III), cv_configfaithful(III))
TAB7 = {
    "AUD": (-0.38, -0.87, -2.32), "CAD": (-1.89, -2.06, -2.10),
    "CHF": ( 2.27,  1.78, -2.10), "GBP": (-0.33, -0.71, -2.11),
    "JPY": ( 0.17, -0.16, -2.10), "NOK": (-0.77, -0.97, -2.15),
    "NZD": (-0.33, -0.74, -2.10), "SEK": (-0.95, -1.39, -2.15),
}
TOL = dict(cbar=0.05, se=0.03, cv=0.03, power=0.03, mzt=0.03)


def bad(x):
    try: return x is None or math.isnan(float(x))
    except (TypeError, ValueError): return True

def chk(name, lit, got, tol, out):
    if bad(got):
        out.append(f"  [FAIL] {name}: manuscript {lit}  output NaN/missing"); return 1
    if abs(float(lit) - float(got)) > tol:
        out.append(f"  [FAIL] {name}: manuscript {lit}  output {float(got):+.4f}  "
                   f"Δ={float(got)-float(lit):+.4f} (tol {tol})"); return 1
    return 0


def table8(boot):
    p = os.path.join(boot, "calib", "surface_ppp_boot.csv")
    print(f"\n== TABLE 8  tab:pppsurface  <-  {p} ==")
    if not os.path.exists(p):
        print("  [MISSING] run: python boot_ppp_cbar.py --full --empirical"); return 1
    df = pd.read_csv(p)
    s = df[(df.m == 2) & (df["T"] == 52)]
    n = 0; out = []
    for lab, (pp, meth, cb, se, cv, pw) in TAB8.items():
        r = s[(s.p == pp) & (s.method == meth)]
        if len(r) == 0:
            out.append(f"  [FAIL] {lab}: no (m=2,T=52,p={pp},{meth}) row"); n += 1; continue
        r = r.iloc[0]
        n += chk(f"{lab} cbar*",  cb, r.get("cbar_star"),             TOL["cbar"],  out)
        n += chk(f"{lab} se",     se, r.get("cbar_star_se"),          TOL["se"],    out)
        n += chk(f"{lab} cv",     cv, r.get("cv5_mzt_at_star"),       TOL["cv"],    out)
        n += chk(f"{lab} powerF", pw, r.get("power_feasible_at_star"),TOL["power"], out)
    print("\n".join(out) if out else "  (all cells within tolerance)")
    print(f"  --> Table 8: {n} mismatch(es)")
    return n


def table7(boot):
    cand = [os.path.join(boot, "empirical", f) for f in
            ("ppp_empirical.csv", "ppp_empirical_prelim.csv")]
    p = next((c for c in cand if os.path.exists(c)), None)
    print(f"\n== TABLE 7  tab:ppp  <-  {p or cand[0]} ==")
    if p is None:
        print("  [MISSING] run: python boot_ppp_cbar.py --full --empirical"); return 1
    df = pd.read_csv(p).set_index("currency")
    n = 0; out = []
    for cur, (m2, m3, cv3) in TAB7.items():
        if cur not in df.index:
            out.append(f"  [FAIL] {cur}: absent from output"); n += 1; continue
        r = df.loc[cur]
        n += chk(f"{cur} MZt(II)",  m2,  r.get("MZt_asym"),          TOL["mzt"], out)
        n += chk(f"{cur} MZt(III)", m3,  r.get("MZt_cal"),           TOL["mzt"], out)
        n += chk(f"{cur} cv(III)",  cv3, r.get("cv_configfaithful"), TOL["cv"],  out)
    print("\n".join(out) if out else "  (all cells within tolerance)")
    print(f"  --> Table 7: {n} mismatch(es)")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot-out", default="boot_out")
    a = ap.parse_args()
    n = table8(a.boot_out) + table7(a.boot_out)
    print(f"\n{'='*60}\nOVERALL: {'ALL RECONCILED' if n==0 else str(n)+' MISMATCH(ES) — investigate'}")
    sys.exit(0 if n == 0 else 1)


if __name__ == "__main__":
    main()
