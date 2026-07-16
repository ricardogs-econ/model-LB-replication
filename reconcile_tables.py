# -*- coding: utf-8 -*-
"""
reconcile_tables.py -- package self-check: manuscript table literals vs the
replication artifacts. Read-only; no simulation is run here.

Paths are resolved relative to THIS file (the package root), so it runs from
anywhere. Checks split into two tiers:

  ALWAYS (shipped artifacts):
    Table 1  tab:cbar   <- cbar_surface.csv        (const, mean over locations)
    Table 2  tab:cv     <- cbar_surface.csv        (MZt 5% cv at optimal cbar)
    Table 6  tab:dates  <- exog_dates.csv
  CONDITIONAL (regenerate first with the section drivers, then re-run me):
    Table 4  tab:ar1-sizepower  <- robustness_out/ar1_size_power.csv
             (python replicate_section5.py ar1)
    Table 5  tab:ar1-recal      <- robustness_out/ar1_recalibration.csv
             (python replicate_section5.py ar1)
    Table 9  tab:hl             <- hl_results_wild.csv
             (python replicate_section6.py hl --boot wild --B 20000)

Tables 7-8 (the PPP empirical block) are validated by reconcile_boot.py after
`python boot_ppp_cbar.py --full --empirical`.

NaN / missing lookups are forced to FAIL (never a silent pass).
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
Ts = [30, 45, 50, 60, 80, 100, 150, 200, 300]


def hl(a):
    a = float(a)
    return math.inf if a >= 1 else math.log(0.5) / math.log(a)

def bad(x):
    try: return x is None or math.isnan(float(x))
    except (TypeError, ValueError): return True

def show(tag, ok, n=None):
    tail = "" if n is None else f" (mismatches: {n})"
    print(f"  [{'PASS' if ok else 'FLAG'}] {tag}{tail}")


def table12(a):
    print("\n== TABLES 1-2  tab:cbar / tab:cv  <-  cbar_surface.csv ==")
    c = a[a.sigma2_method == "const"]
    man1 = {0:[-7.2,-7.1,-7.0,-7.0,-7.0,-6.9,-7.0,-7.1,-7.0],
            1:[-8.7,-8.2,-7.9,-7.8,-7.7,-7.4,-7.4,-7.1,-7.1],
            2:[-10.3,-9.3,-9.0,-8.7,-8.3,-8.0,-7.7,-7.5,-7.3],
            3:[None,-10.4,-10.1,-9.5,-9.0,-8.6,-8.1,-7.8,-7.5],
            4:[None,None,None,-10.5,-9.6,-9.2,-8.6,-8.2,-8.0],
            5:[None,None,None,None,-10.4,-9.4,-8.8,-8.4,-8.1]}
    man2 = {0:[-2.24,-2.20,-2.18,-2.15,-2.13,-2.10,-2.05,-2.03,-2.00],
            1:[-2.56,-2.46,-2.41,-2.37,-2.29,-2.24,-2.18,-2.13,-2.09],
            2:[-2.86,-2.69,-2.64,-2.57,-2.47,-2.39,-2.30,-2.24,-2.17],
            3:[None,-2.94,-2.88,-2.77,-2.65,-2.55,-2.43,-2.34,-2.27],
            4:[None,None,None,-2.96,-2.81,-2.71,-2.54,-2.46,-2.38],
            5:[None,None,None,None,-2.98,-2.82,-2.64,-2.55,-2.44]}
    n1 = n2 = 0
    for m, row in man1.items():
        for T, lit in zip(Ts, row):
            if lit is None: continue
            v = round(c[(c.m == m) & (c["T"] == T)].cbar_otimo.mean(), 1)
            if bad(v) or abs(v - lit) > 0.05:
                print(f"    T1 (m={m},T={T}) man {lit} got {v}"); n1 += 1
    for m, row in man2.items():
        for T, lit in zip(Ts, row):
            if lit is None: continue
            v = round(c[(c.m == m) & (c["T"] == T)].mzt_cv5.mean(), 2)
            if bad(v) or abs(v - lit) > 0.005:
                print(f"    T2 (m={m},T={T}) man {lit} got {v}"); n2 += 1
    show("Table 1 (cbar)", n1 == 0, n1); show("Table 2 (cv)", n2 == 0, n2)
    return n1 + n2


def table6():
    print("\n== TABLE 6  tab:dates  <-  exog_dates.csv ==")
    ed = pd.read_csv(ROOT / "exog_dates.csv")
    man = {"AUD":[1983],"CAD":[1985],"CHF":[1985],"GBP":[1985,1992],"JPY":[1985],
           "NOK":[1985,1992],"NZD":[1985],"SEK":[1985,1992]}
    n = 0
    for cur, yrs in man.items():
        got = sorted(ed[ed.currency == cur].break_year.tolist())
        if got != sorted(yrs): print(f"    {cur}: man {yrs} got {got}"); n += 1
    show("Table 6 (dates)", n == 0, n); return n


def table45():
    p = ROOT / "robustness_out" / "ar1_size_power.csv"
    q = ROOT / "robustness_out" / "ar1_recalibration.csv"
    print("\n== TABLES 4-5  tab:ar1-*  <-  robustness_out/ (rho=0.5) ==")
    if not p.exists() or not q.exists():
        print("    [SKIP] run: python replicate_section5.py ar1   (then re-run)"); return 0
    ar = pd.read_csv(p); ar = ar[ar.rho == 0.5]
    man4 = {(30,0):(0.002,0.033,0.121,0.257,0.472),(30,1):(0.001,0.033,0.046,0.159,0.416),
            (30,2):(0.001,0.039,0.051,0.155,0.430),(60,0):(0.000,0.015,0.086,0.383,0.491),
            (60,1):(0.001,0.019,0.062,0.290,0.521),(60,2):(0.001,0.018,0.036,0.159,0.499),
            (100,0):(0.001,0.016,0.076,0.517,0.532),(100,1):(0.000,0.012,0.069,0.501,0.513),
            (100,2):(0.001,0.010,0.043,0.396,0.479)}
    def g(m,T,meth,col): return float(ar[(ar.m==m)&(ar["T"]==T)&(ar.method==meth)][col].iloc[0])
    n4 = 0
    for (T,m),(cs,cp,ms,mp,pi) in man4.items():
        got=(round(g(m,T,"const","size_emp"),3),round(g(m,T,"const","power_emp"),3),
             round(g(m,T,"maic","size_emp"),3),round(g(m,T,"maic","power_emp"),3),
             round(g(m,T,"maic","power_iid"),3))
        for nm,lit,gv in zip("Csz Cpw Msz Mpw Piid".split(),(cs,cp,ms,mp,pi),got):
            if bad(gv) or abs(lit-gv)>0.0005: print(f"    T4 T={T} m={m} {nm}: man {lit} got {gv}"); n4+=1
    rc = pd.read_csv(q)
    man5 = {(30,0):(-7.20,-7.30,-0.10),(60,0):(-7.00,-6.95,0.05),(100,0):(-6.94,-7.30,-0.36),
            (30,1):(-8.73,-8.12,0.60),(60,1):(-7.76,-7.31,0.45),(100,1):(-7.43,-7.60,-0.17),
            (30,2):(-10.32,-9.58,0.74),(60,2):(-8.67,-8.51,0.16),(100,2):(-7.97,-8.02,-0.05)}
    n5 = 0
    for (T,m),(ci,ca,dl) in man5.items():
        r = rc[(rc.m==m)&(rc["T"]==T)]
        if len(r)==0 or abs(float(r.cbar_iid.iloc[0])-ci)>0.005 or abs(float(r.cbar_AR.iloc[0])-ca)>0.005 or abs(float(r.delta.iloc[0])-dl)>0.005:
            print(f"    T5 T={T} m={m}: man ({ci},{ca},{dl})"); n5+=1
    show("Table 4 (ar1 size/power)", n4==0, n4); show("Table 5 (recalibration)", n5==0, n5)
    return n4 + n5


def table9():
    p = ROOT / "hl_results_wild.csv"
    print("\n== TABLE 9  tab:hl  <-  hl_results_wild.csv (wild, B=20,000) ==")
    if not p.exists():
        print("    [SKIP] run: python replicate_section6.py hl --boot wild --B 20000"); return 0
    h = pd.read_csv(p).set_index("currency")
    man = {"AUD":(0.910,7.3,0.675,1.8,0.8,7.3),"CAD":(0.879,5.4,0.882,5.5,2.9,69.0),
           "CHF":(0.967,20.5,0.904,6.9,3.6,math.inf),"GBP":(0.847,4.2,0.768,2.6,1.2,9.6),
           "JPY":(0.978,31.8,0.814,3.4,2.2,34.3),"NOK":(0.905,6.9,0.805,3.2,1.6,45.9),
           "NZD":(0.907,7.1,0.785,2.9,1.0,11.2),"SEK":(0.901,6.7,0.698,1.9,1.0,5.5)}
    n = 0
    for cur,(aMP,HLMP,aLB,HLLB,clo,chi) in man.items():
        r = h.loc[cur]
        gclo = round(hl(r.alpha_ci_lo_LB),1); gchi = hl(r.alpha_ci_hi_LB)
        gchi = math.inf if math.isinf(gchi) else round(gchi,1)
        checks = {"aMP":(aMP,round(float(r.alpha_mu_MP),3)),"HLMP":(HLMP,round(float(r.HL_scalar_MP),1)),
                  "aLB":(aLB,round(float(r.alpha_mu_LB),3)),"HLLB":(HLLB,round(float(r.HL_scalar_LB),1)),
                  "CIlo":(clo,gclo),"CIhi":(chi,gchi)}
        for nm,(lit,gv) in checks.items():
            eq = (math.isinf(lit) and math.isinf(gv)) or (not math.isinf(lit) and not math.isinf(gv) and abs(lit-gv)<=0.05)
            if not eq: print(f"    {cur} {nm}: man {lit} got {gv}"); n += 1
    show("Table 9 (half-lives)", n == 0, n); return n


def main():
    print("PACKAGE SELF-CHECK -- manuscript tables vs artifacts")
    a = pd.read_csv(ROOT / "cbar_surface.csv")
    total = table12(a) + table6() + table45() + table9()
    print(f"\n{'='*56}\nSHIPPED-ARTIFACT + AVAILABLE tables: "
          f"{'ALL RECONCILED' if total==0 else str(total)+' MISMATCH(ES)'}")
    print("Tables 7-8: run `python reconcile_boot.py --boot-out boot_out`.")
    raise SystemExit(0 if total == 0 else 1)


if __name__ == "__main__":
    main()
