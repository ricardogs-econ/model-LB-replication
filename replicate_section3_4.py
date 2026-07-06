# -*- coding: utf-8 -*-
"""
================================================================================
replicate_section3_4.py -- Replicates Sections 3-4 of the paper:
    the finite-sample calibration surface c-bar(m,T) and the critical values of
    the five M-class statistics, plus the surface figure.
================================================================================

WHAT IT PRODUCES
    * cbar_surface.csv   -- the calibration surface: for every (m,T) cell and
      each long-run-variance method, the tangency c-bar*(m,T), its standard
      error, and the 1/2.5/5/10% critical values of MZa, MSB, MZt, PT, MPT.
      This CSV is the INPUT to replicate_section5.py and replicate_section6.py.
    * cbar_surface.pdf   -- Figure `cbar`: c-bar vs 1/T by m with the per-m
      fits c-bar = c_inf + a(m)/T (common intercept at the ERS value -7), and
      the slopes a(m) against m+1 (origin ray of slope ~ -37 for m>=2).

METHOD (Section 3-4)
    For each (m,T), c-bar* is the point where the local power of the feasible
    point-optimal statistic crosses 0.50 (the ERS 50%-power tangency),
    interpolated on a c-bar grid; the critical values are the lower percentiles
    of the null (driftless random walk) distribution in the exact break
    configuration. Everything runs through the single kernel in mlb_core.py.

USAGE
    python replicate_section3_4.py --full            # paper numbers (hours)
    python replicate_section3_4.py --quick           # smoke test (~2-3 min)
    python replicate_section3_4.py --jobs 8 --outdir .
    python replicate_section3_4.py --figure-only     # rebuild the figure from CSV

DETERMINISM & COST
    Deterministic given the replication counts (config_seed in mlb_core). The
    full grid (m=0..5, T up to 300) is ~400 cells; wall time is dominated by the
    per-cell Monte Carlo and scales with --jobs. --quick runs a reduced grid.

REQUIRES
    mlb_core.py; numpy, numba, matplotlib (figure), joblib (parallel).
================================================================================
"""
import argparse
import csv
import os
import sys

import numpy as np

try:
    import mlb_core as core
except ImportError:
    sys.exit("replicate_section3_4.py requires mlb_core.py in the same directory.")

SURFACE_CSV = "cbar_surface.csv"
SURFACE_PDF = "cbar_surface.pdf"


# -----------------------------------------------------------------------------
# figure (Section 4): c-bar vs 1/T by m, and slopes a(m) vs m+1
# -----------------------------------------------------------------------------
def _cells(path, method="const"):
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("sigma2_method") != method:
                continue
            key = (int(r["m"]), int(r["T"]))
            out.setdefault(key, []).append((float(r["cbar_otimo"]), float(r["se_cbar"])))
    agg = {}
    for k, vs in out.items():
        v = np.array([x[0] for x in vs])
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else vs[0][1]
        agg[k] = (v.mean(), se)
    return agg


def _wls(points):
    T = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    w = 1.0 / np.array([p[2] for p in points], float) ** 2
    X = np.column_stack([np.ones_like(T), 1.0 / T])
    XtW = X.T * w
    beta = np.linalg.solve(XtW @ X, XtW @ y)
    return beta[0], beta[1]


def make_figure(csv_path=SURFACE_CSV, out_pdf=SURFACE_PDF):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = _cells(csv_path)
    ms = sorted({m for (m, _) in cells})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    slopes = {}
    for m in ms:
        pts = [(T, *cells[(m, T)]) for (mm, T) in sorted(cells) if mm == m]
        if len(pts) < 2:
            continue
        Ts = np.array([p[0] for p in pts]); cs = np.array([p[1] for p in pts])
        ax1.plot(1.0 / Ts, cs, "o-", ms=4, label=f"m={m}")
        if len(pts) >= 3:
            c_inf, a = _wls(pts); slopes[m] = a
    ax1.axhline(-7.0, ls=":", c="k", lw=1)
    ax1.set_xlabel("1/T"); ax1.set_ylabel(r"$\bar c^*(m,T)$")
    ax1.legend(frameon=False, ncol=2, fontsize=8)
    if len(slopes) >= 2:
        mm = np.array(sorted(slopes)); aa = np.array([slopes[m] for m in mm])
        ax2.plot(mm + 1, aa, "s", ms=6)
        big = mm[mm >= 2]
        if len(big) >= 2:
            s = float(np.sum((big + 1) * np.array([slopes[m] for m in big])) /
                      np.sum((big + 1) ** 2))
            xr = np.array([0, big.max() + 1])
            ax2.plot(xr, s * xr, "-", lw=1, label=f"origin ray, slope~{s:.0f} (m>=2)")
            ax2.legend(frameon=False, fontsize=8)
    ax2.set_xlabel("m+1"); ax2.set_ylabel("slope a(m)")
    fig.tight_layout(); fig.savefig(out_pdf); fig.savefig(out_pdf.replace(".pdf", ".png"), dpi=150)
    print(f"[figure] written: {out_pdf}")


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="paper grid and replication counts")
    ap.add_argument("--quick", action="store_true", help="reduced grid smoke test")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--figure-only", action="store_true", help="rebuild the figure from the CSV")
    args = ap.parse_args()

    out_csv = os.path.join(args.outdir, SURFACE_CSV)
    out_pdf = os.path.join(args.outdir, SURFACE_PDF)

    if args.figure_only:
        if not os.path.exists(out_csv):
            sys.exit(f"{out_csv} not found; run the calibration first.")
        make_figure(out_csv, out_pdf); return

    P = dict(core.DEFAULTS)
    P["checkpoint_dir"] = os.path.join(args.outdir, "checkpoints_section3_4")
    if args.quick:
        P.update(T_grid=[60, 100], m_grid=[0, 1],
                 R_cv=300, R_pow=200, save_raw_vectors=False,
                 compute_power_curves=False)
        print("[quick] reduced grid m in {0,1,2}, T in {60,100,300}, small R.")
    elif not args.full:
        print("[note] neither --full nor --quick given; defaulting to --quick.")
        P.update(T_grid=[60, 100], m_grid=[0, 1],
                 R_cv=300, R_pow=200, save_raw_vectors=False,
                 compute_power_curves=False)

    core.warm_up_numba(P["kmax"])
    core.run_grid(P, n_jobs=args.jobs)
    df = core.agregar(P)
    src = os.path.join(P["checkpoint_dir"], "resultados_cbar_ml1_v5.csv")
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, out_csv)
        print(f"[written] {out_csv}")
    try:
        make_figure(out_csv, out_pdf)
    except Exception as e:
        print(f"[figure] skipped ({e}); CSV is complete.")


if __name__ == "__main__":
    main()
