# -*- coding: utf-8 -*-
"""
================================================================================
replicate_section3_4.py -- Sections 3-4: finite-sample calibration surface and
    the null-law Monte Carlo that backs Figure 1.
================================================================================

This is a COMPUTE module: it writes CSV artifacts and never draws figures.
All plotting lives in generate_figures.py, which reads the CSVs written here.

WHAT IT PRODUCES
    cbar_surface.csv        (--full / --quick)
        The calibration surface: for every (m,T) cell and each long-run-variance
        method, the tangency c-bar*(m,T), its standard error, and the 1/2.5/5/10%
        critical values of MZa, MSB, MZt, PT, MPT. This CSV is the INPUT to
        replicate_section5.py, replicate_section6.py, generate_figures.py, and
        run_model_lb.py.

    limiting_density.csv    (--limiting-density)
        The processed plotting data behind Figure 1 (the null MZ_t law across m).
        Long format: columns [panel, T, m, x, y]. Two panel kinds:
          panel="density" : Gaussian-KDE density y at abscissa x, for
                            (T=300, m=0..3) -> Fig. 1(a) and (T=60, m=0..3)
                            -> Fig. 1(c);
          panel="ecdf"    : empirical CDF y at abscissa x, for (T=300, m=0..3)
                            -> Fig. 1(b), the rejection-tail panel.
        The heavy Monte Carlo (R draws of the null MZ_t per cell) and the KDE/ECDF
        reduction happen HERE; generate_figures.py only draws the stored lines.

METHOD
    Surface: for each (m,T), c-bar* is the c-bar at which the local power of the
    feasible point-optimal statistic crosses 0.50 (the ERS 50%-power tangency),
    interpolated on a c-bar grid; the critical values are the lower percentiles
    of the null (driftless random walk) law in the exact break configuration.
    Figure-1 law: R replicates of MZ_t under H0 (c=0) for Model LB with m
    equispaced breaks, GLS-detrended at the cell's c-bar*, difference-based
    sigma^2. Before writing, an integrity GATE compares the simulated 5% quantile
    to the tabulated MZ_t CV5 (same surface CSV), cell by cell, in combined Monte
    Carlo standard-error units; it aborts on a systematic mismatch. Every routine
    runs through the single kernel in mlb_core.py.

USAGE
    python replicate_section3_4.py --full               # paper surface (hours)
    python replicate_section3_4.py --quick              # surface smoke test
    python replicate_section3_4.py --limiting-density   # Fig. 1 data (needs the surface CSV; ~minutes at R=50,000)
    python replicate_section3_4.py --limiting-density --quick   # Fig. 1 data, reduced R
    python replicate_section3_4.py --jobs 8 --outdir .

DETERMINISM & COST
    Deterministic given the replication counts and seeds (config_seed in
    mlb_core for the surface; --seed here for the null law). The full surface
    grid (m=0..5, T up to 300) is ~400 cells; the Figure-1 law is 8 cells
    (T in {60,300} x m in {0,1,2,3}) at R draws each. Wall time scales with --jobs
    for the surface; the null law is single-process but numba-compiled.

REQUIRES
    mlb_core.py; numpy, numba, joblib (surface parallelism). No matplotlib.
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
LIMITING_DENSITY_CSV = "limiting_density.csv"

# ---- Figure-1 null-law configuration (matches the manuscript) ----------------
T_LARGE, T_SHORT = 300, 60          # panels (a)/(b) at large T; panel (c) at short T
M_LIST = [0, 1, 2, 3]               # break counts overlaid in Figure 1
R_DENSITY = 50_000                  # production draws per (T,m) cell
R_DENSITY_QUICK = 2_000             # smoke-test draws
DENSITY_SEED = 20260701             # base seed; per-cell seed = base + 1000*T + m
ALPHA = 0.05                        # nominal level anchoring the gate/panel (b)
DENS_GRID = np.linspace(-5.0, 1.5, 400)      # abscissa for the KDE density panels
ECDF_GRID = np.linspace(-3.5, -1.0, 250)     # abscissa for the rejection-tail ECDF


# =============================================================================
# 1. CALIBRATION SURFACE  (Sections 3-4)  ->  cbar_surface.csv
# =============================================================================
def build_surface(args):
    """Run the calibration grid through mlb_core and write cbar_surface.csv."""
    P = dict(core.DEFAULTS)
    P["checkpoint_dir"] = os.path.join(args.outdir, "checkpoints_section3_4")
    if args.quick:
        P.update(T_grid=[60, 100], m_grid=[0, 1],
                 R_cv=300, R_pow=200, save_raw_vectors=False,
                 compute_power_curves=False)
        print("[quick] reduced surface grid (m in {0,1}, T in {60,100}), small R.")
    elif not args.full:
        print("[note] neither --full nor --quick given; defaulting to --quick surface.")
        P.update(T_grid=[60, 100], m_grid=[0, 1],
                 R_cv=300, R_pow=200, save_raw_vectors=False,
                 compute_power_curves=False)

    core.warm_up_numba(P["kmax"])
    core.run_grid(P, n_jobs=args.jobs)
    core.aggregate(P)
    out_csv = os.path.join(args.outdir, SURFACE_CSV)
    src = os.path.join(P["checkpoint_dir"], "cbar_surface.csv")
    if os.path.exists(src):
        import shutil
        shutil.copyfile(src, out_csv)
        print(f"[written] {out_csv}")
    else:
        print(f"[warn] {src} not produced; nothing written to {out_csv}.")


# =============================================================================
# 2. FIGURE-1 NULL LAW  (--limiting-density)  ->  limiting_density.csv
# =============================================================================
def load_surface_cells(path, method="const"):
    """Aggregate the surface CSV over lambda configurations for the given
    sigma^2 method. Returns {(m,T): dict(cbar, cv5, cv5_se, n)} where cbar is the
    mean tangency and cv5/cv5_se are the mean tabulated MZ_t 5% critical value
    and its Monte Carlo standard error (used by the integrity gate)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Surface CSV not found at {path}. Run --full (or --quick) first.")
    from collections import defaultdict
    buckets = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("sigma2_method", "const") != method:
                continue
            m, T = int(r["m"]), int(r["T"])
            cv5se = r.get("mzt_cv5_se", "")
            buckets[(m, T)].append((
                float(r["cbar_otimo"]),
                float(r["mzt_cv5"]),
                float(cv5se) if cv5se not in ("", "nan") else np.nan,
            ))
    cells = {}
    for key, rows in buckets.items():
        arr = np.array(rows)
        cells[key] = dict(cbar=float(np.mean(arr[:, 0])),
                          cv5=float(np.mean(arr[:, 1])),
                          cv5_se=float(np.nanmean(arr[:, 2])),
                          n=len(rows))
    return cells


def simulate_mzt_null(T, m, cbar, R, seed):
    """Draw R replicates of MZ_t under H0 (c=0) for Model LB with m equispaced
    breaks, GLS-detrended at cbar, difference-based sigma^2. The RNG stays in
    this orchestrator (eps precomputed), matching the calibration convention.
    Returns the finite draws (cells that fail the kernel guard are dropped)."""
    lams = [(j + 1) / (m + 1) for j in range(m)]
    bp = core.break_pos_from_lambdas(T, lams)
    Z = core.build_z_nb(T, bp)
    rng = np.random.default_rng(seed)
    out = np.empty(R)
    kept = 0
    for _ in range(R):
        eps = rng.standard_normal(T)
        y = core.gen_dgp_nb(T, bp, 0.0, 0.0, eps)          # c=0, beta_scale=0 -> y=u
        _, _, mzt, _, _, ok = core.mstats_nb(y, Z, cbar, 0, 12)
        if ok > 0.5:
            out[kept] = mzt
            kept += 1
    return out[:kept]


def gate_against_surface(sims, cells, k_sigma=3.0):
    """Integrity gate: compare each cell's simulated 5% quantile of MZ_t to the
    tabulated CV5. The two are independent Monte Carlo estimators of the same
    population quantile, so the standard error of their difference combines both:
        se_delta = sqrt(se_fig^2 + se_prod^2),
    with se_prod the tabulated MC SE and se_fig the delta-method SE of the
    empirical quantile. Passing criterion for the 8-cell family (robust to the
    known m=0 seed-averaging borderline case):
        (i)  the MEDIAN standardized deviation is < 1  (no systematic bias), AND
        (ii) at most ONE cell exceeds k_sigma.
    Returns (passed, rows) with rows = (T, m, q05, cv5, dev, nse, n, ok)."""
    rows, devs = [], []
    for (T, m), draws in sims.items():
        if (m, T) not in cells:
            continue
        q05 = float(np.percentile(draws, 100 * ALPHA))
        cv5 = cells[(m, T)]["cv5"]
        lo = np.percentile(draws, 100 * ALPHA - 2)
        hi = np.percentile(draws, 100 * ALPHA + 2)
        dens = 0.04 / max(hi - lo, 1e-6)
        se_fig = np.sqrt(ALPHA * (1 - ALPHA) / len(draws)) / max(dens, 1e-6)
        se_prod = cells[(m, T)].get("cv5_se", np.nan)
        se_delta = np.sqrt(se_fig ** 2 + (se_prod ** 2 if np.isfinite(se_prod) else 0.0))
        dev = abs(q05 - cv5)
        nse = dev / se_delta if se_delta > 0 else np.inf
        devs.append(nse)
        rows.append((T, m, q05, cv5, dev, nse, len(draws), nse <= k_sigma))
    median_sig = float(np.median(devs)) if devs else np.inf
    n_over = sum(1 for s in devs if s > k_sigma)
    passed = (median_sig < 1.0) and (n_over <= 1)
    return passed, rows


def _kde(x, grid):
    """Gaussian kernel density estimate with a Silverman bandwidth."""
    x = x[np.isfinite(x)]
    s = min(x.std(ddof=1), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.349)
    bw = 0.9 * s * len(x) ** (-0.2)
    dens = np.mean(np.exp(-0.5 * ((grid[:, None] - x[None, :]) / bw) ** 2), axis=1)
    return dens / (bw * np.sqrt(2 * np.pi))


def build_limiting_density(args):
    """Simulate the null MZ_t law, gate it against the surface, reduce it to
    KDE densities and rejection-tail ECDFs, and write limiting_density.csv."""
    surface_csv = os.path.join(args.outdir, SURFACE_CSV)
    cells = load_surface_cells(surface_csv)
    R = R_DENSITY_QUICK if args.quick else R_DENSITY
    seed = args.seed

    print(core.warm_up_numba(12))
    sims = {}
    for T in (T_LARGE, T_SHORT):
        for m in M_LIST:
            if (m, T) not in cells:
                raise KeyError(f"surface CSV lacks cell (m={m}, T={T}); cannot anchor Figure 1.")
            cbar = cells[(m, T)]["cbar"]
            sims[(T, m)] = simulate_mzt_null(T, m, cbar, R, seed + 1000 * T + m)
            print(f"  simulated (T={T}, m={m}): n={len(sims[(T, m)])}, cbar={cbar:.3f}")

    passed, rows = gate_against_surface(sims, cells)
    print("\n[gate] simulated 5% quantile vs tabulated MZt CV5 (combined MC SE):")
    print("   T   m   q05_sim   CV5_csv    dev   (SE units)     n    verdict")
    for T, m, q05, cv5, dev, nse, n, ok in rows:
        print(f"  {T:>3} {m:>2}   {q05:+7.3f}  {cv5:+7.3f}  {dev:6.3f}  {nse:7.2f}   "
              f"{n:>6}   {'ok' if ok else 'CHECK'}")
    if not passed:
        msg = ("[gate] FAILED: simulated quantiles inconsistent with the tabulated "
               "CVs beyond Monte Carlo error (systematic median offset or >1 cell "
               "past 3 SE) -- a kernel/surface mismatch. Not writing the figure data.")
        if not args.quick:
            raise RuntimeError(msg)
        print(msg + "  [continuing: --quick]")
    else:
        print("[gate] PASS: no systematic bias; isolated borderline cells tolerated.")

    out_csv = os.path.join(args.outdir, LIMITING_DENSITY_CSV)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "T", "m", "x", "y"])
        for T in (T_LARGE, T_SHORT):          # density panels (a) T=300, (c) T=60
            for m in M_LIST:
                dens = _kde(sims[(T, m)], DENS_GRID)
                for x, y in zip(DENS_GRID, dens):
                    w.writerow(["density", T, m, f"{x:.5f}", f"{y:.6f}"])
        for m in M_LIST:                       # rejection-tail ECDF panel (b), T=300
            draws = np.sort(sims[(T_LARGE, m)])
            for x in ECDF_GRID:
                F = float(np.mean(draws <= x))
                w.writerow(["ecdf", T_LARGE, m, f"{x:.5f}", f"{F:.6f}"])
    print(f"[written] {out_csv}")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--full", action="store_true", help="paper surface grid and replication counts")
    ap.add_argument("--quick", action="store_true", help="reduced grid / reduced R smoke test")
    ap.add_argument("--limiting-density", action="store_true",
                    help="build Figure-1 null-law data from the surface CSV")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers for the surface")
    ap.add_argument("--outdir", default=".", help="directory for inputs/outputs")
    ap.add_argument("--seed", type=int, default=DENSITY_SEED,
                    help="base seed for the Figure-1 null law")
    args = ap.parse_args()

    if args.limiting_density:
        build_limiting_density(args)
    else:
        build_surface(args)


if __name__ == "__main__":
    main()
