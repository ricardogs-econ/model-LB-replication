#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ar1_sizepower_recal.py -- AR(1) size/power and c-bar recalibration (Model LB).

Producer for the paper's robustness section (section 5):

  NUMBER / OBJECT      PRODUCER (this file)         ARTEFACT
  tab:ar1-sizepower    run_ar1() main path          <outdir>/ar1_size_power.csv
                                                     <outdir>/tab_ar1_sizepower.tex
  stab:recal           run_ar1(--recalibrate)       <outdir>/ar1_recalibration.csv

Frozen golden reference: output/golden/calibration_robustness/ar1_size_power.csv,
ar1_recalibration.csv.

Convention (identical for every cell): the DGP is the Model LB process
u_t = (1 + c/T) u_{t-1} + eps_t (u_0 = 0, so u_1 = eps_1); the statistic is the
point-optimal P_T; the c-bar* anchor is read from the frozen calibration surface
(inputs/cbar_surface.csv). Constants: M_LIST=[0,1,2], T_LIST=[30,60,100],
RHO_LIST=[0.3,0.5,0.6], METHODS=[const,maic], R_ANCHOR_CV=R_ANCHOR_POW=8000,
R_EVAL=10000, R_RECAL_CV=R_RECAL_POW=4000, SEED_BASE=20260624. The i.i.d. anchor
power column is capped at 2000 replications by design.

This is the AR(1) size/power + recalibration core on the numba kernel
(from src.mlb import kernel_numba), imported and never reimplemented. Its
ar1+recal core is the verified production logic; the constants above are the
values behind the printed table.

Only the ar1 subcommand (with --recalibrate) is exposed: the other robustness
experiments (oracle -> tab:serial, trimming -> tab:trimming, power -> tab:power)
are not objects of this door.

USAGE
    python replicate.py robustness           # --quick (read the frozen golden)
    python replicate.py robustness --full    # regenerate ar1 + recalibration
    python section_replications/ar1_sizepower_recal.py ar1 [--recalibrate] [--quick]
    python section_replications/ar1_sizepower_recal.py --selftest   # pure logic
"""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.mlb import kernel_numba as kn  # the ONE production kernel; never reimplemented here
from src.mlb import paths

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False


# ------------------------------------------------------------------------------
# Shared configuration and helpers
# ------------------------------------------------------------------------------
TARGET_POWER = 0.50
TRIM         = 0.15
# The frozen calibration surface, resolved through the single resolver
# (inputs/cbar_surface.csv). produced_outside, regenerable_here=false.
CSV_DEFAULT  = str(paths.resolve("cbar_surface"))

_METHOD_CODE = kn._METHOD_CODE
KMAX = kn.DEFAULTS['kmax']


def make_ar1_eps(rng, T, rho):
    """AR(1) innovations eps_t = rho*eps_{t-1} + e_t, e_t ~ N(0,1).
    rho=0 returns pure i.i.d. e (== the production DGP innovations)."""
    e = rng.standard_normal(T)
    if rho == 0.0:
        return e
    eps = np.empty(T)
    eps[0] = e[0]
    for t in range(1, T):
        eps[t] = rho * eps[t - 1] + e[t]
    return eps


def equi_break_fractions(m, trim=TRIM):
    """m equally-spaced break fractions in [trim, 1-trim] (the production
    equispaced convention). m=0 -> ()."""
    if m == 0:
        return ()
    return tuple(round(trim + (i + 1) * (1 - 2 * trim) / (m + 1), 4) for i in range(m))


def _interp_crossing(cb, pw, target):
    """Locate the interpolated crossing pw == target on a (c-bar, power) curve
    sorted by c-bar. Returns (i, frac) of the first sign change, or None.
    Removes the 0.5 grid quantization (Perron-style tangency interpolation)."""
    for i in range(len(cb) - 1):
        p0, p1 = pw[i], pw[i + 1]
        if (p0 - target) * (p1 - target) <= 0 and p1 != p0:
            return i, (target - p0) / (p1 - p0)
    return None


def _parallel(tasks, fn, jobs, use_parallel):
    if use_parallel and _HAS_JOBLIB and jobs != 1 and len(tasks) > 1:
        return Parallel(n_jobs=jobs)(delayed(fn)(*t) for t in tasks)
    return [fn(*t) for t in tasks]


def _ensure_outdir(outdir):
    os.makedirs(outdir, exist_ok=True)
    return outdir


# ==============================================================================
# AR(1) SIZE AND POWER + RECALIBRATION  (tab:ar1-sizepower, stab:recal)
# ==============================================================================
AR1 = dict(
    M_LIST=[0, 1, 2], T_LIST=[30, 60, 100], RHO_LIST=[0.3, 0.5, 0.6],
    METHODS=["const", "maic"],
    CBAR_GRID=np.round(np.arange(-16.0, -3.4, 0.5), 2),
    R_ANCHOR_CV=8000, R_ANCHOR_POW=8000, R_EVAL=10000,
    R_RECAL_CV=4000, R_RECAL_POW=4000, SEED_BASE=20260624,
)


def _ar1_seed(m, T, rho, method):
    msig = 1 if method == "const" else 2
    return (AR1['SEED_BASE'] + T * 1_000_000 + m * 100_000
            + int(round(rho * 1000)) * 17 + msig * 3) % (2**63 - 1)


def load_cbar_csv(csv_path):
    """Load c-bar*(m,T,method) from the production CSV. {} on failure."""
    if not os.path.exists(csv_path):
        return {}
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        out = {}
        for method in ['const', 'maic']:
            sub = df[df.sigma2_method == method]
            for (m, T), v in sub.groupby(['m', 'T'])['cbar_otimo'].mean().to_dict().items():
                out[(int(m), int(T), method)] = float(v)
        return out
    except Exception as e:
        print(f"  [warn] could not load calibration CSV: {e}")
        return {}


def _ar1_cv_and_power_at(m, T, mc, cbar, seed, r_cv, r_pow, rho_cv, rho_pow,
                         at_c_null=0.0, at_c_alt=None):
    """CV5 under H0 (c=at_c_null) and power at H1 (c=at_c_alt or cbar), paired
    common random numbers, AR(1) rho for each stream. Returns (cv5, power)."""
    lambdas = equi_break_fractions(m)
    bp = kn.break_pos_from_lambdas(T, lambdas)
    Z = kn.build_z_nb(T, bp)
    if at_c_alt is None:
        at_c_alt = cbar
    pt_h0 = np.empty(r_cv)
    for r in range(r_cv):
        rng = np.random.default_rng((seed + r) % (2**63 - 1))
        eps = make_ar1_eps(rng, T, rho_cv)
        y = kn.gen_dgp_nb(T, bp, at_c_null, 0.0, eps)
        _, _, _, pt, _, ok = kn.mstats_nb(y, Z, cbar, mc, KMAX)
        pt_h0[r] = pt if ok > 0.5 else np.nan
    pt_h0 = pt_h0[np.isfinite(pt_h0)]
    if len(pt_h0) < r_cv * 0.5:
        return np.nan, np.nan
    cv5 = float(np.percentile(pt_h0, 5))
    rej = nval = 0
    for r in range(r_pow):
        rng = np.random.default_rng((seed + 10**6 + r) % (2**63 - 1))
        eps = make_ar1_eps(rng, T, rho_pow)
        y = kn.gen_dgp_nb(T, bp, at_c_alt, 0.0, eps)
        _, _, _, pt, _, ok = kn.mstats_nb(y, Z, cbar, mc, KMAX)
        if ok > 0.5 and np.isfinite(pt):
            nval += 1
            rej += (pt < cv5)
    power = rej / nval if nval > 0 else np.nan
    return cv5, power


def ar1_anchor_iid(m, T, method, seed, cbar_lut, r_cv, r_pow, cbar_grid):
    """(cbar_star, cv5, power_iid). Uses the CSV c-bar* if present (only CV5 is
    recomputed -- cheap); otherwise a full i.i.d. tangency grid search."""
    mc = _METHOD_CODE[method]
    if (m, T, method) in cbar_lut:
        cbar_star = cbar_lut[(m, T, method)]
        cv5, power = _ar1_cv_and_power_at(m, T, mc, cbar_star, seed, r_cv,
                                          min(r_pow, 2000), 0.0, 0.0)
        return float(cbar_star), cv5, (power if np.isfinite(power) else TARGET_POWER)

    # full i.i.d. grid search (no CSV)
    cb_l, cv_l, pw_l = [], [], []
    for cbar in cbar_grid:
        cv5, power = _ar1_cv_and_power_at(m, T, mc, cbar, seed, r_cv, r_pow, 0.0, 0.0)
        if np.isfinite(cv5) and np.isfinite(power):
            cb_l.append(cbar); cv_l.append(cv5); pw_l.append(power)
    if not cb_l:
        return np.nan, np.nan, np.nan
    o = np.argsort(cb_l)
    cb, cv, pw = np.array(cb_l)[o], np.array(cv_l)[o], np.array(pw_l)[o]
    cr = _interp_crossing(cb, pw, TARGET_POWER)
    if cr is not None:
        i, f = cr
        return (float(cb[i] + f * (cb[i+1] - cb[i])),
                float(cv[i] + f * (cv[i+1] - cv[i])), TARGET_POWER)
    idx = int(np.argmin(np.abs(pw - TARGET_POWER)))
    return float(cb[idx]), float(cv[idx]), float(pw[idx])


def ar1_size_power_at_rho(m, T, method, rho, cbar_star, cv5, seed, r_eval):
    """Empirical size (H0) and power (H1) at rho, at the iid-calibrated (cbar*,cv5)."""
    mc = _METHOD_CODE[method]
    bp = kn.break_pos_from_lambdas(T, equi_break_fractions(m))
    Z = kn.build_z_nb(T, bp)

    def _rate(c_dgp, seed_off):
        rej = nval = 0
        for r in range(r_eval):
            rng = np.random.default_rng((seed + seed_off + r) % (2**63 - 1))
            eps = make_ar1_eps(rng, T, rho)
            y = kn.gen_dgp_nb(T, bp, c_dgp, 0.0, eps)
            _, _, _, pt, _, ok = kn.mstats_nb(y, Z, cbar_star, mc, KMAX)
            if ok > 0.5 and np.isfinite(pt) and np.isfinite(cv5):
                nval += 1
                rej += (pt < cv5)
        return rej / nval if nval > 0 else np.nan

    return _rate(0.0, 0), _rate(cbar_star, 5 * 10**6)


def ar1_recalibrate(m, T, method, rho, seed, r_cv, r_pow, cbar_grid):
    """Full tangency search WITH s^2_AR estimated at rho>0. Returns cbar_AR."""
    mc = _METHOD_CODE[method]
    cb_l, pw_l = [], []
    for cbar in cbar_grid:
        cv5, power = _ar1_cv_and_power_at(m, T, mc, cbar, seed, r_cv, r_pow, rho, rho)
        if np.isfinite(cv5) and np.isfinite(power):
            cb_l.append(cbar); pw_l.append(power)
    if not cb_l:
        return np.nan
    o = np.argsort(cb_l)
    cb, pw = np.array(cb_l)[o], np.array(pw_l)[o]
    cr = _interp_crossing(cb, pw, TARGET_POWER)
    if cr is not None:
        i, f = cr
        if cb[i] == cb[0] or cb[i+1] == cb[-1]:
            print(f"  [!] recal c-bar* at grid EDGE (m={m},T={T},rho={rho})")
        return float(cb[i] + f * (cb[i+1] - cb[i]))
    idx = int(np.argmin(np.abs(pw - TARGET_POWER)))
    return float(cb[idx])


def run_ar1(args, outdir):
    cfg = dict(AR1)
    m_list, t_list, rho_list, methods = cfg['M_LIST'], cfg['T_LIST'], cfg['RHO_LIST'], cfg['METHODS']
    cbar_grid = cfg['CBAR_GRID']
    r_anchor_cv, r_anchor_pow, r_eval = cfg['R_ANCHOR_CV'], cfg['R_ANCHOR_POW'], cfg['R_EVAL']
    if args.quick:
        m_list, t_list, rho_list, methods = [0, 1], [60], [0.5], ["const"]
        r_anchor_cv = r_anchor_pow = r_eval = 500
        cbar_grid = np.round(np.arange(-12.0, -4.9, 1.0), 2)
        print("[ar1][quick] reduced grid/reps.\n")

    cbar_lut = load_cbar_csv(args.csv)
    print(f"[ar1] anchor c-bar* from CSV: {len(cbar_lut)} entries "
          f"({'ok' if cbar_lut else 'MISSING -> local grid search'})")

    t0 = time.time()
    anchor_tasks = [(m, T, method) for m in m_list for T in t_list for method in methods]

    def _anchor(m, T, method):
        seed = _ar1_seed(m, T, 0.0, method)
        cs, cv5, pw = ar1_anchor_iid(m, T, method, seed, cbar_lut,
                                     r_anchor_cv, r_anchor_pow, cbar_grid)
        return (m, T, method), (cs, cv5, pw)

    print("\n[ar1] Experiment 0: i.i.d. anchor ...")
    anchors = dict(_parallel(anchor_tasks, _anchor, args.jobs, not args.quick))
    for (m, T, method), (cs, cv5, pw) in sorted(anchors.items()):
        print(f"  m={m} T={T:>3} {method:>5}: cbar*={cs:>6.2f} CV5%={cv5:>7.3f} power={pw:.3f}")

    sp_tasks = [(m, T, method, rho)
                for (m, T, method), (cs, _, _) in anchors.items() if np.isfinite(cs)
                for rho in rho_list]

    def _sp(m, T, method, rho):
        cs, cv5, _ = anchors[(m, T, method)]
        seed = _ar1_seed(m, T, rho, method)
        se, pe = ar1_size_power_at_rho(m, T, method, rho, cs, cv5, seed, r_eval)
        return (m, T, method, rho), (se, pe)

    print("\n[ar1] Experiments 1+2: size and power under AR(1) ...")
    sp_res = dict(_parallel(sp_tasks, _sp, args.jobs, not args.quick))
    for (m, T, method, rho), (se, pe) in sorted(sp_res.items()):
        piid = anchors[(m, T, method)][2]
        print(f"  m={m} T={T:>3} {method:>5} rho={rho:.1f}: size={se:.3f} "
              f"power={pe:.3f} (anchor {piid:.3f})")

    # CSV
    csv_path = os.path.join(outdir, "ar1_size_power.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "T", "method", "rho", "cbar_iid", "cv5_iid",
                    "power_iid", "size_emp", "power_emp"])
        for (m, T, method, rho), (se, pe) in sorted(sp_res.items()):
            cs, cv5, piid = anchors[(m, T, method)]
            w.writerow([m, T, method, rho, f"{cs:.2f}", f"{cv5:.4f}", f"{piid:.4f}",
                        f"{se:.4f}" if np.isfinite(se) else "nan",
                        f"{pe:.4f}" if np.isfinite(pe) else "nan"])
    print(f"[ar1] written: {csv_path}")

    # LaTeX tab:ar1-sizepower
    rho_focus = 0.5 if 0.5 in rho_list else rho_list[len(rho_list) // 2]
    tex_path = os.path.join(outdir, "tab_ar1_sizepower.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Empirical size and power of $P_T$ at "
                f"$\\rho={rho_focus:g}$ AR(1) serial correlation, using the "
                "i.i.d.-calibrated $\\bar c$ and $5\\%$ critical value (no "
                "recalibration), with $s^2_{\\mathrm{AR}}$ \\emph{estimated} by "
                "the difference-based (\\texttt{const}) and autoregressive MAIC "
                "(\\texttt{maic}) methods.}\n\\label{tab:ar1-sizepower}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("& & \\multicolumn{2}{c}{\\texttt{const}} & "
                "\\multicolumn{2}{c}{\\texttt{maic}} & \\\\\n")
        f.write("$T$ & $m$ & size & power & size & power & "
                "power$_{\\mathrm{iid}}$ \\\\\n\\midrule\n")
        for T in t_list:
            for m in m_list:
                row = [str(T), str(m)]; p_iid = None
                for method in ["const", "maic"]:
                    key = (m, T, method, rho_focus)
                    if key in sp_res:
                        se, pe = sp_res[key]; row += [f"{se:.3f}", f"{pe:.3f}"]
                        p_iid = anchors[(m, T, method)][2]
                    else:
                        row += ["--", "--"]
                row.append(f"{p_iid:.3f}" if p_iid is not None else "--")
                f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        f.write("\\par\\medskip\\footnotesize "
                f"$R_{{\\mathrm{{eval}}}}={r_eval}$ replications; MC standard "
                "error of a proportion near $0.5$ is approx.\\ $\\sqrt{0.25/R}"
                f"\\approx{np.sqrt(0.25/r_eval):.3f}$. The i.i.d.\\ anchor "
                "(power$_{\\mathrm{iid}}$) is the power at $\\rho=0$ used to "
                "calibrate $\\bar c$ and the critical value.\n\\end{table}\n")
    print(f"[ar1] written: {tex_path}")

    # stab:recal -- recalibration under AR(1)
    if getattr(args, "recalibrate", False):
        print("\n[ar1] Experiment 3: recalibration under AR(1) ...")
        recal_rho = args.recal_rho or [0.5]
        recal_methods = args.recal_method or ["const"]
        r_recal_cv, r_recal_pow = cfg['R_RECAL_CV'], cfg['R_RECAL_POW']
        if args.quick:
            r_recal_cv = r_recal_pow = 500
        rc_tasks = [(m, T, method, rho) for m in m_list for T in t_list
                    for method in recal_methods for rho in recal_rho]

        def _recal(m, T, method, rho):
            seed = _ar1_seed(m, T, rho, method) + 999
            ca = ar1_recalibrate(m, T, method, rho, seed, r_recal_cv, r_recal_pow, cbar_grid)
            ci = anchors.get((m, T, method), (np.nan,))[0]
            return (m, T, method, rho), (ci, ca)

        recal = dict(_parallel(rc_tasks, _recal, args.jobs, not args.quick))
        rc_csv = os.path.join(outdir, "ar1_recalibration.csv")
        with open(rc_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["m", "T", "method", "rho", "cbar_iid", "cbar_AR", "delta"])
            for (m, T, method, rho), (ci, ca) in sorted(recal.items()):
                d = ca - ci if (np.isfinite(ca) and np.isfinite(ci)) else float('nan')
                w.writerow([m, T, method, rho, f"{ci:.2f}", f"{ca:.2f}",
                            f"{d:+.2f}" if np.isfinite(d) else "nan"])
                print(f"  m={m} T={T:>3} {method:>5} rho={rho:.1f}: "
                      f"cbar_iid={ci:.2f} cbar_AR={ca:.2f} delta={d:+.2f}")
        print(f"[ar1] written: {rc_csv}")

    print(f"[ar1] done in {(time.time()-t0)/60:.1f} min")


# ==============================================================================
# MAIN
# ==============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="AR(1) size/power and c-bar recalibration (Model LB). "
                     "AR(1) size/power and c-bar recalibration for Model LB.")
    ap.add_argument("--selftest", action="store_true",
                    help="pure-logic tests (no numba/production needed)")
    sub = ap.add_subparsers(dest="cmd")
    spx = sub.add_parser("ar1")
    spx.add_argument("--quick", action="store_true", help="fast smoke test")
    spx.add_argument("--jobs", type=int, default=-1)
    spx.add_argument("--outdir", default="ar1_sizepower_recal_out")
    spx.add_argument("--csv", default=CSV_DEFAULT, help="calibration CSV (ar1 anchor)")
    spx.add_argument("--recalibrate", action="store_true")
    spx.add_argument("--recal-rho", type=float, nargs="+", default=[0.5])
    spx.add_argument("--recal-method", choices=["const", "maic"],
                     nargs="+", default=["const"])
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.cmd:
        ap.print_help()
        return 0

    print(kn.warm_up_numba())
    print(f"[ar1_sizepower_recal] kernel: src/mlb/kernel_numba.py; joblib="
          f"{'yes' if _HAS_JOBLIB else 'no'}; jobs={args.jobs}")
    outdir = _ensure_outdir(args.outdir)

    t0 = time.time()
    run_ar1(args, outdir)
    print(f"\n[ar1_sizepower_recal] DONE in {(time.time()-t0)/60:.1f} min -> {outdir}/")
    return 0


# ------------------------------------------------------------------------------
# SELF-TEST (pure logic; no numba/production import)
# ------------------------------------------------------------------------------
def _selftest():
    print("=" * 70)
    print("SELF-TEST ar1_sizepower_recal -- pure logic (no numba)")
    print("=" * 70)
    ok = True

    cb = np.array([-10.0, -9.0, -8.0, -7.0, -6.0])
    pw = np.array([0.95, 0.80, 0.55, 0.30, 0.10])   # crosses 0.5 between -8 and -7
    cr = _interp_crossing(cb, pw, 0.50)
    if cr is None:
        print("  [x] crossing not found"); ok = False
    else:
        i, f = cr
        cstar = cb[i] + f * (cb[i+1] - cb[i])
        exp = -8.0 + (0.50 - 0.55) / (0.30 - 0.55) * 1.0
        if abs(cstar - exp) > 1e-9:
            print(f"  [x] crossing c*={cstar:.4f} != {exp:.4f}"); ok = False
        else:
            print(f"  [OK] interpolated crossing c*={cstar:.4f}")

    rng = np.random.default_rng(1)
    e0 = make_ar1_eps(rng, 5, 0.0)
    rng2 = np.random.default_rng(1)
    e1 = make_ar1_eps(rng2, 5, 0.0)
    if not np.allclose(e0, e1):
        print("  [x] make_ar1_eps rho=0 not reproducible"); ok = False
    rng3 = np.random.default_rng(2); T = 6; rho = 0.5
    eps = make_ar1_eps(rng3, T, rho)
    rng4 = np.random.default_rng(2); e = rng4.standard_normal(T)
    rec = np.empty(T); rec[0] = e[0]
    for t in range(1, T):
        rec[t] = rho * rec[t-1] + e[t]
    if not np.allclose(eps, rec):
        print("  [x] make_ar1_eps recursion wrong"); ok = False
    else:
        print("  [OK] make_ar1_eps: iid reproducible and AR(1) recursion exact")

    bf = equi_break_fractions(2, 0.15)
    if not (len(bf) == 2 and abs(bf[0] - (0.15 + 1*0.7/3)) < 1e-3):
        print(f"  [x] equi_break_fractions wrong: {bf}"); ok = False
    else:
        print(f"  [OK] equi_break_fractions(2)={bf}")

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["m", "T", "sigma2_method", "cbar_otimo"])
            w.writerow([0, 100, "const", -6.97]); w.writerow([1, 100, "const", -7.40])
        lut = load_cbar_csv(p)
        if lut.get((0, 100, "const")) != -6.97:
            print(f"  [x] load_cbar_csv wrong: {lut}"); ok = False
        else:
            print("  [OK] load_cbar_csv reads (m,T,method)->cbar")

    print("\n" + ("SELF-TEST: ALL PASSED" if ok else "SELF-TEST: FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
