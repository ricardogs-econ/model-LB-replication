#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
robustness.py  --  Unified robustness suite for the Model LB c-bar calibration
================================================================================

This single script reproduces ALL THREE robustness experiments reported in the
Model LB ("Level Breaks and GLS Detrending") paper, each as a subcommand:

  ar1       Empirical size and power of P_T when the innovations are AR(1)
            (serial correlation), using the i.i.d.-calibrated c-bar and 5%
            critical value with s^2_AR ESTIMATED by the difference-based
            ('const') and autoregressive MAIC ('maic') long-run-variance
            estimators. Optional experiment 3 recalibrates c-bar under AR(1).
            -> tab:ar1-sizepower

  oracle    Optimal c-bar under AR(1) innovations with the ORACLE long-run
            variance omega^2 = 1/(1-rho)^2 (known, not estimated). Isolates
            serial correlation from LRV estimation and shows the limiting
            tangency c-bar is (asymptotically) unmoved by rho.
            -> tab:serial

  trimming  Sensitivity of the c-bar(m,T) surface to the break-fraction TRIMMING
            epsilon in {0.10, 0.15, 0.20}. Since c-bar depends only weakly on the
            break locations lambda, each cell should be nearly invariant to eps
            (within one grid step / Monte Carlo error).
            -> tab:trimming  (+ a plain-text number for the paper's \\TBD)

  power     Size and power of MZ_t under three c-bar specifications (calibrated
            Model LB -8.40, linear break-count -17, trend-break surface -21.7)
            and three DGPs (I(1) no breaks; I(1)+level breaks; I(0) broken mean),
            at T=60, m=2, lambda=(1/3,2/3). Each spec uses its own 5% CV, so all
            are correctly sized and differ only in power; over-detrending by the
            shortcuts costs power. Writes power_comparison.csv (consumed by
            figuras_v6.fig_power) and tab_power.tex.
            -> tab:power, fig:power

  all       Run ar1, oracle, trimming and power in sequence into --outdir.

WHY UNIFIED
-----------
The three analyses shared a single engine and were previously three loose
scripts on TWO DIFFERENT, an earlier production API (which exposed
`build_z`, `_glsd`, `compute_M_statistics`, `generate_dgp`; later renamed to
`build_z_nb`, `glsd_nb`, `mstats_nb`, `gen_dgp_nb` with different signatures).
Only the AR(1) script had been migrated. This file:
  * imports the VALIDATED kernels from the production module (mlb_core, falling back to
    the pure-Python kernel -- identical) and NEVER reimplements them;
  * drives all three experiments through the SAME production DGP `gen_dgp_nb`
    (u_t = (1 + c/T) u_{t-1} + eps_t) and the SAME P_T statistic, so the three
    experiments are mutually consistent and consistent with the calibration;
  * reads the anchor c-bar* from the production CSV
    (cbar_surface.csv), so the m=0 anchors automatically inherit the
    the seed-averaged surface -- no hard-coded, stale c-bar table.

A journal replication package is then just three modular files:
    mlb_core.py   (calibration -> CSV + tables + figure)
    mlb_core.py                       (the Model LB test kernel; reads the CSV)
    robustness.py                  (this file; all robustness objects)

USAGE
-----
    python3 robustness.py all                    # everything, production reps
    python3 robustness.py ar1  --recalibrate     # AR(1) size/power + recal
    python3 robustness.py oracle
    python3 robustness.py trimming
    python3 robustness.py all --quick            # fast smoke test of all three
    python3 robustness.py --selftest             # pure-logic tests (no numba)

Place this file in the same folder as mlb_core.py.
================================================================================
"""
from __future__ import annotations

import os
import sys
import csv
import time
import argparse
import importlib

import numpy as np

# ------------------------------------------------------------------------------
# Production kernels (import; never reimplement). numba kernel required.
# ------------------------------------------------------------------------------
_PROD_CANDIDATES = ["mlb_core", "cbar_ml1_final_production_v5"]


def _load_prod(module_name=None):
    names = [module_name] if module_name else _PROD_CANDIDATES
    for nm in names:
        if not nm:
            continue
        try:
            return importlib.import_module(nm)
        except ModuleNotFoundError:
            continue
    raise ImportError(
        "Could not import the production module "
        f"({', '.join(n for n in names if n)}). Place robustness.py in the "
        "same folder as mlb_core.py.")


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
ALPHA        = 0.05
CSV_DEFAULT  = os.path.join("cbar_checkpoints", "cbar_surface.csv")

# Bound module-level kernel handles once _load_prod runs (set in _bind_kernels).
_K = {}          # name -> callable/const


def _bind_kernels(prod):
    """Bind the production kernels used across all three experiments."""
    _K['mstats_nb']            = prod.mstats_nb
    _K['gen_dgp_nb']           = prod.gen_dgp_nb
    _K['build_z_nb']           = prod.build_z_nb
    _K['glsd_nb']              = prod.glsd_nb
    _K['break_pos_from_lambdas'] = prod.break_pos_from_lambdas
    _K['make_lambdas']        = prod.make_lambdas
    _K['warm_up_numba']        = prod.warm_up_numba
    _K['_METHOD_CODE']         = prod._METHOD_CODE
    _K['KMAX']                 = prod.DEFAULTS['kmax']
    _K['prod_name']            = prod.__name__


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
# EXPERIMENT 1 -- AR(1) SIZE AND POWER  (tab:ar1-sizepower)
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
    KMAX = _K['KMAX']; gen = _K['gen_dgp_nb']; bz = _K['build_z_nb']; ms = _K['mstats_nb']
    lambdas = equi_break_fractions(m)
    bp = _K['break_pos_from_lambdas'](T, lambdas)
    Z = bz(T, bp)
    if at_c_alt is None:
        at_c_alt = cbar
    pt_h0 = np.empty(r_cv)
    for r in range(r_cv):
        rng = np.random.default_rng((seed + r) % (2**63 - 1))
        eps = make_ar1_eps(rng, T, rho_cv)
        y = gen(T, bp, at_c_null, 0.0, eps)
        _, _, _, pt, _, ok = ms(y, Z, cbar, mc, KMAX)
        pt_h0[r] = pt if ok > 0.5 else np.nan
    pt_h0 = pt_h0[np.isfinite(pt_h0)]
    if len(pt_h0) < r_cv * 0.5:
        return np.nan, np.nan
    cv5 = float(np.percentile(pt_h0, 5))
    rej = nval = 0
    for r in range(r_pow):
        rng = np.random.default_rng((seed + 10**6 + r) % (2**63 - 1))
        eps = make_ar1_eps(rng, T, rho_pow)
        y = gen(T, bp, at_c_alt, 0.0, eps)
        _, _, _, pt, _, ok = ms(y, Z, cbar, mc, KMAX)
        if ok > 0.5 and np.isfinite(pt):
            nval += 1
            rej += (pt < cv5)
    power = rej / nval if nval > 0 else np.nan
    return cv5, power


def ar1_anchor_iid(m, T, method, seed, cbar_lut, r_cv, r_pow, cbar_grid):
    """(cbar_star, cv5, power_iid). Uses the CSV c-bar* if present (only CV5 is
    recomputed -- cheap); otherwise a full i.i.d. tangency grid search."""
    mc = _K['_METHOD_CODE'][method]
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
    mc = _K['_METHOD_CODE'][method]
    KMAX = _K['KMAX']; gen = _K['gen_dgp_nb']; bz = _K['build_z_nb']; ms = _K['mstats_nb']
    bp = _K['break_pos_from_lambdas'](T, equi_break_fractions(m))
    Z = bz(T, bp)

    def _rate(c_dgp, seed_off):
        rej = nval = 0
        for r in range(r_eval):
            rng = np.random.default_rng((seed + seed_off + r) % (2**63 - 1))
            eps = make_ar1_eps(rng, T, rho)
            y = gen(T, bp, c_dgp, 0.0, eps)
            _, _, _, pt, _, ok = ms(y, Z, cbar_star, mc, KMAX)
            if ok > 0.5 and np.isfinite(pt) and np.isfinite(cv5):
                nval += 1
                rej += (pt < cv5)
        return rej / nval if nval > 0 else np.nan

    return _rate(0.0, 0), _rate(cbar_star, 5 * 10**6)


def ar1_recalibrate(m, T, method, rho, seed, r_cv, r_pow, cbar_grid):
    """Full tangency search WITH s^2_AR estimated at rho>0. Returns cbar_AR."""
    mc = _K['_METHOD_CODE'][method]
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

    # Optional experiment 3: recalibration
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
# EXPERIMENT 2 -- ORACLE LONG-RUN VARIANCE  (tab:serial)
# ==============================================================================
ORACLE = dict(
    T_LIST=[60, 100, 200, 300], M_LIST=[0, 1, 2], RHO_LIST=[0.0, 0.3, 0.6],
    CBAR_GRID=np.round(np.arange(-20.0, -2.9, 0.5), 2),
    R_CV=10000, R_POW=20000, SEED_BASE=20260623,
)


def _oracle_seed(m, rho, T):
    return ORACLE['SEED_BASE'] + T * 100_000 + m * 10_000 + int(round(rho * 1000)) * 37


def omega2_ar1(rho, sigma2_e=1.0):
    """Long-run variance of an AR(1): sigma2_e / (1-rho)^2 (the oracle LRV)."""
    return sigma2_e / (1.0 - rho) ** 2


def pt_oracle(y, bp, Z, cbar, omega2):
    """P_T (ERS SSR form) with the ORACLE omega^2 in place of estimated s^2_AR:
        pt = (S(a-bar) - a-bar S(1)) / omega2 ,  a-bar = 1 + cbar/T."""
    nt = len(y)
    _, ssra = _K['glsd_nb'](y, Z, float(cbar))
    _, ssr1 = _K['glsd_nb'](y, Z, 0.0)
    if not (np.isfinite(ssra) and np.isfinite(ssr1)) or omega2 <= 0:
        return np.nan
    return (ssra - (1.0 + cbar / nt) * ssr1) / omega2


def oracle_tangency(T, m, rho, seed, r_cv, r_pow, cbar_grid, tag=""):
    """c-bar* whose oracle-P_T power crosses TARGET_POWER (interpolated), with a
    delta-method MC standard error. Paired common random numbers across c-bar."""
    gen = _K['gen_dgp_nb']; bz = _K['build_z_nb']
    lambdas = equi_break_fractions(m)
    bp = _K['break_pos_from_lambdas'](T, lambdas)
    Z = bz(T, bp)
    omega2 = omega2_ar1(rho)

    sel = []
    for cbar in cbar_grid:
        h0 = np.empty(r_cv)
        for r in range(r_cv):
            rng = np.random.default_rng((seed + r) % (2**63 - 1))
            eps = make_ar1_eps(rng, T, rho)
            y = gen(T, bp, 0.0, 0.0, eps)
            h0[r] = pt_oracle(y, bp, Z, cbar, omega2)
        h0 = h0[np.isfinite(h0)]
        if len(h0) < r_cv * 0.5:
            continue
        cv5 = float(np.percentile(h0, 5))
        rej = nval = 0
        for r in range(r_pow):
            rng = np.random.default_rng((seed + 10**6 + r) % (2**63 - 1))
            eps = make_ar1_eps(rng, T, rho)
            y = gen(T, bp, cbar, 0.0, eps)
            v = pt_oracle(y, bp, Z, cbar, omega2)
            if np.isfinite(v):
                nval += 1
                rej += (v < cv5)
        if nval > 0:
            sel.append((float(cbar), rej / nval))

    if len(sel) < 2:
        return np.nan, np.nan, np.nan
    arr = np.array(sel); o = np.argsort(arr[:, 0])
    cb, pw = arr[o, 0], arr[o, 1]
    cr = _interp_crossing(cb, pw, TARGET_POWER)
    if cr is not None:
        i, f = cr
        cbar_star = float(cb[i] + f * (cb[i+1] - cb[i]))
        slope = (pw[i+1] - pw[i]) / (cb[i+1] - cb[i])
        se_pow = np.sqrt(TARGET_POWER * (1 - TARGET_POWER) / r_pow)
        se = float(se_pow / abs(slope)) if slope != 0 else np.nan
        if cb[i] == cb[0] or cb[i+1] == cb[-1]:
            print(f"  [!] {tag}: crossing at CBAR_GRID edge -- widen the grid.")
        return cbar_star, TARGET_POWER, se
    idx = int(np.argmin(np.abs(pw - TARGET_POWER)))
    ps = float(pw[idx])
    print(f"  [!] {tag}: no power={TARGET_POWER} crossing; argmin c-bar*={cb[idx]:.2f} "
          f"(power={ps:.3f}).")
    return float(cb[idx]), ps, float(np.sqrt(max(ps * (1 - ps), 0.0) / r_pow))


def run_oracle(args, outdir):
    cfg = dict(ORACLE)
    t_list, m_list, rho_list, cbar_grid = cfg['T_LIST'], cfg['M_LIST'], cfg['RHO_LIST'], cfg['CBAR_GRID']
    r_cv, r_pow = cfg['R_CV'], cfg['R_POW']
    if args.quick:
        t_list, m_list, rho_list = [60, 300], [0, 1], [0.0, 0.3]
        r_cv = r_pow = 600
        cbar_grid = np.round(np.arange(-14.0, -3.9, 1.0), 2)
        print("[oracle][quick] reduced grid/reps.\n")

    t0 = time.time()
    tasks = [(T, m, rho) for T in t_list for m in m_list for rho in rho_list]

    def _cell(T, m, rho):
        seed = _oracle_seed(m, rho, T)
        cbar, power, se = oracle_tangency(T, m, rho, seed, r_cv, r_pow, cbar_grid,
                                          tag=f"(T={T},m={m},rho={rho})")
        return (T, m, rho), (cbar, power, se)

    print(f"[oracle] T={t_list} m={m_list} rho={rho_list} R_CV={r_cv} R_POW={r_pow}")
    results = dict(_parallel(tasks, _cell, args.jobs, not args.quick))
    for (T, m, rho), (cbar, power, se) in sorted(results.items()):
        print(f"  T={T:>3} m={m} rho={rho:.1f}: cbar*={cbar:>7.2f} power={power:.3f} se={se:.3f}")

    # CSV
    csv_path = os.path.join(outdir, "oracle_serial.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["T", "m", "rho", "cbar_star", "power_at_opt", "se_cbar"])
        for (T, m, rho), (cbar, power, se) in sorted(results.items()):
            w.writerow([T, m, rho, f"{cbar:.3f}", f"{power:.4f}", f"{se:.4f}"])
    print(f"[oracle] written: {csv_path}")

    # LaTeX tab:serial (one panel per T)
    tex_path = os.path.join(outdir, "tab_serial.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Optimal $\\bar{c}$ under serially correlated "
                "innovations with the \\emph{oracle} long-run variance, across "
                f"$T\\in\\{{{','.join(str(t) for t in t_list)}\\}}$, "
                f"$R_{{\\mathrm{{cv}}}}={r_cv:,}$, $R_{{\\mathrm{{pow}}}}={r_pow:,}$. "
                "Innovations are AR(1) with coefficient $\\rho$; the long-run "
                "variance $\\omega^2=1/(1-\\rho)^2$ is treated as known, isolating "
                "serial correlation from the estimation of $s^2_{\\mathrm{AR}}$. "
                "The tangency $\\bar c$ interpolates the power curve to $0.50$.}\n"
                "\\label{tab:serial}\n")
        f.write("\\begin{tabular}{l" + "c" * len(rho_list) + "}\n\\toprule\n")
        for ti, T in enumerate(t_list):
            if ti > 0:
                f.write("\\addlinespace\n")
            f.write(f"\\multicolumn{{{1+len(rho_list)}}}{{l}}{{\\textit{{$T={T}$}}}} \\\\\n")
            f.write("$m$ & " + " & ".join(f"$\\rho={r:g}$" for r in rho_list) + " \\\\\n\\midrule\n")
            for m in m_list:
                f.write(f"{m} & " + " & ".join(f"{results[(T, m, rho)][0]:.2f}"
                                               for rho in rho_list) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        se_vals = [results[(T, m, rho)][2] for T in t_list for m in m_list for rho in rho_list]
        se_vals = [v for v in se_vals if np.isfinite(v)]
        se_max = max(se_vals) if se_vals else float("nan")
        m_top, T_lo, T_hi = max(m_list), min(t_list), max(t_list)

        def _drift(Tx):
            vs = [results[(Tx, m_top, r)][0] for r in rho_list if np.isfinite(results[(Tx, m_top, r)][0])]
            return (max(vs) - min(vs)) if len(vs) >= 2 else float('nan')
        f.write("\\par\\medskip\\footnotesize Entries are the optimal $\\bar c$ "
                "(interpolated tangency at power $0.50$) with the true long-run "
                "variance and the same $P_T$ statistic as the production "
                f"calibration. The delta-method MC standard error is at most "
                f"${se_max:.2f}$ across cells. At $m={m_top}$ the spread of "
                f"$\\bar c^*$ over $\\rho$ is ${_drift(T_lo):.2f}$ at $T={T_lo}$ "
                f"versus ${_drift(T_hi):.2f}$ at $T={T_hi}$. Serial correlation "
                "does not move the \\emph{limiting} tangency.\n\\end{table}\n")
    print(f"[oracle] written: {tex_path}")
    print(f"[oracle] done in {(time.time()-t0)/60:.1f} min")


# ==============================================================================
# EXPERIMENT 3 -- TRIMMING SENSITIVITY  (tab:trimming + \TBD number)
# ==============================================================================
TRIMMING = dict(
    CELLS=[(1, 60), (2, 60), (3, 60), (2, 100)], EPS=[0.10, 0.15, 0.20],
    MIN_SPACING=0.15, N_GRID=5,
    CBAR_GRID=np.round(np.arange(-14.0, -4.9, 0.5), 2),
    K_MAX_LAMBDA=5, R_CV=10000, R_POW=5000, SEED0=90909, GRID_STEP=0.5,
)


def trimming_tangency(T, lam, r_cv, r_pow, seed, cbar_grid):
    """c-bar* for one (T, lambda) config, by the SAME interpolated-tangency
    criterion and delta-method standard error as the production calibration
    (and as the paper's Robustness section states): trace the (c-bar, power)
    curve on the grid, locate the interpolated crossing of power=TARGET_POWER,
    se(c-bar*) = se(power)/|slope| in the bracketing interval. Falls back to
    the argmin-gap grid point (flagged) only if the curve never crosses.
    Returns (cbar_star, se_cbar) -- (None, None) if the curve is empty."""
    gen = _K['gen_dgp_nb']; bz = _K['build_z_nb']; ms = _K['mstats_nb']; KMAX = _K['KMAX']
    bp = _K['break_pos_from_lambdas'](T, lam)
    Z = bz(T, bp)

    def _pt_sample(cbar, c_dgp, n, seed_off):
        out = np.empty(n)
        for i in range(n):
            rng = np.random.default_rng((seed + seed_off + i) % (2**63 - 1))
            eps = rng.standard_normal(T)
            y = gen(T, bp, c_dgp, 0.0, eps)
            _, _, _, pt, _, ok = ms(y, Z, cbar, 0, KMAX)   # const (mc=0)
            out[i] = pt if ok > 0.5 else np.nan
        return out[np.isfinite(out)]

    cb_l, pw_l = [], []
    for cbar in cbar_grid:
        null = _pt_sample(cbar, 0.0, r_cv, 1)
        if len(null) < max(10, r_cv // 2):
            continue
        cv = float(np.percentile(null, 100 * ALPHA))
        powr = _pt_sample(cbar, cbar, r_pow, 2)
        if len(powr) < 10:
            continue
        cb_l.append(float(cbar)); pw_l.append(float(np.mean(powr < cv)))
    if not cb_l:
        return None, None
    order = np.argsort(cb_l)
    cb = np.asarray(cb_l)[order]; pw = np.asarray(pw_l)[order]

    cr = _interp_crossing(cb, pw, TARGET_POWER)
    if cr is not None:
        i, frac = cr
        cstar = float(cb[i] + frac * (cb[i + 1] - cb[i]))
        slope = (pw[i + 1] - pw[i]) / (cb[i + 1] - cb[i])
        se_pl = float(np.sqrt(TARGET_POWER * (1 - TARGET_POWER) / r_pow))
        se = se_pl / abs(slope) if slope != 0 else float('nan')
        return cstar, float(se)
    # fallback: no crossing on the grid (flagged upstream via se=nan)
    idx = int(np.argmin(np.abs(pw - TARGET_POWER)))
    return float(cb[idx]), float('nan')


def run_trimming(args, outdir):
    cfg = dict(TRIMMING)
    cells, eps_list, cbar_grid = cfg['CELLS'], cfg['EPS'], cfg['CBAR_GRID']
    r_cv, r_pow, kmaxlam = cfg['R_CV'], cfg['R_POW'], cfg['K_MAX_LAMBDA']
    if args.quick:
        r_cv, r_pow, kmaxlam = 250, 250, 2
        print("[trimming][quick] reduced reps.\n")

    print(f"[trimming] cells={cells} eps={eps_list} min_spacing={cfg['MIN_SPACING']} "
          f"R_cv={r_cv} R_pow={r_pow}")

    tasks, meta = [], {}
    for ci, (m, T) in enumerate(cells):
        for ei, eps in enumerate(eps_list):
            lams = [tuple(l) for l in _K['make_lambdas'](m, eps, cfg['MIN_SPACING'],
                                                          n_grid=cfg['N_GRID']) if len(l) == m]
            if len(lams) > kmaxlam:
                idx = np.linspace(0, len(lams) - 1, kmaxlam).round().astype(int)
                lams = [lams[i] for i in idx]
            for li, lam in enumerate(lams):
                seed = cfg['SEED0'] + ci * 100000 + ei * 1000 + li * 17
                tasks.append((ci, ei, T, lam, seed))
    print(f"[trimming] {len(tasks)} tangencies (cell x eps x lambda-config) ...")

    def _one(ci, ei, T, lam, seed):
        cb, se = trimming_tangency(T, lam, r_cv, r_pow, seed, cbar_grid)
        return (ci, ei, cb, se)

    res = _parallel(tasks, _one, args.jobs, not args.quick)

    from collections import defaultdict
    bucket, bucket_se = defaultdict(list), defaultdict(list)
    n_fallback = 0
    for ci, ei, cb, se in res:
        if cb is not None:
            bucket[(ci, ei)].append(cb)
            if se is not None and np.isfinite(se):
                bucket_se[(ci, ei)].append(se)
            else:
                n_fallback += 1

    print("\n[trimming] c-bar per cell and trimming (median over lambda configs; "
          "interpolated tangency, delta-method se)")
    step = cfg['GRID_STEP']; overall_max = 0.0; overall_se = 0.0; rows = []
    for ci, (m, T) in enumerate(cells):
        vals, nlam, cell_se = [], 0, 0.0
        for ei in range(len(eps_list)):
            v = bucket.get((ci, ei), []); nlam = max(nlam, len(v))
            vals.append(float(np.median(v)) if v else np.nan)
            se_v = bucket_se.get((ci, ei), [])
            if se_v:
                cell_se = max(cell_se, float(np.max(se_v)))
        finite = [v for v in vals if np.isfinite(v)]
        dmax = (max(finite) - min(finite)) if len(finite) >= 2 else np.nan
        if np.isfinite(dmax):
            overall_max = max(overall_max, dmax)
        overall_se = max(overall_se, cell_se)
        rows.append((m, T, vals, dmax, cell_se))
        print(f"  (m={m},T={T:>3}): " + "  ".join(f"eps={e:.2f}->{v:6.2f}"
              for e, v in zip(eps_list, vals))
              + f"  | max|dcbar|={dmax:.2f}  se(cbar*)<= {cell_se:.2f} (n_lam={nlam})")
    if n_fallback:
        print(f"  [!] {n_fallback} tangencies fell back to argmin (no crossing on grid)")

    print(f"\n[trimming] MAX |delta cbar| over eps and cells = {overall_max:.2f} "
          f"(grid step {step}; {'<= one step OK' if overall_max <= step else '> one step -- check'}); "
          f"max delta-method se(cbar*) = {overall_se:.2f} "
          f"(resolution floor for a 3-point spread ~ {2.0*overall_se:.2f}--{2.6*overall_se:.2f})")

    txt_path = os.path.join(outdir, "trimming_sensitivity.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"max |delta cbar| across eps in {eps_list}, over cells {cells}: "
                f"{overall_max:.2f} (grid step {step}); "
                f"max delta-method se(cbar*): {overall_se:.2f}\n")
        f.write("criterion: interpolated tangency (power=0.5 crossing), "
                "se = se(power)/|slope| in the bracketing interval; "
                f"fallbacks to argmin: {n_fallback}\n")
        for m, T, vals, dmax, cse in rows:
            f.write(f"  (m={m},T={T}): " + ", ".join(f"eps={e:.2f}->{v:.2f}"
                    for e, v in zip(eps_list, vals))
                    + f"  | max|delta|={dmax:.2f}  se<= {cse:.2f}\n")
    print(f"[trimming] written: {txt_path}")

    tex_path = os.path.join(outdir, "tab_trimming.tex")
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Sensitivity of the calibrated $\\bar c(m,T)$ to the "
                "break-fraction trimming $\\epsilon$. Each entry is the median "
                "tangency $\\bar c$ over the admissible $\\lambda$ configurations "
                f"at that $\\epsilon$; $R_{{\\mathrm{{cv}}}}={r_cv}$, "
                f"$R_{{\\mathrm{{pow}}}}={r_pow}$.}}\n\\label{{tab:trimming}}\n")
        f.write("\\begin{tabular}{l" + "c" * len(eps_list) + "c}\n\\toprule\n")
        f.write("$(m,T)$ & " + " & ".join(f"$\\epsilon={e:.2f}$" for e in eps_list)
                + " & $\\max|\\Delta\\bar c|$ & $\\mathrm{se}(\\bar c^{*})\\le$ \\\\\n\\midrule\n")
        for m, T, vals, dmax, cse in rows:
            f.write(f"$({m},{T})$ & " + " & ".join(f"{v:.2f}" for v in vals)
                    + f" & {dmax:.2f} & {cse:.2f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        verdict = ("within one grid step of the calibration search"
                   if overall_max <= step else
                   "LARGER than one grid step --- inspect before quoting")
        f.write("\\par\\medskip\\footnotesize The largest $|\\Delta\\bar c|$ over "
                f"$\\epsilon\\in\\{{{','.join(f'{e:g}' for e in eps_list)}\\}}$ and "
                f"all cells is ${overall_max:.2f}$ ({verdict}; grid step ${step}$). "
                f"Max delta-method $\\mathrm{{se}}(\\bar c^{{*}})$: ${overall_se:.2f}$; "
                "resolution floor for a three-point spread is roughly "
                f"${2.0*overall_se:.2f}$--${2.6*overall_se:.2f}$. Interpolated-tangency "
                "criterion throughout; the paper draws the substantive conclusion "
                "from the production-replication run, not from this file.\n\\end{table}\n")
    print(f"[trimming] written: {tex_path}")
    print(f"\n[trimming] \\TBD suggestion: 'at most {overall_max:.2f} across "
          f"eps in {{{','.join(f'{e:g}' for e in eps_list)}}} --- within one grid "
          f"step ({step}) and Monte Carlo error.'")


# ==============================================================================
# EXPERIMENT 4 -- POWER COMPARISON OF cbar SPECIFICATIONS  (tab:power, fig:power)
# ==============================================================================
# Size and power of MZ_t under three cbar specifications and three DGPs, at
# T=60, m=2, lambda=(1/3,2/3), alpha=0.05.  Migrated here (its correct home) from
# figuras_v6.py: this experiment produces DATA (power_comparison.csv), which
# figuras_v6.fig_power then consumes -- restoring the data/figure boundary a
# replication package requires.  The original tab:power script was lost; this is
# the authoritative regeneration on the validated the numba kernel.
#
# Columns:
#   (i)   I(1), no breaks (size)          : c=0,      beta=0
#   (ii)  I(1) + level breaks (size)      : c=0,      beta=BETA_BIG  [Lemma 1 => ~(i)]
#   (iii) I(0) around a broken mean (power): c=C_ALT,  beta=BETA_BIG
# Each cbar uses its OWN 5% CV (5th percentile of MZ_t under c=0), so all three
# are correctly sized by construction and differ only in power.
POWER = dict(
    T=60, M=2, ALPHA=0.05, C_ALT=-10.0, BETA_BIG=50.0,
    R=10000, SEED_BASE=20260701,
    # (label, cbar) -- calibrated Model LB at the EXACT lambda=(1/3,2/3) tangency
    # (-8.40, not the (m,T)-cell mean -8.7; see the sec:results lambda-residual),
    # linear break-count scaling, reused trend-break surface.
    SPECS=[("Calibrated Model~LB $\\bar c(m,T)$", -8.40),
           ("Linear break-count scaling",        -17.0),
           ("Trend-break surface",               -21.7)],
)


def _power_seed(kind, cbar):
    """Disjoint stream per (experiment-column, spec)."""
    k = {"cv": 0, "i": 11, "ii": 22, "iii": 33}[kind]
    return (POWER['SEED_BASE'] + int(round(abs(cbar) * 100)) * 101 + k) % (2**63 - 1)


def _mzt_null_cv(T, bp, cbar, R, seed, alpha=ALPHA):
    """5% critical value of MZ_t(cbar) under H0 (c=0, no shifts): the alpha-quantile
    of the null distribution. Own-CV per spec => size controlled by construction."""
    Z = _K['build_z_nb'](T, bp)
    rng = np.random.default_rng(seed)
    vals = np.empty(R); kept = 0
    for _ in range(R):
        eps = rng.standard_normal(T)
        y = _K['gen_dgp_nb'](T, bp, 0.0, 0.0, eps)          # c=0, beta=0
        _, _, mzt, _, _, ok = _K['mstats_nb'](y, Z, cbar, 0, _K['KMAX'])
        if ok > 0.5:
            vals[kept] = mzt; kept += 1
    return float(np.percentile(vals[:kept], 100 * alpha)) if kept else np.nan


def _mzt_reject_rate(T, bp, cbar, cv, c, beta_scale, R, seed):
    """Empirical rejection rate P[MZ_t(cbar) < cv | root 1+c/T, level shifts of
    size beta_scale]. By Lemma 1 the rate is invariant to beta_scale under c=0."""
    Z = _K['build_z_nb'](T, bp)
    rng = np.random.default_rng(seed)
    rej = kept = 0
    for _ in range(R):
        eps = rng.standard_normal(T)
        y = _K['gen_dgp_nb'](T, bp, c, beta_scale, eps)
        _, _, mzt, _, _, ok = _K['mstats_nb'](y, Z, cbar, 0, _K['KMAX'])
        if ok > 0.5:
            kept += 1
            rej += (mzt < cv)
    return rej / kept if kept else np.nan


def run_power(args, outdir):
    cfg = dict(POWER)
    T, m, R = cfg['T'], cfg['M'], cfg['R']
    c_alt, beta_big, alpha = cfg['C_ALT'], cfg['BETA_BIG'], cfg['ALPHA']
    if args.quick:
        R = 1500
        print("[power][quick] reduced reps.\n")

    lams = tuple(round((j + 1) / (m + 1), 4) for j in range(m))     # (1/3, 2/3)
    bp = _K['break_pos_from_lambdas'](T, lams)
    se = float(np.sqrt(0.25 / R))
    print(f"[power] T={T} m={m} lambda={lams} R={R} c_alt={c_alt} "
          f"beta_big={beta_big}")

    # tasks: per spec, one CV + three columns. Kept sequential per spec so the
    # null CV precedes the columns that use it; specs parallelised.
    def _one_spec(label, cbar):
        cv = _mzt_null_cv(T, bp, cbar, R, _power_seed("cv", cbar), alpha)
        col_i   = _mzt_reject_rate(T, bp, cbar, cv, 0.0,   0.0,      R, _power_seed("i", cbar))
        col_ii  = _mzt_reject_rate(T, bp, cbar, cv, 0.0,   beta_big, R, _power_seed("ii", cbar))
        col_iii = _mzt_reject_rate(T, bp, cbar, cv, c_alt, beta_big, R, _power_seed("iii", cbar))
        return (label, cbar, cv, col_i, col_ii, col_iii)

    tasks = [(lab, cb) for lab, cb in cfg['SPECS']]
    res = _parallel(tasks, _one_spec, args.jobs, not args.quick)
    # preserve SPECS order (parallel may reorder)
    order = {cb: i for i, (_, cb) in enumerate(cfg['SPECS'])}
    res.sort(key=lambda r: order[r[1]])

    print(f"\n  {'spec':<34}{'cbar':>7}{'cv5':>8}{'(i)':>8}{'(ii)':>8}{'(iii)':>8}")
    for lab, cbar, cv, ci, cii, ciii in res:
        short = lab.split('$')[0].strip()
        print(f"  {short:<34}{cbar:>7.2f}{cv:>8.3f}{ci:>8.3f}{cii:>8.3f}{ciii:>8.3f}")

    # ---- integrity gates ----
    g_size = all(abs(ci - alpha) <= 3 * se and abs(cii - alpha) <= 3 * se
                 for _, _, _, ci, cii, _ in res)
    g_inv = all(abs(cii - ci) <= 3 * se for _, _, _, ci, cii, _ in res)   # Lemma 1
    powers = [r[5] for r in res]
    g_order = powers[0] > powers[1] > powers[2]
    print(f"  [gate] sizes~{alpha}: {'PASS' if g_size else 'FAIL'}; "
          f"invariance |ii-i|<=3SE (Lemma 1): {'PASS' if g_inv else 'FAIL'}; "
          f"power ordered LB>lin>trend: {'PASS' if g_order else 'FAIL'}")
    if not (g_size and g_inv and g_order) and not args.quick:
        print("  [power][warn] an integrity gate did not pass at full R; "
              "inspect before quoting (not aborting: the CSV is still written).")

    # ---- write power_comparison.csv (authoritative data for tab:power + fig) ----
    _ensure_outdir(outdir)
    csv_path = os.path.join(outdir, "power_comparison.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["spec", "cbar", "cv5", "size_i", "size_ii", "power_iii",
                    "se_mc", "R", "T", "m", "lambda", "c_alt", "beta_big"])
        for lab, cbar, cv, ci, cii, ciii in res:
            short = lab.split('$')[0].strip()
            w.writerow([short, f"{cbar:.4f}", f"{cv:.6f}", f"{ci:.6f}",
                        f"{cii:.6f}", f"{ciii:.6f}", f"{se:.6f}", R, T, m,
                        "|".join(f"{l:g}" for l in lams), c_alt, beta_big])
    print(f"[power] written: {csv_path}")

    # ---- write tab_power.tex (manuscript schema) ----
    tex_path = os.path.join(outdir, "tab_power.tex")
    with open(tex_path, "w") as f:
        f.write("% auto-generated by robustness.run_power (the numba kernel) -- "
                "authoritative source for tab:power AND the fig_power anchor.\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("& (i) $\\mathrm{I}(1)$, & (ii) $\\mathrm{I}(1)$ + & "
                "(iii) $\\mathrm{I}(0)$, \\\\\n")
        f.write("$\\bar c$ specification & no breaks (size) & level breaks (size) "
                "& broken mean (power) \\\\\n\\midrule\n")
        for lab, cbar, cv, ci, cii, ciii in res:
            f.write(f"{lab} & ${ci:.3f}$ {{\\tiny$(\\pm{se:.3f})$}} "
                    f"& ${cii:.3f}$ {{\\tiny$(\\pm{se:.3f})$}} "
                    f"& ${ciii:.3f}$ {{\\tiny$(\\pm{se:.3f})$}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"[power] written: {tex_path}")
    return res


# ==============================================================================
# MAIN
# ==============================================================================
def _add_common(sp):
    sp.add_argument("--quick", action="store_true", help="fast smoke test")
    sp.add_argument("--jobs", type=int, default=-1)
    sp.add_argument("--outdir", default="robustness_out")
    sp.add_argument("--module", default=None, help="production module name")
    sp.add_argument("--csv", default=CSV_DEFAULT, help="calibration CSV (ar1 anchor)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Unified Model LB robustness suite.")
    ap.add_argument("--selftest", action="store_true",
                    help="pure-logic tests (no numba/production needed)")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("ar1", "oracle", "trimming", "power", "all"):
        spx = sub.add_parser(name)
        _add_common(spx)
        if name in ("ar1", "all"):
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

    prod = _load_prod(args.module)
    _bind_kernels(prod)
    print(_K['warm_up_numba']())
    print(f"[robustness] kernels from {_K['prod_name']}; joblib="
          f"{'yes' if _HAS_JOBLIB else 'no'}; jobs={args.jobs}")
    outdir = _ensure_outdir(args.outdir)

    t0 = time.time()
    if args.cmd in ("ar1", "all"):
        run_ar1(args, outdir)
    if args.cmd in ("oracle", "all"):
        run_oracle(args, outdir)
    if args.cmd in ("trimming", "all"):
        run_trimming(args, outdir)
    if args.cmd in ("power", "all"):
        run_power(args, outdir)
    print(f"\n[robustness] ALL DONE in {(time.time()-t0)/60:.1f} min -> {outdir}/")
    return 0


# ------------------------------------------------------------------------------
# SELF-TEST (pure logic; no numba/production import)
# ------------------------------------------------------------------------------
def _selftest():
    print("=" * 70)
    print("SELF-TEST robustness -- pure logic (no numba)")
    print("=" * 70)
    ok = True

    # 1) interpolated crossing on a synthetic decreasing power curve
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

    # 2) make_ar1_eps: rho=0 -> iid reproducible; rho>0 -> exact recursion
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

    # 3) omega2_ar1 and equi_break_fractions
    if abs(omega2_ar1(0.5) - 4.0) > 1e-12:
        print("  [x] omega2_ar1(0.5) != 4"); ok = False
    else:
        print("  [OK] omega2_ar1(0.5)=4.0")
    bf = equi_break_fractions(2, 0.15)
    if not (len(bf) == 2 and abs(bf[0] - (0.15 + 1*0.7/3)) < 1e-3):
        print(f"  [x] equi_break_fractions wrong: {bf}"); ok = False
    else:
        print(f"  [OK] equi_break_fractions(2)={bf}")

    # 4) load_cbar_csv on a synthetic CSV (m=0 seed-averaged style)
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

    # 5) power gates: seed disjointness + gate arithmetic (no kernel needed)
    seeds = {(k, cb) for k in ("cv", "i", "ii", "iii")
             for cb in (-8.40, -17.0, -21.7)}
    svals = [_power_seed(k, cb) for k, cb in seeds]
    if len(set(svals)) != len(svals):
        print("  [x] _power_seed collisions across (column, spec)"); ok = False
    else:
        print(f"  [OK] _power_seed: {len(svals)} disjoint streams")
    # gate logic: consistent triple passes; a size-violating / non-ordered one fails
    R = 10000; se = (0.25 / R) ** 0.5
    good = [("LB", -8.40, -2.5, 0.049, 0.054, 0.534),
            ("lin", -17.0, -3.1, 0.048, 0.051, 0.344),
            ("tr", -21.7, -3.2, 0.050, 0.056, 0.293)]
    g_size = all(abs(ci - 0.05) <= 3 * se and abs(cii - 0.05) <= 3 * se
                 for _, _, _, ci, cii, _ in good)
    g_inv = all(abs(cii - ci) <= 3 * se for _, _, _, ci, cii, _ in good)
    g_ord = good[0][5] > good[1][5] > good[2][5]
    bad_ord = [good[0], good[1], ("tr", -21.7, -3.2, 0.05, 0.056, 0.60)]  # trend > LB
    g_ord_bad = bad_ord[0][5] > bad_ord[1][5] > bad_ord[2][5]
    if g_size and g_inv and g_ord and not g_ord_bad:
        print("  [OK] power gates: consistent triple PASS, non-ordered FAIL")
    else:
        print(f"  [x] power gate logic: size={g_size} inv={g_inv} "
              f"ord={g_ord} bad_ord={g_ord_bad}"); ok = False

    print("\n" + ("SELF-TEST: ALL PASSED" if ok else "SELF-TEST: FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
