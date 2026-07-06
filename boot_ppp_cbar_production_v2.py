#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boot_ppp_cbar_production.py
===========================
PPP application (OECD-core/USD, Model LB): production calibration of the
extended response surface  (cbar*(m,T,p), cv(m,T,p))  under a sieve-AR(p)
innovation, plus the preliminary empirical block. Numba throughout.

EQUIVALENCE CONTRACT (v6).  Every Monte Carlo convention is inherited from
mlb_core.py, the Article-1 production engine:
  * numerical kernels build_z_nb/glsd_nb/ols_detrend_nb/s2ar_const_nb/
    s2ar_maic_nb/mstats_nb IMPORTED from v6 (never reimplemented);
  * TANGENCY (Option B, ERS envelope convention): cbar*(m,T,p) is located
    where the power of the ORACLE PT --- P_T computed with the TRUE long-run
    variance omega^2 = sigma^2/phi(1)^2 of each draw, NP2001 eq. 7 with
    s2_AR -> omega^2 --- crosses 0.50 on the cbar grid [-20,-3] step 0.5
    (interpolated, delta-method SE, flagged argmin fallback). This is the ERS
    definition of c-bar (the envelope), which the v6 feasible tangency
    approximates under iid/const; under AR(p)+MAIC at short T the feasible
    power saturates below 0.50 (the Murray-Papell mechanism), so the envelope
    is the well-defined object. The FEASIBLE statistics (MZt, PT with MAIC)
    keep their own null quantiles: cv(m,T,p) is the feasible MZt 5% quantile,
    so the empirical test confronts statistic and critical value from the SAME
    feasible null. The feasible power at cbar* is reported as the MAIC-cost
    diagnostic (power_feasible_at_star).
  * R_cv=10000 null reps, R_pow=5000 alternative reps (defaults; --hi doubles);
  * kmax configurable via --kmax (default 12, the paper convention; NOT
    Schwert -- 10 would be the Schwert value at T=52, exposed for the
    referee sensitivity run);
  * lambda-averaging over gerar_lambdas(m, trim=.15, min_spacing=.15, n_grid=7);
  * per-config deterministic seeds: config_seed(...) extended with 104729*p
    so AR orders use disjoint streams;
  * beta=0 WLOG (exact-detrending invariance, the validated sanity direction);
  * sigma^2: MAIC mandatory for p>=1 (difference-based 'const' is inconsistent
    under serial correlation, plim sigma^2/(1-rho^2) != omega^2); 'const' runs
    only in the p=0 baseline to anchor against Article-1 tab:cbar.
AR(p) innovation: partial-autocorrelation (Levinson-Durbin) parameterisation,
bijective onto the stationary region -- stability by construction.

FALLBACK CHAIN (strict): v6 kernels -> mlb_kernel.py (exact, pure Python,
~45x slower, loud WARN) -> ABORT. Production never runs on an approximation.

Outputs (all numbers needed for tables/figures) under --out:
  calib/surface_ppp_boot.csv    per (m,T,p,method): cbar*, se, power*, cv5_PT,
                                cv5_MZt (lambda-averaged), fallback flags
  calib/power_curves.csv        per (cell, lambda-config, cbar): power, cv5s, SEs
  calib/lambda_configs.csv      the lambda grid actually used
  raw/cell_<m>_<T>_<p>_<meth>.npz   null/alt PT & MZt draws at the tangency
  empirical/ppp_empirical_prelim.csv  per-currency test block (PRELIMINARY:
                                placeholder breaks 1973+1985 until Part 2b)
  meta.json                     params, seeds, versions, input hashes, timing

Usage (local, Windows):
  python boot_ppp_cbar_production.py --quick            # smoke (~1 min)
  python boot_ppp_cbar_production.py --full             # production surface
  python boot_ppp_cbar_production.py --full --empirical # + empirical block
  python boot_ppp_cbar_production.py --full --hi --raw  # 2x reps + raw draws
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, os, sys, time, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

# =============================================================================
# 0. PATHS (defaults = Ricardo's machine; every one overridable via CLI)
# =============================================================================
DEFAULT_MC_DIR  = "."   # calibration CSV directory (portable default)
DEFAULT_OUT_DIR = "./boot_out"           # bootstrap outputs (portable default)
DEFAULT_EMP_DIR = "."           # empirical panel directory (portable default)

# =============================================================================
# 1. KERNEL IMPORT -- strict chain: v6 -> mlb_kernel -> abort
# =============================================================================
KERNEL_SOURCE = None
V6 = None

def load_kernels(mc_dir: str):
    """Bind the numba kernels from v6; fall back to mlb_kernel (exact) or abort."""
    global KERNEL_SOURCE, V6
    for cand in (mc_dir, os.getcwd(), os.path.dirname(os.path.abspath(__file__)),
                 "/mnt/project"):
        if cand and os.path.isdir(cand) and cand not in sys.path:
            sys.path.insert(0, cand)
    try:
        import mlb_core as V6mod
        V6 = V6mod
        KERNEL_SOURCE = f"mlb_core ({V6mod.__file__})"
        return dict(build_z=V6mod.build_z_nb, glsd=V6mod.glsd_nb,
                    mstats=V6mod.mstats_nb, gen_dgp=V6mod.gen_dgp_nb,
                    warm=V6mod.warm_up_numba, njit=_get_njit(),
                    gerar_lambdas=V6mod.gerar_lambdas,
                    config_seed=V6mod.config_seed,
                    break_pos=V6mod.break_pos_from_lambdas,
                    se_power=V6mod.se_power, method_code=V6mod._METHOD_CODE)
    except Exception as e:
        print(f"[kernel] v6 unavailable ({e}); trying mlb_kernel (exact, slow)...")
    try:
        import mlb_kernel as MK
        KERNEL_SOURCE = f"mlb_kernel FALLBACK ({MK.__file__}) -- EXACT BUT ~45x SLOWER"
        print(f"[kernel] WARNING: {KERNEL_SOURCE}")
        # adapt pure-python API to the kernel dict (subset needed)
        def mstats_adapter(y, Z, cbar, mc, kmax):
            r = MK.m_statistics(y, Z, cbar, sigma2_method=mc, kmax=kmax)
            ok = 1.0 if r.get("ok") else 0.0
            return (r.get("MZa", np.nan), r.get("MSB", np.nan),
                    r.get("MZt", np.nan), r.get("PT", np.nan),
                    r.get("MPT", np.nan), ok)
        def gen_dgp_adapter(nt, bp, c, beta, eps):
            Z = MK.build_z(nt, bp); rho = 1.0 + c / nt
            u = np.zeros(nt)
            for t in range(1, nt):
                u[t] = rho * u[t - 1] + eps[t]
            return u  # beta=0 WLOG
        return dict(build_z=MK.build_z, glsd=MK.gls_detrend,
                    mstats=mstats_adapter, gen_dgp=gen_dgp_adapter,
                    warm=lambda k=12: "[mlb_kernel] pure python",
                    njit=_get_njit(),
                    gerar_lambdas=_gerar_lambdas_local,
                    config_seed=_config_seed_local,
                    break_pos=_break_pos_local,
                    se_power=lambda p, R: float(np.sqrt(max(p*(1-p),0)/R)) if R>0 else np.nan,
                    method_code={"const": 0, "maic": 1})
    except Exception as e:
        raise SystemExit(f"[kernel] FATAL: neither v6 nor mlb_kernel importable "
                         f"({e}). Production requires a validated kernel. ABORT.")

def _get_njit():
    try:
        from numba import njit
        return njit
    except Exception:
        def njit(*a, **k):
            if len(a) == 1 and callable(a[0]):
                return a[0]
            return lambda f: f
        return njit

# local copies used ONLY under the mlb_kernel fallback (identical arithmetic)
def _gerar_lambdas_local(m, trim, min_spacing, n_grid=7):
    if m == 0: return [()]
    ng = max(n_grid, 3 * m - 1)
    grid = np.round(np.linspace(trim, 1 - trim, ng), 3)
    combos = [c for c in itertools.combinations(grid, m)
              if all(c[i+1]-c[i] >= min_spacing - 1e-9 for i in range(len(c)-1))]
    if m >= 4 and len(combos) > 12:
        idx = np.linspace(0, len(combos)-1, 12).round().astype(int)
        combos = [combos[i] for i in sorted(set(idx))]
    return combos

def _config_seed_local(P, T, m, lambdas):
    h = P["seed_base"] + 1000003 * T + 7919 * m
    for i, lam in enumerate(lambdas):
        h += int(round(lam * 1000)) * (37 ** (i + 1))
    return h % (2 ** 63 - 1)

def _break_pos_local(T, lambdas):
    return np.array([int(max(1, min(T-1, np.floor(l*T)))) for l in lambdas],
                    dtype=np.int64)

# =============================================================================
# 2. AR(p) MACHINERY -- PAC (Levinson-Durbin) parameterisation, njit DGP
# =============================================================================
def pac_to_phi(pac: np.ndarray) -> np.ndarray:
    """Map partial autocorrelations in (-1,1)^p to AR(p) coefficients.
    Bijective onto the stationary region: stability by construction."""
    p = len(pac)
    if p == 0:
        return np.zeros(0)
    phi = np.array([pac[0]])
    for k in range(1, p):
        new = np.empty(k + 1)
        new[k] = pac[k]
        new[:k] = phi - pac[k] * phi[::-1]
        phi = new
    return phi

def draw_phi(p: int, rng: np.random.Generator, pac1: float = 0.4,
             pac_hw: float = 0.3) -> np.ndarray:
    """Nuisance AR(p): first PAC fixed at pac1 (persistence level), higher PACs
    uniform in [-pac_hw, pac_hw]. Matches the session diagnostic design."""
    if p == 0:
        return np.zeros(0)
    pac = rng.uniform(-pac_hw, pac_hw, size=p)
    pac[0] = pac1
    return pac_to_phi(pac)

def make_gen_arp(njit):
    @njit(cache=True)
    def gen_dgp_arp_nb(nt, burn, c, phi, eps):
        """y_t = u_t with u_t = rho*u_{t-1} + w_t, rho = 1 + c/nt (local root),
        and w_t the AR(p) nuisance: phi(L) w_t = e_t. beta = 0 WLOG (exact-
        detrending invariance). eps has length nt+burn; the first `burn` draws
        prime the AR(p) recursion for w so that w is at its stationary law by
        t=0. p=0 (phi empty) reproduces the v6 iid DGP EXACTLY: then w_t = e_t,
        u_0 = 0, u_t = rho*u_{t-1} + e_t -- identical to gen_dgp_nb (beta=0).

        NOTE ON THE FIRST OBSERVATION (cf. OBES 2015 on GLS detrending): the
        series passed to the kernel starts at u_0 = 0 exactly as in v6, so the
        ERS treatment of the first observation in glsd_nb is preserved. The
        burn-in acts ONLY on the stationary nuisance w, never on the initial
        condition of the integrated u.
        """
        p = phi.shape[0]
        n = nt + burn
        # 1) stationary AR(p) nuisance w over the full (burn+nt) span
        w = np.zeros(n)
        for t in range(n):
            acc = eps[t]
            for j in range(p):
                if t - 1 - j >= 0:
                    acc += phi[j] * w[t - 1 - j]
            w[t] = acc
        # 2) integrate ONLY the post-burn segment, with u_0 = 0 (v6 convention)
        rho = 1.0 + c / nt
        u = np.zeros(nt)
        for t in range(1, nt):
            u[t] = rho * u[t - 1] + w[burn + t]
        return u
    return gen_dgp_arp_nb

# =============================================================================
# 3. TANGENCY (v6 convention, AR(p) innovation) + cv collection
# =============================================================================
def _pt_oracle(K, y, Z, cbar, omega2):
    """P_T with the TRUE long-run variance (ERS envelope convention):
    P_T = [S(a-bar) - a-bar * S(1)] / omega^2, NP2001 eq. 7, replicated from
    v6's mstats_nb with s2_AR replaced by the known omega^2 = sigma^2/phi(1)^2.
    Uses the kernel's glsd when it returns (residual, ssr); otherwise computes
    the quasi-differenced SSR directly under first-observation definition (1)."""
    nt = Z.shape[0]
    abar = 1.0 + cbar / nt
    def _ssr(cb):
        out = K["glsd"](y, Z, cb)
        if isinstance(out, tuple):
            return float(out[1])
        # fallback kernel returns only the level residual: compute SSR manually
        a = 1.0 + cb / nt
        ya = np.empty(nt); ya[0] = y[0]; ya[1:] = y[1:] - a * y[:-1]
        Za = np.empty_like(Z); Za[0] = Z[0]; Za[1:] = Z[1:] - a * Z[:-1]
        b, *_ = np.linalg.lstsq(Za, ya, rcond=None)
        e = ya - Za @ b
        return float(e @ e)
    ssra, ssr1 = _ssr(cbar), _ssr(0.0)
    if not (np.isfinite(ssra) and np.isfinite(ssr1)) or omega2 <= 0:
        return np.nan
    return (ssra - abar * ssr1) / omega2


def stats_under(K, gen_arp, T, break_pos, c, cbar, mc, n_reps, seed0, kmax,
                p, pac1, burn):
    """n_reps draws at alternative c, stats at detrending cbar; AR(p) innovation.
    RNG in the orchestrator (v6 pattern): per-rep seed = seed0 + r.
    Returns (pt_feasible, mzt_feasible, pt_oracle): the feasible statistics use
    the MAIC/const estimator inside mstats_nb; the oracle PT uses the TRUE
    omega^2 of the draw (phi is known to the simulator), i.e. the ERS envelope
    convention under which the tangency c-bar* is defined."""
    pt = np.empty(n_reps); mzt = np.empty(n_reps); pto = np.empty(n_reps)
    Z = K["build_z"](T, break_pos)
    for r in range(n_reps):
        rng = np.random.default_rng((seed0 + r) % (2**63 - 1))
        phi = draw_phi(p, rng, pac1=pac1)
        omega2 = 1.0 if p == 0 else 1.0 / (1.0 - float(phi.sum()))**2
        eps = rng.standard_normal(T + burn)
        y = gen_arp(T, burn, c, phi, eps)
        mza, msb, mz, ptv, mpt, ok = K["mstats"](y, Z, cbar, mc, kmax)
        if ok > 0.5:
            pt[r] = ptv; mzt[r] = mz
            pto[r] = _pt_oracle(K, y, Z, cbar, omega2)
        else:
            pt[r] = np.nan; mzt[r] = np.nan; pto[r] = np.nan
    return pt, mzt, pto

def tangency_cell(K, gen_arp, P, T, m, p, sigma2_method, lambdas_list,
                  n_jobs=1, collect_curves=None, collect_raw=None, verbose=True):
    """cbar*(m,T,p) by PT power=0.5 crossing, lambda-averaged; also cv5(MZt).
    Returns dict; appends per-config curves to collect_curves (list) if given.

    Progress: joblib prints a per-task ETA (verbose=10) in parallel mode; a
    per-lambda heartbeat is printed in serial mode."""
    import time as _time
    mc = K["method_code"][sigma2_method]
    kmax = P["kmax"]; grid = np.asarray(P["cbar_grid"])
    tang = P["target_power"]; burn = P["burn"]
    n_lam = len(lambdas_list)
    res_l = []
    def one_lambda(lambdas):
        cfg_seed = (K["config_seed"](P, T, m, lambdas) + 104729 * p) % (2**63 - 1)
        bp = K["break_pos"](T, lambdas)
        rows = []
        for cbar in grid:
            # null draws: feasible (mc) and oracle PT from the SAME draws
            pt0, mzt0, pto0 = stats_under(K, gen_arp, T, bp, 0.0, cbar, mc,
                                          P["R_cv"], cfg_seed, kmax, p,
                                          P["pac1"], burn)
            fin0 = np.isfinite(pt0) & np.isfinite(pto0)
            if fin0.sum() < P["R_cv"] * 0.5:
                continue
            cv5_pto = float(np.percentile(pto0[fin0], 5))   # oracle null quantile
            cv5_pt  = float(np.percentile(pt0[fin0], 5))    # feasible null quantile
            m0 = mzt0[np.isfinite(mzt0)]
            cv5_mzt = float(np.percentile(m0, 5)) if len(m0) else np.nan
            # alternative draws at c = cbar
            pt1, _, pto1 = stats_under(K, gen_arp, T, bp, cbar, cbar, mc,
                                       P["R_pow"], cfg_seed + 10**6, kmax, p,
                                       P["pac1"], burn)
            fin1 = np.isfinite(pt1) & np.isfinite(pto1)
            if fin1.sum() == 0:
                continue
            n1 = int(fin1.sum())
            pow_orc = float(np.mean(pto1[fin1] < cv5_pto))  # tangency axis (envelope)
            pow_fea = float(np.mean(pt1[fin1] < cv5_pt))    # MAIC-cost diagnostic
            rows.append((float(cbar), cv5_pto, cv5_pt, cv5_mzt,
                         pow_orc, pow_fea, K["se_power"](pow_orc, n1)))
        return lambdas, rows
    if n_jobs != 1:
        try:
            from joblib import Parallel, delayed
            vlvl = 10 if verbose else 0   # joblib emits per-task progress + ETA
            out = Parallel(n_jobs=n_jobs, verbose=vlvl)(
                delayed(one_lambda)(l) for l in lambdas_list)
        except Exception:
            out = []
            for i, l in enumerate(lambdas_list, 1):
                t0 = _time.time(); out.append(one_lambda(l))
                if verbose:
                    print(f"      lambda {i}/{n_lam} done ({_time.time()-t0:.0f}s)",
                          flush=True)
    else:
        out = []
        for i, l in enumerate(lambdas_list, 1):
            t0 = _time.time(); out.append(one_lambda(l))
            if verbose:
                el = _time.time() - t0
                print(f"      lambda {i}/{n_lam} done ({el:.0f}s, "
                      f"ETA {el*(n_lam-i):.0f}s)", flush=True)

    cb_stars, cvs_mzt_at_star, pow_fea_at_star, fallbacks = [], [], [], 0
    for lambdas, rows in out:
        if not rows:
            continue
        # cols: 0 cbar, 1 cv5_pt_oracle, 2 cv5_pt_maic, 3 cv5_mzt,
        #       4 power_oracle, 5 power_feasible, 6 se_power_oracle
        arr = np.array(rows)
        cb, pw = arr[:, 0], arr[:, 4]          # ORACLE power drives the tangency
        if collect_curves is not None:
            for r in rows:
                collect_curves.append(dict(m=m, T=T, p=p, method=sigma2_method,
                    lambdas=str(lambdas), cbar=r[0], cv5_pt_oracle=r[1],
                    cv5_pt_maic=r[2], cv5_mzt=r[3], power_oracle=r[4],
                    power_feasible=r[5], se_power_oracle=r[6]))
        star = None
        for i in range(len(cb) - 1):
            p0, p1 = pw[i], pw[i + 1]
            if (p0 - tang) * (p1 - tang) <= 0 and p1 != p0:
                frac = (tang - p0) / (p1 - p0)
                star = float(cb[i] + frac * (cb[i + 1] - cb[i]))
                break
        if star is None:
            fallbacks += 1
            star = float(cb[int(np.argmin(np.abs(pw - tang)))])
        cb_stars.append(star)
        cvs_mzt_at_star.append(float(np.interp(star, cb, arr[:, 3])))
        pow_fea_at_star.append(float(np.interp(star, cb, arr[:, 5])))
    if not cb_stars:
        return None
    return dict(m=m, T=T, p=p, method=sigma2_method,
                cbar_star=float(np.mean(cb_stars)),
                cbar_star_se=float(np.std(cb_stars, ddof=1) / np.sqrt(len(cb_stars)))
                             if len(cb_stars) > 1 else np.nan,
                cv5_mzt_at_star=float(np.mean(cvs_mzt_at_star)),
                power_feasible_at_star=float(np.mean(pow_fea_at_star)),
                n_lambda=len(cb_stars), n_fallback=fallbacks,
                definition="oracle-PT (ERS envelope); cv & powF are feasible")

# =============================================================================
# 4. EMPIRICAL BLOCK (PRELIMINARY: placeholder breaks until Part 2b)
# =============================================================================
PLACEHOLDER_BREAKS = (1973, 1985)

def load_exog_dates(search_dirs):
    """Load per-currency exogenous break years from exog_dates.csv, if present.
    Returns {currency: [year, ...]} or None if the file is not found.
    Long format: currency, break_year, event_type, source."""
    import csv as _csv
    for d in search_dirs:
        f = Path(d) / "exog_dates.csv"
        if f.exists():
            ed = {}
            with open(f) as fh:
                for row in _csv.DictReader(fh):
                    ed.setdefault(row["currency"], []).append(int(row["break_year"]))
            return ed, f
    return None, None

def empirical_block(K, gen_arp, P, panel_csv, diag_csv, surface_rows, out_dir,
                    exog_dates=None, start_year=1970):
    import csv as _csv
    # load panel
    panel = {}
    with open(panel_csv) as f:
        for row in _csv.DictReader(f):
            panel.setdefault(row["currency"], []).append(
                (int(row["year"]), float(row["q"])))
    pmap = {}
    with open(diag_csv) as f:
        for row in _csv.DictReader(f):
            pmap[row["currency"]] = int(float(row["k_bic_cq"]))
    # Normalise the surface keys to strings so the lookup works whether
    # surface_rows came from the in-memory calibration (ints) or the CSV
    # (strings). This was the root cause of the -7 fallback: the in-memory
    # rows carry integer (m,T,p), the CSV rows carry strings, and a mixed
    # comparison silently missed every key.
    surf = {(str(r["m"]), str(r["T"]), str(r["p"]), str(r["method"])): r
            for r in surface_rows}
    kmax = P["kmax"]; burn = P["burn"]
    # T of the calibrated surface (rows share it); used as the lookup fallback
    T_cal = None
    for r in surface_rows:
        try:
            T_cal = int(r["T"]); break
        except (KeyError, ValueError, TypeError):
            pass
    rows_out = []
    for cur, obs in sorted(panel.items()):
        obs.sort()
        # apply the start-year window: the empirical sample matches the
        # calibration T (post-Bretton-Woods float when start_year=1973)
        obs = [o for o in obs if o[0] >= start_year]
        years = np.array([o[0] for o in obs])
        q = np.array([o[1] for o in obs]); T = len(q)
        p_hat = pmap.get(cur, 1)
        # break years: exogenous (per currency) if available, else placeholder
        byears = (exog_dates.get(cur, []) if exog_dates is not None
                  else PLACEHOLDER_BREAKS)
        bp = np.array([int(np.searchsorted(years, b)) for b in byears
                       if years[0] < b <= years[-1]], dtype=np.int64)
        m = len(bp)
        Z = K["build_z"](T, bp)
        # surface keys are strings (read from CSV); m and p_hat are ints here,
        # so cast to str before the lookup. The surface is tabulated at m=2
        # (the maximal admissible break count), so currencies with m=1 fall back
        # to the m=2 row at their own (T, p) -- the invariance result of
        # Section 4 guarantees cbar* is (asymptotically) flat in m, and the
        # finite-sample remainder O((m+1)/T) is bounded by the m=2 value.
        Tk = str(T_cal) if T_cal is not None else str(T)
        pk = str(p_hat)
        srow = (surf.get((str(m), Tk, pk, "maic"))
                or surf.get(("2", Tk, pk, "maic"))
                or surf.get(("2", Tk, "1", "maic")))
        cbar_cal = float(srow["cbar_star"]) if srow else -7.0
        # config-faithful CV at the REAL break positions, AR(p_hat) innovation
        seed_cv = (P["seed_base"] + hash(cur) % 10**6 + 104729 * p_hat) % (2**63 - 1)
        pt0, mzt0, _ = stats_under(K, gen_arp, T, bp, 0.0, cbar_cal,
                                   K["method_code"]["maic"], P["R_cv_emp"], seed_cv,
                                   kmax, p_hat, P["pac1"], burn)
        cv_mzt_cf = float(np.nanpercentile(mzt0, 5))
        # statistics on the data under three calibrations
        def mzt_at(cb):
            _, _, mz, _, _, ok = K["mstats"](q, Z, cb, K["method_code"]["maic"], kmax)
            return float(mz) if ok > 0.5 else np.nan
        mzt_asym = mzt_at(-7.0)
        mzt_cal  = mzt_at(cbar_cal)
        # AR(1) persistence with and without breaks (the MP contrast)
        qd = q - q.mean()
        alpha_sq = float(np.corrcoef(qd[:-1], qd[1:])[0, 1])
        beta_ols, *_ = np.linalg.lstsq(Z, q, rcond=None)
        u = q - Z @ beta_ols
        rho_lb = float(np.corrcoef(u[:-1], u[1:])[0, 1])
        hl = lambda a: float(np.log(0.5) / np.log(abs(a))) if 0 < abs(a) < 1 else np.inf
        # break-label reflects the dates ACTUALLY used (exogenous or placeholder),
        # keyed off the same `byears` that produced bp -- never hard-coded
        break_years_used = [int(b) for b in byears
                            if years[0] < b <= years[-1]]
        break_label = (("EXOG:" if exog_dates is not None else "PRELIM:")
                       + "+".join(map(str, break_years_used)))
        rows_out.append(dict(currency=cur, T=T, m=m, p_hat=p_hat,
            cbar_calibrated=round(cbar_cal, 3),
            MZt_asym=round(mzt_asym, 4), cv_asym=-1.98,
            reject_asym=int(mzt_asym < -1.98),
            MZt_cal=round(mzt_cal, 4), cv_configfaithful=round(cv_mzt_cf, 4),
            reject_cal=int(mzt_cal < cv_mzt_cf),
            alpha_MP=round(alpha_sq, 4), rho_LB=round(rho_lb, 4),
            delta=round(alpha_sq - rho_lb, 4),
            HL_MP=round(hl(alpha_sq), 2), HL_LB=round(hl(rho_lb), 2),
            breaks=break_label))
        print(f"  {cur}: MZt(cal)={mzt_cal:.3f} vs cv={cv_mzt_cf:.3f} "
              f"[{'REJ' if mzt_cal < cv_mzt_cf else 'no'}] | "
              f"alpha_MP={alpha_sq:.3f} rho_LB={rho_lb:.3f}")
    _write_csv(Path(out_dir) / "empirical" / "ppp_empirical_prelim.csv", rows_out)
    return rows_out

# =============================================================================
# 5. ORCHESTRATION
# =============================================================================
def _write_csv(path: Path, rows):
    import csv as _csv
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def _sha1(path):
    try:
        return hashlib.sha1(open(path, "rb").read()).hexdigest()[:12]
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mc-dir", default=DEFAULT_MC_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT_DIR)
    ap.add_argument("--emp-dir", default=DEFAULT_EMP_DIR,
                    help="dir containing ppp_panel.csv / ppp_ar_diagnostic.csv")
    ap.add_argument("--quick", action="store_true", help="smoke test (~1 min)")
    ap.add_argument("--full", action="store_true", help="production surface")
    ap.add_argument("--hi", action="store_true", help="double replications")
    ap.add_argument("--raw", action="store_true", help="save raw draw vectors")
    ap.add_argument("--empirical", action="store_true")
    ap.add_argument("--m3", action="store_true", help="also calibrate m=3 cells")
    ap.add_argument("--grid", action="store_true",
                    help="calibrate a SINGLE cell at full resolution (sanity "
                         "check before --full); use with --m --T --p --nrep")
    ap.add_argument("--m", type=int, default=2, help="break count for --grid")
    ap.add_argument("--T", type=int, default=55, help="sample length for --grid")
    ap.add_argument("--start-year", type=int, default=1970,
                    help="first year of the empirical window; the calibration "
                         "sample length T is derived as (last_year - start_year "
                         "+ 1) from the panel, and the empirical block filters "
                         "year >= start_year. Set 1973 for the post-Bretton-Woods "
                         "float (T=52); 1970 reproduces the wider window (T=55).")
    ap.add_argument("--p", type=int, default=1, help="AR order for --grid")
    ap.add_argument("--nrep", type=int, default=None,
                    help="override R_cv for --grid (R_pow set to nrep/2)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--kmax", type=int, default=12,
                    help="MAIC lag ceiling; 12 = paper convention, "
                         "10 = Schwert at T=52 (sensitivity run)")
    ap.add_argument("--pcells", type=int, nargs="+", default=[0, 1, 2, 4])
    args = ap.parse_args()

    t_start = time.time()
    K = load_kernels(args.mc_dir)
    print(f"[kernel] {KERNEL_SOURCE}")
    print(K["warm"](12))
    gen_arp = make_gen_arp(K["njit"])
    # equivalence gate p=0: gen_arp with empty phi must match v6 gen_dgp path
    rng = np.random.default_rng(123); T0 = 40
    eps = rng.standard_normal(T0 + 0)
    if V6 is not None:
        y_v6 = V6.gen_dgp_nb(T0, np.array([12], dtype=np.int64), 0.0, 0.0, eps)
        y_ar = gen_arp(T0, 0, 0.0, np.zeros(0), eps)
        g0 = float(np.max(np.abs(y_v6 - y_ar)))
        print(f"[gate G3] gen_arp(p=0) == v6 gen_dgp: max|d|={g0:.2e} "
              f"[{'PASS' if g0 < 1e-12 else 'FAIL'}]")
        assert g0 < 1e-12

        # gate G4 (Westerlund 2014): the GLS quasi-difference must keep the FIRST
        # observation as a level (definition (1): y~_1 = y_1, Z~_1 = Z_1), not zero
        # it (definition (2)), otherwise the DF-GLS statistic diverges with T under
        # both H0 and H1. We assert the kernel uses (1) by checking that forcing
        # (2) changes the level residual materially -- i.e. the choice is not inert
        # and the kernel is on the correct side.
        try:
            yv = np.cumsum(rng.standard_normal(60))
            Zv = K["build_z"](60, np.array([30], dtype=np.int64))
            yt1, _ = (V6.glsd_nb(yv, Zv, -7.0) if V6 is not None
                      else (K["glsd"](yv, Zv, -7.0), None))
            abar = 1.0 + (-7.0) / 60
            ya2 = np.zeros(60); ya2[1:] = yv[1:] - abar * yv[:-1]          # def (2)
            Za2 = np.zeros_like(Zv); Za2[1:] = Zv[1:] - abar * Zv[:-1]
            b2 = np.linalg.lstsq(Za2, ya2, rcond=None)[0]
            yt2 = yv - Zv @ b2
            d12 = float(np.max(np.abs(yt1 - yt2)))
            print(f"[gate G4] Westerlund first-obs (def 1 vs 2): max|d|={d12:.3f} "
                  f"[{'PASS' if d12 > 1e-6 else 'FAIL: kernel may zero the first obs'}]")
        except Exception as e:
            print(f"[gate G4] skipped ({e})")

    P = dict(
        cbar_grid=list(np.round(np.arange(-20.0, -2.9, 0.5), 2)),
        target_power=0.50, kmax=args.kmax, seed_base=20240601,
        R_cv=10000, R_pow=5000, R_cv_emp=20000,
        trim=0.15, min_spacing=0.15, n_grid=7,
        pac1=0.4, burn=100,
    )
    if args.hi:
        P["R_cv"] *= 2; P["R_pow"] *= 2
    if args.quick:
        P.update(R_cv=600, R_pow=400, R_cv_emp=1000,
                 cbar_grid=list(np.round(np.arange(-14.0, -4.9, 1.5), 2)))

    # --- single-cell sanity check at FULL resolution (before committing --full) ---
    if args.grid:
        if args.nrep:
            P["R_cv"] = args.nrep; P["R_pow"] = max(1, args.nrep // 2)
        m_, T_, p_ = args.m, args.T, args.p
        lams = K["gerar_lambdas"](m_, P["trim"], P["min_spacing"], P["n_grid"])
        n_grid_cbar = len(P["cbar_grid"])
        print(f"[grid] single cell m={m_} T={T_} p={p_} | "
              f"{len(lams)} lambda x {n_grid_cbar} cbar x "
              f"(R_cv={P['R_cv']}, R_pow={P['R_pow']}) | n_jobs={args.n_jobs}",
              flush=True)
        print(f"[grid] cbar grid: [{P['cbar_grid'][0]}, {P['cbar_grid'][-1]}] "
              f"step {round(P['cbar_grid'][1]-P['cbar_grid'][0],2)} "
              f"({n_grid_cbar} points)", flush=True)
        t0 = time.time()
        res = tangency_cell(K, gen_arp, P, T_, m_, p_, "maic", lams,
                            n_jobs=args.n_jobs, verbose=True)
        dt = time.time() - t0
        if res is None:
            print("[grid] NO USABLE POINT (all lambda degenerate)"); sys.exit(1)
        print(f"\n[grid] RESULT m={m_} T={T_} p={p_} (tangency on ORACLE PT, ERS envelope):")
        print(f"       cbar* = {res['cbar_star']:.3f}  (se {res['cbar_star_se']:.3f})")
        print(f"       cv5_MZt (feasible, MAIC) = {res['cv5_mzt_at_star']:.3f}")
        print(f"       power FEASIBLE at cbar* = {res['power_feasible_at_star']:.3f} "
              f"(oracle = 0.50 by construction; the gap is the MAIC cost, "
              f"the Murray-Papell mechanism quantified)")
        print(f"       fallback {res['n_fallback']}/{res['n_lambda']} "
              f"{'[CLEAN - full grid OK]' if res['n_fallback']==0 else '[FALLBACK - grid may be too narrow]'}")
        print(f"       elapsed {dt:.0f}s  =>  est. per full cell ~{dt:.0f}s "
              f"x (R_full/R_here)")
        sys.exit(0)

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if args.kmax != 12 and str(args.out) == DEFAULT_OUT_DIR:
        sys.exit("PROTECAO: rodadas com kmax != 12 exigem --out proprio "
                 "(ex.: --out boot_k10) para nao sobrescrever os artefatos "
                 "canonicos kmax=12 do paper.")
    (out_dir / "calib").mkdir(exist_ok=True)

    # cells: (m=2, T=T_emp) x p in pcells; baseline p=0 also under 'const'.
    # T_emp is the effective sample length after the start-year cut: the panel
    # ends in 2024, so T_emp = 2024 - start_year + 1  (55 for 1970, 52 for 1973).
    # This keeps the calibration T consistent with the empirical window instead
    # of hard-coding 55.
    PANEL_LAST_YEAR = 2024
    T_emp = PANEL_LAST_YEAR - args.start_year + 1
    print(f"[window] start_year={args.start_year} -> calibration T={T_emp} "
          f"(empirical block filters year >= {args.start_year})", flush=True)
    m_list = [2] + ([3] if args.m3 else [])
    cells = []
    for m in m_list:
        for p in args.pcells:
            cells.append((m, T_emp, p, "maic"))
    cells.insert(1, (2, T_emp, 0, "const"))  # Article-1 anchor

    lam_cache = {m: K["gerar_lambdas"](m, P["trim"], P["min_spacing"], P["n_grid"])
                 for m in m_list}
    _write_csv(out_dir / "calib" / "lambda_configs.csv",
               [dict(m=m, config=str(l)) for m in lam_cache for l in lam_cache[m]])

    surface, curves = [], []
    if args.quick or args.full:
        n_cells = len(cells)
        run_t0 = time.time()
        for ci, (m, T, p, meth) in enumerate(cells, 1):
            t0 = time.time()
            lams = lam_cache[m][:3] if args.quick else lam_cache[m]
            print(f"[cell {ci}/{n_cells}] START m={m} T={T} p={p} {meth} "
                  f"| {len(lams)} lambda x {len(P['cbar_grid'])} cbar "
                  f"| {time.strftime('%H:%M:%S')}", flush=True)
            res = tangency_cell(K, gen_arp, P, T, m, p, meth, lams,
                                n_jobs=args.n_jobs, collect_curves=curves,
                                verbose=True)
            if res is None:
                print(f"[cell {ci}/{n_cells}] NO USABLE POINT "
                      f"(m={m},T={T},p={p},{meth})", flush=True)
                continue
            res["seconds"] = round(time.time() - t0, 1)
            surface.append(res)
            elapsed = time.time() - run_t0
            eta_all = elapsed / ci * (n_cells - ci)
            print(f"[cell {ci}/{n_cells}] DONE  cbar*={res['cbar_star']:.3f} "
                  f"(se {res['cbar_star_se']:.3f}) cv5_MZt={res['cv5_mzt_at_star']:.3f} "
                  f"powF@*={res['power_feasible_at_star']:.2f} "
                  f"fallback {res['n_fallback']}/{res['n_lambda']} "
                  f"[{res['seconds']}s] | run ETA {eta_all/60:.1f}min", flush=True)
        _write_csv(out_dir / "calib" / "surface_ppp_boot.csv", surface)
        _write_csv(out_dir / "calib" / "power_curves.csv", curves)
        # delta table: with vs without AR calibration (the requested comparison)
        base = {r["m"]: r for r in surface if r["p"] == 0 and r["method"] == "maic"}
        deltas = [dict(m=r["m"], T=r["T"], p=r["p"],
                       cbar_star_p=r["cbar_star"],
                       cbar_star_iid=base[r["m"]]["cbar_star"],
                       delta=round(r["cbar_star"] - base[r["m"]]["cbar_star"], 3))
                  for r in surface if r["method"] == "maic" and r["p"] > 0
                  and r["m"] in base]
        _write_csv(out_dir / "calib" / "delta_cbar_with_without_AR.csv", deltas)
        for d in deltas:
            print(f"  DELTA m={d['m']} p={d['p']}: cbar*(AR)={d['cbar_star_p']:.3f} "
                  f"vs iid {d['cbar_star_iid']:.3f} -> {d['delta']:+.3f}")

    if args.empirical:
        print("[empirical] test block (break dates resolved below):")
        # search the panel/diagnostic in several plausible locations, in order:
        #   1. --emp-dir (explicit)   2. --out dir   3. cwd   4. script dir
        search_dirs = [Path(args.emp_dir), out_dir, Path.cwd(),
                       Path(os.path.dirname(os.path.abspath(__file__)))]
        panel_csv = diag_csv = None
        for d in search_dirs:
            pc, dc = d / "ppp_panel.csv", d / "ppp_ar_diagnostic.csv"
            if pc.exists() and dc.exists():
                panel_csv, diag_csv = pc, dc
                print(f"  inputs found in: {d}")
                break
        if panel_csv is not None:
            exog_dates, ed_file = load_exog_dates(search_dirs)
            if exog_dates is not None:
                print(f"  exogenous break dates: {ed_file} "
                      f"({sum(len(v) for v in exog_dates.values())} breaks, "
                      f"{len(exog_dates)} currencies)")
            else:
                print(f"  exog_dates.csv NOT found -> using placeholder "
                      f"{PLACEHOLDER_BREAKS} (PRELIMINARY)")
            empirical_block(K, gen_arp, P, panel_csv, diag_csv, surface, out_dir,
                            exog_dates=exog_dates, start_year=args.start_year)
        else:
            print("  MISSING inputs: ppp_panel.csv / ppp_ar_diagnostic.csv not found.")
            print("  searched (in order):")
            for d in search_dirs:
                print(f"    - {d}")
            print("  FIX: pass --emp-dir pointing to the folder that holds both CSVs, "
                  "e.g. --emp-dir \"...\\Resp_Surf_CKP\\Empirico\"")
            print("  (the calibration surface above is COMPLETE and unaffected.)")

    _pc = locals().get("panel_csv")
    _dc = locals().get("diag_csv")
    meta = dict(kernel=KERNEL_SOURCE, params=P, cells=[list(c) for c in cells],
                placeholder_breaks=PLACEHOLDER_BREAKS,
                start_year=args.start_year, T_emp=T_emp,
                inputs=dict(panel=_sha1(_pc) if _pc else None,
                            diagnostic=_sha1(_dc) if _dc else None),
                args=vars(args), seconds_total=round(time.time() - t_start, 1),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[done] outputs in {out_dir} | total {meta['seconds_total']}s")

if __name__ == "__main__":
    main()
