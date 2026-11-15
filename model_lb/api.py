"""The Model LB test of the paper applied to a user's own series.

Every number is computed by the code that produces the paper's application
(mlb/ppp_application.py, ppp_sieve.py, ppp_half_lives.py,
ppp_variance_scope.py and mlb/envelope_exact.py, mlb/kernel_numba.py); this
module only removes the application's fixed sample (T = 52 from 1973) and its seed
registry, so that the same procedure runs on any series, at any break dates, with any
seed. With the application's seeds it returns the application's numbers bit for bit.

THE TEST. y_t = mu_0 + sum_j delta_j DU_jt + u_t, DU_jt = 1{t >= b_j} (t = 0, ...,
T - 1), where b_j, the break position, is the number of observations before break j:
the break date is the first observation of the new regime. Null: u_t has a unit root.

  (I)   MZ_t at c-bar = -7 (GLS detrending on (1, DU_1, ..., DU_m), s^2_AR with the
        lag chosen by the MAIC, k_max = 12) against the asymptotic critical value
        -1.98;
  (II)  the same statistic against the 5% quantile of its null distribution simulated
        at the series' own configuration under the series' own sieve: the ADF
        regression of the level-break residual with p lagged differences, p chosen by
        the BIC (k_max = 10) on a common sample; the p-value is the share of null
        draws at or below the statistic. The true size of -1.98 is the share of null
        draws at or below -1.98;
  (III) MZ_t at the exact half-power point c-bar_1/2 of the configuration (the c-bar
        at which the Gaussian envelope's power is one half, computed by Imhof's
        formula) against its own simulated null.

THE NULL DRAWS. Replication r draws eps = standard_normal(T + 100) from
numpy.random.default_rng(seed + r); the AR nuisance Delta u_t = sum_j phi_j Delta
u_{t-j} + eps_t starts at zero 100 periods before the sample and u_0 = 0. A draw on
which the kernel flags a degenerate statistic is excluded from the critical value and
the p-value and counted. The Monte Carlo standard error of a critical value is the
larger of two block estimates (8 and 20 contiguous blocks of draws, the critical value
re-estimated in each block); that of a p-value or a size is binomial. Draws do not
depend on the number of workers.

SCALE AND DEGENERATE SERIES (prepare). MZ_t, the p-values and the decisions do not
depend on the units of y, but the numerical kernel has absolute floors (s^2_AR at
least 1e-10, ridges of 1e-12 in the lag regressions), which move the statistic of a
series whose first differences are tiny, and the null is simulated at unit scale. The
scale of a series is the root mean square of its first differences within regimes
(the differences that cross a break are left out). Inside SCALE_BAND = [1e-3, 1e3]
every function computes on the series as given. Outside the band
the series is divided by its scale before any computation, and the result says so
(key "scale"); the outputs in the units of y (residuals, s^2, BIC values, variances)
are converted back (GBP times 1e-4, 1e-8 and 1e6 gives GBP's statistic, sieve,
p-values and decisions). A series that is
constant, or constant within every regime (it moves only at the breaks), or whose
within-regime differences are below 1e-12 of its magnitude (rounding of the stored
values) is refused with a ValueError.

WARNINGS. Each result carries a list "warnings": T below T_GRID_MIN = 45, where the
paper's grids start (at small T the sieve is often non-stationary or at its bound);
the BIC order at its bound k_max = 10; the MAIC lag at its bound k_max = 12; a
half-power point outside K = [-20, -3], the set on which the paper defines it.

Functions: prepare, statistic, sieve, null_distribution, summarize_null, column_ii,
half_power_point, column_iii, model_lb_test, half_life, holm,
variance_profile_size. Break positions are validated by design_positions.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np

from mlb import app_seeds as A
from mlb import ppp_application as P
from mlb import ppp_half_lives as HL
from mlb import ppp_sieve as S
from mlb import kernel_numba as KN

CBAR = P.CBAR  # -7: the paper's c-bar for every configuration
KMAX_MAIC = P.KMAX  # 12: the MAIC's largest lag
KMAX_BIC = S.K_MAX_BIC  # 10: the sieve's largest order
CV_ASY = P.CV_ASY  # -1.98: the asymptotic 5% critical value of MZ_t, column (I)
R_DEFAULT = P.R_NULL  # 200,000 replications per null
B_HALF_LIFE = HL.B_MAIN  # 19,999 bootstrap replicates per grid node
B_WILD = 19_999  # the wild sieve bootstrap of the variance scope
SEED_DEFAULT = 20260701
SEED_STRIDE = A.STRIDE  # 10**9: the default seed of column (III) is seed + 10**9
MIN_T = 24  # the BIC's ADF regression at p = 10 needs T - 11 > 12 observations
T_GRID_MIN = 45  # the paper's grids start at T = 45; below it a warning
MIN_R = 1_000
SCALE_BAND = (1e-3, 1e3)  # the scale inside which the series is used as given
ROUNDING_REL = 1e-12  # within-regime variation below this share of |y| is rounding
K_SET = (-20.0, -3.0)  # K, on which the paper defines the half-power point
NO_HALF_POWER = ("no half-power point: the exact envelope's power crosses 1/2 in "
                 "neither (-25, -1) nor (-40, -1), or Imhof's quantile fails at this "
                 "configuration (regimes of very few observations)")


# ================================================================== inputs
def _series(y):
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float64).ravel())
    if y.size < MIN_T:
        raise ValueError(f"the series has {y.size} observations; at least {MIN_T}")
    if not np.all(np.isfinite(y)):
        raise ValueError("the series contains NaN or inf")
    return y


def design_positions(T, break_pos):
    """The break positions as a sorted tuple of ints, each in 1 .. T - 1 (a break
    position is the number of observations before the break); duplicates refused."""
    bp = tuple(sorted(int(b) for b in break_pos))
    if len(set(bp)) != len(bp):
        raise ValueError(f"duplicate break positions {bp}")
    if any(b < 1 or b > T - 1 for b in bp):
        raise ValueError(f"break positions must lie in 1..{T - 1}: {bp}")
    return bp


def _within_differences(y, bp):
    edges = (0,) + tuple(bp) + (len(y),)
    return np.concatenate([np.diff(y[a:b]) for a, b in zip(edges[:-1], edges[1:])])


def prepare(y, break_pos):
    """The series as the other functions compute on it, and its scale.

    The scale is the root mean square of the first differences within regimes. A
    series that is constant, constant within every regime, or whose within-regime
    differences are below ROUNDING_REL of its largest |value| is refused
    (ValueError). Inside SCALE_BAND the series is returned as given; outside it, divided
    by its scale (MZ_t, the p-values and the decisions do not depend on the scale; the
    kernel's absolute floors do).

    Returns (y_used, break_pos, scale) with scale = dict(rms, divisor, rescaled,
    band); divisor is 1.0 when the series is used as given."""
    y = _series(y)
    bp = design_positions(len(y), break_pos)
    d = _within_differences(y, bp)
    if d.size < 2:
        raise ValueError("the breaks leave fewer than two first differences inside "
                         "the regimes")
    dmax = float(np.max(np.abs(d)))
    if dmax == 0.0:
        what = ("constant" if not bp else
                "constant within every regime (it moves only at the breaks)")
        raise ValueError(f"the series is {what}: the test needs variation around the "
                         "regime means")
    rms = dmax * float(np.sqrt(np.mean((d / dmax) ** 2)))
    if rms <= ROUNDING_REL * float(np.max(np.abs(y))):
        raise ValueError(
            f"the first differences within regimes (root mean square {rms:.3g}) are "
            f"below {ROUNDING_REL:g} of the series' magnitude: what varies is the "
            "rounding of the stored values"
        )
    lo, hi = SCALE_BAND
    if lo <= rms <= hi:
        return y, bp, dict(rms=rms, divisor=1.0, rescaled=False, band=SCALE_BAND)
    return y / rms, bp, dict(rms=rms, divisor=rms, rescaled=True, band=SCALE_BAND)


def _merge(*lists):
    out = []
    for w in lists:
        for x in w or ():
            if x not in out:
                out.append(x)
    return out


def _warn_T(T):
    if T >= T_GRID_MIN:
        return []
    return [f"T = {T} is below {T_GRID_MIN}, where the paper's grids start: at small T "
            "the BIC sieve is often at its bound or non-stationary; read the result "
            "with care"]


def _warn_maic(k):
    if k < KMAX_MAIC:
        return []
    return [f"the MAIC lag is at its bound k_max = {KMAX_MAIC}: s^2_AR may need more "
            "lags than the paper allows"]


def _warn_bic(p, kmax=KMAX_BIC):
    if p < kmax:
        return []
    return [f"the BIC order is at its bound k_max = {kmax}: the sieve may need more "
            "lags than the paper allows, and the simulated null may then miss part of "
            "the serial correlation"]


def _warn_K(cbar_half):
    lo, hi = K_SET
    if lo <= cbar_half <= hi:
        return []
    return [f"c-bar_1/2 = {cbar_half:.4f} lies outside K = [{lo:g}, {hi:g}], the set "
            "on which the paper defines the half-power point and shows the envelope "
            "to be strictly decreasing; it is the root of the envelope's power = 1/2 "
            "found beyond K"]


@contextlib.contextmanager
def _bound(module, **values):
    """Binds module-level constants for the duration of a call and restores them.

    The application's modules read their sample length (and ppp_half_lives its seed
    registry) from module globals; this is how their functions run at another T. A
    process runs one call at a time (joblib workers are processes), so the binding is
    never seen by a concurrent call."""
    saved = {k: getattr(module, k) for k in values}
    for k, v in values.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(module, k, v)


def _run(fn, args, n, jobs):
    """fn(*args, i0, i1) over n replications: jobs workers, four chunks per worker;
    the draws do not depend on jobs."""
    if jobs <= 1:
        return fn(*args, 0, n)
    from joblib import Parallel, delayed

    edges = np.linspace(0, n, jobs * 4 + 1).astype(int)
    parts = Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
        delayed(fn)(*args, int(a), int(b))
        for a, b in zip(edges[:-1], edges[1:])
        if b > a
    )
    return np.concatenate(parts)


# ================================================================== (1) the statistic
def _statistic(y, bp, cbar):
    T = len(y)
    K, _ = P.kernels()
    Z = K["build_z"](T, np.asarray(bp, dtype=np.int64))
    mza, msb, mzt, pt, mpt, ok = K["mstats"](
        y, Z, float(cbar), K["method_code"]["maic"], KMAX_MAIC
    )
    k = int(KN.maic_select_k_nb(KN.ols_detrend_nb(y, Z), KMAX_MAIC))
    ok = bool(ok > 0.5)
    vals = [float(v) if ok else float("nan") for v in (mzt, mza, msb, pt, mpt)]
    return dict(
        zip(("mzt", "mza", "msb", "pt", "mpt"), vals),
        maic_k=k,
        cbar=float(cbar),
        T=T,
        break_pos=bp,
        ok=ok,
    )


def statistic(y, break_pos, cbar=CBAR):
    """MZ_t of y at the level-break configuration, GLS detrending at c-bar, s^2_AR
    with the MAIC lag (k_max = 12; the lag from the OLS-detrended regression, s^2 on
    the GLS-detrended series), computed on prepare(y, break_pos).

    Returns dict(mzt, mza, msb, pt, mpt, maic_k, cbar, T, break_pos, ok, scale,
    warnings); the statistics are NaN and ok False if the kernel flags a degenerate
    series."""
    yy, bp, sc = prepare(y, break_pos)
    st = _statistic(yy, bp, cbar)
    st.update(scale=sc, warnings=_merge(_warn_T(len(yy)), _warn_maic(st["maic_k"])))
    return st


# ================================================================== (2) the sieve
def _sieve(y, bp, kmax, first_period):
    u = S.lb_residual(y, bp)
    p, bics = S.bic_order(u, kmax)
    fit = S.sieve_fit(u, p, first_period)
    return dict(
        p=int(p),
        bic=[float(bics[j]) for j in range(kmax + 1)],
        phi=np.asarray(fit["phi"], dtype=float),
        se=np.sqrt(np.diag(fit["V"])) if p else np.zeros(0),
        V=fit["V"],
        s2=float(fit["s2"]),
        sum_phi=float(fit["sum_phi"]),
        max_root=float(fit["max_root"]),
        stationary=bool(fit["stationary"]),
        flag09=bool(fit["flag09"]),
        u=u,
        e=fit["e"],
        e_periods=fit["e_years"],
    )


def sieve(y, break_pos, kmax=KMAX_BIC, first_period=0):
    """The sieve of the null: u_hat, the OLS residual of y on (1, DU); the ADF
    regression of Delta u_hat on (1, u_hat_{t-1}, Delta u_hat_{t-1..t-p}); p_hat by
    the BIC over p = 0..kmax on the common sample t = kmax + 1, ..., T - 1 (smallest p
    on a tie); phi_hat, the p_hat lagged-difference coefficients on the regression's
    own sample, with V = s^2 (X'X)^-1, s^2 = SSR / (n - p_hat - 2).

    Computed on prepare(y, break_pos); u, e, s2 and the BIC values are returned in the
    units of y. Returns dict(p, bic (list over p = 0..kmax), phi, se, V, s2, sum_phi,
    max_root, stationary, flag09 (sum phi_hat >= 0.9), u, e, e_periods, scale,
    warnings); e is the residual of the ADF regression at p_hat and e_periods its
    periods, first_period + t."""
    yy, bp, sc = prepare(y, break_pos)
    sv = _sieve(yy, bp, kmax, first_period)
    if sc["rescaled"]:
        f = sc["divisor"]
        sv.update(u=sv["u"] * f, e=sv["e"] * f, s2=sv["s2"] * f * f,
                  bic=[b + 2.0 * math.log(f) for b in sv["bic"]])
    sv.update(scale=sc, warnings=_merge(_warn_T(len(yy)), _warn_bic(sv["p"], kmax)))
    return sv


# ================================================================== (3) column (II)
def _null_chunk(T, bp, phi, cbar, seed, r0, r1):
    with _bound(P, T=int(T)):
        arms = (("u0zero", 0.0, tuple(phi), float(cbar)),)
        return P._chunk(tuple(bp), arms, int(seed), int(r0), int(r1))[:, 0]


def null_distribution(T, break_pos, phi, cbar=CBAR, R=R_DEFAULT, seed=SEED_DEFAULT,
                      jobs=1):
    """R draws of MZ_t at c-bar under the null with the sieve phi (an empty phi is
    the i.i.d. null), at sample length T and the break positions; NaN marks a
    degenerate draw. Replication r draws from numpy.random.default_rng(seed + r)."""
    bp = design_positions(int(T), break_pos)
    if int(R) < 1:
        raise ValueError("R must be positive")
    phi = tuple(float(x) for x in phi)
    return _run(_null_chunk, (int(T), bp, phi, float(cbar), int(seed)), int(R), jobs)


def summarize_null(null, stat):
    """The statistics of a null sample against the data's statistic: the 5% critical
    value (linear interpolation) with its block standard errors, the p-value (the share
    of finite draws <= stat, without +1) with its binomial standard error, the
    decision stat < cv, and the true size of -1.98 with its standard error."""
    null = np.asarray(null, dtype=float)
    cv = P.cv5(null)
    se8, se20, se = P.block_se(P.cv_blocks(null))
    p, p_se, n = P.pvalue(null, float(stat))
    fin = np.isfinite(null)
    size = float(np.mean(null[fin] <= CV_ASY))
    return dict(
        R=int(len(null)),
        cv=cv,
        cv_se8=se8,
        cv_se20=se20,
        cv_se=se,
        p=p,
        p_se=p_se,
        n_valid=n,
        n_nan=int(len(null) - n),
        reject=bool(stat < cv),
        size_asy=size,
        size_asy_se=float(np.sqrt(size * (1 - size) / fin.sum())),
    )


def _check_R(R):
    if int(R) < MIN_R:
        raise ValueError(f"R = {R}: at least {MIN_R} replications")


def column_ii(y, break_pos, R=R_DEFAULT, seed=SEED_DEFAULT, jobs=1):
    """Columns (I) and (II) and the true size of -1.98 for the series.

    Returns dict(statistic=statistic(...), sieve=sieve(...), reject_I, scale,
    warnings, status, and, when the sieve is stationary, simulated=summarize_null(...)
    and seed); with a non-stationary phi_hat nothing is simulated (status says so)."""
    _check_R(R)
    st = statistic(y, break_pos)
    sv = sieve(y, break_pos)
    out = dict(statistic=st, sieve=sv, reject_I=bool(st["mzt"] < CV_ASY),
               scale=st["scale"], warnings=_merge(st["warnings"], sv["warnings"]))
    if not st["ok"]:
        out["status"] = "degenerate statistic"
        return out
    if not sv["stationary"]:
        out["status"] = "not simulated (non-stationary phi_hat)"
        return out
    null = null_distribution(st["T"], st["break_pos"], sv["phi"], CBAR, R, seed, jobs)
    out.update(status="ok", seed=int(seed), simulated=summarize_null(null, st["mzt"]))
    return out


# ================================================================== (4) column (III)
def half_power_point(T, break_pos):
    """The exact half-power point c-bar_1/2 of the configuration: the c-bar at which
    the power of the Gaussian point-optimal envelope (sigma known, u_0 = 0) at the 5%
    level equals 1/2, by Imhof's formula (envelope_exact.half_power_point) with the
    fractions lambda_j = (b_j + 0.5) / T, the bracket (-25, -1), then (-40, -1), and
    xtol 1e-9, as in the application.

    The paper defines c-bar_1/2 on K = [-20, -3]; a root outside K is returned with
    in_K False and a warning. Returns dict(cbar_half, b_T, lam, bracket, in_K,
    warnings)."""
    T = int(T)
    bp = design_positions(T, break_pos)
    with _bound(P, T=T):
        try:
            r = P._cbar_one(bp)
        except SystemExit as e:
            raise RuntimeError(f"{NO_HALF_POWER} [{e}]") from None
    cb = float(r["cbar"])
    w = _warn_K(cb)
    return dict(cbar_half=cb, b_T=float(r["bT"]), lam=tuple(r["lam"]),
                bracket=(r["lo"], r["hi"]), in_K=not w, warnings=w)


def column_iii(y, break_pos, R=R_DEFAULT, seed=SEED_DEFAULT + SEED_STRIDE, jobs=1,
               cbar_half=None, sieve_fit=None):
    """Column (III): MZ_t at c-bar_1/2 against its null simulated under the sieve at
    the same c-bar. cbar_half and sieve_fit are computed if not given.

    Returns dict(cbar_half, statistic=statistic(..., cbar_half), warnings, status and,
    when the sieve is stationary, simulated=summarize_null(...) and seed)."""
    _check_R(R)
    yy, bp, _ = prepare(y, break_pos)
    if cbar_half is None:
        cbar_half = half_power_point(len(yy), bp)["cbar_half"]
    sv = sieve(y, bp) if sieve_fit is None else sieve_fit
    st = statistic(y, bp, cbar=cbar_half)
    out = dict(cbar_half=float(cbar_half), statistic=st,
               warnings=_merge(st["warnings"], _warn_K(float(cbar_half))))
    if not st["ok"]:
        out["status"] = "degenerate statistic"
        return out
    if not sv["stationary"]:
        out["status"] = "not simulated (non-stationary phi_hat)"
        return out
    null = null_distribution(len(yy), bp, sv["phi"], cbar_half, R, seed, jobs)
    out.update(status="ok", seed=int(seed), simulated=summarize_null(null, st["mzt"]))
    return out


def model_lb_test(y, break_pos, R=R_DEFAULT, seed=SEED_DEFAULT, seed_iii=None,
                  jobs=1, iii=True):
    """Columns (I), (II) and, if iii, (III) of the paper's table for one series: the
    dict of column_ii with the keys "iii" (column_iii's dict, or None) and
    "half_power"; "warnings" collects every warning. seed_iii defaults to
    seed + 10**9."""
    out = column_ii(y, break_pos, R, seed, jobs)
    out["iii"] = None
    if iii:
        s3 = int(seed) + SEED_STRIDE if seed_iii is None else int(seed_iii)
        try:
            hp = half_power_point(out["statistic"]["T"], out["statistic"]["break_pos"])
        except RuntimeError as e:
            out["half_power"] = None
            out["iii"] = dict(cbar_half=float("nan"), statistic=None, warnings=[],
                              status=f"not computed: {NO_HALF_POWER}", error=str(e))
            return out
        out["half_power"] = hp
        out["iii"] = column_iii(y, break_pos, R, s3, jobs, hp["cbar_half"],
                                out["sieve"])
        out["warnings"] = _merge(out["warnings"], hp["warnings"],
                                 out["iii"]["warnings"])
    return out


# ================================================================== (5) half-lives
class _NodeSeeds:
    """Stands in for the seed registry inside ppp_half_lives: grid node slot r (node
    k uses r = k + 200) draws from numpy.random.default_rng(seed + r)."""

    def __init__(self, seed):
        self.seed = int(seed)

    def rng(self, item, r):
        return np.random.default_rng(self.seed + int(r))


def half_life(y, break_pos, scheme="rec", model="LB", p=None, B=B_HALF_LIFE,
              seed=SEED_DEFAULT):
    """The grid-t median-unbiased estimate of the sum of the AR coefficients alpha and
    its 95% confidence set (Hansen 1999), with the half-lives HL = ln 0.5 / ln alpha,
    as in the paper (ppp_half_lives.grid_t).

    model "LB" works on the level-break residual of y, "MP" on y demeaned (no
    breaks); the AR(p + 1) in levels with a constant, p the sieve order p_hat of the
    level-break residual unless given (the paper uses it for both models). scheme
    "rec": the recursive bootstrap (residuals drawn with replacement), "wild":
    Rademacher weights. Nodes alpha_k = k / 200 from k = 100 to the cap k = 210
    (alpha = 1.05), extended downward by blocks of 20; B replicates per node, with
    (B + 1) / 40 an integer (19,999 by default). Node k draws from
    numpy.random.default_rng(seed + k + 200). Computed on prepare(y, break_pos)
    (alpha and the half-lives do not depend on the scale).

    Returns ppp_half_lives.grid_t's summary (alpha_ols, alpha_mu, hl_mu, alpha_lo,
    alpha_hi, hl_lo, hl_hi, k_lo, k_hi, upper_open, ci_above_cap, mu_at_cap, ...) with
    p, scheme, model, scale and warnings added, and the nodes under "nodes". Raises
    RuntimeError when the grid reaches its lower bound (alpha = -1) or the set is
    empty."""
    if scheme not in ("rec", "wild") or model not in ("LB", "MP"):
        raise ValueError("scheme is 'rec' or 'wild'; model is 'LB' or 'MP'")
    if int(B) < 39 or (int(B) + 1) % 40:
        raise ValueError(f"B = {B}: (B + 1) / 40 must be an integer")
    yy, bp, sc = prepare(y, break_pos)
    T = len(yy)
    warnings = _warn_T(T)
    if p is None:
        p = _sieve(yy, bp, KMAX_BIC, 0)["p"]
        warnings = _merge(warnings, _warn_bic(p))
    cell_bp = bp if model == "LB" else ()
    with _bound(HL, T=T, A=_NodeSeeds(seed)):
        try:
            summ, nodes = HL.grid_t(yy, cell_bp, int(p), "user", scheme, int(B))
        except SystemExit as e:
            raise RuntimeError(str(e)) from None
    return dict(summ, p=int(p), scheme=scheme, model=model, B=int(B), seed=int(seed),
                scale=sc, warnings=warnings, nodes=nodes)


def holm(pvalues):
    """Holm's step-down adjustment across series: {name: p} -> {name: adjusted p}
    (ties in p broken by name)."""
    return P.holm({k: float(v) for k, v in pvalues.items()})


# ================================================================== (5) variance scope
def _vnull_chunk(T, bp, phi, rel, seed, r0, r1):
    from mlb import ppp_variance_scope as VS

    with _bound(VS, T=int(T)):
        return VS._vchunk(tuple(bp), tuple(phi), np.asarray(rel), int(seed), r0, r1)


def _vwild_chunk(T, bp, phi, e, seed, b0, b1):
    from mlb import ppp_variance_scope as VS

    with _bound(VS, T=int(T)):
        return VS._wchunk(tuple(bp), tuple(phi), np.asarray(e), int(seed), b0, b1)


def variance_profile_size(y, break_pos, float_breaks=(), first_period=0,
                          R=R_DEFAULT, seed=SEED_DEFAULT, B=B_WILD, seed_wild=None,
                          cv_ii=None, jobs=1):
    """What a variance that changes at the regime dates does to columns (I) and (II)
    (the paper's variance scope, rule V).

    Rule V, on the break dates before any variance is estimated: a variance regime has
    at least h = ceil(0.15 T) observations; a date whose first or last regime is
    shorter is dropped; of two dates less than h apart the adoption of a float stays
    (float_breaks: the break positions that are adoptions of a float), otherwise the
    earlier. sigma_k^2 is the mean of the squared sieve residuals e_t over regime k;
    the profile rel_t = sigma_k(t) / sqrt(mean sigma^2). The null is simulated twice
    on the same draws (seed): "homo", the null of column (II), and "profile", eps
    scaled by rel_t (the burn-in by the first regime's scale). The wild sieve
    bootstrap (B replicates, Rademacher weights, seed_wild, default seed + 10**9) is
    column (IV-W). first_period labels the periods of the dates returned. Computed on
    prepare(y, break_pos); sigma2 is returned in the units of y.

    Returns dict(variance_dates (periods), variance_positions, n_residuals, status,
    scale, warnings, and when status is "ok": sigma2, variance_ratio, rel, homo,
    profile (with size_I = the size of -1.98, size_IIown = the size of (II) at the
    homoscedastic arm's own critical value, and size_IIreg at cv_ii if given), wild)."""
    from mlb import ppp_variance_scope as VS

    _check_R(R)
    yy, bp, sc = prepare(y, break_pos)
    T = len(yy)
    if not set(int(b) for b in float_breaks) <= set(bp):
        raise ValueError("float_breaks must be among the break positions")
    start = int(first_period)
    sv = _sieve(yy, bp, KMAX_BIC, start)
    st = _statistic(yy, bp, CBAR)
    taus = tuple(start + b for b in bp)
    floats = {start + int(b) for b in float_breaks}
    vd = VS.variance_dates(taus, floats, start=start, n_obs=T, h=math.ceil(0.15 * T))
    out = dict(variance_dates=vd, variance_positions=tuple(t - start for t in vd),
               mzt=st["mzt"], p=sv["p"], phi=sv["phi"], scale=sc,
               warnings=_merge(_warn_T(T), _warn_bic(sv["p"]),
                               _warn_maic(st["maic_k"])))
    if not sv["stationary"]:
        out["status"] = "not simulated (non-stationary phi_hat)"
        return out
    fit = dict(e=sv["e"], e_years=sv["e_periods"])
    with _bound(VS, T=T, START=start):
        pf = VS.profile(fit, vd)
    out["n_residuals"] = pf["n_k"]
    if pf["status"] != "ok":
        out["status"] = pf["status"]
        return out
    phi = tuple(float(x) for x in sv["phi"])
    vn = _run(_vnull_chunk, (T, bp, phi, tuple(pf["rel"]), int(seed)), int(R), jobs)
    sw = int(seed) + SEED_STRIDE if seed_wild is None else int(seed_wild)
    vw = _run(_vwild_chunk, (T, bp, phi, tuple(sv["e"]), sw), int(B), jobs)
    stat = st["mzt"]
    f2 = sc["divisor"] ** 2
    out.update(
        status="ok",
        sigma2=pf["s2"] * f2 if sc["rescaled"] else pf["s2"],
        variance_ratio=pf["ratio"],
        rel=pf["rel"],
        R=int(R),
        B=int(B),
        seed=int(seed),
        seed_wild=sw,
        homo=VS.stats_of(vn[:, 0], stat, cv_ii),
        profile=VS.stats_of(vn[:, 1], stat, cv_ii, cv_own=P.cv5(vn[:, 0])),
        wild=VS.stats_of(vw[:, 0], stat, cv_ii),
    )
    return out
