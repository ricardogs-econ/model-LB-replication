# -*- coding: utf-8 -*-
"""
================================================================================
src/mlb/kernel_numba.py -- Numerical kernel for the Model LB point-optimal GLS unit-root
               test (Carrion-i-Silvestre, Kim & Perron 2009, no-trend case).
================================================================================

This is the SINGLE canonical implementation of the test and its finite-sample
calibration used by every replication driver in this package
(replicate_section3_4.py, replicate_section5.py, replicate_section6.py) and by
the stand-alone tool run_model_lb.py. It is a library, not a script to run
directly (though `python src/mlb/kernel_numba.py --selftest` runs its validation gates).

Model LB: y_t = Z_t' theta + u_t, u_t = (1 + c/T) u_{t-1} + eps_t, with the
no-trend regressor set Z_t = [1, DU_{1,t}, ..., DU_{m,t}] where
DU_{j,t} = 1{t > tau_j}. Under GLS quasi-differencing each level dummy becomes
an impulse, orthogonal across break dates; the point-optimal test is
asymptotically invariant to the number and location of the breaks and the
optimal detrending parameter keeps the ERS (1996) demeaned value c-bar = -7 for
every configuration. In finite samples the optimum departs from -7 by an
O((m+1)/T) remainder, which this kernel calibrates.

CONTENTS
    Numerical kernels (@njit): build_z_nb, glsd_nb, ols_detrend_nb,
        s2ar_const_nb, s2ar_maic_nb, mstats_nb, gen_dgp_nb.
    Calibration driver: calibrate_config / run_grid / aggregate (the c-bar(m,T)
        surface with the critical values of the five M-class statistics).
    Design helpers: make_lambdas, enumerate_configs, config_seed, warm-up.
    Validation: `--selftest` (empirical size ~ 0.05; magnitude-invariance
        |dMZt| ~ 1e-16; power at c=-10 > size; DU = 1{t>tau}).

DETERMINISM
    All Monte Carlo uses config_seed(P,T,m,lambda) = seed_base + disjoint
    offsets; results are bit-reproducible on a fixed platform for a given R.

REFERENCE
    Carrion-i-Silvestre, J.L., Kim, D., Perron, P. (2009). GLS-based unit root
    tests with multiple structural breaks under both the null and the
    alternative hypotheses. Econometric Theory 25(6), 1754-1792.
================================================================================
"""
from __future__ import annotations
import os, sys, time, pickle, argparse, warnings, itertools, zlib
import numpy as np

# -----------------------------------------------------------------------------
# Numba: import and REAL application. If unavailable, a pure-Python njit shim
# with identical semantics is used, and the script says so at startup.
# -----------------------------------------------------------------------------
try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    prange = range
    def njit(*args, **kwargs):
        # Decorator shim: supports @njit and @njit(...) forms.
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

warnings.filterwarnings("ignore")

# =============================================================================
# DEFAULTS  (documented)
# =============================================================================
DEFAULTS = dict(
    T_grid       = [30, 45, 50, 60, 80, 100, 150, 200, 300],
    m_grid       = [0, 1, 2, 3, 4, 5],
    m_min_T      = {0: 30, 1: 30, 2: 30, 3: 45, 4: 60, 5: 80},
    trim         = 0.15,
    min_spacing  = 0.15,
    n_grid       = 7,
    cbar_grid    = list(np.round(np.arange(-20.0, -2.9, 0.5), 2)),
    target_power = 0.50,
    R_cv         = 10000,   # null reps for critical values
    R_pow        = 5000,    # alternative reps for power (SE ~ 0.0071 at p=0.5)
    R_curve      = 3000,    # reps per point on the (diagnostic) power curves
    c_grid_power = list(np.round(np.arange(-30.0, -1.9, 2.0), 1)),
    kmax         = 12,
    sigma2_methods = ['const', 'maic'],   # 'const' is the headline; 'maic' robustness
    compute_power_curves = True,
    save_raw_vectors     = True,
    seed_base    = 20240601,
    checkpoint_dir = "cbar_checkpoints",
    # ---- per-configuration refinement of the tangency search ----------------
    # Motivation: m=0 has NO averaging over break locations lambda (zero breaks
    # => a single configuration), so it is the only grid row unprotected against
    # single-draw Monte Carlo noise. The power curve there has local slope
    # ~0.08-0.10 per unit of c-bar near power=0.5 (measured), so one MC error of
    # ~0.008 (R_pow=5000) displaces the interpolated tangency by ~0.10 in c-bar.
    # With the uniform 0.5 grid this shows up as a few-tenths wobble of m=0
    # around the ERS constant -7.0, largest at the T extremes. The fix is purely
    # a precision allocation: cheap configurations (few lambda, low m) get more
    # replications and a finer c-bar grid local to their expected optimum. This
    # changes NOTHING in the estimand -- only the Monte Carlo resolution with
    # which the same fixed point is located -- and leaves m>=1 (already smooth,
    # protected by lambda-averaging) at the production settings.
    refine_cheap_configs = True,
    refine_max_lambdas   = 0,      # refine ONLY m=0 (len(lambda)==0): the single
                                   # row with no lambda-averaging, hence the only
                                   # one exposed to single-draw noise. m>=1 already
                                   # averages over 7 lambda configs and is smooth;
                                   # refining it would only burn MC for no gain.
                                   # (Set to 1 to also refine each individual m=1
                                   # configuration, e.g. for a robustness check.)
    refine_R_cv          = 40000,  # null reps for refined configs (4x)
    refine_R_pow         = 40000,  # power reps for refined configs (8x; SE ~ 0.0025)
    refine_grid_step     = 0.1,    # finer c-bar step for refined configs (vs 0.5)
    refine_grid_halfwidth= 1.5,    # c-bar window half-width around the coarse
                                   # optimum. The coarse tangency already locates
                                   # the optimum to within ~0.5, so +-1.5 (31 grid
                                   # points) brackets it with margin. A wider
                                   # window wastes refine_R_pow reps on points far
                                   # from power=0.5 (where power is ~0 or ~1 and
                                   # carries no information about the crossing).
    # ---- native seed-averaging of the m=0 row --------------------------------
    # m=0 has NO lambda-averaging (zero breaks => one configuration), so a single
    # seed is exposed to ~0.07 cross-seed scatter. The per-configuration refinement
    # pins ONE seed tightly (se_cbar~0.027) but does not average ACROSS seeds; the
    # m=0 row is instead run
    # over m0_seeds independent streams and reports the mean; se <- sd/sqrt(K).
    # This changes nothing for m>=1 (already smooth via lambda-averaging).
    m0_seed_averaging = True,
    m0_seeds          = 20,
    m0_seed_stride    = 10 ** 9,   # must exceed max_T*1000003 + internal span
                                   # (~3.0e8 at T=300): guarantees disjoint seed
                                   # streams (config_seed = seed_base+1000003*T+..).
)

# =============================================================================
# 1. NUMERICAL KERNELS  (genuine @njit; Model LB: Z = [1, DU_1, ..., DU_m])
# =============================================================================

@njit(cache=True)
def build_z_nb(nt, break_pos):
    """Design matrix Z = [1, DU_1, ..., DU_m], DU_j = 1{t > tau_j}.
    break_pos is an int array of the m break dates (1-based positions)."""
    m = break_pos.shape[0]
    Z = np.empty((nt, m + 1))
    for t in range(nt):
        Z[t, 0] = 1.0
    for j in range(m):
        bp = break_pos[j]
        if bp < 1:
            bp = 1
        if bp > nt - 1:
            bp = nt - 1
        for t in range(nt):
            Z[t, j + 1] = 1.0 if (t + 1) > bp else 0.0   # t+1: 1-based time
    return Z


@njit(cache=True)
def glsd_nb(y, Z, cbar):
    """GLS detrending at a-bar = 1 + c-bar/T.
    Returns (yt, ssr): yt = y - Z @ bhat (level residual, for M-stats),
    ssr = SSR of the quasi-differenced regression (for P_T / MPT).
    Element-wise to stay inside nopython mode and match the GAUSS engine."""
    nt = Z.shape[0]
    ncol = Z.shape[1]
    abar = 1.0 + cbar / nt

    # Quasi-difference y and Z: first row kept as level, rest (x_t - abar x_{t-1}).
    ya = np.empty(nt)
    ya[0] = y[0]
    for t in range(1, nt):
        ya[t] = y[t] - abar * y[t - 1]

    Za = np.empty((nt, ncol))
    for j in range(ncol):
        Za[0, j] = Z[0, j]
    for t in range(1, nt):
        for j in range(ncol):
            Za[t, j] = Z[t, j] - abar * Z[t - 1, j]

    # Normal equations  (Za' Za) bhat = Za' ya
    AtA = np.zeros((ncol, ncol))
    Aty = np.zeros(ncol)
    for i in range(ncol):
        for k in range(ncol):
            s = 0.0
            for t in range(nt):
                s += Za[t, i] * Za[t, k]
            AtA[i, k] = s
        sy = 0.0
        for t in range(nt):
            sy += Za[t, i] * ya[t]
        Aty[i] = sy
    for i in range(ncol):
        AtA[i, i] += 1e-10            # ridge for numerical safety

    bhat = np.linalg.solve(AtA, Aty)

    # Level residual yt = y - Z bhat  (used by MZa/MSB/MZt and the boundary term)
    yt = np.empty(nt)
    for t in range(nt):
        pred = 0.0
        for j in range(ncol):
            pred += Z[t, j] * bhat[j]
        yt[t] = y[t] - pred

    # SSR of the quasi-differenced regression
    ssr = 0.0
    for t in range(nt):
        pred = 0.0
        for j in range(ncol):
            pred += Za[t, j] * bhat[j]
        d = ya[t] - pred
        ssr += d * d
    return yt, ssr


@njit(cache=True)
def ols_detrend_nb(y, Z):
    """OLS-detrended level residual y - Z @ bhat_ols.  Used for sigma^2: the
    projection removes Z@beta EXACTLY, so the long-run-variance input is
    invariant to the break magnitudes (algebraic, independent of the GLS lemma)."""
    nt = Z.shape[0]
    ncol = Z.shape[1]
    AtA = np.zeros((ncol, ncol))
    Aty = np.zeros(ncol)
    for i in range(ncol):
        for k in range(ncol):
            s = 0.0
            for t in range(nt):
                s += Z[t, i] * Z[t, k]
            AtA[i, k] = s
        sy = 0.0
        for t in range(nt):
            sy += Z[t, i] * y[t]
        Aty[i] = sy
    for i in range(ncol):
        AtA[i, i] += 1e-12
    bhat = np.linalg.solve(AtA, Aty)
    res = np.empty(nt)
    for t in range(nt):
        pred = 0.0
        for j in range(ncol):
            pred += Z[t, j] * bhat[j]
        res[t] = y[t] - pred
    return res


@njit(cache=True)
def s2ar_const_nb(yt_ols):
    """k=0 long-run variance: sample variance of the first difference of the
    OLS-detrended series (ddof=1). Under i.i.d. innovations this is consistent
    and equals s2ar with zero lags. This is the headline ('const') estimator."""
    nt = yt_ols.shape[0]
    n = nt - 1
    if n < 2:
        return 1e-10
    # diff
    mean = 0.0
    for t in range(1, nt):
        mean += (yt_ols[t] - yt_ols[t - 1])
    mean /= n
    ss = 0.0
    for t in range(1, nt):
        d = (yt_ols[t] - yt_ols[t - 1]) - mean
        ss += d * d
    v = ss / (n - 1)
    return v if v > 1e-10 else 1e-10


@njit(cache=True)
def s2ar_maic_nb(yt_ols, kmax):
    """Autoregressive spectral-density long-run variance with MAIC lag selection
    (Ng-Perron 2001). Input is the OLS-detrended series (Perron's note: OLS, not
    GLS). Faithful to the s2ar convention of the calibration kernel.
    Returns s2ar = s2e / (1 - sum_b)^2 at the MAIC-selected k."""
    nt = yt_ols.shape[0]
    nef = nt - kmax - 1
    if nef < 5:
        # fall back to const estimator
        return s2ar_const_nb(yt_ols)

    # Build regressors: dy_t on [y_{t-1}, dy_{t-1}, ..., dy_{t-kmax}]
    ny = nt - 1
    dy = np.empty(ny)
    for t in range(ny):
        dy[t] = yt_ols[t + 1] - yt_ols[t]

    # design rows aligned so that lag kmax is available: effective sample
    neff = ny - kmax
    # columns: 0 -> y_{t-1} (level), 1..kmax -> dy lags
    X = np.empty((neff, kmax + 1))
    d0 = np.empty(neff)
    for i in range(neff):
        # row r corresponds to dy index (kmax + i)
        idx = kmax + i
        d0[i] = dy[idx]
        X[i, 0] = yt_ols[idx]          # y_{t-1} aligned to dy[idx] = y[idx+1]-y[idx]
        for k in range(1, kmax + 1):
            X[i, k] = dy[idx - k]

    sumy = 0.0
    for i in range(neff):
        sumy += X[i, 0] * X[i, 0]

    best_k = 0
    best_mic = 1e18
    best_s2e = 1e18
    # Try k = 0..kmax, fit OLS of d0 on first (k+1) columns, compute MAIC.
    for k in range(kmax + 1):
        nc = k + 1
        # normal equations on the first nc columns
        AtA = np.zeros((nc, nc))
        Aty = np.zeros(nc)
        for a in range(nc):
            for b in range(nc):
                s = 0.0
                for i in range(neff):
                    s += X[i, a] * X[i, b]
                AtA[a, b] = s
            sy = 0.0
            for i in range(neff):
                sy += X[i, a] * d0[i]
            Aty[a] = sy
        for a in range(nc):
            AtA[a, a] += 1e-12
        bk = np.linalg.solve(AtA, Aty)
        # residual variance
        ss = 0.0
        for i in range(neff):
            pred = 0.0
            for a in range(nc):
                pred += X[i, a] * bk[a]
            e = d0[i] - pred
            ss += e * e
        s2e = ss / nef
        if s2e <= 0.0:
            s2e = 1e-12
        b0 = bk[0]
        tau = (b0 * b0) * sumy / s2e
        mic = np.log(s2e) + 2.0 * (k + tau) / nef
        if mic < best_mic:
            best_mic = mic
            best_k = k
            best_s2e = s2e

    # Refit at best_k to get the AR coefficients on dy lags (for the sum)
    nc = best_k + 1
    AtA = np.zeros((nc, nc))
    Aty = np.zeros(nc)
    for a in range(nc):
        for b in range(nc):
            s = 0.0
            for i in range(neff):
                s += X[i, a] * X[i, b]
            AtA[a, b] = s
        sy = 0.0
        for i in range(neff):
            sy += X[i, a] * d0[i]
        Aty[a] = sy
    for a in range(nc):
        AtA[a, a] += 1e-12
    bopt = np.linalg.solve(AtA, Aty)
    s2 = best_s2e
    if best_k > 0:
        bsum = 0.0
        for k in range(1, nc):
            bsum += bopt[k]
        arsum = 1.0 - bsum
        if abs(arsum) > 1e-6:
            sar = s2 / (arsum * arsum)
        else:
            sar = s2
    else:
        sar = s2
    return sar if sar > 1e-10 else 1e-10


# =============================================================================
# 1b. PERRON & QU (2007) OLS-GLS LONG-RUN VARIANCE  (Phase-5 correction)
# =============================================================================
# Perron, P. and Z. Qu (2007), "A simple modification to improve the finite
# sample properties of Ng and Perron's unit root tests", Economics Letters
# 94:12-19. Their recipe: the MAIC lag order is built entirely from the
# OLS-detrended ADF regression (their eq. 4), while the regression that
# produces s2_AR runs on the GLS-detrended series (their eq. 2). They state
# that running BOTH stages on OLS-detrended data yields tests that are too
# conservative in common sample sizes and do not recommend it.
#
# The pre-Phase-5 kernel ran BOTH stages on OLS-detrended data (the variant PQ
# reject), which is why it disagreed with the published tab:cv. Mechanism,
# measured: OLS detrending fits segment means to a random walk, so the
# detrended series inherits jumps at the break dates and the difference-based
# estimator collects m increments of order T; E[s2] settles at ~1.31/1.39/1.42
# for m=1/2/3 and is FLAT in T, whereas under GLS detrending E[s2] = 1.00
# everywhere because the quasi-differenced dummy is an impulse.
#
# The superseded functions s2ar_const_nb and s2ar_maic_nb above are RETAINED,
# unmodified, so the pre-correction numbers stay reproducible for the
# comparison reported in the kernel-convention report. They are no longer on
# any statistic's path.


@njit(cache=True)
def s2ar_diff_nb(yt):
    """SUPERSEDED 2026-08-09 (T-7). Retained off-path; do not call
    from the production dispatch.

    Computes sum (dy)^2 / (T-1) on the series passed: the plain sum of squared
    first differences, with no regression and no mean removal.

    WHY IT IS SUPERSEDED, and why the old docstring was itself the defect.
    This function used to describe itself as "the k=0 member of the PQ family".
    It is not. In eq:s2ar (the manuscript ) the
    regression is

        Delta ytilde_t = b_0 ytilde_{t-1} + sum_{j=1}^{k} b_j Delta ytilde_{t-j} + e_t

    and the level term b_0 ytilde_{t-1} is NOT inside the sum over j. At k = 0
    the SUM is empty but the LEVEL TERM REMAINS, so the k=0 member is the
    residual variance of the ADF regression, not the sample variance of
    Delta ytilde_t. The manuscript's prose says otherwise at v4:464-465 and
    v4:521-523; the equation and the prose disagree, and the author has ruled
    that THE EQUATION GOVERNS. See s2ar_k0_eq_nb.

    That old docstring is the propagation vector worth recording: the
    manuscript's incorrect gloss was copied into a code comment, and from
    there it legitimised the implementation.

    Measured cost of the difference (mean relative, random walk, c-bar = -7):
    10.5% at T=30, 4.2% at T=60, 2.4% at T=100, 0.28% at T=800. Scaled by T it
    stabilises at ~2.2 -- an O(1/T) gap, and 2.2 rather than ~1 because the
    regression on ytilde_{t-1} explains excess variance under the null: the
    Dickey-Fuller signature, which is the very channel the M class is built to
    control.
    """
    nt = yt.shape[0]
    if nt < 3:
        return 1e-10
    ss = 0.0
    for t in range(1, nt):
        d = yt[t] - yt[t - 1]
        ss += d * d
    v = ss / (nt - 1)
    return v if v > 1e-10 else 1e-10


@njit(cache=True)
def s2ar_k0_eq_nb(yt):
    """MEASUREMENT INSTRUMENT, not a superseded path, and not production.

    Author's ruling 2026-08-09 (which reversed an earlier same-day decision):
    the production baseline stays `s2ar_diff_nb`, and the MANUSCRIPT is
    corrected instead. This function is retained not as a discarded
    alternative but because it is what MEASURES the bias that justifies that
    choice -- see the numbers at the end of this docstring.

    Why the reversal, in one line: deciding which TEXT is authoritative is not
    deciding which OBJECT is correct. eq:s2ar is authoritative as text; under
    H0, where critical values live, the plain difference variance is the better
    estimator of the two.

    WHAT IT COMPUTES: the k=0 member of eq:s2ar as the EQUATION defines it
    (the manuscript ) --

        s2_AR(k=0) = residual variance of  Delta ytilde_t = b_0 ytilde_{t-1} + e_t

    -- because the level regressor b_0 ytilde_{t-1} is NOT inside the sum over
    j and therefore survives at k = 0. Written from the equation by ordinary
    least squares on the single regressor, deliberately NOT by delegating to
    s2ar_at_k_nb: the two must agree to machine precision, and that agreement
    is only a test if they are computed independently. They do agree, to
    9.7e-16 over 3,200 comparisons.

    THE MEASUREMENT IT EXISTS FOR. Against s2ar_diff_nb the relative gap is
    exactly decomposable, to 2.1e-15 per replication:

        TSS = SSR * (1 + t^2/(T-2))     =>     rel * T  ->  1 + E[t^2]

    where t is the t-ratio on b_0, i.e. the DF-GLS statistic. The "1" is the
    divisor difference (T-1 here against T there); the remainder is the
    Dickey-Fuller downward bias of the ADF residual variance in short samples.
    Measured under H0 at c-bar = -7: 10.5% at T=30, 2.4% at T=100, 0.28% at
    T=800.

    That is the number the manuscript's justification of the baseline rests on,
    so this function is production-adjacent evidence, not dead code. Do not
    DIVISOR: T (nt), matching sigma_hat^2_ek of Ng-Perron (2001) / Perron-Qu
    (2007) and s2ar_at_k_nb's estimation slot -- while the baseline
    s2ar_diff_nb divides by T-1, the number of terms. eq:s2ar does not specify
    a divisor anywhere in the .tex; that undeclared divisor is HALF the
    measured gap, which is why it is stated here explicitly.

    Do not delete this function and do not wire it into the dispatch.
    """
    nt = yt.shape[0]
    ny = nt - 1
    if ny < 5:
        return 1e-10

    # OLS of dy_t on ytilde_{t-1}: b0 = sum(x*y)/sum(x*x), one regressor.
    sxx = 0.0
    sxy = 0.0
    for t in range(ny):
        x = yt[t]                 # ytilde_{t-1} aligned to dy_t = yt[t+1]-yt[t]
        d = yt[t + 1] - yt[t]
        sxx += x * x
        sxy += x * d
    b0 = sxy / sxx if sxx > 1e-12 else 0.0

    ss = 0.0
    for t in range(ny):
        e = (yt[t + 1] - yt[t]) - b0 * yt[t]
        ss += e * e
    s2e = ss / nt                 # divisor T, as sigma_hat^2_ek
    # (1 - sum_{j=1}^{0} b_j)^2 = 1, so s2_AR = sigma_hat^2_e exactly.
    return s2e if s2e > 1e-10 else 1e-10


@njit(cache=True)
def maic_select_k_nb(yt_ols, kmax):
    """MAIC lag order from the OLS-detrended ADF regression (PQ eq. 4).

    Selection only: identical arithmetic to the selection loop of
    s2ar_maic_nb (same fixed-kmax alignment, same tau_T(k) and sigma2_k), so
    the order chosen is unchanged by the Phase-5 correction. Returns k.
    """
    nt = yt_ols.shape[0]
    # sigma_check^2_k: COMMON sample for every k, divisor (T - kmax).
    # Ng-Perron (2001) / Perron-Qu (2007); the common sample is also what
    # the manuscript requires ("modified AIC ...
    # minimized over a common sample up to kmax=12"). Corrected 2026-08-03
    # from (T - kmax - 1): the divisor is NOT neutral for the argmin,
    # because tau_T(k) is proportional to 1/sigma_check^2_k.
    nef = nt - kmax
    if nef < 5:
        return 0

    ny = nt - 1
    dy = np.empty(ny)
    for t in range(ny):
        dy[t] = yt_ols[t + 1] - yt_ols[t]

    neff = ny - kmax
    X = np.empty((neff, kmax + 1))
    d0 = np.empty(neff)
    for i in range(neff):
        idx = kmax + i
        d0[i] = dy[idx]
        X[i, 0] = yt_ols[idx]
        for k in range(1, kmax + 1):
            X[i, k] = dy[idx - k]

    sumy = 0.0
    for i in range(neff):
        sumy += X[i, 0] * X[i, 0]

    best_k = 0
    best_mic = 1e18
    for k in range(kmax + 1):
        nc = k + 1
        AtA = np.zeros((nc, nc))
        Aty = np.zeros(nc)
        for a in range(nc):
            for b in range(nc):
                s = 0.0
                for i in range(neff):
                    s += X[i, a] * X[i, b]
                AtA[a, b] = s
            sy = 0.0
            for i in range(neff):
                sy += X[i, a] * d0[i]
            Aty[a] = sy
        for a in range(nc):
            AtA[a, a] += 1e-12
        bk = np.linalg.solve(AtA, Aty)
        ss = 0.0
        for i in range(neff):
            pred = 0.0
            for a in range(nc):
                pred += X[i, a] * bk[a]
            e = d0[i] - pred
            ss += e * e
        s2e = ss / nef
        if s2e <= 0.0:
            s2e = 1e-12
        b0 = bk[0]
        tau = (b0 * b0) * sumy / s2e
        mic = np.log(s2e) + 2.0 * (k + tau) / nef
        if mic < best_mic:
            best_mic = mic
            best_k = k
    return best_k


@njit(cache=True)
def s2ar_at_k_nb(yt, k_sel, kmax):
    """AR spectral s2_AR at a GIVEN order, estimated on the series passed
    (eq. s2ar of the manuscript called on the
    GLS-detrended series).

    sigma_hat^2_ek: the k-SPECIFIC sample (starting at k+1) and divisor T,
    per Ng-Perron (2001) / Perron-Qu (2007). This is deliberately NOT the
    estimator used for lag selection: selection uses sigma_check^2_k on the
    common kmax sample with divisor (T - kmax). Corrected 2026-08-03, when
    this function still used the common sample and the (T - kmax - 1)
    divisor -- the selection estimator in the estimation slot.
    """
    nt = yt.shape[0]
    ny = nt - 1
    neff = ny - k_sel
    if neff < 5:
        return s2ar_diff_nb(yt)

    dy = np.empty(ny)
    for t in range(ny):
        dy[t] = yt[t + 1] - yt[t]

    nc = k_sel + 1
    X = np.empty((neff, nc))
    d0 = np.empty(neff)
    for i in range(neff):
        idx = k_sel + i
        d0[i] = dy[idx]
        X[i, 0] = yt[idx]
        for k in range(1, nc):
            X[i, k] = dy[idx - k]

    AtA = np.zeros((nc, nc))
    Aty = np.zeros(nc)
    for a in range(nc):
        for b in range(nc):
            s = 0.0
            for i in range(neff):
                s += X[i, a] * X[i, b]
            AtA[a, b] = s
        sy = 0.0
        for i in range(neff):
            sy += X[i, a] * d0[i]
        Aty[a] = sy
    for a in range(nc):
        AtA[a, a] += 1e-12
    bopt = np.linalg.solve(AtA, Aty)

    ss = 0.0
    for i in range(neff):
        pred = 0.0
        for a in range(nc):
            pred += X[i, a] * bopt[a]
        e = d0[i] - pred
        ss += e * e
    s2e = ss / nt                       # sigma_hat^2_ek: divisor T
    if s2e <= 0.0:
        s2e = 1e-12

    if k_sel > 0:
        bsum = 0.0
        for k in range(1, nc):
            bsum += bopt[k]
        arsum = 1.0 - bsum
        if abs(arsum) > 1e-6:
            sar = s2e / (arsum * arsum)
        else:
            sar = s2e
    else:
        sar = s2e
    return sar if sar > 1e-10 else 1e-10


@njit(cache=True)
def mstats_nb(y, Z, cbar, sigma2_method, kmax):
    """All five CKP M-statistics for Model LB: returns (mza, msb, mzt, pt, mpt,
    ok). ok=0.0 signals a degenerate draw (caller discards). sigma2_method:
    0 -> 'const', 1 -> 'maic'.

    Rejection convention: ALL reject for SMALL values (very negative for MZa/MZt;
    small for MSB/PT/MPT) -> the critical value is the lower percentile.

    Long-run variance: Perron & Qu (2007) OLS-GLS (see section 1b above).
    Phase-5 correction, 2026-08-03: s2_AR moved from the OLS-detrended to the
    GLS-detrended series; MAIC lag selection stays on the OLS-detrended
    regression. This CHANGES NUMBERS; every affected value is listed in
    the renumbering audit.
    """
    nt = Z.shape[0]
    yt, ssra = glsd_nb(y, Z, cbar)
    if not np.isfinite(ssra):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Long-run variance, PQ OLS-GLS. Both branches estimate s2 on the
    # GLS-detrended level residual yt; the MAIC ORDER comes from the
    # OLS-detrended regression (invariant to theta).
    if sigma2_method == 0:
        # 0 = the GLOSS (v4:464-465, :521-523): plain sum of squared
        # differences. SUPERSEDED by the equation-faithful estimator (code 2)
        # but still the production default until the dispatch is aligned; see
        # s2ar_diff_nb's docstring.
        sar = s2ar_diff_nb(yt)
    elif sigma2_method == 2:
        # 2 = the equation-faithful estimator, added 2026-08-09, ADDITIVE: the
        # k=0 member of eq:s2ar AS THE EQUATION DEFINES IT (v4:512-514), i.e.
        # retaining the level regressor b_0 ytilde_{t-1}. Added as a THIRD code
        # rather than by replacing code 0, so that the test of whether the
        # printed tab:cv survives this estimator can be run BEFORE the
        # production path is switched.
        # Switching first and testing after would destroy the comparison.
        sar = s2ar_k0_eq_nb(yt)
    else:
        yt_ols = ols_detrend_nb(y, Z)
        k_sel = maic_select_k_nb(yt_ols, kmax)
        sar = s2ar_at_k_nb(yt, k_sel, kmax)

    # sum of squares of the GLS-detrended level series (lagged)
    denom = 0.0
    for t in range(nt - 1):
        denom += yt[t] * yt[t]
    if denom <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # NORMALISATION FIX 2026-08-03 (Phase-5 ruling): the paper
    # defines the M class with T^{-1} and T^{-2} throughout -- see
    # the manuscript (eq. Mstats), which is Ng and
    # Perron's (2001) definition. This kernel previously used bt = nt - 1
    # for both, which contradicted eq. (4) and was internally inconsistent
    # with its own MPT line below (already T^{-1}). It is a DEFECT, not an
    # alternative convention: the published tables are on the T convention,
    # so this fix makes the kernel agree with them rather than moving them.
    # The superseded path is retained, off-path, as mstats_bt_tminus1_nb.
    bt = nt
    sumyt2 = denom / (bt * bt)        # T^{-2} sum y_{t-1}^2
    if sumyt2 <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    ytT = yt[nt - 1]                  # y-tilde_T (last detrended value)

    mza = (ytT * ytT / bt - sar) / (2.0 * sumyt2)
    msb = np.sqrt(sumyt2 / sar)
    mzt = mza * msb

    # P_T (ERS, SSR form, NP2001 eq.7):  [S(a-bar) - a-bar S(1)] / s2ar
    _, ssr1 = glsd_nb(y, Z, 0.0)      # S(1): a-bar=1 at cbar=0
    abar = 1.0 + cbar / nt
    pt = (ssra - abar * ssr1) / sar

    # MPT, DEMEANED form (NP2001 eq.9, p=0):  [c-bar^2 T^-2 sum yt^2 - c-bar ytT^2/T]/s2ar
    #   (the (1 - c-bar) form is the DETRENDED eq.10 form, wrong for Model LB.)
    mpt = (cbar * cbar * sumyt2 - cbar * ytT * ytT / nt) / sar

    return mza, msb, mzt, pt, mpt, 1.0


@njit(cache=True)
def mstats_old_s2_nb(y, Z, cbar, sigma2_method, kmax):
    """OLD sigma^2 convention (OLS-OLS), retained off-path for the 2x2.

    Identical to mstats_nb EXCEPT that s2_AR is taken from the OLS-detrended
    series by the pre-Phase-5 estimators (s2ar_const_nb / s2ar_maic_nb) --
    the variant Perron and Qu (2007) reject and that the kernel used before
    2026-08-03.

    The T normalisation is DELIBERATELY the corrected one (bt = nt): the 2x2
    has two axes, initial condition and sigma^2, and the printed
    tables are already on the T convention, so putting the
    superseded (T-1) here would test a combination that never produced
    anything published.

    Nothing calls this; it exists so the attribution runs can select the old
    sigma^2 arm explicitly. Defaults are untouched.
    """
    nt = Z.shape[0]
    yt, ssra = glsd_nb(y, Z, cbar)
    if not np.isfinite(ssra):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # OLD: both stages on the OLS-detrended series.
    yt_ols = ols_detrend_nb(y, Z)
    if sigma2_method == 0:
        sar = s2ar_const_nb(yt_ols)
    else:
        sar = s2ar_maic_nb(yt_ols, kmax)

    denom = 0.0
    for t in range(nt - 1):
        denom += yt[t] * yt[t]
    if denom <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    bt = nt
    sumyt2 = denom / (bt * bt)
    if sumyt2 <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    ytT = yt[nt - 1]
    mza = (ytT * ytT / bt - sar) / (2.0 * sumyt2)
    msb = np.sqrt(sumyt2 / sar)
    mzt = mza * msb
    _, ssr1 = glsd_nb(y, Z, 0.0)
    abar = 1.0 + cbar / nt
    pt = (ssra - abar * ssr1) / sar
    mpt = (cbar * cbar * sumyt2 - cbar * ytT * ytT / nt) / sar
    return mza, msb, mzt, pt, mpt, 1.0


@njit(cache=True)
def mstats_old_s2_old_norm_nb(y, Z, cbar, sigma2_method, kmax):
    """OLD sigma^2 (OLS-OLS) AND OLD normalisation (bt = nt - 1), off-path.

    The third axis of the attribution. Added because the top-left 2x2
    cell reproduced c-bar* and power_feas exactly while leaving the printed
    critical values +0.043 to +0.061 too negative, uniformly - the signature
    of a systematic rescaling of MZ_t rather than a sigma^2 or IC residue,
    since power (a rejection frequency of the statistic against its own cv)
    cancels a common factor and cv does not.

    Predicted target: in the left tail MZ_alpha's numerator is dominated by
    -s^2, so printed/new tends to 1/r with r = T/(T-1) = 52/51, i.e. 0.9808.
    Observed printed/new at the 5% point: 0.9793, 0.9791, 0.9801, 0.9781
    (mean 0.9792) - short of 0.9808 by 0.0016, with the right sign, because
    the scalar reduction is exact only in the tail limit while the
    y_T^2 / bt term still contributes at the 5% quantile.

    Nothing calls this. Defaults are untouched.
    """
    nt = Z.shape[0]
    yt, ssra = glsd_nb(y, Z, cbar)
    if not np.isfinite(ssra):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    yt_ols = ols_detrend_nb(y, Z)
    if sigma2_method == 0:
        sar = s2ar_const_nb(yt_ols)
    else:
        sar = s2ar_maic_nb(yt_ols, kmax)

    denom = 0.0
    for t in range(nt - 1):
        denom += yt[t] * yt[t]
    if denom <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    bt = nt - 1                       # OLD normalisation
    sumyt2 = denom / (bt * bt)
    if sumyt2 <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    ytT = yt[nt - 1]
    mza = (ytT * ytT / bt - sar) / (2.0 * sumyt2)
    msb = np.sqrt(sumyt2 / sar)
    mzt = mza * msb
    _, ssr1 = glsd_nb(y, Z, 0.0)
    abar = 1.0 + cbar / nt
    pt = (ssra - abar * ssr1) / sar
    mpt = (cbar * cbar * sumyt2 - cbar * ytT * ytT / nt) / sar
    return mza, msb, mzt, pt, mpt, 1.0


@njit(cache=True)
def mstats_bt_tminus1_nb(y, Z, cbar, sigma2_method, kmax):
    """SUPERSEDED normalisation, retained off-path (Phase-5).

    Identical to mstats_nb except that the lagged sum of squares is divided
    by (T-1)^2 and the boundary term by (T-1) -- the pre-2026-08-03
    behaviour, which contradicts the manuscript .
    Kept so the pre-fix numbers stay reproducible for the record;
    nothing calls it.
    """
    nt = Z.shape[0]
    yt, ssra = glsd_nb(y, Z, cbar)
    if not np.isfinite(ssra):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if sigma2_method == 0:
        sar = s2ar_diff_nb(yt)
    else:
        yt_ols = ols_detrend_nb(y, Z)
        sar = s2ar_at_k_nb(yt, maic_select_k_nb(yt_ols, kmax), kmax)
    denom = 0.0
    for t in range(nt - 1):
        denom += yt[t] * yt[t]
    if denom <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    bt = nt - 1
    sumyt2 = denom / (bt * bt)
    if sumyt2 <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ytT = yt[nt - 1]
    mza = (ytT * ytT / bt - sar) / (2.0 * sumyt2)
    msb = np.sqrt(sumyt2 / sar)
    mzt = mza * msb
    _, ssr1 = glsd_nb(y, Z, 0.0)
    abar = 1.0 + cbar / nt
    pt = (ssra - abar * ssr1) / sar
    mpt = (cbar * cbar * sumyt2 - cbar * ytT * ytT / nt) / sar
    return mza, msb, mzt, pt, mpt, 1.0


@njit(cache=True)
def gen_dgp_nb(nt, break_pos, c, beta_scale, eps):
    """y = Z @ beta + u, with u_t = (1 + c/T) u_{t-1} + eps_t.

    INITIAL CONDITION (corrected 2026-08-03): the recursion is
    initialised at the PRE-SAMPLE value u_0 = 0, so the first in-sample
    observation is u_1 = eps_1. This is the manuscript's declared convention:
    the manuscript (= v4:701) states the DGP is
    "initialized at u_0 = 0 (the ERS convention; a non-negligible
    initialization of order T^{1/2} would alter the demeaned limit below)",
    and the proof of Theorem 1(ii) at v2:721-722 (= v4:725-726) DEPENDS on it:
    "the estimated intercept converges to y_1 = eps_1 = O_p(1)".

    Until this fix the loop started at t = 1 over a zero-initialised array,
    which set the first OBSERVATION to zero (u_1 = 0) and discarded eps_1
    entirely -- a different DGP from the one Theorem 1 is proved under, and
    the reason the canonical kernel did not reproduce tab:power/Figure_1.
    The superseded path is retained
    off-path as gen_dgp_u1zero_nb.

    eps is a precomputed N(0,1) vector (so the RNG stays in the orchestrator,
    where seeding is reproducible). Returns y."""
    Z = build_z_nb(nt, break_pos)
    ncol = Z.shape[1]
    u = np.zeros(nt)
    rho = 1.0 + c / nt
    u[0] = eps[0]                       # u_1 = eps_1, from u_0 = 0
    for t in range(1, nt):
        u[t] = rho * u[t - 1] + eps[t]
    y = np.empty(nt)
    for t in range(nt):
        acc = 0.0
        for j in range(ncol):
            acc += Z[t, j] * beta_scale     # beta is constant beta_scale in every coord
        y[t] = acc + u[t]
    return y


@njit(cache=True)
def gen_dgp_u1zero_nb(nt, break_pos, c, beta_scale, eps):
    """SUPERSEDED initial condition, retained off-path.

    Sets the first in-sample observation to zero (u_1 = 0) and never uses
    eps[0] -- the pre-2026-08-03 behaviour, which contradicts
    the manuscript and the proof at :721-722. Kept so the
    pre-fix numbers stay reproducible; nothing calls it.
    """
    Z = build_z_nb(nt, break_pos)
    ncol = Z.shape[1]
    u = np.zeros(nt)
    rho = 1.0 + c / nt
    for t in range(1, nt):
        u[t] = rho * u[t - 1] + eps[t]
    y = np.empty(nt)
    for t in range(nt):
        acc = 0.0
        for j in range(ncol):
            acc += Z[t, j] * beta_scale
        y[t] = acc + u[t]
    return y


# =============================================================================
# 2. PYTHON ORCHESTRATION  (seeding, grids, calibration, persistence)
# =============================================================================

def warm_up_numba(kmax=12):
    """Compile every kernel once and verify (when numba is present) that
    compilation really occurred. Returns a banner string."""
    nt = 40
    bp = np.array([12, 24], dtype=np.int64)
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(nt)
    y = gen_dgp_nb(nt, bp, 0.0, 0.0, eps)
    Z = build_z_nb(nt, bp)
    _ = glsd_nb(y, Z, -7.0)
    _ = ols_detrend_nb(y, Z)
    _ = s2ar_const_nb(y)
    _ = s2ar_maic_nb(y, kmax)
    _ = mstats_nb(y, Z, -7.0, 0, kmax)
    _ = mstats_nb(y, Z, -7.0, 1, kmax)
    if _HAS_NUMBA:
        # signatures present => compiled
        compiled = (len(mstats_nb.signatures) > 0 and len(glsd_nb.signatures) > 0)
        assert compiled, "NUMBA PRESENT BUT KERNELS DID NOT COMPILE -- aborting."
        return "[numba] kernels compiled and verified (nopython, cache=True)."
    return "[no-numba] running pure-Python fallback (identical arithmetic, slower)."


def make_lambdas(m, trim, min_spacing, n_grid=7):
    """m break fractions in [trim, 1-trim] with spacing >= min_spacing. m=0 -> [()]."""
    if m == 0:
        return [()]
    ng = max(n_grid, 3 * m - 1)
    grid = np.round(np.linspace(trim, 1 - trim, ng), 3)
    combos = []
    for combo in itertools.combinations(grid, m):
        if all(combo[i + 1] - combo[i] >= min_spacing - 1e-9
               for i in range(len(combo) - 1)):
            combos.append(tuple(combo))
    if m >= 4 and len(combos) > 12:
        idx = np.linspace(0, len(combos) - 1, 12).round().astype(int)
        combos = [combos[i] for i in sorted(set(idx))]
    return combos


# 2026-08-10: the Phase-3 rename's old name, kept as an ALIAS.
# `robustness_v6.py` binds kernels by attribute name and asks for
# `gerar_lambdas`. The port of that file is allowed exactly two differences
# from the original -- the kernel import target and the --csv default -- so the
# missing name has to be supplied HERE rather than edited THERE. Verified a pure
# rename: the two bodies are identical after substituting the name
# (tools/kernel_diff_mc.py), so this alias adds no behaviour.
gerar_lambdas = make_lambdas


def enumerate_configs(P):
    """All (T, m, lambdas) with T >= m_min_T[m]."""
    cfgs = []
    for m in P['m_grid']:
        for T in P['T_grid']:
            if T < P['m_min_T'].get(m, 30):
                continue
            for lambdas in make_lambdas(m, P['trim'], P['min_spacing'], P['n_grid']):
                cfgs.append((T, m, lambdas))
    return cfgs


def config_seed(P, T, m, lambdas):
    """Deterministic per-config seed (positional weights 37^i)."""
    h = P['seed_base'] + 1000003 * T + 7919 * m
    for i, lam in enumerate(lambdas):
        h += int(round(lam * 1000)) * (37 ** (i + 1))
    return h % (2 ** 63 - 1)


def break_pos_from_lambdas(T, lambdas):
    return np.array([int(max(1, min(T - 1, np.floor(lam * T)))) for lam in lambdas],
                    dtype=np.int64)


def se_power(p, R):
    if R <= 0 or not np.isfinite(p):
        return np.nan
    return float(np.sqrt(max(p * (1 - p), 0.0) / R))


def se_quantile_bootstrap(sample, q, n_boot=500, seed=0):
    """Percentile-bootstrap SE of the q-th percentile (q in 0..100)."""
    a = np.asarray(sample, dtype=float)
    if len(a) < 100:
        return np.nan
    rng = np.random.default_rng(seed)
    n = len(a)
    qs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        qs[b] = np.percentile(a[idx], q)
    return float(np.std(qs, ddof=1))


_METHOD_CODE = {'const': 0, 'maic': 1}

# 2026-08-10: the NAME of the s^2 kernel each dispatch code resolves to.
# `sigma2_method` records the ARM the caller asked for; it does not record the
# ESTIMATOR, and the two stopped being interchangeable when code 2 was added.
# The const surface had to be dated against git to establish which kernel ran,
# because no artefact field carried it. Recording
# the name in every result means no future run needs that argument. Keep this
# table adjacent to the dispatch in mstats_nb: if a code is added there and not
# here, the field reads 'unknown', which is the correct answer rather than a
# stale one.
_S2_KERNEL_NAME = {0: 's2ar_diff_nb', 1: 's2ar_maic (maic_select_k_nb + s2ar_at_k_nb)',
                   2: 's2ar_k0_eq_nb'}


def _stats_under(P, T, break_pos, c, cbar, method_code, n_reps, seed0, kmax):
    """Run n_reps draws at alternative c, evaluating M-stats at detrending cbar.
    Returns dict of arrays (one per statistic). RNG seeded reproducibly."""
    out = {k: np.empty(n_reps) for k in ['mza', 'msb', 'mzt', 'pt', 'mpt']}
    for r in range(n_reps):
        rng = np.random.default_rng((seed0 + r) % (2 ** 63 - 1))
        eps = rng.standard_normal(T)
        y = gen_dgp_nb(T, break_pos, c, 0.0, eps)
        Z = build_z_nb(T, break_pos)
        mza, msb, mzt, pt, mpt, ok = mstats_nb(y, Z, cbar, method_code, kmax)
        if ok > 0.5:
            out['mza'][r] = mza; out['msb'][r] = msb; out['mzt'][r] = mzt
            out['pt'][r] = pt;  out['mpt'][r] = mpt
        else:
            for k in out:
                out[k][r] = np.nan
    return out


def _tangency_on_grid(P, T, break_pos, mc, kmax, cfg_seed, cbar_grid,
                      r_cv, r_pow, seed_offset=0):
    """Locate the interpolated power=target tangency on a given c-bar grid.
    Returns (cbar_star, poder_star, se_cbar_star, fallback_used, cb_arr,
    pw_arr, cv_arr, se_arr) or None if no usable point. Factored out of
    calibrate_config_sigma so it can be called twice (coarse then refined).
    seed_offset lets the refined pass use independent draws from the coarse
    pass, so the two are not coupled."""
    tang = P['target_power']
    cb_list, cv_list, pw_list, sep_list = [], [], [], []
    for cbar in cbar_grid:
        null = _stats_under(P, T, break_pos, 0.0, cbar, mc, r_cv,
                            cfg_seed + seed_offset, kmax)['pt']
        null = null[np.isfinite(null)]
        if len(null) < r_cv * 0.5:
            continue
        cv5 = float(np.percentile(null, 5))
        alt = _stats_under(P, T, break_pos, cbar, cbar, mc, r_pow,
                           cfg_seed + 10 ** 6 + seed_offset, kmax)['pt']
        alt = alt[np.isfinite(alt)]
        if len(alt) == 0:
            continue
        power = float(np.mean(alt < cv5))
        cb_list.append(cbar); cv_list.append(cv5)
        pw_list.append(power); sep_list.append(se_power(power, len(alt)))

    if not cb_list:
        return None
    order = np.argsort(cb_list)
    cb_arr = np.array(cb_list)[order]; pw_arr = np.array(pw_list)[order]
    cv_arr = np.array(cv_list)[order]; se_arr = np.array(sep_list)[order]

    cbar_star = poder_star = se_cbar_star = None
    fallback_used = False
    for i in range(len(cb_arr) - 1):
        p0, p1 = pw_arr[i], pw_arr[i + 1]
        if (p0 - tang) * (p1 - tang) <= 0 and p1 != p0:
            c0, c1 = cb_arr[i], cb_arr[i + 1]
            frac = (tang - p0) / (p1 - p0)
            cbar_star = float(c0 + frac * (c1 - c0))
            poder_star = float(tang)
            slope = (p1 - p0) / (c1 - c0)
            se_pl = np.sqrt(tang * (1 - tang) / r_pow)
            se_cbar_star = float(se_pl / abs(slope)) if slope != 0 else float('nan')
            if c0 == cb_arr[0] or c1 == cb_arr[-1]:
                fallback_used = True
            break
    if cbar_star is None:
        fallback_used = True
        idx = int(np.argmin(np.abs(pw_arr - tang)))
        cbar_star = float(cb_arr[idx]); poder_star = float(pw_arr[idx])
        se_cbar_star = float(se_power(poder_star, r_pow))
    return (cbar_star, poder_star, se_cbar_star, fallback_used,
            cb_arr, pw_arr, cv_arr, se_arr)


def calibrate_config_sigma(T, m, lambdas, P, sigma2_method):
    """Calibrate c-bar* and critical values for one config and one sigma^2 method.
    Selection rule: interpolated tangency of power=target on the c-bar grid
    (Perron style), with a delta-method SE; grid-argmin only as a flagged
    fallback. Returns a result dict.

    For cheap configurations (few lambda points -- in practice m=0, which
    has no lambda-averaging and is the only row exposed to single-draw noise),
    a SECOND tangency pass is run on a finer c-bar grid local to the coarse
    optimum, with more replications. This sharpens the Monte Carlo resolution
    of the SAME fixed point; it does not change the estimand. See DEFAULTS for
    the rationale and the measured slope that motivates it."""
    mc = _METHOD_CODE[sigma2_method]
    kmax = P['kmax']
    cfg_seed = config_seed(P, T, m, lambdas)
    break_pos = break_pos_from_lambdas(T, lambdas)
    cbar_grid = np.array(P['cbar_grid'])

    # ---- 1) Coarse tangency on the production grid ---------------------------
    coarse = _tangency_on_grid(P, T, break_pos, mc, kmax, cfg_seed, cbar_grid,
                               P['R_cv'], P['R_pow'])
    if coarse is None:
        return None
    (cbar_star, poder_star, se_cbar_star, fallback_used,
     cb_arr, pw_arr, cv_arr, se_arr) = coarse

    # ---- 1r) Refined tangency for cheap configs ------------------------------
    refined = False
    if (P.get('refine_cheap_configs', False)
            and len(lambdas) <= P.get('refine_max_lambdas', 1)
            and not fallback_used):
        hw = P['refine_grid_halfwidth']; step = P['refine_grid_step']
        fine_grid = np.round(
            np.arange(cbar_star - hw, cbar_star + hw + step / 2, step), 3)
        # keep within the global grid's range
        gmin, gmax = float(cbar_grid.min()), float(cbar_grid.max())
        fine_grid = fine_grid[(fine_grid >= gmin) & (fine_grid <= gmax)]
        fine = _tangency_on_grid(
            P, T, break_pos, mc, kmax, cfg_seed, fine_grid,
            P['refine_R_cv'], P['refine_R_pow'], seed_offset=7 * 10 ** 6)
        if fine is not None and not fine[3]:  # accept only a clean (non-fallback) refine
            (cbar_star, poder_star, se_cbar_star, _fb,
             cb_arr, pw_arr, cv_arr, se_arr) = fine
            refined = True

    se_poder_star = float(np.interp(cbar_star, cb_arr, se_arr))

    # ---- 2) Critical values of all five stats at c-bar* (+ SE, + raw) --------
    r_cv_final = P['refine_R_cv'] if refined else P['R_cv']
    dist = _stats_under(P, T, break_pos, 0.0, cbar_star, mc, r_cv_final,
                        cfg_seed + 2 * 10 ** 6, kmax)
    cvs, cvs_se = {}, {}
    for k, a in dist.items():
        a = a[np.isfinite(a)]
        if len(a) < 100:
            cvs[k] = {q: np.nan for q in ['1%', '2.5%', '5%', '10%']}
            cvs_se[k] = {q: np.nan for q in ['1%', '2.5%', '5%', '10%']}
            continue
        cvs[k] = {q: float(np.percentile(a, float(q[:-1])))
                  for q in ['1%', '2.5%', '5%', '10%']}
        cvs_se[k] = {q: se_quantile_bootstrap(a, float(q[:-1]),
                     seed=cfg_seed + (zlib.crc32(k.encode("utf-8")) % 1000)) for q in ['1%', '2.5%', '5%', '10%']}

    # ---- 3) Invariance sanity (Lemma): beta=0 vs beta=10, identical stat -----
    inv_dev = _invariance_check(T, break_pos, cbar_star, cfg_seed, mc, kmax)

    # ---- 4) Power curves (diagnostic) ----------------------------------------
    power_curve = None
    if P['compute_power_curves']:
        power_curve = {k: [] for k in dist}
        power_curve['c_grid'] = list(P['c_grid_power'])
        cv5 = {k: cvs[k]['5%'] for k in dist}
        for c_true in P['c_grid_power']:
            d = _stats_under(P, T, break_pos, c_true, cbar_star, mc, P['R_curve'],
                             cfg_seed + 3 * 10 ** 6 + int(abs(c_true)) * 10 ** 4, kmax)
            for k in dist:
                a = d[k][np.isfinite(d[k])]
                c5 = cv5[k]
                power_curve[k].append(float(np.mean(a < c5)) if (len(a) and np.isfinite(c5))
                                      else np.nan)

    raw = ({k: dist[k][np.isfinite(dist[k])].astype(np.float32) for k in dist}
           if P['save_raw_vectors'] else None)

    # The CALIBRATION-grid curve the tangency search actually walked. Added
    # 2026-08-07; the shipped surface carries none of this, and `power_curve`
    # above is a different, coarser object (`c_grid_power`, step 2.0, labelled
    # diagnostic). Without these four arrays no artefact can show how close the
    # search came to the target at each node it evaluated -- which is exactly
    # what the diagnostic needed and could not get. All four are already
    # computed by _tangency_on_grid; they were simply discarded here.
    #
    # They also let a reader tell the TWO conditions `fallback_used` conflates
    # apart (kernel_numba.py:1004-1011): a crossing found inside a boundary
    # interval (poder_star == target exactly, c-bar interpolated off-node) is a
    # different object from no crossing at all (c-bar sits ON a node,
    # poder_star < target).
    calib_curve = dict(
        cbar=[float(x) for x in cb_arr],
        power=[float(x) for x in pw_arr],
        cv5_pt=[float(x) for x in cv_arr],
        se_power=[float(x) for x in se_arr],
        grid_is_refined=bool(refined),
    )

    return dict(
        T=T, m=m, lambdas=lambdas, sigma2_method=sigma2_method,
        s2_kernel=_S2_KERNEL_NAME.get(_METHOD_CODE.get(sigma2_method), 'unknown'),
        cbar_star=cbar_star, poder_star=poder_star, se_poder_star=se_poder_star,
        se_cbar_star=se_cbar_star, fallback_used=fallback_used, refined=refined,
        cvs=cvs, cvs_se=cvs_se, inv_dev=inv_dev, power_curve=power_curve,
        calib_curve=calib_curve,
        cfg_seed=int(cfg_seed), raw=raw,
    )


def _invariance_check(T, break_pos, cbar, seed, method_code, kmax):
    """Exact test of Lemma 'invariance': same innovation u (a pure random walk),
    vary only beta in {0, 10}; the statistic (MZt) must be identical."""
    rng = np.random.default_rng((seed + 7) % (2 ** 63 - 1))
    eps = rng.standard_normal(T)
    vals = []
    for beta_scale in (0.0, 10.0):
        y = gen_dgp_nb(T, break_pos, 0.0, beta_scale, eps)  # c=0 -> random walk
        Z = build_z_nb(T, break_pos)
        mza, msb, mzt, pt, mpt, ok = mstats_nb(y, Z, cbar, method_code, kmax)
        vals.append(mzt if ok > 0.5 else np.nan)
    if any(not np.isfinite(v) for v in vals):
        return np.nan
    return float(abs(vals[0] - vals[1]))


# --- seed-averaging helpers for the m=0 row ----------------------------------
_M0_STATS = ('mza', 'msb', 'mzt', 'pt', 'mpt')
_M0_QS = ('1%', '2.5%', '5%', '10%')


def _m0_nanmean(xs):
    a = np.asarray(list(xs), float)
    return float(np.nanmean(a)) if np.any(np.isfinite(a)) else float('nan')


def _m0_nan_se(xs, K):
    a = np.asarray(list(xs), float)
    if np.sum(np.isfinite(a)) < 2:
        return float('nan')
    return float(np.nanstd(a, ddof=1) / np.sqrt(K))


def guard_m0_stride(P):
    """m0_seed_stride must exceed max_T*1000003 + the internal seed-offset span of
    one m=0 calibration (refine at +7e6; CVs at +2e6; power at +3e6 + |c|*1e4), so
    that config_seed = seed_base + 1000003*T + ... yields DISJOINT streams across
    the K seeds. Raises if too small (no silent, correlated averaging)."""
    max_T = max(P['T_grid'])
    cg = P.get('c_grid_power', [])
    internal = 7_000_000 + 3_000_000 + (int(max(abs(c) for c in cg) * 10_000) if cg else 0)
    need = max_T * 1_000_003 + internal
    stride = int(P['m0_seed_stride'])
    if stride <= need:
        raise ValueError(
            f"[G_STRIDE] m0_seed_stride={stride:,} is too small: m=0 seed streams "
            f"overlap for T up to {max_T} (need > {need:,}). Use >= {10**9:,}.")
    return need


def _aggregate_m0_method(rs, T, method):
    """Aggregate K per-method result dicts `r` (calibrate_config_sigma output, same
    (m=0, T, method), one per seed) into a single seed-averaged r, in the EXACT
    schema aggregate/save_result consume: values -> mean over seeds; each SE
    -> cross-seed sd/sqrt(K); inv_dev -> max; refined -> all; fallback -> any."""
    K = len(rs)
    cvs = {s: {} for s in _M0_STATS}
    cvs_se = {s: {} for s in _M0_STATS}
    for s in _M0_STATS:
        for q in _M0_QS:
            vals = [r['cvs'].get(s, {}).get(q, np.nan) for r in rs]
            cvs[s][q] = _m0_nanmean(vals)
            cvs_se[s][q] = _m0_nan_se(vals, K)
    finite_inv = [abs(r['inv_dev']) for r in rs if np.isfinite(r['inv_dev'])]
    return dict(
        T=T, m=0, lambdas=(), sigma2_method=method,
        cbar_star=_m0_nanmean([r['cbar_star'] for r in rs]),
        poder_star=_m0_nanmean([r['poder_star'] for r in rs]),
        se_poder_star=_m0_nan_se([r['poder_star'] for r in rs], K),
        se_cbar_star=_m0_nan_se([r['cbar_star'] for r in rs], K),
        fallback_used=int(any(int(r.get('fallback_used', 0)) for r in rs)),
        refined=int(all(int(r.get('refined', 0)) for r in rs)),
        cvs=cvs, cvs_se=cvs_se,
        inv_dev=float(np.nanmax(finite_inv)) if finite_inv else float('nan'),
        # Neither curve is aggregated across seeds: each of the K seeds walked
        # the grid with its own draws, so a mean of the K curves would not be
        # the curve any calibration actually saw. Kept None, as power_curve
        # already was, rather than fabricated. The per-seed curves are not
        # retained -- if they are ever wanted, the m=0 path must persist them
        # before aggregation, not after.
        power_curve=None, calib_curve=None, cfg_seed=-K, raw=None,
    )


def calibrate_config(T, m, lambdas, P):
    out = {'T': T, 'm': m, 'lambdas': lambdas, 'by_sigma2': {}}
    if (m == 0 and P.get('m0_seed_averaging', False)
            and int(P.get('m0_seeds', 1)) > 1):
        # m=0 has one configuration (no lambda). Calibrate over K independent
        # seed streams (seed_base + k*m0_seed_stride) and report the seed-average.
        # Native: the pickle written here is already seed-averaged, so CSV/tables/
        # figure derive from it directly. The K loop is serial (one m=0 config is
        # one joblib task in run_grid; the 9 m=0 tasks parallelize across workers).
        K = int(P['m0_seeds']); stride = int(P['m0_seed_stride']); base = int(P['seed_base'])
        for method in P['sigma2_methods']:
            rs = []
            for k in range(K):
                Pk = dict(P); Pk['seed_base'] = base + k * stride
                r = calibrate_config_sigma(T, m, lambdas, Pk, method)
                if r is not None:
                    rs.append(r)
            if len(rs) >= 2:
                out['by_sigma2'][method] = _aggregate_m0_method(rs, T, method)
            elif rs:
                out['by_sigma2'][method] = rs[0]     # degenerate (K effectively 1)
    else:
        for method in P['sigma2_methods']:
            res = calibrate_config_sigma(T, m, lambdas, P, method)
            if res is not None:
                out['by_sigma2'][method] = res
    return out if out['by_sigma2'] else None


# =============================================================================
# 3. CHECKPOINTING (resume) AND ORCHESTRATION
# =============================================================================
def config_key(T, m, lambdas):
    lam = '_'.join(str(round(l, 3)) for l in lambdas)
    return f"T{T}_m{m}_lambdas_{lam}"


def pkl_path(P, key):  return os.path.join(P['checkpoint_dir'], f"{key}.pkl")
def npz_path(P, key):  return os.path.join(P['checkpoint_dir'], f"{key}_raw.npz")


def save_result(P, res):
    key = config_key(res['T'], res['m'], res['lambdas'])
    raw_blocks = {}
    for method, r in res['by_sigma2'].items():
        if r.get('raw') is not None:
            for stat, vec in r['raw'].items():
                raw_blocks[f"{method}__{stat}"] = vec
            r = dict(r); r['raw'] = None
            res['by_sigma2'][method] = r
    if raw_blocks:
        np.savez_compressed(npz_path(P, key), **raw_blocks)
    with open(pkl_path(P, key), 'wb') as fh:
        pickle.dump(res, fh)


def already_done(P, T, m, lambdas):
    return os.path.exists(pkl_path(P, config_key(T, m, lambdas)))


def run_grid(P, n_jobs=-1, chunk_size=None):
    """
    Runs the pending configuration grid with guaranteed progress reporting.

    The grid is split into batches; `Parallel` is opened once as a context
    manager (reusing the worker pool across batches, without re-paying the
    spawn cost) and called once per batch. Each `parallel(...)` call blocks
    until its batch finishes and then returns control to the main process,
    so a progress print after every batch does not depend on worker stdout
    being forwarded or on joblib's internal verbosity -- unlike a single
    `Parallel(...)` call over the whole `todo` list, which only returns
    control when everything finishes and can leave the terminal silent for
    hours on a full production grid. Progress is also written to
    `<checkpoint_dir>/progress.log` (append, immediate flush), for `tail -f`
    when the script runs in background/nohup.
    """
    os.makedirs(P['checkpoint_dir'], exist_ok=True)
    cfgs = enumerate_configs(P)
    todo = [c for c in cfgs if not already_done(P, *c)]
    n_total = len(todo)
    log_path = os.path.join(P['checkpoint_dir'], "progress.log")

    def _log(msg):
        print(msg, flush=True)
        try:
            with open(log_path, 'a') as f:
                f.write(msg + "\n")
        except OSError:
            pass  # never let a logging failure bring down the computation

    _log(f"Configs: {len(cfgs)} total | {n_total} pending | "
         f"{len(cfgs) - n_total} already done (resume).")
    if n_total == 0:
        _log("Nothing pending.")
        return
    t0 = time.time()

    def _do(cfg):
        T, m, lambdas = cfg
        res = calibrate_config(T, m, lambdas, P)
        if res is not None:
            save_result(P, res)
        return cfg

    def _progress_line(n_done):
        el = time.time() - t0
        rate = n_done / el if el > 0 else 0.0
        eta = (n_total - n_done) / rate if rate > 0 else float('nan')
        return (f"  [{n_done}/{n_total}] ({100*n_done/n_total:5.1f}%)  "
                f"{el:7.0f}s decorridos  ~{eta:7.0f}s restantes  "
                f"{rate:5.3f} cfg/s")

    if _HAS_JOBLIB and n_jobs != 1 and n_total > 1:
        # numba kernels are process-safe; loky spawns fresh interpreters.
        # Progress is reported PER CONFIG (not per chunk) using joblib's
        # streaming generator, so a single expensive config (e.g. the refined
        # m=0 cells) does not block all visible progress. The unordered
        # generator yields each result as soon as its worker finishes.
        n_done = 0
        try:
            results = Parallel(n_jobs=n_jobs, backend="loky",
                               return_as="generator_unordered")(
                delayed(_do)(c) for c in todo)
            for _cfg in results:
                n_done += 1
                if n_done % max(1, n_total // 100) == 0 or n_done <= 5 \
                        or n_done == n_total:
                    _log(_progress_line(n_done))
        except TypeError:
            # joblib too old for return_as: fall back to small-chunk batches.
            if chunk_size is None:
                workers = n_jobs if n_jobs > 0 else (os.cpu_count() or 4)
                chunk_size = max(workers, 1)
            with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
                for start in range(0, n_total, chunk_size):
                    chunk = todo[start:start + chunk_size]
                    parallel(delayed(_do)(c) for c in chunk)
                    n_done += len(chunk)
                    _log(_progress_line(n_done))
    else:
        for i, c in enumerate(todo, 1):
            _do(c)
            _log(f"{_progress_line(i)}  {config_key(*c)}")
    _log(f"Grid done in {time.time() - t0:.0f}s.")


# =============================================================================
# 4. AGGREGATION + ALL PAPER OBJECTS
# =============================================================================
def aggregate(P):
    import pandas as pd
    rows = []
    vm_max = 0.0
    for f in sorted(os.listdir(P['checkpoint_dir'])):
        if not f.endswith(".pkl"):
            continue
        with open(os.path.join(P['checkpoint_dir'], f), 'rb') as fh:
            res = pickle.load(fh)
        for method, r in res['by_sigma2'].items():
            lam = r['lambdas']
            row = dict(
                config_key=config_key(r['T'], r['m'], lam),
                T=r['T'], m=r['m'], sigma2_method=method,
                cbar_otimo=r['cbar_star'], poder_no_otimo=r['poder_star'],
                se_poder=r['se_poder_star'], se_cbar=r.get('se_cbar_star', np.nan),
                fallback_used=int(r.get('fallback_used', False)),
                refined=int(r.get('refined', False)),
                inv_dev=r['inv_dev'], cfg_seed=r['cfg_seed'],
            )
            for i in range(5):
                row[f'lambda{i+1}'] = lam[i] if i < len(lam) else np.nan
            for stat in ['mza', 'msb', 'mzt', 'pt', 'mpt']:
                for q, tag in [('1%', 'cv1'), ('2.5%', 'cv2_5'),
                               ('5%', 'cv5'), ('10%', 'cv10')]:
                    row[f'{stat}_{tag}'] = r['cvs'].get(stat, {}).get(q, np.nan)
                    row[f'{stat}_{tag}_se'] = r['cvs_se'].get(stat, {}).get(q, np.nan)
            rows.append(row)
            if np.isfinite(r['inv_dev']):
                vm_max = max(vm_max, abs(r['inv_dev']))

    df = pd.DataFrame(rows).sort_values(['sigma2_method', 'T', 'm']).reset_index(drop=True)
    out = os.path.join(P['checkpoint_dir'], "cbar_surface.csv")
    df.to_csv(out, index=False)
    print(f"\nAggregated: {out} | {len(df)} rows")

    print(f"[Invariance] max |MZt(beta=0) - MZt(beta=10)| over grid = {vm_max:.2e} "
          f"(expected ~ 0).")
    if vm_max > 1e-6:
        print("  [!] WARNING: statistic NOT invariant to break magnitudes.")
    else:
        print("  [OK] statistic invariant across the grid (Lemma confirmed).")

    m0 = df[df.m == 0]
    if len(m0):
        for meth in P['sigma2_methods']:
            mm = m0[m0.sigma2_method == meth]
            if len(mm):
                print(f"[ERS check] m=0 ({meth}): c-bar in "
                      f"[{mm.cbar_otimo.min():.2f}, {mm.cbar_otimo.max():.2f}] (expect ~ -7)")
        print(f"[Power check] global mean power at optimum = "
              f"{df.poder_no_otimo.mean():.4f} (target {P['target_power']}).")
    return df


def fit_intercepts_1_over_T(df, sigma2_method='const'):
    """Per-m fit  c-bar(T) = c_inf + a(m)/T  by OLS, returning the intercept
    c_inf, slope a(m), and their standard errors. This is the source of the
    '-6.99/-7.07/-7.03' intercept numbers and now their SEs."""
    import pandas as pd
    d = df[df.sigma2_method == sigma2_method]
    out = []
    for m in sorted(d.m.unique()):
        dm = d[d.m == m].groupby('T')['cbar_otimo'].mean().reset_index()
        if len(dm) < 3:
            out.append(dict(m=int(m), c_inf=np.nan, c_inf_se=np.nan,
                            a=np.nan, a_se=np.nan, n=len(dm)))
            continue
        x = 1.0 / dm['T'].values
        y = dm['cbar_otimo'].values
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = max(len(y) - 2, 1)
        s2 = float(resid @ resid) / dof
        XtX_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(s2 * XtX_inv))
        out.append(dict(m=int(m), c_inf=float(beta[0]), c_inf_se=float(se[0]),
                        a=float(beta[1]), a_se=float(se[1]), n=len(dm)))
    res = pd.DataFrame(out)
    print("\n[1/T extrapolation] per-m intercepts c_inf (expect ~ -7 across m):")
    for _, r in res.iterrows():
        if np.isfinite(r['c_inf']):
            print(f"  m={int(r['m'])}: c_inf={r['c_inf']:+.2f} (SE {r['c_inf_se']:.2f}), "
                  f"a={r['a']:+.1f} (SE {r['a_se']:.1f}), n_T={int(r['n'])}")
    return res


def surface_diagnostics(P, df, sigma2_method='const'):
    """R^2 of the (m,T) lookup; R^2 and condition number of the lambda-only CKP
    surface for m=1; rank deficiency note for m>=2. Sources the paper's
    R^2=0.91, lambda-spread~0.24, R^2~0.16, condition-number ~2e3 numbers.

    Provenance: adjudicated 2026-07-12 against the then-production surface
    (427 configs, 46 (m,T) cells; const method): R2_mT=0.911, lam_spread=0.241,
    R2_lambda_m1=0.163, cond_lambda_m1=2162. lam_spread is the mean absolute
    within-(m,T)-cell deviation of cbar_otimo from the cell mean, over m>=1,
    const branch. That 2026-07-12 file is now legacy/cbar_surface.csv (C4);
    these R^2/spread numbers are PRE-correction and recompute off the
    corrected cbar_surface.csv -- a diagnostic, not a printed-table number."""
    d = df[df.sigma2_method == sigma2_method].reset_index(drop=True)
    y = d.cbar_otimo.values

    pred_mT = d.groupby(['m', 'T'])['cbar_otimo'].transform('mean').values
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2_mT = 1 - np.sum((y - pred_mT) ** 2) / ss_tot if ss_tot > 0 else np.nan

    # within-cell lambda spread (the "~0.24")
    grp = d.groupby(['m', 'T'])['cbar_otimo']
    within = grp.transform(lambda v: v - v.mean())
    lam_spread = float(np.mean(np.abs(within[d.m >= 1]))) if (d.m >= 1).any() else np.nan

    # lambda-only CKP surface for m=1: degree-4 polynomial in lambda_1
    diag = dict(r2_mT=float(r2_mT), lam_spread=lam_spread)
    d1 = d[d.m == 1].dropna(subset=['lambda1'])
    if len(d1) >= 6:
        lam1 = d1['lambda1'].values
        y1 = d1['cbar_otimo'].values
        Xp = np.column_stack([lam1 ** k for k in range(5)])  # 1, l, l^2, l^3, l^4
        b1, *_ = np.linalg.lstsq(Xp, y1, rcond=None)
        pred1 = Xp @ b1
        ss1 = np.sum((y1 - y1.mean()) ** 2)
        r2_lam_m1 = 1 - np.sum((y1 - pred1) ** 2) / ss1 if ss1 > 0 else np.nan
        cond_m1 = float(np.linalg.cond(Xp))
        diag['r2_lambda_m1'] = float(r2_lam_m1)
        diag['cond_lambda_m1'] = cond_m1
    else:
        diag['r2_lambda_m1'] = np.nan
        diag['cond_lambda_m1'] = np.nan

    # rank check for m=5 (61-regressor CKP form vs distinct configs)
    d5 = d[d.m == 5]
    diag['m5_distinct_configs'] = int(d5[['lambda1','lambda2','lambda3','lambda4','lambda5']]
                                      .drop_duplicates().shape[0]) if len(d5) else 0

    print(f"\n[Surface diagnostics, {sigma2_method}]")
    print(f"  R^2 of (m,T) lookup           = {diag['r2_mT']:.3f}   (paper: 0.91)")
    print(f"  within-cell |lambda| spread   = {diag['lam_spread']:.3f}   (paper: ~0.24)")
    print(f"  R^2 lambda-only surface (m=1) = {diag['r2_lambda_m1']:.3f}   (paper: ~0.16)")
    print(f"  cond. number lambda surf (m=1)= {diag['cond_lambda_m1']:.1f}   (paper: ~2e3)")
    print(f"  m=5 distinct break configs    = {diag['m5_distinct_configs']}   "
          f"(61-term CKP form is rank-deficient)")
    return diag


def make_latex_tables(P, df, intercepts, diag, sigma2_method='const'):
    """Table 1 (c-bar surface) and Table 2 (MZt 5% CVs) as LaTeX, plus a small
    macro file with the headline numbers so the paper can \\input them.

    Note: the three files must be written under method-dependent names. Since
    post_process() calls this function once per method in
    P['sigma2_methods']=['const','maic'], writing under FIXED names would let
    'maic' (called last) overwrite the 'const' artifacts (the headline files).
    In cells where MAIC collapses (T=30; see robustness) that can differ by >10
    in c-bar and make the 1/T intercepts appear to contradict Proposition 1,
    when the correct ('const') numbers confirm it. Fix: 'const' writes the
    CANONICAL names (those the paper \\input{}s); any other method writes to
    names suffixed _{sigma2_method}, keeping both without conflict.
    """
    d = df[df.sigma2_method == sigma2_method]
    out_dir = P['checkpoint_dir']
    suffix = "" if sigma2_method == 'const' else f"_{sigma2_method}"

    piv = d.groupby(['m', 'T'])['cbar_otimo'].mean().unstack().round(1)
    with open(os.path.join(out_dir, f"tab_cbar_mT{suffix}.tex"), 'w') as f:
        f.write(piv.to_latex(na_rep='---', float_format="%.1f",
                caption=f"Optimal $\\bar c(m,T)$ for Model~LB ($\\sigma^2$: {sigma2_method}).",
                label=f"tab:cbar{suffix}"))

    pivc = d.groupby(['m', 'T'])['mzt_cv5'].mean().unstack().round(2)
    with open(os.path.join(out_dir, f"tab_cv5_mzt{suffix}.tex"), 'w') as f:
        f.write(pivc.to_latex(na_rep='---', float_format="%.2f",
                caption=f"$\\mathrm{{MZ}}_t$ 5\\% critical values at optimal "
                        f"$\\bar c$ ($\\sigma^2$: {sigma2_method}).",
                label=f"tab:cv{suffix}"))

    # headline macros (the numbers hardcoded in the prose)
    def _row(m):
        r = intercepts[intercepts.m == m]
        return (float(r['c_inf'].values[0]) if len(r) else np.nan)
    macro_tag = "" if sigma2_method == 'const' else sigma2_method.capitalize()
    with open(os.path.join(out_dir, f"headline_numbers{suffix}.tex"), 'w') as f:
        f.write(f"% auto-generated headline numbers ({sigma2_method}; "
                f"refresh prose to match)\n")
        for m in [1, 2, 3]:
            v = _row(m)
            if np.isfinite(v):
                f.write(f"\\newcommand{{\\cinfMonem{m}{macro_tag}}}{{{v:.2f}}}\n")
        f.write(f"\\newcommand{{\\RsqMT{macro_tag}}}{{{diag.get('r2_mT', float('nan')):.2f}}}\n")
        f.write(f"\\newcommand{{\\RsqLamMone{macro_tag}}}{{{diag.get('r2_lambda_m1', float('nan')):.2f}}}\n")
        cond_val = diag.get('cond_lambda_m1', float('nan'))
        cond_str = f"{cond_val:.3g}" if np.isfinite(cond_val) else "nan"
        f.write(f"\\newcommand{{\\condLamMone{macro_tag}}}{{{cond_str}}}\n")
        f.write(f"\\newcommand{{\\lamSpread{macro_tag}}}{{{diag.get('lam_spread', float('nan')):.2f}}}\n")
    print(f"[tables] wrote tab_cbar_mT{suffix}.tex, tab_cv5_mzt{suffix}.tex, "
          f"headline_numbers{suffix}.tex in {out_dir}/")


def post_process(P):
    """Run all aggregation + paper-object generation from existing checkpoints."""
    df = aggregate(P)
    for method in P['sigma2_methods']:
        inter = fit_intercepts_1_over_T(df, method)
        diag = surface_diagnostics(P, df, method)
        make_latex_tables(P, df, inter, diag, method)
    return df


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# SINGLE-SERIES TEST INTERFACE  (used by run_model_lb.py)
# -----------------------------------------------------------------------------
# Thin layer over the SAME kernel (mstats_nb) that produces the paper's tables,
# so a user testing one series gets numbers on the identical convention: the
# long-run variance is estimated on the OLS-detrended series (the Perron-Qu
# hybrid of Section 2), NOT the GLS-detrended one. c-bar is read from a
# calibration CSV by nearest (m,T) or falls back to the ERS demeaned value -7;
# critical values are simulated on the fly in the exact break configuration.
# =============================================================================
ALPHA_DEFAULT = 0.05
CBAR_ERS_DEMEANED = -7.0
_STAT_NAMES = ("MZa", "MSB", "MZt", "PT", "MPT")


def cbar_from_csv(csv_path, m, T, method="const"):
    """Nearest-(m,T) lookup of (cbar_star, cv5_MZt) from a calibration CSV with
    columns m,T,sigma2_method,cbar_otimo,mzt_cv5 (the schema this kernel writes).
    Returns (cbar, cv5_MZt) or None."""
    import csv as _csv
    best, bestd = None, 1e18
    with open(csv_path, newline="") as fh:
        for r in _csv.DictReader(fh):
            if r.get("sigma2_method", method) != method:
                continue
            try:
                mm, TT = int(r["m"]), int(r["T"])
            except (KeyError, ValueError):
                continue
            if mm != m:
                continue
            d = abs(TT - T)
            if d < bestd:
                cv = r.get("mzt_cv5", "")
                best = (float(r["cbar_otimo"]),
                        float(cv) if cv not in ("", None) else None)
                bestd = d
    return best


def critical_values_on_the_fly(nt, break_pos, cbar, method_code, kmax,
                               R=9999, seed=20260701, alpha=0.05):
    """Simulate the null (driftless random walk with the SAME level-break design)
    and return the lower-percentile critical values of the five statistics in the
    exact (nt, break_pos) configuration. Uses gen_dgp_nb + mstats_nb (the kernel)."""
    Z = build_z_nb(nt, np.asarray(break_pos, dtype=np.int64))
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(R) for k in _STAT_NAMES}
    kept = 0
    for _ in range(R):
        eps = rng.standard_normal(nt)
        y = gen_dgp_nb(nt, np.asarray(break_pos, dtype=np.int64), 0.0, 0.0, eps)
        mza, msb, mzt, pt, mpt, ok = mstats_nb(y, Z, cbar, method_code, kmax)
        if ok > 0.5:
            draws["MZa"][kept] = mza; draws["MSB"][kept] = msb
            draws["MZt"][kept] = mzt; draws["PT"][kept] = pt
            draws["MPT"][kept] = mpt; kept += 1
    q = 100.0 * alpha
    return {k: float(np.percentile(v[:kept], q)) for k, v in draws.items()}


def run_test(y, break_pos, sigma2_method="const", kmax=12, cbar_csv=None,
             cbar_override=None, alpha=ALPHA_DEFAULT, R_null=9999, seed=20260701):
    """Run the Model LB test on series y with KNOWN, EXOGENOUS break positions
    (0-based interior). Returns a dict with the statistics, critical values and a
    per-statistic verdict. Same kernel and s^2 convention as the paper's tables."""
    y = np.asarray(y, dtype=float)
    nt = len(y); m = len(break_pos)
    mc = 0 if sigma2_method == "const" else 1
    bp = np.asarray(break_pos, dtype=np.int64)

    cbar, cv5_csv, source = None, None, None
    if cbar_override is not None:
        cbar, source = float(cbar_override), "user-supplied"
    elif cbar_csv:
        got = cbar_from_csv(cbar_csv, m, nt, sigma2_method)
        if got is not None:
            cbar, cv5_csv = got
            source = f"calibration CSV (m={m}, T={nt}, nearest)"
    if cbar is None:
        cbar, source = CBAR_ERS_DEMEANED, f"ERS demeaned default ({CBAR_ERS_DEMEANED})"

    Z = build_z_nb(nt, bp)
    mza, msb, mzt, pt, mpt, ok = mstats_nb(y, Z, cbar, mc, kmax)
    if ok < 0.5:
        raise RuntimeError("Degenerate statistics on the observed series.")
    # ADF-GLS t on the GLS-detrended series (reported alongside; not an M-stat)
    stats = dict(MZa=mza, MSB=msb, MZt=mzt, PT=pt, MPT=mpt)

    if cv5_csv is not None and abs(alpha - 0.05) < 1e-9:
        cvs = {"MZt": cv5_csv}
        cv_source = f"tabulated MZt CV from CSV; others simulated (R={R_null})"
        others = critical_values_on_the_fly(nt, bp, cbar, mc, kmax, R=R_null,
                                             seed=seed, alpha=alpha)
        for k in _STAT_NAMES:
            cvs.setdefault(k, others[k])
    else:
        cvs = critical_values_on_the_fly(nt, bp, cbar, mc, kmax, R=R_null,
                                         seed=seed, alpha=alpha)
        cv_source = f"on-the-fly null simulation (R={R_null})"

    verdict = {k: ("reject" if stats[k] < cvs[k] else "fail") for k in _STAT_NAMES}
    return dict(nt=nt, m=m, break_pos=list(map(int, break_pos)), cbar=cbar,
                cbar_source=source, sigma2_method=sigma2_method,
                stats=stats, cvs=cvs, cv_source=cv_source, verdict=verdict,
                alpha=alpha)


def print_report(res):
    print("=" * 70)
    print("  Model LB point-optimal GLS unit-root test  (no-trend)")
    print("=" * 70)
    print(f"  sample length T          : {res['nt']}")
    print(f"  number of breaks m       : {res['m']}")
    print(f"  break positions (0-based): {res['break_pos']}")
    print(f"  c-bar                    : {res['cbar']:.3f}  [{res['cbar_source']}]")
    # C5/D5, 2026-08-10. This line PRINTED "(OLS-detrended, Perron-Qu)" -- the
    # convention correction 1 (e821f7d) moved away from -- to every user of the
    # tool, on stdout, under a value they chose. Same propagation vector as
    # s2ar_diff_nb's old docstring, one step worse because it is not a comment.
    #
    # And the replacement has to be PER ARM: the OLS-GLS hybrid describes where
    # the MAIC ORDER comes from, and the 'const' arm selects no order at all.
    # One sentence covering both would be the same kind of near-miss.
    _lrv = {
        'const': "plain difference variance on the GLS-detrended series, "
                 "sum (Delta ytilde_t)^2 / (T-1)",
        'maic': "Perron-Qu OLS-GLS: MAIC order from the OLS-detrended "
                "regression, s^2 on the GLS-detrended series",
    }.get(res['sigma2_method'], "convention not recorded")
    print(f"  long-run variance        : {res['sigma2_method']} ({_lrv})")
    print(f"  critical values          : {res['cv_source']}")
    print(f"  significance level       : {int(res['alpha']*100)}%")
    print("-" * 70)
    print(f"  {'statistic':<8}{'value':>12}{'crit. val.':>14}   decision")
    print("-" * 70)
    for k in ("PT", "MPT", "MZa", "MSB", "MZt"):
        v, c, d = res["stats"][k], res["cvs"][k], res["verdict"][k]
        flag = "reject H0 (stationary)" if d == "reject" else "fail to reject (unit root)"
        print(f"  {k:<8}{v:>12.4f}{c:>14.4f}   {flag}")
    print("-" * 70)
    n = sum(1 for k in res["verdict"] if res["verdict"][k] == "reject")
    print(f"  {n}/5 M-statistics reject the unit root at the {int(res['alpha']*100)}% level.")
    print("  Small values reject; break dates are treated as KNOWN and EXOGENOUS.")
    print("=" * 70)


def _selftest_single():
    """Validation gates for the single-series interface (fast)."""
    print("=" * 60); print("SELF-TEST kernel_numba -- single-series interface"); print("=" * 60)
    ok = True
    warm_up_numba(12)
    rng = np.random.default_rng(0); T = 120; bp = np.array([40, 80], dtype=np.int64)
    Z = build_z_nb(T, bp)
    # (1) empirical size ~ alpha under H0
    rej = 0; R = 2000
    cv = critical_values_on_the_fly(T, bp, -7.0, 0, 12, R=2000, seed=1)["MZt"]
    for _ in range(R):
        y = gen_dgp_nb(T, bp, 0.0, 0.0, rng.standard_normal(T))
        _, _, mzt, _, _, o = mstats_nb(y, Z, -7.0, 0, 12)
        rej += (o > 0.5 and mzt < cv)
    size = rej / R
    print(f"  (1) empirical size (MZt) ~ 0.05: {size:.3f}  [{'OK' if 0.02<size<0.10 else 'CHECK'}]")
    ok &= 0.02 < size < 0.10
    # (2) magnitude-invariance: statistics unchanged by break height
    y0 = gen_dgp_nb(T, bp, 0.0, 0.0, rng.standard_normal(T))
    r_small = mstats_nb(y0, Z, -7.0, 0, 12)
    ybig = y0.copy()
    for j, tau in enumerate(bp):
        ybig[tau:] += 50.0 * (j + 1)
    r_big = mstats_nb(ybig, Z, -7.0, 0, 12)
    dmzt = abs(r_small[2] - r_big[2])
    print(f"  (2) magnitude-invariance |dMZt|={dmzt:.2e}  [{'OK' if dmzt<1e-8 else 'CHECK'}]")
    ok &= dmzt < 1e-8
    # (3) power at c=-10 exceeds size
    rej = 0
    for _ in range(R):
        eps = rng.standard_normal(T)
        y = gen_dgp_nb(T, bp, -10.0, 0.0, eps)
        _, _, mzt, _, _, o = mstats_nb(y, Z, -7.0, 0, 12)
        rej += (o > 0.5 and mzt < cv)
    power = rej / R
    print(f"  (3) power at c=-10 > size: {power:.3f} > {size:.3f}  [{'OK' if power>size else 'CHECK'}]")
    ok &= power > size
    print("=" * 60); print("SELF-TEST:", "ALL PASSED" if ok else "FAILURES"); print("=" * 60)
    return ok


def main():
    # By default stdout is FULLY buffered when not attached to a terminal
    # (e.g. `nohup python3 ... > log.txt &`) -- nothing appears until the
    # buffer fills or the process ends, which alone explains "no progress
    # printed" in an hours-long run. Forcing line buffering fixes this for
    # ALL prints in the script, not just those in run_grid.
    sys.stdout.reconfigure(line_buffering=True)

    ap = argparse.ArgumentParser(description="Calibration of c-bar for Model LB")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--chunk-size", type=int, default=None,
                    help="configs per batch on the parallel path (default: "
                         "2x the number of workers). Smaller batches report "
                         "progress more frequently.")
    ap.add_argument("--speed", action="store_true", help="quick smoke test (subgrid)")
    ap.add_argument("--selftest", action="store_true", help="run the single-series validation gates and exit")
    ap.add_argument("--outdir", default=DEFAULTS['checkpoint_dir'])
    ap.add_argument("--post", action="store_true",
                    help="only aggregate + build tables/figure from checkpoints")
    args = ap.parse_args()
    if getattr(args, "selftest", False):
        import sys as _sys
        _sys.exit(0 if _selftest_single() else 1)

    P = dict(DEFAULTS)
    P['checkpoint_dir'] = args.outdir

    print(warm_up_numba(P['kmax']))
    print(f"[joblib] {'available' if _HAS_JOBLIB else 'NOT available (serial)'}")

    if args.speed:
        P.update(T_grid=[30, 60, 100], m_grid=[0, 1, 2], R_cv=600, R_pow=400,
                 R_curve=200, compute_power_curves=False,
                 refine_R_cv=1500, refine_R_pow=1500,  # keep refine path active but cheap
                 sigma2_methods=['const'], m0_seeds=3,
                 checkpoint_dir=args.outdir + "_speed")
        print("[speed] reduced grid / reps for a smoke test (refine path active, cheap).")

    if P.get('m0_seed_averaging', False) and int(P.get('m0_seeds', 1)) > 1:
        need = guard_m0_stride(P)
        print(f"[calibration] m=0 native seed-averaging: K={P['m0_seeds']} seeds, "
              f"stride={P['m0_seed_stride']:,} (> {need:,}).")

    if args.post:
        post_process(P)
        return

    run_grid(P, n_jobs=args.jobs, chunk_size=args.chunk_size)
    post_process(P)


if __name__ == "__main__":
    main()
