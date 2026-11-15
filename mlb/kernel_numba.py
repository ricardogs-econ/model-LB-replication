# -*- coding: utf-8 -*-
"""
================================================================================
mlb/kernel_numba.py -- Numerical kernel for the Model LB point-optimal GLS unit-root
               test (Carrion-i-Silvestre, Kim & Perron 2009, no-trend case).
================================================================================

The package's implementation of the M statistics (with the two long-run variance
arms), of the local-to-unity data-generating process (gen_dgp_nb) and of the
configuration grid: the modules of mlb/ and the functions of model_lb/ call it. It is
a library.

Model LB: y_t = Z_t' theta + u_t, u_t = (1 + c/T) u_{t-1} + eps_t, with the
no-trend regressor set Z_t = [1, DU_{1,t}, ..., DU_{m,t}] where
DU_{j,t} = 1{t > tau_j}. Under GLS quasi-differencing each level dummy becomes
an impulse plus a tail of order c-bar/T, and the Gram matrix of the
quasi-differenced design departs from the identity by a term of order (m+1)/T
(the paper's Lemma 1, lem:design). The point-optimal test is asymptotically
invariant to the number and location of the breaks, and the half-power point
of the envelope, the detrending parameter the paper locates by a level
condition, converges to the root of the limit envelope, the ERS (1996) demeaned
value of about -7, for every configuration. In finite samples it departs from
that limit by O(1/T) (Proposition 1, prop:flat); Table 1 of the paper gives it
exactly (mlb/envelope_exact.py).

CONTENTS
    Numerical kernels (@njit): build_z_nb, glsd_nb, ols_detrend_nb,
        s2ar_diff_nb, maic_select_k_nb, s2ar_at_k_nb, mstats_nb, gen_dgp_nb.
    Draws: _stats_under (the five statistics over n replications of one
        configuration); warm_up_numba compiles the kernels.
    The configuration grid: DEFAULTS, make_lambdas, enumerate_configs,
        config_seed, config_key, break_pos_from_lambdas; _METHOD_CODE maps the
        long-run variance arms 'const' and 'maic' to the codes 0 and 1 of
        mstats_nb.

DETERMINISM
    _stats_under draws replication r from numpy's default_rng(seed0 + r), so
    its draws are bit-reproducible on a fixed platform and do not depend on how
    the replications are split into calls; config_seed(P, T, m, lambdas) gives
    a configuration's seed from P['seed_base'].

REFERENCE
    Carrion-i-Silvestre, J.L., Kim, D., Perron, P. (2009). GLS-based unit root
    tests with multiple structural breaks under both the null and the
    alternative hypotheses. Econometric Theory 25(6), 1754-1792.
================================================================================
"""
from __future__ import annotations
import itertools
import numpy as np
from numba import njit

from .break_dates import break_dates

# =============================================================================
# DEFAULTS: the configuration grid, the MAIC maximum lag and the seed base
# =============================================================================
DEFAULTS = dict(
    T_grid       = [45, 50, 60, 80, 100, 150, 200, 300],
    m_grid       = [0, 1, 2, 3, 4, 5],
    m_min_T      = {0: 45, 1: 45, 2: 45, 3: 45, 4: 60, 5: 80},
    trim         = 0.15,
    min_spacing  = 0.15,
    n_grid       = 7,
    kmax         = 12,
    seed_base    = 20240601,
)

# =============================================================================
# 1. NUMERICAL KERNELS  (@njit; Model LB: Z = [1, DU_1, ..., DU_m])
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
    Element-wise, to stay inside nopython mode."""
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
    """OLS-detrended level residual y - Z @ bhat_ols, the input of the MAIC lag
    order (maic_select_k_nb). The projection removes Z @ beta exactly, so the lag
    order is invariant to the break magnitudes by algebra alone (supplement S2.1,
    rem:invariance)."""
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


# =============================================================================
# 1b. PERRON & QU (2007) OLS-GLS LONG-RUN VARIANCE
# =============================================================================
# Perron, P. and Z. Qu (2007), "A simple modification to improve the finite
# sample properties of Ng and Perron's unit root tests", Economics Letters
# 94:12-19. Their recipe: the MAIC lag order is built entirely from the
# OLS-detrended ADF regression (their eq. 4), while the regression that
# produces s2_AR runs on the GLS-detrended series (their eq. 2). They state
# that running BOTH stages on OLS-detrended data yields tests that are too
# conservative in common sample sizes and do not recommend it.
#
# Under level breaks the OLS-detrended series is also the wrong input for the
# difference-based estimator: OLS detrending fits segment means to a random
# walk, so the detrended series inherits jumps at the break dates and the
# estimator collects m increments of order T, so E[s2] stays above 1 and does
# not fall with T, whereas under GLS detrending E[s2] = 1 because the
# quasi-differenced dummy is an impulse. The estimators
# below therefore run on the GLS-detrended series; only the MAIC lag order
# comes from the OLS-detrended regression.


@njit(cache=True)
def s2ar_diff_nb(yt):
    """The difference-based long-run variance s^2_Delta = sum (Delta ytilde_t)^2 /
    (T - 1) of the series passed: the plain sum of squared first differences,
    with no regression and no mean removal. It is the 'const' arm (code 0) of
    mstats_nb, called on the GLS-detrended series.

    It is not the k = 0 member of the autoregressive estimator of eq:s2ar, whose
    regression keeps the level term b_0 ytilde_{t-1} when the sum over j is
    empty.
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
def maic_select_k_nb(yt_ols, kmax):
    """MAIC lag order from the OLS-detrended ADF regression (PQ eq. 4): the
    common fixed-kmax sample, tau_T(k) and sigma2_k of Ng and Perron (2001).
    Selection only. Returns k.
    """
    nt = yt_ols.shape[0]
    # sigma_check^2_k: COMMON sample for every k, divisor (T - kmax), as in
    # Ng and Perron (2001) and Perron and Qu (2007); the paper selects k <= 12
    # by the MAIC on the OLS-detrended series (sec:lags). The divisor is not
    # neutral for the argmin, because tau_T(k) is proportional to
    # 1/sigma_check^2_k.
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
def s2ar_at_k_nb(yt, k_sel):
    """AR spectral s2_AR at a GIVEN order, estimated on the series passed
    (eq. s2ar of the paper, called on the GLS-detrended series).

    sigma_hat^2_ek: the k-SPECIFIC sample (starting at k+1) and divisor T,
    per Ng-Perron (2001) / Perron-Qu (2007). This is deliberately NOT the
    estimator used for lag selection: selection uses sigma_check^2_k on the
    common kmax sample with divisor (T - kmax).
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

    Long-run variance: Perron & Qu (2007) OLS-GLS (see section 1b above): s2_AR
    on the GLS-detrended series, the MAIC lag order from the OLS-detrended
    regression.
    """
    nt = Z.shape[0]
    yt, ssra = glsd_nb(y, Z, cbar)
    if not np.isfinite(ssra):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Long-run variance, PQ OLS-GLS. Both branches estimate s2 on the
    # GLS-detrended level residual yt; the MAIC ORDER comes from the
    # OLS-detrended regression (invariant to theta).
    if sigma2_method == 0:
        # 0: the difference-based s^2_Delta, the plain sum of squared
        # differences (see s2ar_diff_nb's docstring).
        sar = s2ar_diff_nb(yt)
    else:
        yt_ols = ols_detrend_nb(y, Z)
        k_sel = maic_select_k_nb(yt_ols, kmax)
        sar = s2ar_at_k_nb(yt, k_sel)

    # sum of squares of the GLS-detrended level series (lagged)
    denom = 0.0
    for t in range(nt - 1):
        denom += yt[t] * yt[t]
    if denom <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # The M class is normalized by T^{-1} and T^{-2} throughout, the paper's
    # definition (eq. Mstats) and Ng and Perron's (2001); the MPT line below
    # uses the same T^{-1}.
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
def gen_dgp_nb(nt, c, eps):
    """y = u, with u_t = (1 + c/T) u_{t-1} + eps_t: Model LB with theta = 0. The
    GLS-detrended statistics are invariant to theta, so the deterministic terms
    are not added; the break dates enter through Z (build_z_nb), not through y.

    INITIAL CONDITION: the recursion starts at the PRE-SAMPLE value u_0 = 0, so
    the first in-sample observation is u_1 = eps_1. u_0 = 0 is the paper's
    convention (as in ERS; rem:init), under which Theorem 1 and the half-power
    points are stated: under the null the statistics are exactly invariant to
    u_0, under the alternative it affects power.

    eps is a precomputed N(0,1) vector (so the RNG stays in the orchestrator,
    where seeding is reproducible). Returns y."""
    u = np.zeros(nt)
    rho = 1.0 + c / nt
    u[0] = eps[0]                       # u_1 = eps_1, from u_0 = 0
    for t in range(1, nt):
        u[t] = rho * u[t - 1] + eps[t]
    return u


# =============================================================================
# 2. PYTHON HELPERS  (warm-up, the configuration grid, seeds, draws)
# =============================================================================

def warm_up_numba(kmax=12):
    """Compile every kernel once and verify that compilation really occurred."""
    nt = 40
    bp = np.array([12, 24], dtype=np.int64)
    rng = np.random.default_rng(0)
    eps = rng.standard_normal(nt)
    y = gen_dgp_nb(nt, 0.0, eps)
    Z = build_z_nb(nt, bp)
    _ = glsd_nb(y, Z, -7.0)
    _ = ols_detrend_nb(y, Z)
    _ = mstats_nb(y, Z, -7.0, 0, kmax)
    _ = mstats_nb(y, Z, -7.0, 1, kmax)
    # signatures present => compiled
    compiled = (len(mstats_nb.signatures) > 0 and len(glsd_nb.signatures) > 0)
    assert compiled, "THE KERNELS DID NOT COMPILE -- aborting."


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
    return combos


def enumerate_configs(P):
    """All (T, m, lambdas) with T >= m_min_T[m]."""
    cfgs = []
    for m in P['m_grid']:
        for T in P['T_grid']:
            if T < P['m_min_T'][m]:
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
    """tau_j = clamp(floor(lambda_j T), 1, T - 1), exact (see break_dates.py)."""
    return np.array(break_dates(T, lambdas), dtype=np.int64)


_METHOD_CODE = {'const': 0, 'maic': 1}


def _stats_under(T, break_pos, c, cbar, method_code, n_reps, seed0, kmax):
    """Run n_reps draws at alternative c, evaluating M-stats at detrending cbar.
    Returns dict of arrays (one per statistic). RNG seeded reproducibly."""
    out = {k: np.empty(n_reps) for k in ['mza', 'msb', 'mzt', 'pt', 'mpt']}
    for r in range(n_reps):
        rng = np.random.default_rng((seed0 + r) % (2 ** 63 - 1))
        eps = rng.standard_normal(T)
        y = gen_dgp_nb(T, c, eps)
        Z = build_z_nb(T, break_pos)
        mza, msb, mzt, pt, mpt, ok = mstats_nb(y, Z, cbar, method_code, kmax)
        if ok > 0.5:
            out['mza'][r] = mza; out['msb'][r] = msb; out['mzt'][r] = mzt
            out['pt'][r] = pt;  out['mpt'][r] = mpt
        else:
            for k in out:
                out[k][r] = np.nan
    return out


# =============================================================================
# 3. CONFIGURATION KEYS
# =============================================================================
def config_key(T, m, lambdas):
    lam = '_'.join(str(round(l, 3)) for l in lambdas)
    return f"T{T}_m{m}_lambdas_{lam}"
