"""Robust tests for level shifts: Harvey, Leybourne and Taylor (2010) and
Carrion-i-Silvestre and Gadea (2023).

A line-by-line port of the MATLAB functions distributed by Carrion-i-Silvestre and
Gadea with their article (J.L. Carrion-i-Silvestre and M.D. Gadea, "Testing for
multiple level shifts with an integrated or stationary noise component", Journal of
Applied Econometrics 38(6), 2023, 801-819; folder functions_empirical of their
replication files), which also hold their implementation of the procedure of
D.I. Harvey, S.J. Leybourne and A.M.R. Taylor, "Robust methods for detecting multiple
level breaks in autocorrelated time series", Journal of Econometrics 157, 2010,
342-358. The function names are those of the MATLAB files, in lower case:

  HLT_test_empirical.m           hlt_test
  locate_breaks_HLT.m            locate_breaks_hlt
  lrv_u.m, lrv_e.m               lrv_u, lrv_e
  Ftest_0_m_max.m                ftest_0_m_max
  MultipleBreaksJointEstimationGSSR.m, MultipleBreaksEstimation.m
                                 multiple_breaks_joint, multiple_breaks_one_at_a_time
  MultipleBreaksInference.m      multiple_breaks_inference
  locate_breaks_BP.m             locate_breaks_bp
  MeanBreak_segment.m            mean_break_segment
  dating.m, dating_impulses.m    dating, dating_impulses
  RS_BP_1.m                      rs_bp_1
  vec_SSR_combinacions.m         vec_ssr_combinations
  mestima_HLT.m                  estimate_breaks_hlt
  lrv_e_np.m, lrv_u_np.m         lrv_e_np, lrv_u_np

Break points are the authors' T_B: the 1-based position of the last observation of the
old regime; the first observation of the new regime is T_B + 1, and in calendar years
tau = years[T_B] (0-based indexing of the year vector).

Numerical choices of the MATLAB code, kept here:
  * round() is MATLAB's (halves away from zero), see matlab_round;
  * max/min return the FIRST index on ties (np.argmax/np.argmin do the same);
  * the HLT ranking sorts |M_j| ascending with a stable sort and takes the LAST row,
    so among equal values the later date is taken;
  * A\\b for an over-determined system is MATLAB's QR solver: Householder QR with
    column pivoting (LAPACK dgeqp3), numerical rank = #{|R_ii| > max(m, n) eps(|R_11|)}
    and the basic solution (zero coefficients on the columns beyond the rank), see
    mldivide. It matters in lrv_u: its final regression shifts each impulse dummy by
    [1, 1, k_opt - 1] (the authors' code; the lag selection uses 1, ..., k_opt - 1),
    so for k_opt >= 2 lagged dummy columns are repeated and the regression is rank
    deficient. The coefficient on the lagged level that the variance uses is still
    identified, and the basic solution gives it; an SVD solve that keeps singular
    values of order eps does not;
  * lag1 puts NaN in the first position and shiftr fills with zeros; every NaN is
    trimmed before a regression;
  * critical values: the HLT (2010) constants for the windows 0.10 and 0.15 that the
    package uses (from HLT_test_empirical.m), the response surface of RS_BP_1.m for
    the trimming 0.15 (RS_COEF), and the authors' tables of inputs/level_shift_cv/
    for the trimming 0.15, read as numbers; table rows are selected by exact
    equality with the trimming, as in the MATLAB code;
  * the decisions are strict inequalities (statistic > critical value).
"""

import math

import numpy as np
from scipy import linalg

from . import paths

CV_DIR = paths.INPUTS / "level_shift_cv"

# HLT (2010) settings by window m, as in HLT_test_empirical.m and mestima_HLT.m, for
# the windows the package uses
HLT_WINDOWS = (0.10, 0.15)
HLT_NB_MAX = (8, 5)
HLT_CV_S1 = (0.579, 0.679)
HLT_CV_S0 = (23.400, 18.550)
HLT_KAPPA = (1.003, 1.018)
HLT_TAU_L, HLT_TAU_U = 0.15, 0.85

# Ftest_0_m_max.m: trimming -> the suffix of its tables (only 0.15 is shipped)
FTEST_TABLES = {0.15: 5}

# RS_BP_1.m: response-surface coefficients for the trimming 0.15 (39 x 4), the 10, 5,
# 2.5 and 1% levels
RS_COEF = np.array([
    [0.19976, 0.27097, 0.33898, 0.43429],
    [0.069355, 0.093165, 0.12036, 0.15654],
    [-1.0506, -1.4234, -1.7713, -2.2994],
    [-0.91861, -1.2528, -1.5945, -2.0573],
    [-0.70643, -0.97381, -1.2634, -1.6337],
    [-1.9091, -2.5557, -3.1133, -4.0907],
    [1.2826, 1.7183, 2.0755, 2.7067],
    [-0.77756, -1.0353, -1.2296, -1.6271],
    [1.0776, 1.4462, 1.7515, 2.3082],
    [0.37166, 0.49962, 0.61979, 0.79848],
    [0.48756, 0.65365, 0.82408, 1.0529],
    [0.5096, 0.63972, 0.77859, 1.0099],
    [0.10593, 0.17229, 0.30655, 0.37909],
    [0.46473, 0.61601, 0.76995, 0.99354],
    [0.47885, 0.60557, 0.74677, 0.96125],
    [4.9633, 6.7998, 8.4449, 11.0321],
    [1.3883, 1.8497, 2.3827, 3.118],
    [-0.64643, -1.0735, -1.3225, -1.803],
    [-0.7465, -0.85568, -0.84964, -1.2024],
    [-1.0441, -0.87488, -1.1403, -0.92061],
    [8.2439, 10.7604, 13.0505, 16.8277],
    [-7.9025, -11.3078, -14.2963, -19.0513],
    [-9.7196, -13.0662, -16.8865, -22.1729],
    [3.9658, 6.6857, 8.081, 11.0277],
    [-1.0112, -1.9845, -2.7759, -3.4772],
    [6.5205, 6.1799, 7.8481, 7.212],
    [-19.4391, -25.0053, -30.2275, -38.4969],
    [13.6587, 20.0593, 25.6652, 34.7477],
    [27.5466, 37.3052, 48.1805, 63.7459],
    [-10.5665, -18.0597, -21.4575, -29.3801],
    [1.0398, 2.6807, 4.0275, 5.0804],
    [-15.9105, -15.9008, -19.9233, -19.1424],
    [28.7593, 36.978, 44.7113, 56.8095],
    [-9.6853, -14.4132, -18.544, -25.3558],
    [-27.1268, -36.9519, -47.5982, -63.5589],
    [10.4566, 18.0688, 21.1916, 29.1702],
    [-0.72268, -1.9049, -2.9781, -3.806],
    [13.5808, 14.0014, 17.3964, 17.1208],
    [-16.0248, -20.6229, -24.9596, -31.6979],
])

# the CISG statistics in the order of locate_breaks_BP.m, with the break-date matrix
# each one reads (0: BP_1 / mat_Tbe, 1: first modified, 2: second modified)
CISG_STATISTICS = ("F_1", "F_1a", "F_1b", "F_0", "F_Ua", "F_Ub")
CISG_DATE_SOURCE = (0, 1, 2, 0, 1, 2)
CISG_LEVELS = ("10", "05")  # rows of the inference outputs
FTEST_LEVEL_ROWS = (("10", 0), ("05", 1), ("01", 3))  # rows of the 4-row tables


# --------------------------------------------------------------------- helpers
def matlab_round(v):
    """MATLAB round: to the nearest integer, halves away from zero."""
    a = abs(float(v))
    r = math.floor(a)
    if a - r >= 0.5:
        r += 1
    return int(r) if v >= 0 else -int(r)


def k_max_rule(t):
    """The lag bound of the authors' application, round(4 (T/100)^(1/4))."""
    return matlab_round(4 * (t / 100) ** 0.25)


def mldivide(a, b):
    """MATLAB's A\\b for an over-determined system (rows > columns): Householder QR
    with column pivoting, numerical rank #{|R_ii| > max(m, n) eps(|R_11|)}, basic
    solution."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m, n = a.shape
    if m <= n:
        raise ValueError(f"mldivide: {m} x {n} is not over-determined")
    q, r, piv = linalg.qr(a, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    tol = max(m, n) * np.spacing(d[0]) if d[0] > 0 else 0.0
    rank = int(np.sum(d > tol))
    x = np.zeros(n)
    if rank:
        z = linalg.solve_triangular(r[:rank, :rank], q[:, :rank].T @ b)
        x[piv[:rank]] = z
    return x


def lag1(z):
    """[NaN; z(1:T-1)]."""
    z = np.asarray(z, dtype=float)
    return np.concatenate(([np.nan], z[:-1]))


def shift_columns(v, shifts):
    """shiftr(v', s, 0)': column i is v shifted down by s[i] (up when negative),
    filled with zeros."""
    v = np.asarray(v, dtype=float)
    t = len(v)
    out = np.empty((t, len(shifts)))
    for i, s in enumerate(shifts):
        s = int(s)
        if s == 0:
            out[:, i] = v
        elif s > 0:
            out[:s, i] = 0.0
            out[s:, i] = v[: t - s]
        else:
            out[: t + s, i] = v[-s:]
            out[t + s:, i] = 0.0
    return out


def _impulses(t, dates):
    """T x len(dates) matrix with a one at position date + 1 (1-based) of column j:
    matrix_equal_by_elem(seqa(1, 1, T), (dates + 1)')."""
    tr = np.arange(1, t + 1, dtype=float)
    return (tr[:, None] == (np.asarray(dates, dtype=float) + 1.0)[None, :]).astype(
        float
    )


def _steps(t, dates):
    """T x len(dates) matrix of [zeros(date, 1); ones(T - date, 1)] columns."""
    dates = np.atleast_1d(np.asarray(dates, dtype=float))
    out = np.zeros((t, len(dates)))
    for i, d in enumerate(dates):
        out[int(d):, i] = 1.0
    return out


def _isin(values, pool):
    pool = np.asarray(pool, dtype=float)
    return (np.asarray(values, dtype=float)[:, None] == pool[None, :]).any(axis=1)


def _largest_jumps(y, m, tau_l, tau_u):
    """The first stage of HLT_test_empirical.m and mestima_HLT.m: the dates of the
    largest |Delta y| in [tau_l T + 1, tau_u T + 1], one at a time, each excluding
    the positions within round(m T) - 1 of the previous one."""
    t = len(y)
    i_m = HLT_WINDOWS.index(m)
    nb_max = HLT_NB_MAX[i_m]
    t_l = matlab_round(tau_l * t)
    t_u = matlab_round(tau_u * t)
    sep = matlab_round(m * t)
    tr = np.arange(1, t + 1, dtype=float)
    dy = y - lag1(y)
    tdy = dy[t_l:t_u + 1]
    ttr = tr[t_l:t_u + 1]
    dates = [ttr[int(np.argmax(np.abs(tdy)))] - 1.0]
    for j in range(2, nb_max + 1):
        excl = dates[j - 2] - sep + 1 + np.arange(2 * sep - 1, dtype=float)
        keep = ~_isin(ttr, excl + 1.0)
        if not keep.any():
            break
        tdy = tdy[keep]
        ttr = ttr[keep]
        dates.append(ttr[int(np.argmax(np.abs(tdy)))] - 1.0)
    return np.array(dates)


# ------------------------------------------------------------------------ HLT
def lrv_u(z, kmax, sel_k, dum, brk):
    """lrv_u.m: the autoregressive long-run variance of the levels residual z
    (Dickey-Fuller regression with the impulse dummies dum when brk == 1), lag
    selected by BIC over 1..kmax when sel_k == 1. Returns (lrv, k_opt)."""
    z = np.asarray(z, dtype=float)
    dum = np.asarray(dum, dtype=float).reshape(len(z), -1)
    n = len(z)
    lz = lag1(z)
    dz = z - lz

    def design(k, dum_shifts):
        if brk == 1:
            if k == 1:
                return np.column_stack([lz, dum])
            blocks = [lz[:, None], dum, shift_columns(dz, range(1, k))]
            blocks += [shift_columns(dum[:, j], dum_shifts)
                       for j in range(dum.shape[1])]
            return np.column_stack(blocks)
        if k == 1:
            return lz[:, None]
        return np.column_stack([lz, shift_columns(dz, range(1, k))])

    if sel_k == 0:
        k_opt = kmax
    else:
        bic = np.zeros(kmax)
        for k in range(1, kmax + 1):
            x = design(k, list(range(1, k)))[kmax:]
            dep = dz[kmax:]
            b = mldivide(x, dep)
            r = dep - x @ b
            s2 = (r @ r) / (n - kmax - x.shape[1])
            bic[k - 1] = np.log(s2) + (k * math.log(n - kmax) / (n - kmax))
        k_opt = int(np.argmin(bic)) + 1
    # the final regression shifts the dummies by [1, 1, k_opt - 1], as lrv_u.m does
    x = design(k_opt, [1, 1, k_opt - 1])[k_opt:]
    dep = dz[k_opt:]
    b = mldivide(x, dep)
    r = dep - x @ b
    if brk == 1:
        sig2 = (r @ r) / (n - (2 + dum.shape[1]) * k_opt)
    else:
        sig2 = (r @ r) / (n - k_opt)
    return sig2 / (b[0] ** 2), k_opt


def lrv_e(z, kmax, sel_k):
    """lrv_e.m: the autoregressive long-run variance of the differences residual z,
    lag selected by BIC over 1..kmax when sel_k == 1. Returns (lrv, k_opt)."""
    z = np.asarray(z, dtype=float)
    n = len(z)
    lz = lag1(z)
    dz = z - lz

    def design(k):
        if k == 1:
            return lz[:, None]
        return np.column_stack([lz, shift_columns(dz, range(1, k))])

    if sel_k == 0:
        k_opt = kmax
    else:
        bic = np.zeros(kmax)
        for k in range(1, kmax + 1):
            x = design(k)[kmax:]
            dep = dz[kmax:]
            b = mldivide(x, dep)
            r = dep - x @ b
            s2 = (r @ r) / (n - kmax - x.shape[1])
            bic[k - 1] = np.log(s2) + (k * math.log(n - kmax) / (n - kmax))
        k_opt = int(np.argmin(bic)) + 1
    x = design(k_opt)[k_opt:]
    dep = dz[k_opt:]
    b = mldivide(x, dep)
    r = dep - x @ b
    sig2 = (r @ r) / (n - k_opt - x.shape[1])
    return sig2 / (b[0] ** 2), k_opt


def hlt_test(y, m, k_max, sel_k):
    """HLT_test_empirical.m: the S_1, S_0 and union (U) tests of Harvey, Leybourne
    and Taylor (2010) with window m, at 5% against the asymptotic critical values.

    Returns a dict with the MATLAB outputs: s1v, s0v (one per candidate, in rank
    order), nb_S1, nb_S0, nb_U, brk_dates (the candidates' T_B), brk_dates_S1,
    brk_dates_S0, brk_dates_U, omega_u, omega_e, k_opt_u, k_opt_e; and the
    decisions rej_S1, rej_S0, rej_U per candidate and the constants used."""
    y = np.asarray(y, dtype=float)
    t = len(y)
    i_m = HLT_WINDOWS.index(m)
    nb_max = HLT_NB_MAX[i_m]
    cv_s0, cv_s1, kappa = HLT_CV_S0[i_m], HLT_CV_S1[i_m], HLT_KAPPA[i_m]
    t_l = matlab_round(HLT_TAU_L * t)
    t_u = matlab_round(HLT_TAU_U * t)
    wind = matlab_round(m * t / 2)
    sep = matlab_round(m * t)
    tr = np.arange(1, t + 1, dtype=float)

    # dummies at the largest jumps, for the long-run variances
    jumps = _largest_jumps(y, m, HLT_TAU_L, HLT_TAU_U)
    d_mat = _impulses(t, jumps)
    du_mat = np.cumsum(d_mat, axis=0)
    x = np.column_stack([np.ones(t), du_mat])
    b = mldivide(x, y)
    r = y - x @ b
    omega_u, k_opt_u = lrv_u(r, k_max, sel_k, d_mat, 1)
    dy = y - lag1(y)
    dep = dy[1:]
    x = d_mat[1:]
    b = mldivide(x, dep)
    dr = dep - x @ b
    omega_e, k_opt_e = lrv_e(dr, k_max, sel_k)

    # M_j: mean of the next wind observations minus mean of the last wind ones
    leads = shift_columns(y, [-s for s in range(1, wind + 1)]).T
    lags = shift_columns(y, list(range(0, wind))).T
    m_j = np.mean(leads, axis=0) - np.mean(lags, axis=0)

    h_val = np.abs(m_j[t_l - 1:t_u])
    h_tr = tr[t_l - 1:t_u]
    order = np.argsort(h_val, kind="stable")
    s_tr = h_tr[order]
    cands = [s_tr[-1]]
    for j in range(1, nb_max):
        nogo = cands[j - 1] - sep + 1 + np.arange(2 * sep - 1, dtype=float)
        keep = ~_isin(s_tr, nogo)
        if not keep.any():
            break
        s_tr = s_tr[keep]
        cands.append(s_tr[-1])
    cands = np.array(cands)
    hh = np.abs(m_j[cands.astype(int) - 1])
    s0v = (t ** 0.5) * hh / math.sqrt(omega_u)
    s1v = (t ** -0.5) * hh / math.sqrt(omega_e)
    rej_s1 = s1v > cv_s1
    rej_s0 = s0v > cv_s0
    rej_u = (s1v > (kappa * cv_s1)) | (s0v > (kappa * cv_s0))
    return dict(
        s1v=s1v, s0v=s0v,
        nb_S1=int(rej_s1.sum()), nb_S0=int(rej_s0.sum()), nb_U=int(rej_u.sum()),
        brk_dates=cands, brk_dates_S1=cands[rej_s1], brk_dates_S0=cands[rej_s0],
        brk_dates_U=cands[rej_u],
        omega_u=omega_u, omega_e=omega_e, k_opt_u=k_opt_u, k_opt_e=k_opt_e,
        rej_S1=rej_s1, rej_S0=rej_s0, rej_U=rej_u,
        cv_S1=cv_s1, cv_S0=cv_s0, kappa=kappa, nb_max=nb_max, wind=wind, sep=sep,
        t_L=t_l, t_U=t_u, dummy_dates=jumps,
    )


def _assign_column(mat, col, values):
    """mat(1:n, col) = values, growing mat with zero rows as MATLAB does."""
    n = len(values)
    if n > mat.shape[0]:
        mat = np.vstack([mat, np.zeros((n - mat.shape[0], mat.shape[1]))])
    mat[:n, col] = values
    return mat


def locate_breaks_hlt(nb_s1, nb_s0, nb_u, brk_s1, brk_s0, brk_u, m_max, years):
    """locate_breaks_HLT.m: (dates_HLT, position_HLT), m_max x 3 (columns S_1, S_0,
    U); dates are years[T_B - 1], the year of the last pre-break observation."""
    years = np.asarray(years, dtype=float)
    dates = np.full((m_max, 3), np.nan)
    pos = np.full((m_max, 3), np.nan)
    for col, (n, brk) in enumerate(((nb_s1, brk_s1), (nb_s0, brk_s0), (nb_u, brk_u))):
        if n != 0:
            brk = np.asarray(brk, dtype=float)
            dates = _assign_column(dates, col, years[brk.astype(int) - 1])
            pos = _assign_column(pos, col, brk)
    return dates, pos


# ----------------------------------------------------------------------- CISG
def vec_ssr_combinations(x):
    """vec_SSR_combinacions.m: rows (i, j, (sum_{t=i..j} x_t)^2 / (j - i + 1)) for
    all 1 <= i < j <= T, in nchoosek order."""
    x = np.asarray(x, dtype=float)
    t = len(x)
    s = np.cumsum(x)
    sl = np.concatenate(([0.0], s[:-1]))
    rows = []
    for i in range(1, t):
        for j in range(i + 1, t + 1):
            rows.append((float(i), float(j), (s[j - 1] - sl[i - 1]) ** 2 / (j - i + 1)))
    return np.array(rows)


def mean_break_segment(mc, t_inf, t_sup, t, eps_trimming):
    """MeanBreak_segment.m: the sup statistic for one mean break inside the segment
    [t_inf, t_sup]. Returns (Test_segment, Test_segment_mod_a, Test_segment_mod_b,
    T_inferior, Tbe_segment, T_superior); the first, second, third and fifth are
    None when no break is admissible in the segment (MATLAB's empty)."""
    t_desc = matlab_round(t * eps_trimming) - 1
    c0, c1 = mc[:, 0], mc[:, 1]
    seg1 = mc[(c0 == t_inf) & ((t_inf + t_desc - 1) < c1) & (c1 <= (t_sup - t_desc))]
    seg2 = np.zeros_like(seg1)
    for i in range(seg1.shape[0]):
        sel = mc[(c0 == (seg1[i, 1] + 1)) & (c1 == t_sup)]
        if sel.shape[0] != 1:
            raise ValueError("MeanBreak_segment: the second segment is not unique")
        seg2[i] = sel[0]
    ssr_h1 = seg1[:, 2] + seg2[:, 2]
    ssr_h0 = mc[(c0 == t_inf) & (c1 == t_sup), 2]
    if seg1.shape[0] == 0:
        return None, None, None, t_inf, None, t_sup
    vals = t ** (-2) * (ssr_h1 - ssr_h0)
    idx = int(np.argmax(vals))
    test = vals[idx]
    tbe = seg1[idx, 1]
    mod_a = (t_sup - t_inf + 1) ** (-2) * (t ** 2 * test)
    mod_b = (t_sup - t_inf + 1) ** (-1) * (t * test)
    return test, mod_a, mod_b, t_inf, tbe, t_sup


def _ssr_recursive(start, y, z, h, last):
    """The ssr subfunction of dating.m: recursive-residual SSRs of the segments
    [start, r], r = start + h - 1, ..., last (1-based), stored at position r."""
    vecssr = np.zeros(last)
    zs = z[start - 1:start + h - 1]
    ys = y[start - 1:start + h - 1]
    inv1 = np.linalg.inv(zs.T @ zs)
    delta1 = inv1 @ (zs.T @ ys)
    res = ys - zs @ delta1
    vecssr[start + h - 2] = res @ res
    r = start + h
    while r <= last:
        zr = z[r - 1]
        v = y[r - 1] - zr @ delta1
        invz = inv1 @ zr
        f = 1 + zr @ invz
        delta2 = delta1 + (invz * v) / f
        inv2 = inv1 - np.outer(invz, invz) / f
        inv1 = inv2
        delta1 = delta2
        vecssr[r - 1] = vecssr[r - 2] + v * v / f
        r += 1
    return vecssr


def _parti(start, b1, b2, last, bigvec, bigt):
    """The parti subfunction of dating.m: the optimal one-break partition of
    [start, last] with the break in [b1, b2]. Returns (ssrmin, date)."""
    dvec = np.zeros(bigt)
    ini = (start - 1) * bigt - (start - 2) * (start - 1) // 2 + 1
    for j in range(b1, b2 + 1):
        jj = j - start
        k = j * bigt - (j - 1) * j // 2 + last - j
        dvec[j - 1] = bigvec[ini + jj - 1] + bigvec[k - 1]
    seg = dvec[b1 - 1:b2]
    mi = int(np.argmin(seg))
    return seg[mi], (b1 - 1) + mi + 1


def dating(y, z, h, m, q, bigt):
    """dating.m (Bai and Perron 1998): the break points that minimize the global SSR
    of y on z, for 1..m breaks, minimal segment h. Returns (glb, datevec, bigvec);
    column k of datevec holds the k break points."""
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float).reshape(len(y), q)
    datevec = np.zeros((m, m))
    optdat = np.zeros((bigt, m))
    optssr = np.zeros((bigt, m))
    dvec = np.zeros(bigt)
    glb = np.zeros(m)
    bigvec = np.zeros(bigt * (bigt + 1) // 2)
    for i in range(1, bigt - h + 2):
        vecssr = _ssr_recursive(i, y, z, h, bigt)
        lo = (i - 1) * bigt + i - (i - 1) * i // 2
        hi = i * bigt - (i - 1) * i // 2
        bigvec[lo - 1:hi] = vecssr[i - 1:bigt]
    if m == 1:
        ssrmin, datx = _parti(1, h, bigt - h, bigt, bigvec, bigt)
        datevec[0, 0] = datx
        glb[0] = ssrmin
        return glb, datevec, bigvec
    for j1 in range(2 * h, bigt + 1):
        ssrmin, datx = _parti(1, h, j1 - h, j1, bigvec, bigt)
        optssr[j1 - 1, 0] = ssrmin
        optdat[j1 - 1, 0] = datx
    glb[0] = optssr[bigt - 1, 0]
    datevec[0, 0] = optdat[bigt - 1, 0]
    for ib in range(2, m + 1):
        if ib == m:
            jlast = bigt
            for jb in range(ib * h, jlast - h + 1):
                dvec[jb - 1] = (optssr[jb - 1, ib - 2]
                                + bigvec[(jb + 1) * bigt - jb * (jb + 1) // 2 - 1])
            seg = dvec[ib * h - 1:jlast - h]
            mi = int(np.argmin(seg))
            optssr[jlast - 1, ib - 1] = seg[mi]
            optdat[jlast - 1, ib - 1] = (ib * h - 1) + mi + 1
        else:
            for jlast in range((ib + 1) * h, bigt + 1):
                for jb in range(ib * h, jlast - h + 1):
                    dvec[jb - 1] = (optssr[jb - 1, ib - 2]
                                    + bigvec[jb * bigt - jb * (jb - 1) // 2
                                             + jlast - jb - 1])
                seg = dvec[ib * h - 1:jlast - h]
                mi = int(np.argmin(seg))
                optssr[jlast - 1, ib - 1] = seg[mi]
                optdat[jlast - 1, ib - 1] = (ib * h - 1) + mi + 1
        datevec[ib - 1, ib - 1] = optdat[bigt - 1, ib - 1]
        for i in range(1, ib):
            xx = ib - i
            datevec[xx - 1, ib - 1] = optdat[int(datevec[xx, ib - 1]) - 1, xx - 1]
        glb[ib - 1] = optssr[bigt - 1, ib - 1]
    return glb, datevec, bigvec


def dating_impulses(y, h, m):
    """dating_impulses.m: the global-SSR dates of 1..m impulses in y (the first
    differences of the series), minimal distance h. Returns (glb, datevec, optdat)."""
    y = np.asarray(y, dtype=float)
    bigt = len(y)
    datevec = np.zeros((m, m))
    optdat = np.zeros((bigt, m))
    optssr = np.zeros((bigt, m))
    dvec = np.zeros((bigt, 2))
    glb = np.zeros(m)
    init_segment = np.zeros((bigt, m))
    y_sq = y ** 2
    s_sq = np.cumsum(y_sq)
    if m == 1:
        ind = int(np.argmax(y_sq[h - 1:bigt - h])) + 1
        datevec[0, 0] = ind + h - 1
        glb[0] = s_sq[bigt - 1] - y_sq[int(datevec[0, 0]) - 1]
        return glb, datevec, optdat
    for j1 in range(2 * h, bigt + 1):
        ind = int(np.argmax(y_sq[h - 1:j1 - h])) + 1
        optdat[j1 - 1, 0] = ind + h - 1
        optssr[j1 - 1, 0] = s_sq[j1 - 1] - y_sq[int(optdat[j1 - 1, 0]) - 1]
    glb[0] = optssr[bigt - 1, 0]
    datevec[0, 0] = optdat[bigt - 1, 0]
    init_segment[:, 0] = 1
    for ib in range(2, m + 1):
        jlasts = [bigt] if ib == m else range((ib + 1) * h, bigt + 1)
        for jlast in jlasts:
            for jb in range(ib * h, jlast - h + 1):
                ind = int(np.argmax(y_sq[jb - 1:jlast - h])) + 1
                temp_ssr = (s_sq[jlast - 1] - s_sq[jb - 1]) - y_sq[ind + jb - 2]
                dvec[jb - 1, 0] = optssr[jb - 1, ib - 2] + temp_ssr
                dvec[jb - 1, 1] = jb + ind - 1
            seg = dvec[ib * h - 1:jlast - h, 0]
            mi = int(np.argmin(seg))
            optssr[jlast - 1, ib - 1] = seg[mi]
            optdat[jlast - 1, ib - 1] = dvec[ib * h - 1 + mi, 1]
            init_segment[jlast - 1, ib - 1] = ib * h + (mi + 1) - 1
        datevec[ib - 1, ib - 1] = optdat[bigt - 1, ib - 1]
        initial = bigt
        for i in range(1, ib):
            xx = ib - i
            index_start = int(init_segment[initial - 1, xx])
            datevec[xx - 1, ib - 1] = optdat[index_start - 1, xx - 1]
            initial = index_start
        glb[ib - 1] = optssr[bigt - 1, ib - 1]
    return glb, datevec, optdat


def estimate_breaks_hlt(y, m, eps_trimming):
    """mestima_HLT.m: the break points of the largest |Delta y|, one at a time
    (window m, trimming eps_trimming)."""
    y = np.asarray(y, dtype=float)
    return _largest_jumps(y, m, eps_trimming, 1 - eps_trimming)


def _bandwidth_and_lrv(e, u_null, n):
    """The common tail of lrv_e_np.m and lrv_u_np.m: the AR(1) plug-in bandwidth of
    the quadratic spectral kernel from e (length n) and the long-run variance of
    u_null with that kernel, divided by n."""
    eb = e[:n - 1]
    ef = e[1:n]
    ae = (eb @ ef) / (eb @ eb)
    ee = ef - eb * ae
    se = math.sqrt((ee @ ee) / (n - 1))
    ad = (se ** 2) / (1 - ae) ** 4
    a2 = 4 * (ae * se / (1 - ae) ** 4) ** 2 / ad
    eband = 1.3221 * ((a2 * n) ** 0.2)
    jb = np.arange(1, n, dtype=float) / eband
    jband = jb * 1.2 * np.pi
    kern = ((np.sin(jband) / jband - np.cos(jband)) / (jband ** 2)) * 3
    sig = u_null @ u_null
    lam = 0.0
    for j in range(1, n):
        lam = lam + (u_null[:n - j] @ u_null[j:n]) * kern[j - 1]
    omega = sig + lam + lam
    return omega / n, eband


def lrv_e_np(x, vec_tb, lrv_est):
    """lrv_e_np.m: the nonparametric long-run variance of Delta x after impulse
    dummies at vec_tb + 1 (lrv_est == 1: residuals under the alternative; 0: Delta x
    itself). Returns (lrv, bandwidth)."""
    x = np.asarray(x, dtype=float)
    t = len(x)
    d_mat = _impulses(t, np.atleast_1d(vec_tb))
    dx = x[1:] - x[:-1]
    x_reg = d_mat[1:]
    b = mldivide(x_reg, dx)
    u_alt = dx - x_reg @ b
    u_null = dx if lrv_est == 0 else u_alt
    return _bandwidth_and_lrv(u_alt, u_null, t - 1)


def lrv_u_np(x, vec_tb, lrv_est, tbe_ho):
    """lrv_u_np.m: the nonparametric long-run variance of x around level shifts.
    The bandwidth uses the residuals with shifts at vec_tb; the variance uses those
    residuals (lrv_est == 1) or the residuals with shifts at tbe_ho (lrv_est == 0;
    the demeaned series when tbe_ho[0] == 0). Returns (lrv, bandwidth)."""
    x = np.asarray(x, dtype=float)
    t = len(x)
    x_reg = np.column_stack([np.ones(t), _steps(t, vec_tb)])
    b = mldivide(x_reg, x)
    u_alt = x - x_reg @ b
    if lrv_est == 0:
        tbe_ho = np.atleast_1d(np.asarray(tbe_ho, dtype=float))
        if tbe_ho[0] == 0:
            u_null = x - np.mean(x)
        else:
            x_reg = np.column_stack([np.ones(t), _steps(t, tbe_ho)])
            b = mldivide(x_reg, x)
            u_null = x - x_reg @ b
    else:
        u_null = u_alt
    return _bandwidth_and_lrv(u_alt, u_null, t)


def _load_table(name):
    return np.loadtxt(CV_DIR / name, ndmin=2)


def ftest_tables(eps_trimming, m_max):
    """The critical values and kappa constants of Ftest_0_m_max.m:
    (mat_cv_BP, mat_cv_BP_UDmax, mat_cv_BP_WDmax, mat_kappa, cv_BP_1, cv_BP_0)."""
    if eps_trimming not in FTEST_TABLES:
        raise ValueError(f"no Ftest_0 tables shipped for trimming {eps_trimming}")
    suffix = FTEST_TABLES[eps_trimming]
    cv_bp_1 = _load_table(f"cv_BP_1_Ftest_0_m_{suffix}.dat")
    cv_bp_0 = _load_table(f"cv_BP_0_Ftest_0_m_{suffix}.dat")
    kappa = _load_table(f"kappa_Ftest_0_m_{suffix}.dat")
    mat_cv = np.column_stack([cv_bp_1[:, :m_max], cv_bp_0[:, :m_max]])
    cv_ud = np.column_stack([cv_bp_1[:, m_max], cv_bp_0[:, m_max]])
    cv_wd = np.column_stack([cv_bp_1[:, m_max + 1], cv_bp_0[:, m_max + 1]])
    return mat_cv, cv_ud, cv_wd, kappa, cv_bp_1, cv_bp_0


def ftest_0_m_max(x, m_max, eps_trimming, gssr_dating):
    """Ftest_0_m_max.m: the BP_1 (I(1) noise) and BP_0 (I(0) noise) tests of zero
    against m = 1..m_max breaks and their UDmax and WDmax. gssr_dating = 1 dates the
    breaks for the long-run variances by the global SSR of the impulses in Delta x
    (joint strategy), 0 by mestima_HLT (one at a time).

    Returns a dict: mat_BP (m_max x 2), datevec, BP_UDmax (2,), BP_WDmax (4 x 2),
    mat_cv_BP, mat_cv_BP_UDmax, mat_cv_BP_WDmax, mat_kappa, omega_e, omega_u,
    vec_Tb."""
    x = np.asarray(x, dtype=float)
    t = len(x)
    t_discard = matlab_round(eps_trimming * t)
    _, datevec, _ = dating(x, np.ones((t, 1)), t_discard, m_max, 1, t)
    if gssr_dating == 1:
        _, mat_tb, _ = dating_impulses(x[1:] - x[:-1], t_discard, m_max)
        vec_tb = mat_tb[:, m_max - 1]
    else:
        vec_tb = estimate_breaks_hlt(x, eps_trimming, eps_trimming)
    omega_e, _ = lrv_e_np(x, vec_tb, 1)
    omega_u, _ = lrv_u_np(x, vec_tb, 0, 0)
    mat_bp = np.zeros((m_max, 2))
    mc = vec_ssr_combinations(x)
    ssr_h0 = mc[t - 2, 2]
    for m in range(1, m_max + 1):
        ssr_h1 = 0.0
        tb = [0.0, *datevec[:m, m - 1], float(t)]
        for k in range(1, m + 2):
            tk, tk1 = tb[k - 1], tb[k]
            ini = (tk * t - 1 / 2 * tk * (tk + 1)) + 1
            loc = tk1 - tk - 2
            ssr_h1 = ssr_h1 + mc[int(ini + loc) - 1, 2]
        mat_bp[m - 1, 0] = t ** (-2) * (ssr_h1 - ssr_h0) / (m * omega_e)
        mat_bp[m - 1, 1] = (ssr_h1 - ssr_h0) / (m * omega_u)
    mat_cv, cv_ud, cv_wd, kappa, cv_bp_1, cv_bp_0 = ftest_tables(eps_trimming, m_max)
    bp_udmax = mat_bp.max(axis=0)
    w1 = np.tile(cv_bp_1[:, [0]], (1, m_max)) / cv_bp_1[:, :m_max]
    w0 = np.tile(cv_bp_0[:, [0]], (1, m_max)) / cv_bp_0[:, :m_max]
    wd1 = np.array([max(w1[i, j] * mat_bp[j, 0] for j in range(m_max))
                    for i in range(4)])
    wd0 = np.array([max(w0[i, j] * mat_bp[j, 1] for j in range(m_max))
                    for i in range(4)])
    return dict(
        mat_BP=mat_bp, datevec=datevec, BP_UDmax=bp_udmax,
        BP_WDmax=np.column_stack([wd1, wd0]), mat_cv_BP=mat_cv,
        mat_cv_BP_UDmax=cv_ud, mat_cv_BP_WDmax=cv_wd, mat_kappa=kappa,
        omega_e=omega_e, omega_u=omega_u, vec_Tb=np.asarray(vec_tb, dtype=float),
    )


def ftest_union_decisions(res):
    """The union decisions on the UDmax and WDmax tests of zero breaks, at 10, 5 and
    1%, as in the authors' main_JST_Ftest_0.m: UDmax rejects when either BP_1 or BP_0
    exceeds its critical value times kappa column 6; WDmax when BP_1 exceeds its
    critical value times kappa column 6 or BP_0 its critical value times kappa
    column 7. Returns (rej_UDmax, rej_WDmax), each a tuple of three booleans."""
    ud, wd = res["BP_UDmax"], res["BP_WDmax"]
    cv_ud, cv_wd, kap = res["mat_cv_BP_UDmax"], res["mat_cv_BP_WDmax"], res["mat_kappa"]
    r_ud, r_wd = [], []
    for _, a in FTEST_LEVEL_ROWS:
        r_ud.append(bool((ud[0] > cv_ud[a, 0] * kap[a, 5])
                         | (ud[1] > cv_ud[a, 1] * kap[a, 5])))
        r_wd.append(bool((wd[a, 0] > cv_wd[a, 0] * kap[a, 5])
                         | (wd[a, 1] > cv_wd[a, 1] * kap[a, 6])))
    return tuple(r_ud), tuple(r_wd)


def _keep_final(mat):
    keep = mat.sum(axis=0) > 1
    return mat[np.ix_(keep, keep)]


def _sequential(x, m_max, eps_trimming, joint):
    """The common body of MultipleBreaksJointEstimationGSSR.m (joint) and
    MultipleBreaksEstimation.m (one at a time)."""
    x = np.asarray(x, dtype=float)
    t = len(x)
    mc = vec_ssr_combinations(x)
    datevec = None
    if joint:
        t_discard = matlab_round(eps_trimming * t)
        _, datevec, _ = dating_impulses(x[1:] - x[:-1], t_discard, m_max)
        vec_tb = datevec[:, m_max - 1]
    else:
        vec_tb = estimate_breaks_hlt(x, eps_trimming, eps_trimming)
    omega_e, _ = lrv_e_np(x, vec_tb, 1)
    bp1 = np.full(m_max, np.nan)
    bp1a = np.full(m_max, np.nan)
    bp1b = np.full(m_max, np.nan)
    bp0 = np.full(m_max, np.nan)
    tbe = np.zeros((m_max, m_max))
    tbea = np.zeros((m_max, m_max))
    tbeb = np.zeros((m_max, m_max))
    for m in range(m_max):
        if m == 0:
            orig, ma, mb, _, seg, _ = mean_break_segment(mc, 1, t, t, eps_trimming)
            bp1[0] = orig / omega_e
            bp1a[0] = ma / omega_e
            bp1b[0] = mb / omega_e
            tbe[0, 0] = tbea[0, 0] = tbeb[0, 0] = seg
            omega_u, _ = lrv_u_np(x, vec_tb, 0, 0)
            bp0[0] = t ** 2 * orig / omega_u
        elif tbe[:m, m - 1].sum() > 0:
            held = datevec[:m, m - 1] if joint else tbe[:m, m - 1]
            omega_u, _ = lrv_u_np(x, vec_tb, 0, held)
            bounds = [0.0, *held, float(t)]
            comp, comp_a, comp_b, comp_0 = [], [], [], []
            for i_m in range(m + 1):
                t_inf = bounds[i_m] + 1
                t_sup = bounds[i_m + 1]
                orig, ma, mb, t_inf, seg, t_sup = mean_break_segment(
                    mc, t_inf, t_sup, t, eps_trimming)
                if orig is None:
                    for c in (comp, comp_a, comp_b, comp_0):
                        c.append((m, 0.0, t_inf, 0.0, t_sup))
                else:
                    comp.append((m, orig / omega_e, t_inf, seg, t_sup))
                    comp_a.append((m, ma / omega_e, t_inf, seg, t_sup))
                    comp_b.append((m, mb / omega_e, t_inf, seg, t_sup))
                    comp_0.append((m, t ** 2 * orig / omega_u, t_inf, seg, t_sup))
            comp, comp_a, comp_b, comp_0 = (np.array(c, dtype=float)
                                            for c in (comp, comp_a, comp_b, comp_0))
            if np.sum(comp[:, 1]) == 0:
                continue
            i = int(np.argmax(comp[:, 1]))
            bp1[m] = comp[i, 1]
            bp0[m] = comp_0[i, 1]
            prev = held if joint else tbe[:m, m - 1]
            tbe[:m + 1, m] = np.sort(np.append(prev, comp[i, 3]))
            i = int(np.argmax(comp_a[:, 1]))
            bp1a[m] = comp_a[i, 1]
            prev = held if joint else tbea[:m, m - 1]
            tbea[:m + 1, m] = np.sort(np.append(prev, comp_a[i, 3]))
            i = int(np.argmax(comp_b[:, 1]))
            bp1b[m] = comp_b[i, 1]
            prev = held if joint else tbeb[:m, m - 1]
            tbeb[:m + 1, m] = np.sort(np.append(prev, comp_b[i, 3]))
    out = dict(mat_BP_1=bp1, mat_Tbe=_keep_final(tbe), mat_BP_1_mod_a=bp1a,
               mat_Tbe_mod_a=_keep_final(tbea), mat_BP_1_mod_b=bp1b,
               mat_Tbe_mod_b=_keep_final(tbeb), mat_BP_0=bp0, omega_e=omega_e,
               vec_Tb=np.asarray(vec_tb, dtype=float))
    if joint:
        out["datevec"] = datevec
    return out


def multiple_breaks_joint(x, m_max, eps_trimming):
    """MultipleBreaksJointEstimationGSSR.m: the sequential BP_1 (and its two
    modifications) and BP_0 statistics, the conditional breaks taken from the
    global-SSR dating of the impulses in Delta x (datevec)."""
    return _sequential(x, m_max, eps_trimming, joint=True)


def multiple_breaks_one_at_a_time(x, m_max, eps_trimming):
    """MultipleBreaksEstimation.m: the sequential statistics with the breaks
    estimated one at a time."""
    return _sequential(x, m_max, eps_trimming, joint=False)


def rs_bp_1(m, lam, eps_trimming):
    """RS_BP_1.m: response-surface critical values (10, 5, 2.5, 1%) of the
    sequential BP_1 test of m against m + 1 breaks, given the break fractions."""
    lam = np.sort(np.atleast_1d(np.asarray(lam, dtype=float)))
    if m == 0:
        lam = np.zeros(4)
    if m > 0 and m != len(lam):
        raise ValueError("RS_BP_1: length(lam) has to be equal to m")
    if len(lam) < 4:
        lam = np.concatenate([lam, np.zeros(4 - len(lam))])
    ld = lam[1:4] - lam[0:3]
    p1 = lam[0] * lam[1:4]
    p2 = lam[1] * lam[2:4]
    p3 = np.array([lam[2] * lam[3]])
    x_regs = np.concatenate([
        [1.0, m + 1.0], lam, ld ** 2, ld ** 3, ld ** 4, p1, p2, p3,
        p1 ** 2, p2 ** 2, p3 ** 2, p1 ** 3, p2 ** 3, p3 ** 3, p1 ** 4, p2 ** 4, p3 ** 4,
    ])
    if eps_trimming == 0.15:
        return x_regs @ RS_COEF
    raise ValueError("RS_BP_1: only the trimming 0.15 is shipped")


def inference_tables(eps_trimming):
    """The rows of the authors' tables for the trimming (4 levels x 10 breaks):
    (cv_F_1_a, cv_F_1_b, cv_F_0, kappa_U_a, kappa_U_b)."""
    out = []
    for name in ("cv_F_1_a.dat", "cv_F_1_b.dat", "cv_F_0.dat", "mat_kappa_U_a.dat",
                 "mat_kappa_U_b.dat"):
        tab = _load_table(name)
        out.append(tab[tab[:, 0] == eps_trimming, 2:])
    return tuple(out)


def multiple_breaks_inference(bp1, bp1a, bp1b, bp0, mat_tbe, t, eps_trimming):
    """MultipleBreaksInference.m: the number of breaks at 10% and 5% for the six
    statistics (F_1, F_1a, F_1b, F_0 and the unions F_Ua, F_Ub): the largest i such
    that the tests of 0, ..., i - 1 against one more break all reject. Returns
    (counts: dict statistic -> (n10, n05), rejections: dict statistic -> array of
    stat_est x 2)."""
    stat_est = int(np.sum(np.isfinite(bp1)))
    cv_a, cv_b, cv_0, k_a, k_b = inference_tables(eps_trimming)
    rho = {s: np.zeros((stat_est, 2), dtype=bool) for s in CISG_STATISTICS}
    num = {s: [0, 0] for s in CISG_STATISTICS}
    for i in range(1, stat_est + 1):
        lam = np.array([0.0]) if i == 1 else mat_tbe[:i - 1, i - 2] / t
        vec_cv = rs_bp_1(i - 1, lam, eps_trimming)
        c = i - 1
        rho["F_1"][c] = vec_cv[:2] < bp1[c]
        rho["F_1a"][c] = cv_a[:2, c] < bp1a[c]
        rho["F_1b"][c] = cv_b[:2, c] < bp1b[c]
        rho["F_0"][c] = cv_0[:2, c] < bp0[c]
        rho["F_Ua"][c] = ((cv_a[:2, c] * k_a[:2, c]) < bp1a[c]) | (
            (cv_0[:2, c] * k_a[:2, c]) < bp0[c])
        rho["F_Ub"][c] = ((cv_b[:2, c] * k_b[:2, c]) < bp1b[c]) | (
            (cv_0[:2, c] * k_b[:2, c]) < bp0[c])
        for s in CISG_STATISTICS:
            temp = np.cumsum(rho[s][:i].astype(float), axis=0)
            for lv in (0, 1):
                if temp[i - 1, lv] == i:
                    num[s][lv] = i
    return {s: tuple(v) for s, v in num.items()}, rho


def _pad_nan(mat, m_max):
    m = mat.shape[0]
    if m >= m_max:
        return mat
    out = np.full((m_max, m_max), np.nan)
    out[:m, :m] = mat
    return out


def locate_breaks_bp(counts, tbe, tbea, tbeb, m_max, years):
    """locate_breaks_BP.m: (dates_BP, position_BP), m_max x 6 x 2 (statistics in
    CISG_STATISTICS order; levels 10%, 5%); dates are years[T_B - 1]."""
    years = np.asarray(years, dtype=float)
    mats = [_pad_nan(np.asarray(a, dtype=float), m_max) for a in (tbe, tbea, tbeb)]
    dates = np.full((m_max, 6, 2), np.nan)
    pos = np.full((m_max, 6, 2), np.nan)
    for s, name in enumerate(CISG_STATISTICS):
        src = mats[CISG_DATE_SOURCE[s]]
        for lv in (0, 1):
            n = counts[name][lv]
            if n != 0:
                col = src[:n, n - 1]
                if np.isnan(col).any():
                    raise ValueError("locate_breaks_BP: no break dates for this count")
                pos[:n, s, lv] = col
                dates[:n, s, lv] = years[col.astype(int) - 1]
    return dates, pos
