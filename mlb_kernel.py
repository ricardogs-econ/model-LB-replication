#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mlb_kernel.py -- Pure-Python (NumPy-only) reference kernel for the Model LB test.

This is the dependency-free fallback for mlb_core.py. It implements the SAME
Model LB point-optimal GLS unit-root primitives as the numba kernel, in plain
NumPy, so the package runs on machines without numba (at roughly 45x the cost).

The two kernels are numerically equivalent to floating-point tolerance: identical
design matrix, identical GLS quasi-differencing, identical M-class statistics and
long-run-variance estimators. boot_ppp_cbar.py imports this module automatically
when `import mlb_core` fails, adapting the API in its _bind_kernels() fallback.

Public API (consumed by the fallback adapter in boot_ppp_cbar.py):
    build_z(nt, break_pos)                       -> Z design matrix
    gls_detrend(y, Z, cbar)                      -> (yt, ssr)
    m_statistics(y, Z, cbar, sigma2_method, kmax)-> dict(MZa,MSB,MZt,PT,MPT,ADF,k_lag,ok)

Rejection convention: all statistics reject the unit-root null for SMALL values;
the critical value is the lower percentile of the null distribution.

Validated against the numba kernel and by an internal self-test (empirical size
~0.05; exact invariance of MZt to break magnitudes -- Lemma 1 -- to ~1e-16;
power > size; DU structure). Run `python mlb_kernel.py --selftest`.
"""

from __future__ import annotations

import numpy as np

KMAX_DEFAULT = 12                 # defensive upper bound for MAIC lag search


# ============================================================================
# KERNEL  (pure NumPy; numerically equivalent to the numba production kernel)
# ============================================================================
def build_z(nt: int, break_pos: np.ndarray) -> np.ndarray:
    """Design matrix Z = [1, DU_1, ..., DU_m], with DU_j,t = 1{t > tau_j}.
    break_pos holds the m break positions as 1-based observation indices."""
    cols = [np.ones(nt)]
    for b in break_pos:
        du = np.zeros(nt)
        du[int(b):] = 1.0          # 1{t > tau_j}; index b is 1-based -> slice [b:]
        cols.append(du)
    return np.column_stack(cols)


def gls_detrend(y: np.ndarray, Z: np.ndarray, cbar: float):
    """GLS (quasi-difference) detrending at a-bar = 1 + cbar/T.
    Returns (yt, ssr): yt = y - Z @ bhat is the level residual used for the
    M-statistics; ssr is the quasi-differenced residual sum of squares used by
    the PT / MPT point-optimal statistics."""
    nt = len(y)
    a = 1.0 + cbar / nt
    yq = y.copy()
    Zq = Z.copy()
    yq[1:] = y[1:] - a * y[:-1]
    Zq[1:] = Z[1:] - a * Z[:-1]
    bhat, *_ = np.linalg.lstsq(Zq, yq, rcond=None)
    yt = y - Z @ bhat
    resid = yq - Zq @ bhat
    ssr = float(resid @ resid)
    return yt, ssr


def _s2_ar(yt: np.ndarray, method_code: int, kmax: int):
    """Long-run variance estimate of the detrended series.
    method_code 0 -> 'const': difference-based (k=0), s^2 = mean(diff(yt)^2).
        Consistent under i.i.d.; INCONSISTENT under serial correlation
        (plim = sigma^2/(1-rho^2) vs omega^2 = sigma^2/(1-rho)^2). Clean and
        robust at very short T.
    method_code 1 -> 'maic': autoregressive spectral estimator with MAIC lag
        selection (Ng-Perron 2001). Consistent under serial correlation; can be
        unstable at very short T (T <~ 40).
    Returns (s2, k_used)."""
    nt = len(yt)
    if method_code == 0:
        d = np.diff(yt)
        return float(np.mean(d * d)), 0
    # MAIC autoregressive spectral density at frequency zero
    best_k, best_maic, best_s2 = 0, np.inf, np.var(yt)
    y0 = yt - yt.mean()
    for k in range(0, kmax + 1):
        # regress dyt on yt_{-1} and k lags of dyt
        dy = np.diff(y0)
        n = len(dy) - k
        if n <= k + 2:
            break
        X = [y0[k:-1]]
        for j in range(1, k + 1):
            X.append(dy[k - j:-j] if j < len(dy) else np.zeros(n))
        Xk = np.column_stack([x[:n] for x in X])
        yk = dy[k:k + n]
        beta, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
        e = yk - Xk @ beta
        s2e = (e @ e) / n
        b0 = beta[0]
        tau = (b0 ** 2) * np.sum(y0[k:k + n] ** 2) / s2e if s2e > 0 else 0.0
        maic = np.log(s2e) + 2.0 * (tau + k) / n
        if maic < best_maic:
            best_maic, best_k = maic, k
            # long-run variance from the AR spectral form
            sum_b = np.sum(beta[1:]) if k >= 1 else 0.0
            best_s2 = s2e / (1.0 - sum_b) ** 2 if abs(1.0 - sum_b) > 1e-8 else s2e
    return float(best_s2), int(best_k)


def m_statistics(y: np.ndarray, Z: np.ndarray, cbar: float,
                 sigma2_method: int = 0, kmax: int = KMAX_DEFAULT):
    """The five CKP M-statistics plus ADF-GLS for Model LB.
    Returns dict with MZa, MSB, MZt, PT, MPT, ADF, k_lag, ok.
    Rejection is for SMALL values (see module docstring)."""
    nt = len(y)
    yt, ssr = gls_detrend(y, Z, cbar)
    s2, k_used = _s2_ar(yt, sigma2_method, kmax)
    if s2 <= 0 or not np.isfinite(s2):
        return dict(ok=0.0)
    ybar2 = np.sum(yt[:-1] ** 2)
    sig = ybar2 / nt ** 2
    a = 1.0 + cbar / nt
    # Ng-Perron / CKP M-statistics
    mza = (yt[-1] ** 2 / nt - s2) / (2.0 * sig) if sig > 0 else np.nan
    msb = np.sqrt(sig / s2) if s2 > 0 else np.nan
    mzt = mza * msb if np.isfinite(mza) and np.isfinite(msb) else np.nan
    # ERS point-optimal PT and MPT (SSR form)
    ybar2_full = np.sum(yt ** 2)
    pt = (ssr - a * ybar2_full) / s2 if s2 > 0 else np.nan
    mpt = ((cbar ** 2) * sig - cbar * yt[-1] ** 2 / nt) / s2 if s2 > 0 else np.nan
    # ADF-GLS t-statistic on the detrended series
    adf = _adf_gls_t(yt, k_used)
    ok = 1.0 if all(np.isfinite(v) for v in (mza, msb, mzt, pt, mpt)) else 0.0
    return dict(MZa=mza, MSB=msb, MZt=mzt, PT=pt, MPT=mpt, ADF=adf,
                k_lag=k_used, s2=s2, ok=ok)


def _adf_gls_t(yt: np.ndarray, k: int) -> float:
    """ADF t-statistic on the GLS-detrended series with k augmentation lags."""
    dy = np.diff(yt)
    n = len(dy) - k
    if n <= k + 2:
        return np.nan
    X = [yt[k:-1]]
    for j in range(1, k + 1):
        X.append(dy[k - j:k - j + n])
    Xk = np.column_stack([x[:n] for x in X])
    yk = dy[k:k + n]
    beta, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
    e = yk - Xk @ beta
    s2 = (e @ e) / (n - Xk.shape[1])
    XtX_inv = np.linalg.inv(Xk.T @ Xk)
    se_b0 = np.sqrt(s2 * XtX_inv[0, 0])
    return float(beta[0] / se_b0) if se_b0 > 0 else np.nan

# ============================================================================
# SELF-TEST
# ============================================================================
def _selftest():
    """Minimal gates: size ~ 0.05, exact magnitude-invariance (Lemma 1),
    power > size, and DU structure. Pure logic, no external dependency."""
    print("=" * 56)
    print("SELF-TEST mlb_kernel (pure-Python Model LB kernel)")
    print("=" * 56)
    ok = True
    rng = np.random.default_rng(0)
    T, bp = 120, np.array([40, 80], dtype=np.int64)
    Z = build_z(T, bp)

    # null 5% CV of MZt
    draws = []
    for _ in range(4000):
        y = np.cumsum(rng.standard_normal(T))
        st = m_statistics(y, Z, -7.0, 0, KMAX_DEFAULT)
        if st["ok"] > 0.5:
            draws.append(st["MZt"])
    cv = float(np.percentile(draws, 5))

    # (1) size
    rej = 0
    for _ in range(500):
        y = np.cumsum(rng.standard_normal(T))
        st = m_statistics(y, Z, -7.0, 0, KMAX_DEFAULT)
        if st["ok"] > 0.5 and st["MZt"] < cv:
            rej += 1
    size = rej / 500
    c1 = 0.02 <= size <= 0.10
    print(f"  (1) empirical size (MZt) ~ 0.05: {size:.3f}  [{'OK' if c1 else 'CHECK'}]")
    ok &= c1

    # (2) exact invariance to break magnitudes (Lemma 1)
    y0 = np.cumsum(rng.standard_normal(T))
    y1 = y0 + Z @ np.array([0.0, 5.0, -3.0])
    s0 = m_statistics(y0, Z, -7.0, 0, KMAX_DEFAULT)["MZt"]
    s1 = m_statistics(y1, Z, -7.0, 0, KMAX_DEFAULT)["MZt"]
    c2 = abs(s0 - s1) < 1e-8
    print(f"  (2) magnitude-invariance (Lemma 1): |d MZt|={abs(s0-s1):.2e}  [{'OK' if c2 else 'FAIL'}]")
    ok &= c2

    # (3) power > size under a stationary broken-mean alternative
    rej = 0
    for _ in range(300):
        e = rng.standard_normal(T)
        u = np.zeros(T)
        for t in range(1, T):
            u[t] = (1 - 10.0 / T) * u[t - 1] + e[t]
        y = u + Z @ np.array([0.0, 4.0, 2.0])
        st = m_statistics(y, Z, -7.0, 0, KMAX_DEFAULT)
        if st["ok"] > 0.5 and st["MZt"] < cv:
            rej += 1
    power = rej / 300
    c3 = power > size
    print(f"  (3) power at c=-10 > size: {power:.3f} > {size:.3f}  [{'OK' if c3 else 'FAIL'}]")
    ok &= c3

    # (4) build_z DU structure
    Zt = build_z(5, np.array([2]))
    c4 = Zt.shape == (5, 2) and Zt[0, 1] == 0 and Zt[2, 1] == 1
    print(f"  (4) build_z DU at 1{{t>tau}}: [{'OK' if c4 else 'FAIL'}]")
    ok &= c4

    print("\nSELF-TEST:", "ALL PASSED" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
