"""Pesaran's cross-sectional dependence (CD) statistic, for the PPP application.

ppp_application.py (stage data) calls the two functions below on the residuals of the
univariate Model LB fits it has already computed: the first differences of the
level-break residuals (Delta u_hat, design E) and the sieve residuals e_hat over the
universe's common sample (Section S5.3, ssub:app-mult). No panel model is estimated:
the dependence induced by the common (US-dollar) numeraire is reported, and inference
is not built on it (under exogenous level breaks each statistic's limiting law is
invariant to the common level component, so the dependence bears only on the
aggregate count).

THE STATISTIC (Pesaran 2004, 2015). For N series with pairwise residual correlations
rho_hat_ij over the T_ij overlapping periods,

    CD = sqrt( 2 / (N(N-1)) ) * sum_{i<j} sqrt(T_ij) * rho_hat_ij   ~  N(0,1)

under H0: cross-sectional independence (E[rho_ij] = 0 for all i != j); |CD| large
indicates cross-sectional dependence. It does not require T > N and is valid for the
small-N, moderate-T panel here (N = 8 or 9, T = 52). cd_statistic also returns the
two-sided p-value, the number of pairs and the mean and mean absolute pairwise
correlations.
"""

from __future__ import annotations

import numpy as np


# ============================================================================
# Pesaran CD
# ============================================================================
def pairwise_corr(U: dict) -> tuple:
    """Pairwise correlations of the residual series in dict U {name: array}.
    Series are aligned by position (the panel is balanced here). Returns
    (names, rho matrix, T_ij matrix) using only overlapping non-NaN periods."""
    names = sorted(U)
    N = len(names)
    rho = np.full((N, N), np.nan)
    Tij = np.zeros((N, N), dtype=int)
    for a in range(N):
        for b in range(a + 1, N):
            ua, ub = U[names[a]], U[names[b]]
            m = np.isfinite(ua) & np.isfinite(ub)
            t = int(m.sum())
            if t >= 3:
                xa, xb = ua[m] - ua[m].mean(), ub[m] - ub[m].mean()
                denom = np.sqrt((xa @ xa) * (xb @ xb))
                r = float(xa @ xb / denom) if denom > 0 else np.nan
                rho[a, b] = rho[b, a] = r
                Tij[a, b] = Tij[b, a] = t
    return names, rho, Tij


def cd_statistic(rho: np.ndarray, Tij: np.ndarray) -> dict:
    """Pesaran CD = sqrt(2/(N(N-1))) * sum_{i<j} sqrt(T_ij) rho_ij ~ N(0,1).
    Returns the statistic, two-sided p-value, and the average |rho|."""
    N = rho.shape[0]
    iu = np.triu_indices(N, k=1)
    r = rho[iu]
    t = Tij[iu].astype(float)
    ok = np.isfinite(r)
    r, t = r[ok], t[ok]
    npair = len(r)
    cd = np.sqrt(2.0 / (N * (N - 1))) * np.sum(np.sqrt(t) * r)
    # two-sided normal p-value
    from math import erfc, sqrt
    pval = erfc(abs(cd) / sqrt(2.0))
    return dict(CD=float(cd), pvalue=float(pval), n_pairs=int(npair),
                mean_abs_rho=float(np.mean(np.abs(r))),
                mean_rho=float(np.mean(r)))
