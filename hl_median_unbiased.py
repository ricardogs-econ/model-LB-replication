#!/usr/bin/env python3
# =============================================================================
# hl_median_unbiased.py
# -----------------------------------------------------------------------------
# Median-unbiased half-life estimation for the PPP application of the Model LB
# apparatus (Article 1).  Produces, per currency, the persistence of the real
# exchange rate under two deterministic specifications --- level breaks at
# exogenous currency-regime dates (LB) versus a constant mean (MP, the
# Murray-Papell reading) --- with:
#
#   (1) Andrews & Chen (1994) approximately median-unbiased estimator of the
#       sum of autoregressive coefficients alpha (the persistence), correcting
#       the O(1/T) Nickell-Kendall median bias that plagues OLS at T ~ 52;
#   (2) Hansen (1999) grid-t bootstrap confidence interval for alpha, valid
#       uniformly in alpha INCLUDING the unit-root neighbourhood --- the CI can
#       have an upper limit at (or above) 1, mapping to an INFINITE upper
#       half-life, which is precisely the Murray-Papell phenomenon;
#   (3) the impulse-response half-life (regime-wise: computed on the residual
#       after the level shifts are removed), reported alongside the scalar
#       ln(0.5)/ln(alpha) half-life;
#   (4) the key diagnostic for the paper's Level-2 claim: does modelling the
#       exogenous breaks COLLAPSE the half-life CI from [., infinity) to
#       [., finite)?
#
# EQUIVALENCE.  The deterministic detrending (constant + level dummies at the
# exogenous break positions) is built exactly as in boot_ppp_cbar_production.py
# / the the numba kernel (searchsorted on the break years, DU_j = 1{t >= tau_j}), so
# the residual u_t here is the same object whose AR(1) correlation the boot
# reports as rho_LB.  The AR(p)/median-unbiased/grid-bootstrap machinery is new
# and self-contained (numba-accelerated), and does not touch the GLS/M-statistic
# kernels, which are irrelevant to half-life estimation.
#
# Registro epistemico.  Half-life is a function of a POINT estimate of
# persistence; it is a distinct object from the unit-root TEST verdict.  A
# finite point half-life coexists with non-rejection of alpha = 1 (finite
# median-unbiased alpha, but a CI for alpha that includes 1).  This module
# reports the CI honestly; it does not convert non-rejection into a PPP verdict.
#
# Usage:
#   python hl_median_unbiased.py --panel ppp_panel.csv --dates exog_dates.csv \
#          --diag ppp_ar_diagnostic.csv --out hl_results.csv
#   python hl_median_unbiased.py --selftest        # validation gates only
# =============================================================================
from __future__ import annotations
import argparse, csv, sys, time
from pathlib import Path
import numpy as np

# ---- optional numba (graceful fallback to pure-python njit no-op) -----------
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:                                             # pragma: no cover
    _HAVE_NUMBA = False
    def njit(*a, **k):
        def wrap(f): return f
        return wrap if (a and callable(a[0])) is False else a[0]

RNG_SEED = 20260704


# =============================================================================
# 1. CORE ESTIMATION KERNELS (numba)
# =============================================================================
@njit(cache=True)
def _ols(X, y):
    """Plain OLS via normal equations; returns (beta, resid, s2_diag_XtXinv).
    Small, well-conditioned designs (T ~ 52, k <= 5) -> normal equations fine."""
    XtX = X.T @ X
    Xty = X.T @ y
    XtXinv = np.linalg.inv(XtX)
    beta = XtXinv @ Xty
    resid = y - X @ beta
    return beta, resid, np.diag(XtXinv)


@njit(cache=True)
def adf_fit(u, p):
    """Fit the ADF-form AR(p) on a (mean-zero, detrended) residual u:
        du_t = (alpha-1) u_{t-1} + sum_{j=1}^{p-1} c_j du_{t-j} + e_t.
    Returns (alpha, se_alpha, c[0:p-1], sigma_e).  No deterministic terms:
    u is already detrended, so its AR is estimated without a constant.
    Effective sample t = p .. n-1 (0-indexed): n-p usable observations."""
    n = u.shape[0]
    du = np.empty(n - 1)
    for t in range(1, n):
        du[t - 1] = u[t] - u[t - 1]
    # regression rows for t = p .. n-1  (du index t-1 = p-1 .. n-2)
    T_eff = n - p
    k = p                                   # [u_{t-1}, du_{t-1}, ..., du_{t-p+1}]
    X = np.empty((T_eff, k))
    y = np.empty(T_eff)
    row = 0
    for t in range(p, n):
        y[row] = du[t - 1]                  # du_t
        X[row, 0] = u[t - 1]                # u_{t-1}  -> coefficient (alpha-1)
        for j in range(1, p):
            X[row, j] = du[t - 1 - j]       # du_{t-j}
        row += 1
    beta, resid, dg = _ols(X, y)
    dof = T_eff - k
    s2 = (resid @ resid) / dof if dof > 0 else np.nan
    se_rho = np.sqrt(s2 * dg[0]) if s2 > 0 else np.nan
    alpha = 1.0 + beta[0]
    c = beta[1:].copy()
    return alpha, se_rho, c, np.sqrt(s2)


@njit(cache=True)
def _ar_coeffs_from_adf(alpha, c):
    """Recover level AR coefficients a_1..a_p from ADF parametrization
    (alpha = sum a_i ; c_j = -sum_{i=j+1}^p a_i).  Inversion:
        a_p = -c_{p-1};  a_i = c_{i-1} - c_i  (i=2..p-1);  a_1 = alpha - sum_{i>=2} a_i.
    p = len(c)+1."""
    p = c.shape[0] + 1
    a = np.zeros(p)
    if p == 1:
        a[0] = alpha
        return a
    a[p - 1] = -c[p - 2]
    for i in range(p - 1, 1, -1):           # i = p-1 .. 2  (1-indexed)
        a[i - 1] = c[i - 2] - c[i - 1]
    ssum = 0.0
    for i in range(1, p):
        ssum += a[i]
    a[0] = alpha - ssum
    return a


@njit(cache=True)
def irf_half_life(a, hmax=2000):
    """First-crossing IRF half-life from level AR coefficients a_1..a_p.
    psi_0 = 1 ; psi_h = sum_{i=1}^{min(h,p)} a_i psi_{h-i}.  Returns the
    smallest h with psi_h <= 0.5, or a sentinel (hmax) if never crossed."""
    p = a.shape[0]
    psi = np.zeros(hmax + 1)
    psi[0] = 1.0
    for h in range(1, hmax + 1):
        s = 0.0
        for i in range(1, min(h, p) + 1):
            s += a[i - 1] * psi[h - i]
        psi[h] = s
        if psi[h] <= 0.5:
            return float(h)
    return float(hmax)


@njit(cache=True)
def _simulate_ar_adf(alpha, c, sigma, e, burn):
    """Simulate an AR(p) in ADF parametrization given alpha, c, and a vector of
    innovations e (already length T+burn).  Returns the length-T tail.
    Uses the level-form recursion for stability."""
    a = _ar_coeffs_from_adf(alpha, c)
    p = a.shape[0]
    n = e.shape[0]
    u = np.zeros(n)
    for t in range(n):
        s = 0.0
        for i in range(1, p + 1):
            if t - i >= 0:
                s += a[i - 1] * u[t - i]
        u[t] = s + sigma * e[t]
    return u[burn:]


# =============================================================================
# 2. ANDREWS-CHEN (1994) MEDIAN-UNBIASED ESTIMATOR
# =============================================================================
@njit(cache=True)
def _median_of_alphahat(alpha0, c, sigma, T, p, nsim, seed):
    """Monte-Carlo median of the OLS alpha-hat when the true persistence is
    alpha0 (c, sigma fixed at their sample values -> 'approximately' MU).
    Gaussian innovations (Andrews-Chen convention)."""
    np.random.seed(seed)
    burn = 50
    out = np.empty(nsim)
    for s in range(nsim):
        e = np.random.standard_normal(T + burn)
        u = _simulate_ar_adf(alpha0, c, sigma, e, burn)
        ah, _, _, _ = adf_fit(u, p)
        out[s] = ah
    return np.median(out)


def andrews_chen_mu(alpha_hat, c, sigma, T, p, nsim=1500, ngrid=25, seed=RNG_SEED):
    """Approximately median-unbiased estimator: invert the median function
    m(alpha) = median(alpha_hat | alpha).  m is monotone increasing; we build
    it on a grid and interpolate m^{-1}(alpha_hat).  The grid extends slightly
    above 1 so that near-unit-root alpha_hat can be inverted.  If alpha_hat
    exceeds m at the top of the grid, alpha_MU is capped at the grid top
    (persistence at/above unity -> infinite half-life)."""
    grid = np.linspace(0.50, 1.02, ngrid)
    m = np.empty(ngrid)
    for i in range(ngrid):
        m[i] = _median_of_alphahat(grid[i], c, sigma, T, p, nsim, seed + i)
    # enforce monotonicity (MC noise can create tiny non-monotone kinks)
    for i in range(1, ngrid):
        if m[i] < m[i - 1]:
            m[i] = m[i - 1] + 1e-9
    if alpha_hat <= m[0]:
        return float(grid[0]), grid, m
    if alpha_hat >= m[-1]:
        return float(grid[-1]), grid, m
    alpha_mu = float(np.interp(alpha_hat, m, grid))
    return alpha_mu, grid, m


# =============================================================================
# 3. HANSEN (1999) GRID-t BOOTSTRAP CONFIDENCE INTERVAL
# =============================================================================
@njit(cache=True)
def _restricted_residuals(u, alpha0, p):
    """Residuals of the ADF regression with alpha CONSTRAINED to alpha0:
    regress (du_t - (alpha0-1) u_{t-1}) on [du_{t-1..t-p+1}] (no intercept;
    for p=1 there are no regressors -> residual is the constrained du itself).
    Returns (c_restricted[0:p-1], resid_centered, sigma)."""
    n = u.shape[0]
    du = np.empty(n - 1)
    for t in range(1, n):
        du[t - 1] = u[t] - u[t - 1]
    T_eff = n - p
    ystar = np.empty(T_eff)
    row = 0
    for t in range(p, n):
        ystar[row] = du[t - 1] - (alpha0 - 1.0) * u[t - 1]
        row += 1
    if p == 1:
        c = np.zeros(0)
        resid = ystar.copy()
    else:
        Z = np.empty((T_eff, p - 1))
        row = 0
        for t in range(p, n):
            for j in range(1, p):
                Z[row, j - 1] = du[t - 1 - j]
            row += 1
        c, resid, _ = _ols(Z, ystar)
    m = 0.0
    for i in range(T_eff):
        m += resid[i]
    m /= T_eff
    for i in range(T_eff):
        resid[i] -= m                       # center
    sig = np.sqrt((resid @ resid) / T_eff)
    return c, resid, sig


@njit(cache=True)
def _grid_t_boot(u, alpha0, p, c_r, resid_r, sigma_r, B, seed):
    """Bootstrap distribution of t(alpha0) = (alpha_hat* - alpha0)/se* under the
    restricted model (alpha imposed = alpha0).  Residual bootstrap: resample the
    centered restricted residuals, rebuild u* via the level-form recursion with
    alpha0 and c_r, refit unrestricted, collect t*."""
    np.random.seed(seed)
    n = u.shape[0]
    burn = 50
    nr = resid_r.shape[0]
    tstar = np.empty(B)
    for b in range(B):
        e = np.empty(n + burn)
        for i in range(n + burn):
            e[i] = resid_r[np.random.randint(nr)] / sigma_r  # unit-scale draws
        ustar = _simulate_ar_adf(alpha0, c_r, sigma_r, e, burn)
        ah, se, _, _ = adf_fit(ustar, p)
        tstar[b] = (ah - alpha0) / se if se > 0 else 0.0
    return tstar


@njit(cache=True)
def _grid_t_wild(u, alpha0, p, c_r, resid_r, sigma_r, B, seed, scheme):
    """WILD grid-t: as _grid_t_boot but the innovation at time t is the RESTRICTED
    residual at position t times a mean-zero, unit-variance multiplier eta_t,
    e*_t = e_hat_{r,t} * eta_t, preserving the conditional heteroskedasticity of
    the residual (Var(e*_t | e_hat) = e_hat_{r,t}^2).  No burn-in resampling: the
    multiplicative scheme keeps the temporal alignment of the variance, which is
    the point of the wild bootstrap under heteroskedasticity (Cavaliere-Taylor
    2008 for the unit-root neighbourhood).  scheme: 0 = Rademacher (eta in {-1,+1}
    w.p. 1/2; matches moments to 2nd order, best for near-symmetric residuals,
    Davidson-Flachaire 2008); 1 = Mammen two-point (matches the 3rd moment, for
    skewed residuals).  The series is initialised at the restricted residuals'
    own scale and run for n steps; the first p observations reuse the data's
    initial values so the level recursion is well defined."""
    np.random.seed(seed)
    n = u.shape[0]
    nr = resid_r.shape[0]                    # = n - p
    a = _ar_coeffs_from_adf(alpha0, c_r)
    pp = a.shape[0]
    mm_a = -(np.sqrt(5.0) - 1.0) / 2.0       # Mammen support points
    mm_b = (np.sqrt(5.0) + 1.0) / 2.0
    mm_pa = (np.sqrt(5.0) + 1.0) / (2.0 * np.sqrt(5.0))   # P(eta = mm_b)
    tstar = np.empty(B)
    for b in range(B):
        ustar = np.zeros(n)
        for t in range(n):
            # innovation: restricted residual at aligned position * multiplier
            if t < p:
                e_t = resid_r[t % nr] if nr > 0 else 0.0
            else:
                e_t = resid_r[t - p]         # aligned restricted residual
            if scheme == 0:                  # Rademacher
                eta = 1.0 if np.random.random() < 0.5 else -1.0
            else:                            # Mammen
                eta = mm_b if np.random.random() < mm_pa else mm_a
            innov = e_t * eta
            s = 0.0
            for i in range(1, pp + 1):
                if t - i >= 0:
                    s += a[i - 1] * ustar[t - i]
            ustar[t] = s + innov
        ah, se, _, _ = adf_fit(ustar, p)
        tstar[b] = (ah - alpha0) / se if se > 0 else 0.0
    return tstar



def hansen_grid_ci(u, p, level=0.95, ngrid=40, B=999, seed=RNG_SEED,
                   refine=True, boot="recursive"):
    """Grid-t bootstrap CI for alpha (Hansen 1999).  For each alpha0 on the
    grid, compute the observed t(alpha0) and the bootstrap quantiles of
    t*(alpha0) under alpha0 imposed; alpha0 is in the CI iff the observed t
    lies within [q_lo, q_hi].  Returns (alpha_lo, alpha_hi, includes_one).

    The upper edge often sits in the unit-root neighbourhood, where a coarse
    grid makes the includes_one verdict flip with Monte-Carlo noise.  We (i) use
    a grid refined to step 0.005 on [0.90, 1.05], and (ii) base includes_one on
    the CONTINUOUS upper edge relative to 1 (>= 1 - tol), not on whether a grid
    node happens to land above 1 -- so the verdict is stable in B."""
    alpha_hat, se_hat, c_hat, _ = adf_fit(u, p)
    if refine:
        lo_part = np.linspace(0.40, 0.90, 26, endpoint=False)
        hi_part = np.arange(0.90, 1.051, 0.005)
        grid = np.concatenate([lo_part, hi_part])
    else:
        grid = np.linspace(0.40, 1.05, ngrid)
    ng = grid.shape[0]
    a = (1.0 - level) / 2.0
    wild_scheme = 1 if boot == "wild-mammen" else 0
    in_ci = np.zeros(ng, dtype=bool)
    for i in range(ng):
        a0 = grid[i]
        c_r, resid_r, sig_r = _restricted_residuals(u, a0, p)
        if boot == "recursive":
            tstar = _grid_t_boot(u, a0, p, c_r, resid_r, sig_r, B, seed + i)
        else:
            tstar = _grid_t_wild(u, a0, p, c_r, resid_r, sig_r, B, seed + i,
                                 wild_scheme)
        q_lo = np.quantile(tstar, a)
        q_hi = np.quantile(tstar, 1.0 - a)
        t_obs = (alpha_hat - a0) / se_hat if se_hat > 0 else 0.0
        in_ci[i] = (q_lo <= t_obs <= q_hi)
    if not in_ci.any():
        return np.nan, np.nan, False
    idx = np.where(in_ci)[0]
    alpha_lo = float(grid[idx.min()])
    alpha_hi = float(grid[idx.max()])
    # includes_one with a small tolerance tied to the grid step, so a verdict
    # that hinges on the last node (the JPY/CAD borderline) is read as "touches
    # unity" rather than flickering: collapse requires the upper edge to clear
    # unity by at least one grid step.
    tol = 0.005
    includes_one = bool(alpha_hi >= 1.0 - tol)
    return alpha_lo, alpha_hi, includes_one


# =============================================================================
# 4. HALF-LIFE MAPPING
# =============================================================================
def half_life(alpha):
    """Scalar (AR(1)-approx) half-life ln(0.5)/ln(alpha); inf if alpha>=1."""
    if 0.0 < alpha < 1.0:
        return float(np.log(0.5) / np.log(alpha))
    return np.inf


def hl_ci(alpha_lo, alpha_hi):
    """Map an alpha CI to a half-life CI.  HL is increasing in alpha, so
    HL_lo = HL(alpha_lo), HL_hi = HL(alpha_hi); HL_hi = inf if alpha_hi>=1."""
    return half_life(alpha_lo), half_life(alpha_hi)


# =============================================================================
# 5. DESIGN MATRIX (equivalence with the boot: constant + level dummies)
# =============================================================================
def build_Z(years, break_years, with_breaks=True):
    """Z = [1, DU_1, ..., DU_m] with DU_j = 1{year >= tau_j}, exactly as the
    boot builds it (searchsorted / indicator on the break years).  If
    with_breaks=False, Z = [1] (the Murray-Papell constant-mean reading)."""
    T = len(years)
    cols = [np.ones(T)]
    if with_breaks:
        for b in break_years:
            if years[0] < b <= years[-1]:
                cols.append((years >= b).astype(float))
    return np.column_stack(cols)


def detrend(q, Z):
    """Remove the deterministic component: return the OLS residual q - Z @ (Z\\Z q).
    Z holds the constant and, if with_breaks, the level dummies at the break years."""
    beta, *_ = np.linalg.lstsq(Z, q, rcond=None)
    return q - Z @ beta


# =============================================================================
# 6. PER-CURRENCY ANALYSIS
# =============================================================================
def analyze(q, years, break_years, p, nsim=1500, B=999, seed=RNG_SEED,
            boot="recursive", also_wild=False):
    """Full analysis for one currency under LB (with breaks) and MP (constant).
    Returns a dict of persistence, median-unbiased alpha, half-lives, IRF
    half-life, and the Hansen CI (with the includes-one flag).  If also_wild,
    additionally computes the wild-bootstrap CI (Rademacher) alongside the
    baseline `boot` interval, for the heteroskedasticity-robustness table."""
    out = {}
    for tag, wb in (("LB", True), ("MP", False)):
        Z = build_Z(years, break_years, with_breaks=wb)
        u = detrend(q, Z)
        T = len(u)
        alpha_hat, se, c, sigma = adf_fit(u, p)
        alpha_mu, _, _ = andrews_chen_mu(alpha_hat, c, sigma, T, p, nsim=nsim,
                                         seed=seed)
        a_lvl = _ar_coeffs_from_adf(alpha_mu, c)
        hl_scalar = half_life(alpha_mu)
        hl_irf = irf_half_life(a_lvl)
        a_lo, a_hi, inc1 = hansen_grid_ci(u, p, B=B, seed=seed, boot=boot)
        hl_lo, hl_hi = hl_ci(a_lo, a_hi)
        rec = dict(alpha_ols=alpha_hat, alpha_mu=alpha_mu,
                   hl_scalar=hl_scalar, hl_irf=hl_irf,
                   alpha_ci=(a_lo, a_hi), hl_ci=(hl_lo, hl_hi),
                   includes_one=inc1, p=p, T=T)
        if also_wild:
            wa_lo, wa_hi, winc1 = hansen_grid_ci(u, p, B=B, seed=seed,
                                                 boot="wild")
            whl_lo, whl_hi = hl_ci(wa_lo, wa_hi)
            rec["wild_alpha_ci"] = (wa_lo, wa_hi)
            rec["wild_hl_ci"] = (whl_lo, whl_hi)
            rec["wild_includes_one"] = winc1
        out[tag] = rec
    return out


# =============================================================================
# 7. VALIDATION GATES
# =============================================================================
def selftest():
    print("=" * 70)
    print("SELF-TEST — validation gates")
    print("=" * 70)
    rng = np.random.default_rng(0)

    # G1: median function monotone increasing
    c0 = np.zeros(0)
    T = 52
    grid = np.linspace(0.5, 1.0, 11)
    ms = [_median_of_alphahat(a, c0, 1.0, T, 1, 600, 1) for a in grid]
    mono = all(ms[i] <= ms[i + 1] + 1e-6 for i in range(len(ms) - 1))
    print(f"[G1] median function monotone in alpha: "
          f"{'PASS' if mono else 'FAIL'}  m(0.5)={ms[0]:.3f} m(1.0)={ms[-1]:.3f}")

    # G2: median-unbiased recovery on simulated AR(1) with known alpha
    true_alpha = 0.85
    biases_ols, biases_mu = [], []
    for r in range(200):
        e = rng.standard_normal(T + 50)
        u = _simulate_ar_adf(true_alpha, c0, 1.0, e, 50)
        ah, _, ch, sg = adf_fit(u, 1)
        amu, _, _ = andrews_chen_mu(ah, ch, sg, T, 1, nsim=400, seed=r)
        biases_ols.append(ah - true_alpha)
        biases_mu.append(amu - true_alpha)
    med_ols = np.median(biases_ols); med_mu = np.median(biases_mu)
    print(f"[G2] AR(1) alpha={true_alpha}: median bias OLS={med_ols:+.4f} "
          f"-> MU={med_mu:+.4f}  "
          f"[{'PASS' if abs(med_mu) < abs(med_ols) else 'FAIL'}] "
          f"(MU closer to 0)")

    # G3: IRF half-life matches scalar for AR(1)
    a1 = np.array([0.9])
    hl_i = irf_half_life(a1); hl_s = np.log(0.5) / np.log(0.9)
    print(f"[G3] AR(1) IRF-HL={hl_i:.2f} vs scalar={hl_s:.2f}  "
          f"[{'PASS' if abs(hl_i - round(hl_s)) <= 1 else 'FAIL'}]")

    # G4: ADF-form alpha ~ AR(1) correlation for p=1 (consistency w/ boot rho_LB)
    e = rng.standard_normal(T + 50)
    u = _simulate_ar_adf(0.8, c0, 1.0, e, 50)
    ah, _, _, _ = adf_fit(u, 1)
    rho_corr = np.corrcoef(u[:-1], u[1:])[0, 1]
    print(f"[G4] p=1: ADF alpha={ah:.3f} vs corr rho={rho_corr:.3f} "
          f"(diff {abs(ah-rho_corr):.3f})  "
          f"[{'PASS' if abs(ah - rho_corr) < 0.10 else 'FAIL'}]")

    # G5: Hansen CI coverage sanity on a unit-root series (should include 1)
    e = rng.standard_normal(T + 50)
    u_rw = np.cumsum(e)[50:]
    a_lo, a_hi, inc1 = hansen_grid_ci(u_rw, 1, B=299, seed=7)
    print(f"[G5] random walk: Hansen CI=[{a_lo:.3f},{a_hi:.3f}] includes 1: "
          f"{inc1}  [{'PASS' if inc1 else 'FAIL'}]")

    # G6: AR coeff recovery (ADF <-> level parametrization round-trip)
    c_test = np.array([0.3, -0.1])          # p=3
    a_lvl = _ar_coeffs_from_adf(0.9, c_test)
    alpha_back = a_lvl.sum()
    print(f"[G6] ADF<->level round-trip: alpha={alpha_back:.6f} (target 0.9) "
          f"[{'PASS' if abs(alpha_back - 0.9) < 1e-9 else 'FAIL'}]")

    # G7: under HOMOSKEDASTICITY, wild CI ~ recursive CI (wild must not distort
    #     when there is no heteroskedasticity to correct)
    e = rng.standard_normal(T + 50)
    u_h = _simulate_ar_adf(0.85, c0, 1.0, e, 50)
    rlo, rhi, _ = hansen_grid_ci(u_h, 1, B=699, seed=11, boot="recursive")
    wlo, whi, _ = hansen_grid_ci(u_h, 1, B=699, seed=11, boot="wild")
    close = abs(rlo - wlo) < 0.06 and abs(rhi - whi) < 0.06
    print(f"[G7] homosk.: recursive=[{rlo:.3f},{rhi:.3f}] "
          f"wild=[{wlo:.3f},{whi:.3f}]  "
          f"[{'PASS' if close else 'FAIL'}] (should agree)")

    # G8: under HETEROSKEDASTICITY (variance doubles mid-sample), the recursive
    #     bootstrap MIS-sizes while the wild tracks it.  We check coverage of the
    #     true alpha over replications: wild coverage should be closer to 0.95.
    true_a = 0.8
    cov_rec = cov_wild = 0; NR = 120
    for r in range(NR):
        ei = rng.standard_normal(T + 50)
        ei[(T + 50) // 2:] *= 2.2                    # variance break
        uh = _simulate_ar_adf(true_a, c0, 1.0, ei, 50)
        rl, rh, _ = hansen_grid_ci(uh, 1, B=199, seed=r, boot="recursive")
        wl, wh, _ = hansen_grid_ci(uh, 1, B=199, seed=r, boot="wild")
        cov_rec += (rl <= true_a <= rh)
        cov_wild += (wl <= true_a <= wh)
    cr, cw = cov_rec / NR, cov_wild / NR
    print(f"[G8] heterosk.: coverage recursive={cr:.2f} wild={cw:.2f} "
          f"(nominal 0.95)  "
          f"[{'PASS' if abs(cw - 0.95) <= abs(cr - 0.95) + 0.02 else 'FAIL'}] "
          f"(wild at least as good)")
    print("=" * 70)


# =============================================================================
# 8. MAIN
# =============================================================================
def load_panel(path):
    """Load the real-exchange-rate panel: CSV with columns year and one column per
    currency (log real exchange rate). Returns (years, {currency: series})."""
    panel = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            panel.setdefault(row["currency"], []).append(
                (int(row["year"]), float(row["q"])))
    return panel


def load_dates(path):
    """Load the exogenous break dates: CSV mapping each currency to its regime-change
    year(s). Returns {currency: [break_year, ...]}."""
    ed = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ed.setdefault(row["currency"], []).append(int(row["break_year"]))
    return ed


def load_p(path):
    """Load the per-currency ADF lag order p (CSV: currency, p). Returns {currency: p}."""
    pm = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            pm[row["currency"]] = max(1, int(float(row["k_bic_cq"])))
    return pm


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", default="ppp_panel.csv")
    ap.add_argument("--start-year", type=int, default=1970,
                    help="first year of the window; observations before it are "
                         "dropped (set 1973 for the post-Bretton-Woods float).")
    ap.add_argument("--dates", default="exog_dates.csv")
    ap.add_argument("--diag", default="ppp_ar_diagnostic.csv")
    ap.add_argument("--out", default="hl_results.csv")
    ap.add_argument("--nsim", type=int, default=1500,
                    help="MC draws for the median function")
    ap.add_argument("--B", type=int, default=999,
                    help="grid-bootstrap replications")
    ap.add_argument("--boot", default="recursive",
                    choices=["recursive", "wild", "wild-mammen"],
                    help="grid-t bootstrap scheme: recursive (i.i.d., "
                         "homoskedastic baseline) or wild (Rademacher / Mammen, "
                         "heteroskedasticity-robust)")
    ap.add_argument("--also-wild", action="store_true",
                    help="report the wild (Rademacher) CI alongside the "
                         "baseline for the robustness table")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if _HAVE_NUMBA:
        # warm up the JIT so timing/measurement is clean
        _ = adf_fit(np.random.standard_normal(30), 1)
    if args.selftest:
        selftest(); return

    panel = load_panel(args.panel)
    dates = load_dates(args.dates)
    pmap = load_p(args.diag)

    rows = []
    print(f"{'cur':4}{'p':>2}{'m':>3} | "
          f"{'a_MP':>7}{'HL_MP':>7}{'CI_MP':>16} | "
          f"{'a_LB':>7}{'HL_LB':>7}{'CI_LB':>16} | collapse?")
    print("-" * 92)
    for cur in sorted(panel):
        obs = sorted(panel[cur])
        obs = [o for o in obs if o[0] >= args.start_year]   # start-year window
        years = np.array([o[0] for o in obs])
        q = np.array([o[1] for o in obs])
        bry = dates.get(cur, [])
        p = pmap.get(cur, 1)
        t0 = time.time()
        res = analyze(q, years, bry, p, nsim=args.nsim, B=args.B,
                      boot=args.boot, also_wild=args.also_wild)
        mp, lb = res["MP"], res["LB"]
        def fmt_ci(d):
            """Format a half-life CI for printing (handles the infinite upper
            bound when the persistence CI reaches the unit root)."""
            lo, hi = d["hl_ci"]
            hi_s = "inf" if not np.isfinite(hi) else f"{hi:.1f}"
            lo_s = "inf" if not np.isfinite(lo) else f"{lo:.1f}"
            return f"[{lo_s},{hi_s}]"
        # the Level-2 diagnostic: MP CI infinite, LB CI finite -> collapse
        collapse = (mp["includes_one"] and not lb["includes_one"])
        m = len([b for b in bry if years[0] < b <= years[-1]])
        print(f"{cur:4}{p:>2}{m:>3} | "
              f"{mp['alpha_mu']:>7.3f}{mp['hl_scalar']:>7.1f}{fmt_ci(mp):>16} | "
              f"{lb['alpha_mu']:>7.3f}{lb['hl_scalar']:>7.1f}{fmt_ci(lb):>16} | "
              f"{'YES' if collapse else 'no'}  [{time.time()-t0:.0f}s]")
        rows.append(dict(currency=cur, p=p, m=m,
            alpha_ols_MP=round(mp["alpha_ols"], 4), alpha_mu_MP=round(mp["alpha_mu"], 4),
            HL_scalar_MP=round(mp["hl_scalar"], 2) if np.isfinite(mp["hl_scalar"]) else "inf",
            HL_irf_MP=round(mp["hl_irf"], 2),
            alpha_ci_lo_MP=round(mp["alpha_ci"][0], 4), alpha_ci_hi_MP=round(mp["alpha_ci"][1], 4),
            HL_ci_hi_MP=("inf" if not np.isfinite(mp["hl_ci"][1]) else round(mp["hl_ci"][1], 2)),
            MP_includes_one=int(mp["includes_one"]),
            alpha_ols_LB=round(lb["alpha_ols"], 4), alpha_mu_LB=round(lb["alpha_mu"], 4),
            HL_scalar_LB=round(lb["hl_scalar"], 2) if np.isfinite(lb["hl_scalar"]) else "inf",
            HL_irf_LB=round(lb["hl_irf"], 2),
            alpha_ci_lo_LB=round(lb["alpha_ci"][0], 4), alpha_ci_hi_LB=round(lb["alpha_ci"][1], 4),
            HL_ci_hi_LB=("inf" if not np.isfinite(lb["hl_ci"][1]) else round(lb["hl_ci"][1], 2)),
            LB_includes_one=int(lb["includes_one"]),
            collapse=int(collapse)))

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    n_collapse = sum(r["collapse"] for r in rows)
    print("-" * 92)
    print(f"[done] {len(rows)} currencies | CI collapse (MP inf -> LB finite) "
          f"in {n_collapse}/{len(rows)} | written {args.out}")


if __name__ == "__main__":
    main()
