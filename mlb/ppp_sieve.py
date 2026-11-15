"""The level-break residual, the sieve and the AR in levels of the PPP application.

Conventions:

  * design Z = (1, DU_1, ..., DU_m), DU_jt = 1{t >= tau_j}: tau_j is the first
    predominantly post-event year and break_pos_j = tau_j - 1973 is the number of
    pre-break observations. In 0-based time t0 = year - 1973,
    DU_jt = 1{t0 >= break_pos_j}: the convention of the kernel's build_z_nb;
  * LB residual u_hat: OLS of q on Z;
  * sieve order p_hat: BIC in the ADF regression of Delta u on
    (1, u_{t-1}, Delta u_{t-1..t-p}), p = 0, ..., K_MAX_BIC = 10, on the common sample
    t = K_MAX_BIC + 1, ..., T - 1 (0-based), so that every order is fitted on the same
    observations;
  * sieve coefficients phi_hat: the coefficients of the p_hat lagged differences in that
    regression at p_hat, on its own sample t = p_hat + 1, ..., T - 1; under H0,
    Delta u_t = sum_j phi_j Delta u_{t-j} + e_t, an AR(p_hat + 1) in levels with a unit
    root;
  * the AR in levels: AR(p + 1) of y on a constant,
    y_t = c + sum_{j=1}^{p+1} a_j y_{t-j} + e_t; alpha = sum_j a_j is the sum of the
    autoregressive coefficients (= 1 + the coefficient on y_{t-1} of the ADF form with p
    lagged differences). The half-life producer and the test use the same AR.

Replacing q by -q flips u_hat and the ADF regressors (the intercept's coefficient flips
with them) and leaves the BIC, phi_hat and alpha unchanged, so no result depends on the
sign of q.
"""
import numpy as np

K_MAX_BIC = 10


def lb_design(T, break_pos):
    """Z = (1, DU_1, ..., DU_m) with DU_j = 1{t0 >= break_pos_j}, t0 = 0, ..., T - 1."""
    t0 = np.arange(T)
    if not all(1 <= int(b) <= T - 1 for b in break_pos):
        raise ValueError(f"break positions must lie in 1..{T - 1}")
    cols = [np.ones(T)] + [(t0 >= int(b)).astype(float) for b in break_pos]
    return np.column_stack(cols)


def lb_residual(q, break_pos):
    """u_hat: the OLS residual of q on the level-break design."""
    q = np.asarray(q, dtype=float)
    Z = lb_design(len(q), break_pos)
    b, *_ = np.linalg.lstsq(Z, q, rcond=None)
    return q - Z @ b


def adf_regression(u, p, first):
    """Delta u_t on (1, u_{t-1}, Delta u_{t-1}, ..., Delta u_{t-p}) for 0-based
    t = first, ..., T - 1.

    Returns (y, X); first >= p + 1 so every lag exists.
    """
    u = np.asarray(u, dtype=float)
    du = np.diff(u)                                  # du[t - 1] = u[t] - u[t - 1]
    T = len(u)
    assert first >= p + 1
    rows = range(first, T)
    y = np.array([du[t - 1] for t in rows])
    X = np.array([[1.0, u[t - 1]] + [du[t - 1 - j] for j in range(1, p + 1)]
                  for t in rows])
    return y, X


def bic_order(u, kmax=K_MAX_BIC):
    """p_hat = argmin_p BIC(p), p = 0..kmax, on the common sample
    t = kmax + 1, ..., T - 1.

    BIC(p) = ln(SSR_p / n) + k_p ln(n) / n, with k_p = p + 2 regressors and n common
    observations; the smallest p wins a tie. Returns (p_hat, {p: BIC(p)}).
    """
    bics = {}
    for p in range(kmax + 1):
        y, X = adf_regression(u, p, kmax + 1)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        ssr = float(np.sum((y - X @ b) ** 2))
        n = len(y)
        bics[p] = np.log(ssr / n) + (p + 2) * np.log(n) / n
    p_hat = min(bics, key=lambda p: (bics[p], p))
    return p_hat, bics


def sieve_phi(u, p):
    """phi_hat: the p lagged-difference coefficients of the ADF regression at order p
    (own sample)."""
    if p == 0:
        return np.zeros(0)
    y, X = adf_regression(u, p, p + 1)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[2:2 + p].copy()


def ar_levels(y, p):
    """AR(p + 1) of y on a constant: returns (const, a_1..a_{p+1}, alpha = sum a_j)."""
    y = np.asarray(y, dtype=float)
    k = p + 1
    T = len(y)
    Y = y[k:]
    X = np.column_stack([np.ones(T - k)] + [y[k - j:T - j] for j in range(1, k + 1)])
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return float(b[0]), b[1:].copy(), float(np.sum(b[1:]))


# ---------------------------------------------------------------------------------
# The application's tests and half-lives: the sieve fit with its covariance, the level
# coefficients, the impulse-response half-life, the Rogoff band and the
# general-to-specific order.
# ---------------------------------------------------------------------------------
GS_TCRIT = 1.96  # general-to-specific: 5% two-sided on the last lag
IRF_HMAX = 100_000  # the IRF is followed at most this many periods


def companion_max_root(phi):
    """max |eigenvalue| of the companion matrix of an AR(p) with coefficients phi;
    0 for p = 0. The AR is stationary iff it is < 1."""
    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    if p == 0:
        return 0.0
    C = np.zeros((p, p))
    C[0, :] = phi
    if p > 1:
        C[1:, :-1] = np.eye(p - 1)
    return float(np.max(np.abs(np.linalg.eigvals(C))))


def sieve_fit(u, p, first_year=1973):
    """The sieve regression at order p on its own sample t = p + 1, ..., T - 1.

    Returns a dict: phi (= sieve_phi(u, p), the same lstsq), V = s^2 (X'X)^-1 on the
    phi block with s^2 = SSR / (n - k), k = p + 2 (dof correction), e (the residual
    of the unrestricted ADF regression) and e_years (first_year + t), sum_phi, flag09
    (sum_phi >= 0.9, the registered flag), max_root (companion) and stationary
    (max_root < 1).
    """
    y, X = adf_regression(u, p, p + 1)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    s2 = float(e @ e) / (n - k)
    V = s2 * np.linalg.inv(X.T @ X)[2:, 2:] if p else np.zeros((0, 0))
    phi = b[2:2 + p].copy() if p else np.zeros(0)
    sum_phi = float(np.sum(phi))
    max_root = companion_max_root(phi)
    T = len(np.asarray(u))
    return dict(phi=phi, V=V, s2=s2, e=e, e_years=first_year + np.arange(p + 1, T),
                sum_phi=sum_phi, flag09=sum_phi >= 0.9, max_root=max_root,
                stationary=max_root < 1.0)


def gs_order(u, kmax=K_MAX_BIC, tcrit=GS_TCRIT):
    """General-to-specific order: from p = kmax down to 1, the first p whose last
    lagged difference has |t| > tcrit, in the ADF regression with a constant on the
    common sample t = kmax + 1, ..., T - 1 (s^2 with dof correction); 0 if none."""
    for p in range(kmax, 0, -1):
        y, X = adf_regression(u, p, kmax + 1)
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = y - X @ b
        n, k = X.shape
        s2 = float(e @ e) / (n - k)
        se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[k - 1, k - 1])
        if abs(b[k - 1] / se) > tcrit:
            return p
    return 0


def level_coeffs(rho, phi):
    """The level coefficients of (1 - rho L) Phi(L), Phi(L) = 1 - sum_j phi_j L^j:
    a_1 = rho + phi_1, a_j = phi_j - rho phi_{j-1} (j = 2..p), a_{p+1} = -rho phi_p.
    sum_j a_j = 1 - (1 - rho)(1 - sum_j phi_j)."""
    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    a = np.empty(p + 1)
    if p == 0:
        a[0] = rho
        return a
    a[0] = rho + phi[0]
    for j in range(1, p):
        a[j] = phi[j] - rho * phi[j - 1]
    a[p] = -rho * phi[p - 1]
    return a


def adf_to_levels(alpha, c):
    """The level coefficients of the ADF form
    Delta y_t = (alpha - 1) y_{t-1} + sum_{j=1}^p c_j Delta y_{t-j} + e_t:
    a_1 = alpha + c_1, a_j = c_j - c_{j-1} (j = 2..p), a_{p+1} = -c_p; sum = alpha."""
    c = np.asarray(c, dtype=float)
    p = len(c)
    a = np.empty(p + 1)
    if p == 0:
        a[0] = alpha
        return a
    a[0] = alpha + c[0]
    for j in range(1, p):
        a[j] = c[j] - c[j - 1]
    a[p] = -c[p - 1]
    return a


def irf_half_life(a, hmax=IRF_HMAX):
    """The impulse-response half-life of the AR with level coefficients a: psi_0 = 1,
    psi_h = sum_i a_i psi_{h-i}; the first passage below 1/2, h* = (h - 1)
    + (psi_{h-1} - 1/2) / (psi_{h-1} - psi_h) at the first h with psi_h <= 1/2
    (linear interpolation). inf if sum a >= 1 or if the IRF has not crossed by hmax."""
    a = [float(x) for x in a]
    if sum(a) >= 1.0:
        return float("inf")
    k = len(a)
    psi = [1.0]
    for h in range(1, hmax + 1):
        s = 0.0
        for i in range(1, min(h, k) + 1):
            s += a[i - 1] * psi[h - i]
        psi.append(s)
        if s <= 0.5:
            prev = psi[h - 1]
            return (h - 1) + (prev - 0.5) / (prev - s)
    return float("inf")


def alpha_h(h):
    """The AR(1) root with half-life h: 0.5^(1/h)."""
    return 0.5 ** (1.0 / h)


def c_band(h, phi, T=52):
    """The Rogoff-band alternative in the sum metric: c_{h,i} = T (alpha_h - 1)
    / (1 - sum phi), so that the level coefficients of (1 - rho L) Phi(L) with
    rho = 1 + c/T sum to alpha_h. With phi empty it is T (0.5^(1/h) - 1) exactly."""
    return T * (0.5 ** (1.0 / h) - 1.0) / (1.0 - float(np.sum(phi)))


RHO_GRID_STEP = 0.001  # the scan of rho_irf over (-1, 1)
RHO_XTOL = 1e-12
RHO_ROOT_TOL = 1e-9  # a bracket is a root only if |HL - h| <= this at brentq's point


def rho_irf(h, phi):
    """Sensitivity (a): the rho in (-1, 1) at which the IRF half-life of
    (1 - rho L) Phi(L) equals h. The IRF half-life is scanned on rho = -1 + k 0.001,
    k = 1..1999; every sign change of HL - h between two finite values is refined by
    brentq (xtol 1e-12) and kept only if |HL - h| <= 1e-9 there (a jump of the first
    passage is not a root). Returns the kept root nearest alpha_h, or None."""
    from scipy.optimize import brentq

    def g(r):
        return irf_half_life(level_coeffs(r, phi)) - h

    grid = [-1.0 + k * RHO_GRID_STEP for k in range(1, int(round(2 / RHO_GRID_STEP)))]
    vals = [g(r) for r in grid]
    roots = []
    for r0, r1, v0, v1 in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
        if not (np.isfinite(v0) and np.isfinite(v1)):
            continue
        if v0 == 0.0:
            roots.append(r0)
            continue
        if v0 * v1 < 0:
            r = brentq(g, r0, r1, xtol=RHO_XTOL)
            if abs(g(r)) <= RHO_ROOT_TOL:
                roots.append(r)
    if not roots:
        return None
    ah = alpha_h(h)
    return min(roots, key=lambda r: (abs(r - ah), r))
