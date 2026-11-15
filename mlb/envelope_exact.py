"""The envelope half-power point, computed EXACTLY -- no Monte Carlo.

Three implementation choices: the residual-maker is built by QR (not `pinv`),
Imhof's integral is evaluated by `quad` with a stated tolerance that is written to
the manifest, and the conventions 1-4 below are read out of this package's own
kernel.

THE METHOD
----------
Computed exactly by the quadratic-form representation L_T = xi' A_T xi of
Supplement S3.2 (the displacement bound), Step 1 (quadratic form), following Imhof
(1961) as in Dufour and King (1991). The object is ERS's: ERS (1996, p. 819) define
the statistic as S(abar) - S(1), and at p. 814 they say they took the asymptotic
route deliberately, citing Dufour and King (1991) for the exact one. The machinery is
Imhof (1961), Davies (1980), Dufour and King (1991), Tanaka (1996, Ch. 5).

WHAT THIS COMPUTES
------------------
The calibration object is the invariant Gaussian MP test of c = 0 against
c = cbar with sigma^2 KNOWN:

    L_T(cbar) = [S(abar) - S(1)] / sigma^2,    abar = 1 + cbar/T,
    S(a)      = min_b || D_a y - D_a Z b ||^2 .

An object of that form is a quadratic form in Gaussians with a KNOWN matrix.
Its distribution is therefore a weighted sum of independent chi^2_1 with
weights the eigenvalues of that matrix, its CDF is Imhof (1961), and the
half-power point is a root of two exactly computable nested functions; this
module is the source of tab:cbar.

By invariance to (mu, delta), L_T = e' K_c e with

    K_c = L_c' (Q_abar - Q_1) L_c,   Q_a = D_a' M_{D_a Z} D_a,
    (L_c)_{ts} = a^(t-s) 1{t >= s},  u_0 = 0.

CONVENTIONS
-----------
1. BREAK DATES. `mlb/kernel_numba.py`
   (`break_pos_from_lambdas`): tau_j = clamp(floor(lambda_j * T), 1, T-1),
   1-based. `kernel_numba.py` (`build_z_nb`): Z = [1, DU_1..DU_m] with
   DU_j[t] = 1{t_1based > tau_j} -- STRICTLY greater, and the same clamp
   re-applied. Both are reproduced here exactly. Another convention (`>=`, or
   `round` instead of `floor`) moves the break dates, and with them the design
   and cbar_half, so this is not a detail.

2. INITIAL CONDITION. `kernel_numba.py` (`gen_dgp_nb`): u_0 = 0, so
   u_1 = eps_1. That is what `(L_c)_{ts} = a^(t-s) 1{t >= s}` encodes.

3. QUASI-DIFFERENCING. `kernel_numba.py` (`glsd_nb`): the first row is
   kept as a LEVEL and rows 2..T are x_t - a x_{t-1}. That is `_D(a, T)` here.

4. S(1) IS AT a = 1. `kernel_numba.py` (`mstats_nb`, the P_T statistic)
   computes S(1) as `glsd_nb(y, Z, 0.0)`, i.e. cbar = 0 hence abar = 1.

5. TEST DIRECTION AND LEVEL. The MP test rejects for SMALL L_T, at the
   ALPHA = 5% LEFT tail of its null at c = 0, and its power is taken on the
   diagonal c = cbar; the half-power point is where that power equals
   TARGET_POWER = 0.50 (both defined below).

ONE DELIBERATE DIFFERENCE FROM THE KERNEL
-----------------------------------------
`glsd_nb` adds a 1e-10 ridge to the normal equations "for numerical safety"
(`kernel_numba.py`). This module does NOT: the exact object is defined
without it, and the ridge is what breaks exact shift-equivariance.
It is a difference, and it is stated here.

`scipy.integrate.quad` is used for Imhof's integral with an explicit tolerance
rather than Davies' algorithm, and that tolerance is written into the run
manifest.
"""
import numpy as np
from scipy import integrate, linalg, optimize

from .break_dates import break_dates

# Imhof integration tolerance. Stated here, recorded in the manifest, and
# reported with every number this module produces.
IMHOF_EPSABS = 1e-12
IMHOF_EPSREL = 1e-10
IMHOF_LIMIT = 2000

BREAK_CONVENTION = ("tau_j = clamp(floor(lambda_j*T), 1, T-1); "
                    "DU_j[t] = 1{t_1based > tau_j}")
U0_CONVENTION = "u_0 = 0 (u_1 = eps_1)"
ALPHA = 0.05
TARGET_POWER = 0.50

_EIG_TOL = 1e-12


def break_positions(T, lambdas):
    """Mirror of `kernel_numba.break_pos_from_lambdas`."""
    return break_dates(T, lambdas)


def design_Z(T, lambdas, deterministic=True):
    """Mirror of `kernel_numba.build_z_nb`: Z = [1, DU_1..DU_m].

    `deterministic=False` returns a T x 0 design: NO regressors at all, which
    is ERS's equation (3), the case their Figure 1 simulates. There
    S(abar) - S(1) reduces to
        cbar^2 T^-2 ||y_{-1}||^2 - 2 cbar T^-1 y_{-1}' Delta y,
    and the half-power point of THAT object is the -7 ERS publish. It is the
    validation case for this whole module: it has no demeaning, so nothing
    absorbs the O(1/T) demeaning effect.
    """
    if not deterministic:
        return np.zeros((T, 0))
    t1 = np.arange(1, T + 1)
    cols = [np.ones(T)]
    for tau in break_positions(T, lambdas):
        cols.append((t1 > tau).astype(float))
    return np.column_stack(cols)


def _D(a, T):
    """Quasi-differencing operator; first row kept as a level (`glsd_nb`)."""
    M = np.eye(T)
    M[np.arange(1, T), np.arange(0, T - 1)] = -a
    return M


def _Lc(c, T):
    """u = L_c e for u_t = a u_{t-1} + e_t with u_0 = 0, a = 1 + c/T."""
    a = 1.0 + c / T
    P = np.arange(T)[:, None] - np.arange(T)[None, :]
    return np.where(P >= 0, a ** np.clip(P, 0, None), 0.0)


def _Q(a, T, Z):
    """D_a' M_{D_a Z} D_a, with the residual-maker built by QR.

    QR, not `pinv`: a thin QR of D_a Z gives M = I - Q Q' directly, at a third
    of the cost and without forming a pseudo-inverse. Column-pivoted, and
    columns whose R diagonal falls below the tolerance are dropped, so a
    rank-deficient design degrades to the projection on its actual column
    space instead of returning silent nonsense.
    """
    Da = _D(a, T)
    if Z.shape[1] == 0:            # no regressors: M = I (ERS eq. 3)
        return Da.T @ Da
    Za = Da @ Z
    Qm, Rm, _ = linalg.qr(Za, mode="economic", pivoting=True)
    d = np.abs(np.diag(Rm))
    k = int(np.sum(d > d[0] * 1e-12)) if d.size and d[0] > 0 else 0
    Qm = Qm[:, :k]
    M = np.eye(T) - Qm @ Qm.T
    return Da.T @ M @ Da


def eig_K(c_dgp, cbar, T, Z, A):
    """Eigenvalues of K_c = L_c' (Q_abar - Q_1) L_c, zeros dropped."""
    L = _Lc(c_dgp, T)
    K = L.T @ A @ L
    w = linalg.eigvalsh(0.5 * (K + K.T))
    return w[np.abs(w) > _EIG_TOL]


def imhof_cdf(x, lam):
    """P(sum_i lam_i chi^2_1 < x), Imhof (1961).

    rho is accumulated in log space so T = 2400 does not overflow.
    """
    lam = np.asarray(lam)

    def f(u):
        theta = 0.5 * np.sum(np.arctan(lam * u)) - 0.5 * x * u
        logrho = 0.25 * np.sum(np.log1p((lam * u) ** 2))
        return np.sin(theta) * np.exp(-logrho) / u

    val, _ = integrate.quad(f, 0.0, np.inf, limit=IMHOF_LIMIT,
                            epsabs=IMHOF_EPSABS, epsrel=IMHOF_EPSREL)
    return 0.5 - val / np.pi


def quantile(lam, p, xtol=1e-10):
    lo = float(np.sum(np.minimum(lam, 0)) - 1.0)
    hi = float(np.sum(np.maximum(lam, 0)) + 1.0)
    return optimize.brentq(lambda x: imhof_cdf(x, lam) - p, lo, hi, xtol=xtol)


def power_at(cbar, T, Z, alpha=ALPHA):
    """(power on the diagonal c = cbar, the alpha-level null quantile b_T)."""
    A = _Q(1.0 + cbar / T, T, Z) - _Q(1.0, T, Z)
    b = quantile(eig_K(0.0, cbar, T, Z, A), alpha)
    return imhof_cdf(b, eig_K(cbar, cbar, T, Z, A)), b


def half_power_point(T, lambdas=(), alpha=ALPHA, target=TARGET_POWER,
                     bracket=(-25.0, -1.0), xtol=1e-6, deterministic=True):
    """cbar_half(m, T, lambda) and the b_T at it. Exact; no Monte Carlo."""
    Z = design_Z(T, lambdas, deterministic)
    box = {}

    def g(cb):
        pw, b = power_at(cb, T, Z, alpha)
        box[cb] = b
        return pw - target

    cb = optimize.brentq(g, bracket[0], bracket[1], xtol=xtol)
    return float(cb), float(power_at(cb, T, Z, alpha)[1])


def conventions():
    """Everything a manifest needs to say about how these numbers were made."""
    return dict(break_convention=BREAK_CONVENTION, u0=U0_CONVENTION,
                alpha=ALPHA, target_power=TARGET_POWER,
                imhof_epsabs=IMHOF_EPSABS, imhof_epsrel=IMHOF_EPSREL,
                imhof_limit=IMHOF_LIMIT, eig_tol=_EIG_TOL,
                projection="column-pivoted QR, rank tol 1e-12 * R[0,0]",
                ridge="none (the kernel's glsd_nb adds 1e-10; see the module "
                      "docstring)")


# ------------------------------------------------------------------------------
# AR(1) innovations. An extension: nothing above this line depends on it.
#
# DGP of ar1_size_power.py and ar1_recalibration.py, which is `make_ar1_eps`
# of mlb/ar1_size_power.py:
#     v_t = rho v_{t-1} + e_t,  v_0 = 0 (v_1 = e_1);
#     u_t = (1 + c/T) u_{t-1} + v_t,  u_0 = 0;   e ~ N(0, I).
# Hence u = L_c P_rho e with (P_rho)_{ts} = rho^(t-s) 1{t >= s}. The statistic
# is unchanged, L_T = S(abar) - S(1); only the law of y moves, so
#     L_T = e' K e,   K = (L_c P_rho)' (Q_abar - Q_1) (L_c P_rho),
# still a Gaussian quadratic form with a known matrix, still Imhof. Dividing by
# a variance does not move the half-power point: the critical value and the
# power come from the same family.
#
# rho == 0 CALLS THE FUNCTIONS ABOVE, unchanged; the check surface_agreement of
# ar1_recalibration.py proves it against cbar_surface_exact.csv.
# ------------------------------------------------------------------------------
AR1_CONVENTION = "v_t = rho v_{t-1} + e_t, v_0 = 0 (v_1 = e_1); u_0 = 0"


def _Pr(rho, T):
    """v = P_rho e for v_t = rho v_{t-1} + e_t, v_0 = 0."""
    P = np.arange(T)[:, None] - np.arange(T)[None, :]
    return np.where(P >= 0, float(rho) ** np.clip(P, 0, None), 0.0)


def eig_K_ar1(c_dgp, cbar, T, Z, rho, A):
    """Eigenvalues of (L_c P_rho)' (Q_abar - Q_1) (L_c P_rho), zeros dropped."""
    if rho == 0.0:
        return eig_K(c_dgp, cbar, T, Z, A)
    L = _Lc(c_dgp, T) @ _Pr(rho, T)
    K = L.T @ A @ L
    w = linalg.eigvalsh(0.5 * (K + K.T))
    return w[np.abs(w) > _EIG_TOL]


def power_at_ar1(cbar, T, Z, rho, alpha=ALPHA):
    """(power on the diagonal c = cbar, the alpha-level null quantile), both
    under AR(1) innovations with coefficient rho."""
    if rho == 0.0:
        return power_at(cbar, T, Z, alpha)
    A = _Q(1.0 + cbar / T, T, Z) - _Q(1.0, T, Z)
    b = quantile(eig_K_ar1(0.0, cbar, T, Z, rho, A), alpha)
    return imhof_cdf(b, eig_K_ar1(cbar, cbar, T, Z, rho, A)), b


def half_power_point_ar1(T, lambdas=(), rho=0.0, alpha=ALPHA, target=TARGET_POWER,
                         bracket=(-25.0, -1.0), xtol=1e-6):
    """cbar_half and b_T under AR(1) innovations. rho = 0 is half_power_point."""
    if rho == 0.0:
        return half_power_point(T, lambdas, alpha, target, bracket, xtol)
    Z = design_Z(T, lambdas)

    def g(cb):
        return power_at_ar1(cb, T, Z, rho, alpha)[0] - target

    cb = optimize.brentq(g, bracket[0], bracket[1], xtol=xtol)
    return float(cb), float(power_at_ar1(cb, T, Z, rho, alpha)[1])
