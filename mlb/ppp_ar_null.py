"""The AR(p) null of the PPP application (Model LB, numba kernel).

load_kernels and make_gen_arp are what ppp_application.py binds; stats_under is the
simulator of one configuration, against which ppp_application.py's draws can be checked
(MZ_t of every replication).

Conventions: every statistic is the kernel's (mlb/kernel_numba.py, imported and
never reimplemented); replication r draws from numpy.random.default_rng(seed0 + r); the
recursion of u starts at u_0 = 0 and the burn-in acts on the stationary AR(p) nuisance
only.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore")


def load_kernels():
    """The kernel functions the application uses, and numba's njit."""
    from mlb import kernel_numba as kn
    return dict(build_z=kn.build_z_nb, mstats=kn.mstats_nb, njit=njit,
                method_code=kn._METHOD_CODE)


def make_gen_arp(njit):
    @njit(cache=True)
    def gen_dgp_arp_nb(nt, burn, c, phi, eps):
        """y_t = u_t with u_t = rho*u_{t-1} + w_t, rho = 1 + c/nt (local root),
        and w_t the AR(p) nuisance: phi(L) w_t = e_t. beta = 0 WLOG (exact-
        detrending invariance). eps has length nt+burn; the first `burn` draws
        prime the AR(p) recursion for w so that w is at its stationary law by
        t=0. p=0 (phi empty) reproduces the iid DGP EXACTLY: then w_t = e_t,
        u_0 = 0, u_t = rho*u_{t-1} + e_t -- identical to gen_dgp_nb.

        NOTE ON THE FIRST OBSERVATION: the recursion is initialized at the
        PRE-SAMPLE value u_0 = 0, so the first in-sample observation is
        u_1 = w_1 -- the paper's convention (initialized at u_0 = 0, the ERS
        convention, so that the estimated intercept converges to
        y_1 = eps_1). The burn-in acts ONLY on the stationary nuisance w,
        never on the initial condition of the integrated u.
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
        # 2) integrate the post-burn segment from the PRE-SAMPLE u_0 = 0,
        #    so u_1 = w_1 (the kernel's convention)
        rho = 1.0 + c / nt
        u = np.zeros(nt)
        u[0] = w[burn]
        for t in range(1, nt):
            u[t] = rho * u[t - 1] + w[burn + t]
        return u
    return gen_dgp_arp_nb


def stats_under(K, gen_arp, T, break_pos, c, cbar, mc, n_reps, seed0, kmax, phi, burn):
    """MZ_t of n_reps draws at alternative c, detrended at cbar, with the AR(p)
    nuisance phi: replication r draws eps = standard_normal(T + burn) from
    default_rng((seed0 + r) mod (2^63 - 1)); NaN where the kernel's ok flag is 0."""
    mzt = np.empty(n_reps)
    Z = K["build_z"](T, break_pos)
    for r in range(n_reps):
        rng = np.random.default_rng((seed0 + r) % (2**63 - 1))
        eps = rng.standard_normal(T + burn)
        y = gen_arp(T, burn, c, phi, eps)
        mza, msb, mz, ptv, mpt, ok = K["mstats"](y, Z, cbar, mc, kmax)
        mzt[r] = mz if ok > 0.5 else np.nan
    return mzt
