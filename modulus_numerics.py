#!/usr/bin/env python3
"""
modulus_numerics.py -- Empirical modulus of the power family (Theorem S2.1).

Observable implication of the asymptotic equicontinuity proved as Theorem S2.1
in the online supplement: the empirical modulus

    M_T(h) = max over cells (m, lambda) and pairs |cbar' - cbar| <= h
             of | Pi^T_lambda(cbar') - Pi^T_lambda(cbar) |

should (i) shrink as h decreases and (ii) be stable across T. This script
estimates Pi^T_lambda(cbar) by Monte Carlo over a grid of cbar and reports
M_T(h).

Design (as stated in the supplement's table note):
  * iid N(0,1) innovations; u_0 = 0; break magnitudes and the intercept are
    set to zero by exact invariance (Lemma 1 of the main text);
  * oracle point-optimal statistic S = S(abar) - abar S(1) with known
    long-run variance (sigma^2 = 1) -- the finite-sample analogue of the
    functional L of eq:power-def, isolating the geometry of the power family
    from variance-estimation noise;
  * the null slice c = 0 pins down the critical value b^T_lambda(cbar) (the
    lower 5% quantile); the alternative slice c = cbar reads off the power
    Pi^T_lambda(cbar) = Pr_{c=cbar}[ S < b^T_lambda(cbar) ];
  * common random numbers: the SAME innovation matrix is used for every cbar
    within a cell (on the null slice, the same paths; on the alternative
    slice, the same shocks filtered at the root 1 + cbar/T), so that
    adjacent power DIFFERENCES have much lower variance than independent
    estimators would.

Usage:
    python modulus_numerics.py --selftest
    python modulus_numerics.py --nrep 40000 --csv-out modulus_table.csv --latex
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import lfilter

# ----------------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------------
CBAR_GRID = np.round(np.arange(-20.0, -3.0 + 1e-9, 0.25), 2)   # K = [-20, -3]
H_GRID = (0.25, 0.5, 1.0, 2.0)
# small sample is the core of the paper: T = 45 is the admissibility floor of
# the application (condition C2), T = 52 is the exact span of the PPP panel,
# and T = 30 sits below the admissibility floor.
T_GRID = (30, 45, 52, 60, 150, 300)
# cells (m, lambdas): includes m = 0 (constant-only family) and interior
# configurations of Lambda_eps (eps = 0.15 in the main text's calibration)
CELLS: Tuple[Tuple[float, ...], ...] = (
    (),                    # m = 0: {Pi^T_0}
    (0.3,), (0.5,), (0.7,),
    (0.3, 0.6), (0.25, 0.75),
)
ALPHA = 0.05


def config_seed(T: int, lams: Sequence[float], slice_tag: str) -> int:
    """Seed derived from the configuration only (exact reproducibility)."""
    key = f"T={T};lams={tuple(lams)};slice={slice_tag};v1"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


# ----------------------------------------------------------------------------
# 2. Core: design, quasi-difference, oracle point-optimal statistic
# ----------------------------------------------------------------------------
def build_Z(T: int, lams: Sequence[float]) -> np.ndarray:
    """Constant + m step dummies DU_t = 1{t > tau}, tau = floor(lam T)."""
    cols = [np.ones(T)]
    for lam in lams:
        tau = int(np.floor(lam * T))
        du = np.zeros(T)
        du[tau:] = 1.0
        cols.append(du)
    return np.column_stack(cols)


def quasi_diff_mat(X: np.ndarray, a: float) -> np.ndarray:
    """Quasi-difference the columns of X (T x k): the first row stays in levels."""
    Xq = np.empty_like(X)
    Xq[0] = X[0]
    Xq[1:] = X[1:] - a * X[:-1]
    return Xq


def quasi_diff_paths(U: np.ndarray, a: float) -> np.ndarray:
    """Quasi-difference each row of U (nrep x T)."""
    Uq = np.empty_like(U)
    Uq[:, 0] = U[:, 0]
    Uq[:, 1:] = U[:, 1:] - a * U[:, :-1]
    return Uq


def rss_batch(Yq: np.ndarray, Zq: np.ndarray) -> np.ndarray:
    """RSS of the GLS regression of each row of Yq on Zq: ||y||^2 - ||P y||^2."""
    G = Zq.T @ Zq
    R = np.linalg.cholesky(np.linalg.inv(G))
    B = (Yq @ Zq) @ R                       # nrep x (m+1), ||B||^2 = y'P y
    return np.einsum("ij,ij->i", Yq, Yq) - np.einsum("ij,ij->i", B, B)


def point_optimal(U: np.ndarray, Z: np.ndarray, cbar: float) -> np.ndarray:
    """S = S(abar) - abar S(1), with known long-run variance (= 1)."""
    T = U.shape[1]
    abar = 1.0 + cbar / T
    s_ab = rss_batch(quasi_diff_paths(U, abar), quasi_diff_mat(Z, abar))
    s_1 = rss_batch(quasi_diff_paths(U, 1.0), quasi_diff_mat(Z, 1.0))
    return s_ab - abar * s_1


def simulate_paths(E: np.ndarray, rho: float) -> np.ndarray:
    """u_t = rho u_{t-1} + eps_t, u_0 = 0 (exact recursion via lfilter)."""
    return lfilter([1.0], [1.0, -rho], E, axis=1)


# ----------------------------------------------------------------------------
# 3. Power per cell and the empirical modulus
# ----------------------------------------------------------------------------
def power_curve(T: int, lams: Sequence[float], nrep: int
                ) -> Tuple[np.ndarray, float]:
    """Pi^T_lambda(cbar) on the CBAR_GRID; returns (powers, se_max)."""
    Z = build_Z(T, lams)
    rng0 = np.random.default_rng(config_seed(T, lams, "null"))
    rng1 = np.random.default_rng(config_seed(T, lams, "alt"))
    E0 = rng0.standard_normal((nrep, T))
    E1 = rng1.standard_normal((nrep, T))
    U0 = simulate_paths(E0, 1.0)            # null slice: shared across all cbar
    powers = np.empty(len(CBAR_GRID))
    for i, cb in enumerate(CBAR_GRID):
        s_null = point_optimal(U0, Z, cb)
        b = np.quantile(s_null, ALPHA)
        U1 = simulate_paths(E1, 1.0 + cb / T)
        s_alt = point_optimal(U1, Z, cb)
        powers[i] = np.mean(s_alt < b)
    se_max = float(np.sqrt(np.max(powers * (1 - powers)) / nrep))
    return powers, se_max


def empirical_modulus(curves: Dict[Tuple[int, Tuple[float, ...]], np.ndarray]
                      ) -> Dict[int, Dict[float, float]]:
    """M_T(h) = max over cells and grid pairs with |diff| <= h."""
    out: Dict[int, Dict[float, float]] = {}
    for T in T_GRID:
        out[T] = {}
        for h in H_GRID:
            m = 0.0
            for (TT, lams), p in curves.items():
                if TT != T:
                    continue
                for i in range(len(CBAR_GRID)):
                    for j in range(i + 1, len(CBAR_GRID)):
                        if CBAR_GRID[j] - CBAR_GRID[i] <= h + 1e-9:
                            m = max(m, abs(p[j] - p[i]))
            out[T][h] = m
    return out


# ----------------------------------------------------------------------------
# 4. Self-tests
# ----------------------------------------------------------------------------
def selftest() -> int:
    ok = True
    # (i) quasi-difference preserves shape and values on a manual example
    x = np.array([[1.0, 2.0, 4.0]])
    xq = quasi_diff_paths(x, 0.5)
    ok &= np.allclose(xq, [[1.0, 1.5, 3.0]])
    # (ii) Gram of the quasi-differenced level design (a = 1) = I_{m+1} (lem:design)
    Z = build_Z(200, (0.3, 0.7))
    Zq = quasi_diff_mat(Z, 1.0)
    ok &= np.allclose(Zq.T @ Zq, np.eye(3), atol=1e-12)
    # (iii) null mean of S at m = 0, cbar = -7 versus the exact trace of A_T
    sys.path.insert(0, ".")
    from assumption1_numerics import A_general
    T, cb, nrep = 120, -7.0, 60000
    tr = np.trace(A_general(T, [], cb, 0.0))
    rng = np.random.default_rng(config_seed(T, (), "selftest"))
    U = simulate_paths(rng.standard_normal((nrep, T)), 1.0)
    mc = float(np.mean(point_optimal(U, build_Z(T, []), cb)))
    ok &= abs(mc - tr) < 0.15
    print(f"selftest: quasi-diff ok; Gram=I ok; "
          f"E[S] MC {mc:.3f} vs tr A_T {tr:.3f} -> {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


# ----------------------------------------------------------------------------
# 5. CLI
# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nrep", type=int, default=40000)
    ap.add_argument("--csv-out", type=str, default=None)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--tgrid", type=str, default=None,
                    help="comma-separated list of T (overrides T_GRID); "
                         "seeds are per-configuration, so partial runs merged "
                         "afterwards coincide with a single joint run")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    global T_GRID
    if args.tgrid:
        T_GRID = tuple(int(x) for x in args.tgrid.split(","))

    curves: Dict[Tuple[int, Tuple[float, ...]], np.ndarray] = {}
    se_all: List[float] = []
    for T in T_GRID:
        for lams in CELLS:
            p, se = power_curve(T, lams, args.nrep)
            curves[(T, tuple(lams))] = p
            se_all.append(se)
            print(f"T={T:3d} m={len(lams)} lam={lams}: "
                  f"Pi[-20]={p[0]:.3f} Pi[-7]={p[np.argmin(np.abs(CBAR_GRID+7))]:.3f} "
                  f"Pi[-3]={p[-1]:.3f}", flush=True)
    mod = empirical_modulus(curves)
    se_max = max(se_all)

    print("\nEmpirical modulus M_T(h): rows T, columns h")
    print("T     " + "".join(f"h={h:<6}" for h in H_GRID))
    for T in T_GRID:
        print(f"{T:<6}" + "".join(f"{mod[T][h]:<8.3f}" for h in H_GRID))
    print(f"max MC SE across all powers: {se_max:.4f}")

    if args.csv_out:
        with open(args.csv_out, "w") as f:
            f.write("T,h,modulus\n")
            for T in T_GRID:
                for h in H_GRID:
                    f.write(f"{T},{h},{mod[T][h]:.4f}\n")
            f.write(f"# max_MC_SE,{se_max:.5f}\n")
        print(f"csv saved to {args.csv_out}")

    if args.latex:
        print("\n% tabular for the supplement:")
        print(r"\begin{tabular}{@{}lcccc@{}}")
        print(r"\toprule")
        print(r"& \multicolumn{4}{c}{$h=|\cb'-\cb|$} \\")
        print(r"\cmidrule(l){2-5}")
        print(r"$T$ & 0.25 & 0.5 & 1 & 2 \\")
        print(r"\midrule")
        for T in T_GRID:
            row = " & ".join(f"{mod[T][h]:.3f}" for h in H_GRID)
            print(f"{T} & {row} " + r"\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
