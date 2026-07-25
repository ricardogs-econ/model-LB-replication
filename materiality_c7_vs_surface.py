#!/usr/bin/env python3
"""
materiality_c7_vs_surface.py

Does the finite-sample surface c-bar(m,T) matter *over simply using -7*?
This is the referee's decisive question. We compare, under Model LB
(constant + m level-shift dummies, breaks at lambda_j = j/(m+1)), three
recipes for the demeaned GLS MZ_t test at nominal 5%:

  (A)  c-bar = -7  with the ASYMPTOTIC 5% critical value (-1.98)
         -> the naive "just use -7" recipe.  Size distortion is the story.
  (A') c-bar = -7  with its own finite-sample 5% cv (size-corrected)
         -> isolates the pure power cost of suboptimal detrending.
  (B)  c-bar = c-bar*(m,T)  with its own finite-sample 5% cv
         -> the paper's finite-sample pair.

c-bar*(m,T): the paper's own iid-calibrated values (supplement Table S1),
so the experiment tests the paper's surface, not a re-derivation.

Detrending is exact-linear: y~ = M y with
  M = I - Z (Zt'Zt)^{-1} Zt' D_abar,   Zt = D_abar Z,  D_abar the quasi-diff.
By Lemma 1 (theta-invariance) set y = u, so we simulate u only.

Pure numpy. Deterministic seed. iid N(0,1) innovations.
"""
import numpy as np

CV_ASY = -1.98            # ERS/NP demeaned MZ_t 5% asymptotic cv (Ng-Perron 2001)
C_ALT  = -10.0            # common local alternative for the power comparison
NREP   = 200_000
SEED   = 20260720

# paper's iid-calibrated c-bar*(m,T)  (supplement Table S1, col c_iid)
CBAR_STAR = {
    (0, 30): -7.20, (0, 60): -7.00, (0, 100): -6.94,
    (1, 30): -8.73, (1, 60): -7.76, (1, 100): -7.43,
    (2, 30): -10.32, (2, 60): -8.67, (2, 100): -7.97,
}


def build_Z(T, m):
    cols = [np.ones(T)]
    for j in range(1, m + 1):
        lam = j / (m + 1)
        tau = int(np.floor(lam * T))
        du = np.zeros(T); du[tau:] = 1.0          # DU_t = 1{t > tau}, t=1..T
        cols.append(du)
    return np.column_stack(cols)


def Dmat(T, a):
    D = np.eye(T)
    for t in range(1, T):
        D[t, t - 1] = -a
    return D


def detrender(T, m, cbar):
    """Fixed T x T matrix M with y~ = M y (GLS-detrended level)."""
    a = 1.0 + cbar / T
    Z = build_Z(T, m)
    D = Dmat(T, a)
    Zt = D @ Z
    G = Zt.T @ Zt
    M = np.eye(T) - Z @ np.linalg.solve(G, Zt.T @ D)
    return M


def Lmat(T, c):
    a = 1.0 + c / T
    L = np.zeros((T, T)); pw = a ** np.arange(T)
    for t in range(T):
        L[t, :t + 1] = pw[:t + 1][::-1]
    return L


def mzt(Ytil, T):
    """MZ_t for each column of the detrended-level array Ytil (T x N).
    Ytil[i] = y~_{i+1}, i=0..T-1."""
    ss = np.sum(Ytil * Ytil, axis=0)
    yT2 = Ytil[-1] ** 2
    Sy2 = (ss - yT2) / T ** 2                       # T^{-2} sum_{t=1}^{T} y~_{t-1}^2
    dy = np.diff(Ytil, axis=0)                      # y~_t - y~_{t-1}, t=2..T
    s2 = np.sum(dy * dy, axis=0) / (T - 1)          # difference-based long-run var
    MZa = (yT2 / T - s2) / (2.0 * Sy2)
    MSB = np.sqrt(Sy2 / s2)
    return MZa * MSB


def simulate(T, m, cbar, c_dgp, E):
    """Return MZ_t draws: detrend at cbar, data u = L_c E under root 1+c/T."""
    M = detrender(T, m, cbar)
    U = Lmat(T, c_dgp) @ E                          # T x N
    Ytil = M @ U
    return mzt(Ytil, T)


def run_cell(m, T, rng):
    cstar = CBAR_STAR[(m, T)]
    E0 = rng.standard_normal((T, NREP))             # null draws
    E1 = rng.standard_normal((T, NREP))             # alternative draws

    # null: statistics under c=0
    mz_null_7  = simulate(T, m, -7.0, 0.0, E0)
    mz_null_st = simulate(T, m, cstar, 0.0, E0)
    cv7  = np.quantile(mz_null_7,  0.05)
    cvst = np.quantile(mz_null_st, 0.05)
    size_A  = np.mean(mz_null_7 < CV_ASY)           # -7 + asymptotic cv
    size_Ap = np.mean(mz_null_7 < cv7)              # = 0.05 by construction (check)
    size_B  = np.mean(mz_null_st < cvst)            # = 0.05 by construction (check)

    # alternative c = C_ALT
    mz_alt_7  = simulate(T, m, -7.0, C_ALT, E1)
    mz_alt_st = simulate(T, m, cstar, C_ALT, E1)
    powA  = np.mean(mz_alt_7  < CV_ASY)             # naive recipe (oversized)
    powAp = np.mean(mz_alt_7  < cv7)                # -7, size-corrected
    powB  = np.mean(mz_alt_st < cvst)              # c* , size-corrected
    return dict(m=m, T=T, cstar=cstar, cv7=cv7, cvst=cvst,
                size_A=size_A, size_Ap=size_Ap, size_B=size_B,
                powA=powA, powAp=powAp, powB=powB)


def main():
    rng = np.random.default_rng(SEED)
    print(f"Model LB, demeaned MZ_t, nominal 5%, N={NREP:,}, alt c={C_ALT}, "
          f"asymptotic cv={CV_ASY}")
    print(f"{'m':>2} {'T':>4} {'cbar*':>7} | "
          f"{'size(-7,asy)':>12} {'size(-7,FS)':>11} {'size(c*,FS)':>11} | "
          f"{'pow(-7,asy)':>11} {'pow(-7,FS)':>10} {'pow(c*,FS)':>10} {'dPow':>6}")
    rows = []
    for m in (0, 1, 2):
        for T in (30, 60, 100):
            r = run_cell(m, T, rng)
            rows.append(r)
            print(f"{r['m']:>2} {r['T']:>4} {r['cstar']:>7.2f} | "
                  f"{r['size_A']*100:>11.1f}% {r['size_Ap']*100:>10.1f}% "
                  f"{r['size_B']*100:>10.1f}% | "
                  f"{r['powA']*100:>10.1f}% {r['powAp']*100:>9.1f}% "
                  f"{r['powB']*100:>9.1f}% {(r['powB']-r['powAp'])*100:>5.1f}")
    return rows


if __name__ == "__main__":
    main()
