"""materiality_c7_vs_surface.py -- the shared exact-linear MZ_t core.

Pure numpy, self-contained (no kernel import): the exact-linear GLS detrender
`M` (with `y~ = M y`), the local-to-unity companion `L`, and the demeaned MZ_t
statistic, used by the Figure-1 (fig:overdetrend) and Figure-2 (fig:size)
producers. This is the frozen core on the correct side of the sigma^2
convention (long-run variance on the GLS-detrended series, Perron & Qu 2007) --
what the published tab:power / Figure_1 / Figure_2 were computed with; verified
over all 54 tab:cv cells at R=200,000 (mean |diff| 0.0047 against the printed
table). By Lemma 1 (theta-invariance) we simulate u only.
"""
import numpy as np

CV_ASY = -1.98            # ERS/NP demeaned MZ_t 5% asymptotic cv (Ng-Perron 2001)
C_ALT  = -10.0            # common local alternative for the power comparison


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
    """MZ_t draws: detrend at cbar, data u = L_c E under root 1+c/T."""
    M = detrender(T, m, cbar)
    U = Lmat(T, c_dgp) @ E
    return mzt(M @ U, T)
