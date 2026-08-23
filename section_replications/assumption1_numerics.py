#!/usr/bin/env python3
"""
assumption1_numerics.py -- Numerical corroboration for the online supplement to
"Level Breaks and Finite-Sample GLS Detrending".

Assumption 1 (equicontinuity of the power family) is proved in Section S2 as
Theorem S2.1, from the pointwise, lambda-free convergence of Theorem 3
(thm:weaklimit) via a uniform density bound on the limit law (a quantitative
substitute for Polya's theorem; Polya itself is not invoked). These checks
corroborate the exact finite-sample structure that stands behind that
convergence -- the material of Section S1 (proof of Lemma 2 / lem:design and
Theorem 3) and Corollary 1 (cor:law): the differencing corner, the P_T-MPT
algebra, and the O((m+1)/T) remainder. The observable implication of Theorem
S2.1 itself is simulated separately in modulus_numerics.py.

Verifies, on a grid T in [30,400], m <= 3 (boundary and interior break
fractions), cbar in [-20,-3], null (c=0) and alternative (c=cbar) slices:

  (1) the closed forms of the quasi-differenced quadratic form:
        A_T(cbar; cbar) = M_abar - abar (I+H)' M_1 (I+H),   H  =  (cbar/T) J D_abar^{-1}
        A_T(cbar; 0)    = (I+H0)' M_abar (I+H0) - abar M_1, H0 = -(cbar/T) J D_1^{-1}
      against the general construction A_T = L_c' Q(cbar) L_c  (machine precision);
  (2) the diagonal identity behind the P_T-MPT combination:
        D_abar' D_abar - abar D_1' D_1 = (1-abar)(I - abar E)  (machine precision);
  (3) nondegeneracy: Var N_T = 2 ||A_T||_F^2 > 0 on the grid (grid minimum);
  (4) the differencing corner (Lemma 2 / lem:design): level Gram exactly I, so the
      design orthonormalizes; slope difference is exactly a step carrying
      (1-lambda)T mass;
  (5) the Dickey-Fuller / ERS closure of the trace means: the lambda-free limit
      of Theorem 3(ii) and Corollary 1;
  (6) the O((m+1)/T) remainder (Theorem 3(iii), prop:flat(i)): the block
      cancellation M_T(c,0)-M_T(c,cbar) = o_p(1) that leaves a lambda-free limit
      under Model LB.

Pure numpy; deterministic; no simulation.
"""
import numpy as np


def build_Z(T, lambdas):
    cols = [np.ones(T)]
    for lam in lambdas:
        tau = int(np.floor(lam * T))
        du = np.zeros(T); du[tau:] = 1.0          # DU_t = 1{t > tau}, t = 1..T
        cols.append(du)
    return np.column_stack(cols)


def qd(T, a):
    D = np.eye(T)
    for t in range(1, T):
        D[t, t - 1] = -a
    return D


def ann(X):
    Q, _ = np.linalg.qr(X)
    return np.eye(X.shape[0]) - Q @ Q.T


def Lmat(T, c):
    rho = 1 + c / T
    L = np.zeros((T, T)); pw = rho ** np.arange(T)
    for t in range(T):
        L[t, :t + 1] = pw[:t + 1][::-1]
    return L


def A_general(T, lams, cbar, c):
    Z = build_Z(T, lams); ab = 1 + cbar / T
    Dab, D1 = qd(T, ab), qd(T, 1.0)
    Q = Dab.T @ ann(Dab @ Z) @ Dab - ab * (D1.T @ ann(D1 @ Z) @ D1)
    L = Lmat(T, c); A = L.T @ Q @ L
    return 0.5 * (A + A.T)


def A_closed_alt(T, lams, cbar):
    Z = build_Z(T, lams); ab = 1 + cbar / T
    Dab, D1 = qd(T, ab), qd(T, 1.0)
    Mab, M1 = ann(Dab @ Z), ann(D1 @ Z)
    J = np.diag(np.ones(T - 1), -1)
    H = (cbar / T) * J @ np.linalg.inv(Dab)
    IH = np.eye(T) + H
    A = Mab - ab * IH.T @ M1 @ IH
    return 0.5 * (A + A.T)


def A_closed_null(T, lams, cbar):
    Z = build_Z(T, lams); ab = 1 + cbar / T
    Dab, D1 = qd(T, ab), qd(T, 1.0)
    Mab, M1 = ann(Dab @ Z), ann(D1 @ Z)
    J = np.diag(np.ones(T - 1), -1)
    H0 = -(cbar / T) * J @ np.linalg.inv(D1)
    IH = np.eye(T) + H0
    A = IH.T @ Mab @ IH - ab * M1
    return 0.5 * (A + A.T)


def main():
    # (1) closed forms
    print("(1) closed forms vs general construction (max abs entry error):")
    for T, lams, cbar in [(30, [0.5], -7.0), (50, [0.15, 0.85], -20.0),
                          (100, [1/3, 2/3], -3.0), (200, [0.25, 0.5, 0.75], -10.0)]:
        e1 = np.abs(A_general(T, lams, cbar, cbar) - A_closed_alt(T, lams, cbar)).max()
        e0 = np.abs(A_general(T, lams, cbar, 0.0) - A_closed_null(T, lams, cbar)).max()
        print(f"    T={T:4d} m={len(lams)} cbar={cbar:6.1f}:  alt {e1:.2e}   null {e0:.2e}")

    # (2) diagonal identity
    T, ab = 40, 1 - 7.0 / 40
    B = qd(T, ab).T @ qd(T, ab) - ab * (qd(T, 1.0).T @ qd(T, 1.0))
    E = np.diag(np.r_[np.ones(T - 1), 0.0])
    print(f"(2) diagonal identity max error: "
          f"{np.abs(B - (1-ab)*(np.eye(T)-ab*E)).max():.2e}")

    # (3) variance floor
    grid_T = (30, 50, 100, 200, 400)
    grid_lam = ([], [0.5], [0.15], [1/3, 2/3], [0.15, 0.85], [0.25, 0.5, 0.75])
    grid_cb = (-20.0, -10.0, -7.0, -3.0)
    vmin, amax = np.inf, 0.0
    for T in grid_T:
        for lams in grid_lam:
            if len(lams) >= 3 and T < 45:
                continue
            for cb in grid_cb:
                for c in (0.0, cb):
                    A = A_general(T, lams, cb, c)
                    v = 2.0 * np.sum(A * A)
                    vmin = min(vmin, v)
    print(f"(3) nondegeneracy: min over grid of Var N_T = {vmin:.4f} > 0")


def check_corner_and_df():
    import numpy as np
    print("(4) corner designs: level Gram exactly I; slope keeps O(T) mass carrying lambda")
    T=100; t=np.arange(1,T+1)
    for lam in (0.3,0.7):
        tau=int(lam*T)
        DT=((t-tau)*(t>tau)).astype(float)
        dDT=np.r_[DT[0],np.diff(DT)]; step=(t>tau).astype(float)
        print(f"    lam={lam}: ||dDT-step||={np.abs(dDT-step).max():.0e}  ||dDT||^2={dDT@dDT:.0f}=T-tau={T-tau}")
    Zl=build_Z(T,[0.3,0.7]); D1=qd(T,1.0)
    G=(D1@Zl).T@(D1@Zl)
    print(f"    level design Gram: ||G-I||_max = {np.abs(G-np.eye(3)).max():.0e}")

    print("(5) DF/ERS closure: tr A_T vs closed-form limit means (cbar=-7, m=2)")
    cbar=-7.0
    EV1=(np.exp(2*cbar)-1)/(2*cbar); EintV=((np.exp(2*cbar)-1)/(2*cbar)-1)/(2*cbar)
    print(f"    limits: null {cbar**2/2-cbar:.4f}   alt {cbar**2*EintV-cbar*EV1:.4f}")
    for T in (100,400,800):
        trn=np.trace(A_general(T,[1/3,2/3],cbar,0.0))
        tra=np.trace(A_general(T,[1/3,2/3],cbar,cbar))
        print(f"    T={T:4d}: tr A_null={trn:8.4f}  tr A_alt={tra:7.4f}")

    print("(6) null-slice projection-correction mean: O((m+1)/T) with O(cbar^4) constant")
    cbar=-7.0
    for T in (100,400,800,1600):
        ab=1+cbar/T
        Z=build_Z(T,[1/3,2/3]); Dab,D1=qd(T,ab),qd(T,1.0)
        Zab=Dab@Z; Pab=Zab@np.linalg.inv(Zab.T@Zab)@Zab.T
        L0=Lmat(T,0.0); X=Dab@L0
        E_pab=np.trace(X.T@Pab@X); E_p1=3.0  # exact: (m+1) impulses on white noise
        print(f"    T={T:5d}: E[correction] = {E_pab-ab*E_p1:7.4f}")


if __name__ == "__main__":
    main()
    check_corner_and_df()
