"""
modulus_feasible.py -- Recomputation of the empirical modulus (the "envelope")
in the design of the application.

WHY THIS RERUN
--------------
Table S1 of the supplement, and the paragraph "The applied span" that reads it,
were produced by `modulus_numerics.py` under three design choices that are NOT
those of the application. The comparison drawn in that paragraph is therefore
not licensed, and this script replaces it.

  (D1) ORACLE vs FEASIBLE. The point-optimal statistic was the oracle
       S = S(abar) - abar S(1), with the long-run variance KNOWN (sigma^2 = 1).
       The application uses the feasible statistic
           P_T(cbar) = [S(abar) - abar S(1)] / s^2(cbar),
       with s^2 estimated from the data. The estimation noise in s^2 moves both
       the null quantile and the curvature of the power family.

  (D2) DIAGONAL FAMILY vs FIXED ALTERNATIVE. The tabulated modulus runs along
       the DIAGONAL family
           Pi^T_lambda(cbar) = Pr_{c=cbar}[ P < b^T_lambda(cbar) ],
       power read AT the tangency point itself. That is the object of
       Theorem S2.1 (Assumption 1). The applied cost the paragraph compared it
       with (0.03-0.04) lives in the FIXED-ALTERNATIVE family
           Pi^T_lambda(cbar; c_alt) = Pr_{c=c_alt}[ P < b^T_lambda(cbar) ],
       with c_alt held fixed while only the detrending parameter moves. These
       are different families; a modulus for one does not bound the other.

  (D3) GLOBAL vs LOCAL MAXIMUM. M_T(h) was a maximum over the whole of
       K = [-20, -3], hence dominated by the steep left region. Near the
       tangency the local variation is much smaller. A global maximum is a
       valid but heavily slack bound for a statement about the calibrated
       region.

  (D4) STATISTIC CLASS. The modulus is computed on the point-optimal statistic,
       because that is the object of eq:power-def and of Theorem S2.1. The
       applied cost it was compared with is measured on the M-class MZ_t --
       the statistic of `tab:power`, of `tab:ppp` and of
       `power_cbar_star_vs_c7.py`. A modulus for the point-optimal family does
       not bound the power variation of MZ_t. The third statistic below closes
       that gap: `mzt` runs the same modulus machinery on the demeaned MZ_t of
       the GLS-detrended level, so the table and the Panel A/B losses finally
       speak about the same test.
       The cell configuration matters for the same reason: `tab:cv`, and hence
       Panels A and B, place the breaks at lambda_j = j/(m+1), which for m = 2
       is (1/3, 2/3) -- a cell ABSENT from the published cell list. Use
       `--cells applied` to include it.

This script returns, in one run and on common random numbers, all six
combinations {oracle, feasible, MZ_t} x {diagonal, fixed-alternative}, each
with a GLOBAL modulus (over K) and a LOCAL one (over K_loc, default [-12,-5]).

EXACT REPRODUCTION OF WHAT IS ALREADY IN THE PAPER
--------------------------------------------------
The primitives (`build_Z`, `quasi_diff_mat`, `quasi_diff_paths`, `rss_batch`,
`simulate_paths`, `config_seed`) and the seeds are IMPORTED from
`modulus_numerics.py`; nothing is reimplemented. The {oracle, diagonal, global}
block reproduces the published Table S1 bit for bit at the same `--nrep`.
`--selftest` checks exactly that, by calling `modulus_numerics.power_curve`.

LONG-RUN VARIANCE
-----------------
The difference-based (`const') rule, identical to the one in
`materiality_c7_vs_surface.py` (function `mzt`, line `s2 = sum(dy*dy)/(T-1)`)
and the one named in the note to Table S2 of the supplement:

    ytil_t    = y_t - z_t' betahat(cbar)          (GLS-detrended level)
    s^2(cbar) = (T-1)^{-1} sum_{t=2}^{T} (ytil_t - ytil_{t-1})^2

Note that s^2 depends on cbar through betahat(cbar): that dependence is the
channel Table S2 isolates. The MAIC/sieve variant is deliberately NOT
implemented here -- it costs one lag selection per replication and does not
change the qualitative reading; if it is wanted, it is a local addition inside
`stats_pair`.

DESIGN (as in the supplement, except where stated)
--------------------------------------------------
  * i.i.d. N(0,1) innovations; u_0 = 0; intercept and break magnitudes set to
    zero by exact invariance (Lemma 1 of the main text);
  * the null slice c = 0 fixes b^T_lambda(cbar), the lower 5% quantile;
  * common random numbers: the SAME innovation matrix serves every cbar within
    a cell, in each of the three slices (null, diagonal, fixed alternative), so
    that adjacent power differences are far more precise than independent
    estimates would be;
  * cells (m, lambda) and the T- and cbar-grids are those of
    `modulus_numerics.py`.

NOISE FLOOR
-----------
M_T(h) is a MAXIMUM over thousands of pairs, so Monte Carlo error enters it
with a positive sign: it is an UPWARD-BIASED estimator of the true modulus.
The script reports an explicit floor, obtained by recomputing the same powers
on two independent halves of the replications and taking
    max_cbar | Pi^(1)(cbar) - Pi^(2)(cbar) | / sqrt(2).
Entries of M_T(h) of the order of the floor do not measure curvature of the
power family; they measure the simulation. This diagnostic did not exist in
the original run and is the main reason not to quote the h = 0.25 column
without qualification.

USAGE
-----
    python modulus_feasible.py --selftest
    python modulus_feasible.py --nrep 40000 --csv-out modulus_feasible.csv --latex
    python modulus_feasible.py --nrep 200000 --tgrid 45,52,60 --cells applied \
        --csv-out modulus_feasible_applied.csv --latex

COST. The loop is |T| x |cells| x 69 cbar x 3 slices, and is roughly linear in
nrep, in the number of cells and in sum(T). Measured references, 7 cells:
    --tgrid 52  --nrep  8000 ...  16 s
    --tgrid 300 --nrep  4000 ...  60 s   (mildly superlinear in T)
so the recommended run below is of the order of 20-30 min, and the full six-T
run at `--nrep 40000` of the order of 20-25 min.
Peak memory is 3 x 69 x nrep x 8 bytes (66 MB at nrep = 40000, 331 MB at
nrep = 200000) plus the path blocks; tune with `--chunk`.
RECOMMENDATION: `--nrep 200000` with `--cells applied` for the rows
T = 45, 52, 60 that the text quotes, so as to sit comfortably above the noise
floor; `--nrep 40000` is enough for the shape of the table.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import argparse
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Primitives come from the already-audited script; nothing is reimplemented.
from section_replications.modulus_numerics import (
    ALPHA,
    CELLS,
    build_Z,
    config_seed,
    quasi_diff_mat,
    quasi_diff_paths,
    rss_batch,
    simulate_paths,
)

# ---------------------------------------------------------------------------
# 1. Configuration (mirrors modulus_numerics.py; K_loc and C_ALT are new)
# ---------------------------------------------------------------------------
CBAR_GRID = np.round(np.arange(-20.0, -3.0 + 1e-9, 0.25), 2)   # K = [-20, -3]
H_GRID: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
T_GRID: Tuple[int, ...] = (30, 45, 52, 60, 150, 300)

# Displacements the paragraph "The applied span" has to quote. They are read
# DIRECTLY, not rounded up to the next tabulated grid point -- that rounding is
# the arithmetic slip which motivated this rerun.
#   0.26, 0.90 : lambda refinement            (|Delta cbar*|)
#   1.67       : -7 -> -8.67   = cbar*(2,60)
#   1.69, 2.21 : AR order                     (|Delta cbar*|)
#   3.32       : -7 -> -10.32  = cbar*(2,30)
#   3.58       : -7 -> -10.58  = tab:ppp column III, AR(1) MAIC
H_QUERY: Tuple[float, ...] = (0.26, 0.90, 1.67, 1.69, 2.21, 3.32, 3.58)

KLOC_DEFAULT = (-12.0, -5.0)     # calibrated region
C_ALT_DEFAULT = -10.0            # fixed alternative (same as power_cbar_star_vs_c7)
CHUNK = 25_000                   # replications per block (memory control)

FAMILIES = ("diag", "alt")
STATS = ("orc", "fea", "mzt")

# Cell lists. `published` is the list of `modulus_numerics.py`, imported so that
# Table S1 reproduces exactly. `applied` adds the tab:cv configuration
# lambda_j = j/(m+1) for m = 2, which is the cell of Panels A and B of
# `power_cbar_star_vs_c7.py` and is absent from the published list.
CELLS_PUBLISHED: Tuple[Tuple[float, ...], ...] = tuple(tuple(c) for c in CELLS)
CELLS_APPLIED: Tuple[Tuple[float, ...], ...] = CELLS_PUBLISHED + (
    (1.0 / 3.0, 2.0 / 3.0),
)


# ---------------------------------------------------------------------------
# 2. Statistics: oracle and feasible, in the same pass
# ---------------------------------------------------------------------------
def detrended_level(U: np.ndarray, Z: np.ndarray, Zq: np.ndarray,
                    Ginv: np.ndarray, Uq: np.ndarray) -> np.ndarray:
    """GLS-detrended level, per replication (nrep x T).

    betahat = (Zq'Zq)^{-1} Zq' Uq  (OLS on the data quasi-differenced at abar),
    ytil = U - Z betahat.  This is the `detrender' of
    `power_cbar_star_vs_c7.py` written row-wise instead of as a T x T matrix.
    """
    beta = (Uq @ Zq) @ Ginv                      # nrep x (m+1)
    return U - beta @ Z.T                        # nrep x T


def s2_from_level(ytil: np.ndarray) -> np.ndarray:
    """Difference-based (`const') long-run variance, per replication:
    s^2 = sum_{t=2}^{T} (ytil_t - ytil_{t-1})^2 / (T-1).
    """
    T = ytil.shape[1]
    d = np.diff(ytil, axis=1)
    return np.einsum("ij,ij->i", d, d) / (T - 1)


def s2_diff(U: np.ndarray, Z: np.ndarray, Zq: np.ndarray, Ginv: np.ndarray,
            Uq: np.ndarray) -> np.ndarray:
    """Convenience wrapper kept for the selftests: s^2 straight from (U, Z)."""
    return s2_from_level(detrended_level(U, Z, Zq, Ginv, Uq))


def mzt_from_level(ytil: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """Demeaned MZ_t on the GLS-detrended level, per replication.

    MZ_t = MZ_alpha * MSB, with
        Sy2      = T^{-2} sum_{t=1}^{T} ytil_{t-1}^2
        MZ_alpha = (T^{-1} ytil_T^2 - s^2) / (2 Sy2)
        MSB      = (Sy2 / s^2)^{1/2}
    Transcribed from `materiality_c7_vs_surface.py` / `power_cbar_star_vs_c7.py`
    (function `mzt`), transposed to the row-per-replication layout used here.
    The statistic rejects for SMALL values, as the point-optimal one does, so
    the same lower-tail quantile serves as critical value.
    """
    T = ytil.shape[1]
    ss = np.einsum("ij,ij->i", ytil, ytil)
    yT2 = ytil[:, -1] ** 2
    Sy2 = (ss - yT2) / T ** 2
    MZa = (yT2 / T - s2) / (2.0 * Sy2)
    MSB = np.sqrt(Sy2 / s2)
    return MZa * MSB


def stats_all(U: np.ndarray, Z: np.ndarray, cbar: float,
              cache1: Dict[str, np.ndarray] | None = None
              ) -> Dict[str, np.ndarray]:
    """Return the three statistics for each row of U, in one pass.

    orc = S(abar) - abar S(1)                        (sigma^2 = 1 known)
    fea = orc / s^2(cbar)                            (difference-based s^2)
    mzt = demeaned MZ_t on the same GLS-detrended level

    `cache1` stores the a = 1 piece, which does not depend on cbar; passing the
    same dict for every cbar of a slice is a pure saving, with no numerical
    effect whatsoever.
    """
    abar = 1.0 + cbar / U.shape[1]

    Zq_a = quasi_diff_mat(Z, abar)
    Uq_a = quasi_diff_paths(U, abar)
    s_ab = rss_batch(Uq_a, Zq_a)

    if cache1 is not None and "s_1" in cache1:
        s_1 = cache1["s_1"]
    else:
        s_1 = rss_batch(quasi_diff_paths(U, 1.0), quasi_diff_mat(Z, 1.0))
        if cache1 is not None:
            cache1["s_1"] = s_1

    p_orc = s_ab - abar * s_1

    Ginv = np.linalg.inv(Zq_a.T @ Zq_a)
    ytil = detrended_level(U, Z, Zq_a, Ginv, Uq_a)
    s2 = s2_from_level(ytil)
    return {"orc": p_orc,
            "fea": p_orc / s2,
            "mzt": mzt_from_level(ytil, s2)}


def stats_pair(U: np.ndarray, Z: np.ndarray, cbar: float,
               cache1: Dict[str, np.ndarray] | None = None
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(P_oracle, P_feasible); thin wrapper on `stats_all`, used by the tests."""
    d = stats_all(U, Z, cbar, cache1)
    return d["orc"], d["fea"]


# ---------------------------------------------------------------------------
# 3. Power curves per cell
# ---------------------------------------------------------------------------
def power_curves_cell(T: int, lams: Sequence[float], nrep: int, c_alt: float,
                      chunk: int = CHUNK
                      ) -> Tuple[Dict[Tuple[str, str], np.ndarray],
                                 Dict[Tuple[str, str],
                                      Tuple[np.ndarray, np.ndarray]]]:
    """Powers on CBAR_GRID for the four (stat, family) combinations.

    Two passes over the replications:
      pass 1 -- null slice (c = 0), stores the statistics and extracts b;
      pass 2 -- alternative slices (c = cbar on the diagonal; c = c_alt fixed),
                counting rejections against the b of pass 1.
    Configuration seeds: identical to those of `modulus_numerics.py` in the
    "null" and "alt" slices; the "altfix" slice uses a new tag and therefore
    does not perturb reproduction of the already-published columns.

    Also returns, for the NOISE FLOOR diagnostic, the same powers computed
    separately on the two independent halves of the pass-2 replications (the
    critical value b remains the full-sample one).
    """
    Z = build_Z(T, lams)
    ng = len(CBAR_GRID)

    # ---- pass 1: null slice -------------------------------------------------
    rng0 = np.random.default_rng(config_seed(T, lams, "null"))
    null = {s: np.empty((ng, nrep)) for s in STATS}
    done = 0
    while done < nrep:
        n = min(chunk, nrep - done)
        U0 = simulate_paths(rng0.standard_normal((n, T)), 1.0)
        cache: Dict[str, np.ndarray] = {}
        for i, cb in enumerate(CBAR_GRID):
            d0 = stats_all(U0, Z, float(cb), cache)
            for s in STATS:
                null[s][i, done:done + n] = d0[s]
        done += n
    b = {s: np.quantile(null[s], ALPHA, axis=1) for s in STATS}
    del null

    # ---- pass 2: alternative slices -----------------------------------------
    rng1 = np.random.default_rng(config_seed(T, lams, "alt"))
    rng2 = np.random.default_rng(config_seed(T, lams, "altfix"))
    keys = [(s, f) for s in STATS for f in FAMILIES]
    # Independent halves: [0, n1) and [n1, nrep). Because the generator fills in
    # C order, splitting the loop at that index yields exactly the same
    # innovations as a single (nrep x T) draw.
    n1 = nrep // 2
    halves = ((0, n1), (n1, nrep))
    hits_half = [{k: np.zeros(ng, dtype=np.int64) for k in keys},
                 {k: np.zeros(ng, dtype=np.int64) for k in keys}]
    for hidx, (a, z) in enumerate(halves):
        done = a
        while done < z:
            n = min(chunk, z - done)
            E1 = rng1.standard_normal((n, T))
            U2 = simulate_paths(rng2.standard_normal((n, T)), 1.0 + c_alt / T)
            cache2: Dict[str, np.ndarray] = {}
            hh = hits_half[hidx]
            for i, cb in enumerate(CBAR_GRID):
                cbf = float(cb)
                # diagonal family: the DGP tracks cbar
                U1 = simulate_paths(E1, 1.0 + cbf / T)
                d1 = stats_all(U1, Z, cbf)
                # fixed-alternative family: DGP at c_alt, only detrending moves
                d2 = stats_all(U2, Z, cbf, cache2)
                for s in STATS:
                    hh[(s, "diag")][i] += int(
                        np.count_nonzero(d1[s] < b[s][i]))
                    hh[(s, "alt")][i] += int(
                        np.count_nonzero(d2[s] < b[s][i]))
            done += n

    full = {k: (hits_half[0][k] + hits_half[1][k]) / nrep for k in keys}
    n2 = nrep - n1
    split = {k: (hits_half[0][k] / max(n1, 1), hits_half[1][k] / max(n2, 1))
             for k in keys}
    return full, split


# ---------------------------------------------------------------------------
# 4. Empirical modulus, global and local
# ---------------------------------------------------------------------------
def _pair_max(p: np.ndarray, idx: np.ndarray, h: float) -> float:
    """max |p_j - p_i| over pairs of the subset idx with |cbar_j - cbar_i| <= h."""
    g = CBAR_GRID[idx]
    v = p[idx]
    best = 0.0
    for i in range(len(idx)):
        j = np.searchsorted(g, g[i] + h + 1e-9, side="right")
        if j > i + 1:
            seg = v[i:j]
            best = max(best, float(seg.max() - seg.min()))
    return best


def modulus(curves: Dict[Tuple[int, Tuple[float, ...]], np.ndarray],
            T: int, hs: Sequence[float], kloc: Tuple[float, float] | None
            ) -> Dict[float, float]:
    """M_T(h) = max over cells and over admissible pairs of the grid."""
    if kloc is None:
        idx = np.arange(len(CBAR_GRID))
    else:
        idx = np.where((CBAR_GRID >= kloc[0] - 1e-9)
                       & (CBAR_GRID <= kloc[1] + 1e-9))[0]
    out: Dict[float, float] = {}
    for h in hs:
        best = 0.0
        for (TT, _lams), p in curves.items():
            if TT != T:
                continue
            best = max(best, _pair_max(p, idx, h))
        out[h] = best
    return out


# ---------------------------------------------------------------------------
# 5. Selftests
# ---------------------------------------------------------------------------
def selftest() -> int:
    ok = True
    msgs: List[str] = []

    # (i) manual quasi-difference (inherited; re-checked here)
    x = np.array([[1.0, 2.0, 4.0]])
    c1 = np.allclose(quasi_diff_paths(x, 0.5), [[1.0, 1.5, 3.0]])
    ok &= c1
    msgs.append(f"(i)   manual quasi-difference ............ {'OK' if c1 else 'FAIL'}")

    # (ii) Gram of the differenced level design (a = 1) equals I_{m+1}
    Zq = quasi_diff_mat(build_Z(200, (0.3, 0.7)), 1.0)
    c2 = np.allclose(Zq.T @ Zq, np.eye(3), atol=1e-12)
    ok &= c2
    msgs.append(f"(ii)  differenced Gram = I_(m+1) ......... {'OK' if c2 else 'FAIL'}")

    # (iii) EXACT REPRODUCTION of the published block {oracle, diagonal},
    #       against modulus_numerics.power_curve on the same seeds.
    import modulus_numerics as mn
    mn.CBAR_GRID = CBAR_GRID
    T, lams, nrep = 45, (0.5,), 4000
    ref, _ = mn.power_curve(T, lams, nrep)
    mine, _split = power_curves_cell(T, lams, nrep, C_ALT_DEFAULT, chunk=1500)
    d = float(np.max(np.abs(mine[("orc", "diag")] - ref)))
    c3 = d == 0.0
    ok &= c3
    msgs.append(f"(iii) oracle/diagonal vs modulus_numerics: "
                f"max|diff| = {d:.1e} ... {'OK (bit for bit)' if c3 else 'FAIL'}")

    # (iv) difference-based s^2: vectorised implementation vs explicit loop
    rng = np.random.default_rng(config_seed(60, (), "selftest_s2"))
    Tt, m_lams, cb = 60, (0.4,), -7.0
    Zt = build_Z(Tt, m_lams)
    U = simulate_paths(rng.standard_normal((200, Tt)), 1.0)
    abar = 1.0 + cb / Tt
    Zq_a = quasi_diff_mat(Zt, abar)
    Uq_a = quasi_diff_paths(U, abar)
    s2v = s2_diff(U, Zt, Zq_a, np.linalg.inv(Zq_a.T @ Zq_a), Uq_a)
    s2l = np.empty(U.shape[0])
    for r in range(U.shape[0]):
        beta = np.linalg.lstsq(Zq_a, quasi_diff_mat(U[r][:, None], abar),
                               rcond=None)[0]
        yt = U[r] - (Zt @ beta).ravel()
        s2l[r] = np.sum(np.diff(yt) ** 2) / (Tt - 1)
    c4 = bool(np.allclose(s2v, s2l, rtol=1e-10, atol=1e-12))
    ok &= c4
    msgs.append(f"(iv)  s^2 vectorised vs loop ............. "
                f"max|diff| = {np.max(np.abs(s2v - s2l)):.1e} ... "
                f"{'OK' if c4 else 'FAIL'}")

    # (v) consistency of s^2: under c = 0 and i.i.d. N(0,1), E[s^2] -> 1
    #     (O(1/T) bias from detrending)
    U2 = simulate_paths(np.random.default_rng(11).standard_normal((20000, Tt)),
                        1.0)
    Uq2 = quasi_diff_paths(U2, abar)
    m5 = float(np.mean(s2_diff(U2, Zt, Zq_a, np.linalg.inv(Zq_a.T @ Zq_a), Uq2)))
    c5 = abs(m5 - 1.0) < 0.05
    ok &= c5
    msgs.append(f"(v)   E[s^2] under c = 0 ................. "
                f"{m5:.4f} (target 1) ... {'OK' if c5 else 'FAIL'}")

    # (vi) identity P_fea = P_orc / s^2
    po, pf = stats_pair(U, Zt, cb)
    c6 = bool(np.allclose(pf * s2v, po, rtol=1e-12, atol=1e-12))
    ok &= c6
    msgs.append(f"(vi)  P_fea * s^2 == P_orc ............... {'OK' if c6 else 'FAIL'}")

    # (vii) size by construction: 5% at the finite-sample critical value
    small, _sp = power_curves_cell(52, (0.5,), 6000, 0.0, chunk=3000)
    sz = float(np.mean(small[("fea", "alt")]))
    c7 = abs(sz - ALPHA) < 0.006
    ok &= c7
    msgs.append(f"(vii) size of the feasible statistic ..... "
                f"{sz:.4f} (target {ALPHA}) ... {'OK' if c7 else 'FAIL'}")

    # (viii) EXTERNAL CROSS-VALIDATION of the MZ_t branch against Panel B of
    #        `power_cbar_star_vs_c7.py` at nrep = 1,000,000 (out_tighter.csv):
    #        T = 52, m = 2, lambda = (1/3, 2/3), c_alt = -10, size-corrected
    #        power of MZ_t
    #            cbar = -7.00  ->  0.4807
    #            cbar = -13.50 ->  0.3874   (loss 9.32 pp)
    #        The two runs use different seeds, so agreement is only up to MC
    #        error; the tolerance below is about 4 s.e. at this nrep.
    ref_b = {-7.00: 0.480668, -13.50: 0.387421}
    nrep8 = 20000
    pb, _spb = power_curves_cell(52, (1.0 / 3.0, 2.0 / 3.0), nrep8,
                                 C_ALT_DEFAULT, chunk=5000)
    got = {}
    for cb in ref_b:
        i = int(np.argmin(np.abs(CBAR_GRID - cb)))
        got[cb] = float(pb[("mzt", "alt")][i])
    dev = max(abs(got[cb] - ref_b[cb]) for cb in ref_b)
    dloss = abs((got[-7.00] - got[-13.50]) - (ref_b[-7.00] - ref_b[-13.50]))
    c8 = dev < 0.015 and dloss < 0.015
    ok &= c8
    msgs.append(f"(viii) MZ_t vs Panel B (N = 1e6): "
                f"Pi(-7) = {got[-7.00]:.4f} vs {ref_b[-7.00]:.4f}, "
                f"Pi(-13.5) = {got[-13.50]:.4f} vs {ref_b[-13.50]:.4f}, "
                f"loss gap = {dloss:.4f} ... {'OK' if c8 else 'FAIL'}")

    print("\n".join(msgs))
    print(f"\nselftest: {'ALL TESTS PASSED' if ok else '*** FAILURE ***'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------------------
def _table(title: str, mods: Dict[int, Dict[float, float]],
           tgrid: Sequence[int], hs: Sequence[float]) -> None:
    print(f"\n{title}")
    print("T      " + "".join(f"h={h:<7}" for h in hs))
    for T in tgrid:
        print(f"{T:<7}" + "".join(f"{mods[T][h]:<9.4f}" for h in hs))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recompute the empirical modulus in the design of the "
                    "application: feasible statistic, fixed-alternative "
                    "family and calibrated region, alongside the published "
                    "oracle/diagonal/global block.")
    ap.add_argument("--nrep", type=int, default=40000)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--c-alt", type=float, default=C_ALT_DEFAULT,
                    help="fixed alternative of the 'alt' family (default -10)")
    ap.add_argument("--kloc", type=str, default="-12,-5",
                    help="calibrated region for the local modulus "
                         "(default -12,-5)")
    ap.add_argument("--tgrid", type=str, default=None,
                    help="comma-separated list of T; seeds are per "
                         "configuration, so partial runs merged together match "
                         "the joint run exactly")
    ap.add_argument("--cells", choices=("published", "applied"),
                    default="published",
                    help="'published' = the six cells of modulus_numerics.py "
                         "(reproduces Table S1 exactly); 'applied' adds the "
                         "tab:cv cell lambda = (1/3, 2/3) of Panels A and B")
    ap.add_argument("--csv-out", type=str, default=None)
    ap.add_argument("--curves-out", type=str, default=None,
                    help="also write the power curves, cbar by cbar")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    tgrid = (tuple(int(x) for x in args.tgrid.split(","))
             if args.tgrid else T_GRID)
    lo, hi = (float(x) for x in args.kloc.split(","))
    kloc = (min(lo, hi), max(lo, hi))
    hs_all = tuple(sorted(set(H_GRID) | set(H_QUERY)))
    cells = CELLS_APPLIED if args.cells == "applied" else CELLS_PUBLISHED

    print("=" * 78)
    print("RECOMPUTED ENVELOPE -- design of the application")
    print("=" * 78)
    print(f"  cbar grid    K = [{CBAR_GRID[0]}, {CBAR_GRID[-1]}], step 0.25 "
          f"({len(CBAR_GRID)} points)")
    print(f"  local region K_loc = [{kloc[0]}, {kloc[1]}]")
    print(f"  cells        {args.cells}, {len(cells)}: "
          + ", ".join(f"m={len(c)}" for c in cells))
    print(f"  T            {tgrid}")
    print(f"  nrep         {args.nrep:,}   (block {args.chunk:,})")
    print("  statistics   orc = point-optimal, sigma^2 = 1 known")
    print("               fea = P_T = orc / s^2, difference-based (`const')")
    print("               mzt = demeaned MZ_t (the statistic of the paper)")
    print("  families     diag = Pr_{c=cbar}[P < b(cbar)]   (Theorem S2.1)")
    print(f"               alt  = Pr_{{c={args.c_alt}}}[P < b(cbar)]  "
          f"(applied cost)")
    print("=" * 78, flush=True)

    keys = [(s, f) for s in STATS for f in FAMILIES]
    curves: Dict[Tuple[str, str],
                 Dict[Tuple[int, Tuple[float, ...]], np.ndarray]] = {
        k: {} for k in keys}
    noise: Dict[Tuple[str, str], Dict[int, float]] = {
        k: {T: 0.0 for T in tgrid} for k in keys}
    t0 = time.time()
    for T in tgrid:
        for lams in cells:
            c, sp = power_curves_cell(T, lams, args.nrep, args.c_alt,
                                      args.chunk)
            for k, v in c.items():
                curves[k][(T, tuple(lams))] = v
            # Noise floor: same cbar, two independent halves. The between-half
            # difference has twice the variance of a full-sample pair, hence
            # the division by sqrt(2).
            for k, (p1, p2) in sp.items():
                noise[k][T] = max(noise[k][T],
                                  float(np.max(np.abs(p1 - p2))) / np.sqrt(2.0))
            i7 = int(np.argmin(np.abs(CBAR_GRID + 7.0)))
            lamtxt = "(" + ",".join(f"{x:.3f}" for x in lams) + ")"
            print(f"T={T:3d} m={len(lams)} lam={lamtxt}: "
                  f"diag_fea(-7)={c[('fea','diag')][i7]:.4f}  "
                  f"alt_fea(-7)={c[('fea','alt')][i7]:.4f}  "
                  f"alt_mzt(-7)={c[('mzt','alt')][i7]:.4f}  "
                  f"[{time.time()-t0:6.1f}s]", flush=True)

    se = float(np.sqrt(0.25 / args.nrep))
    mods: Dict[Tuple[str, str, str], Dict[int, Dict[float, float]]] = {}
    for s in STATS:
        for f in FAMILIES:
            mods[(s, f, "global")] = {
                T: modulus(curves[(s, f)], T, hs_all, None) for T in tgrid}
            mods[(s, f, "local")] = {
                T: modulus(curves[(s, f)], T, hs_all, kloc) for T in tgrid}

    for s in STATS:
        for f in FAMILIES:
            for sc in ("global", "local"):
                _table(f"M_T(h) -- statistic {s}, family {f}, scope {sc}"
                       + (f" (K_loc = [{kloc[0]},{kloc[1]}])"
                          if sc == "local" else ""),
                       mods[(s, f, sc)], tgrid, H_GRID)

    for s in ("fea", "mzt"):
        print("\n" + "-" * 78)
        print(f"READINGS AT THE APPLIED DISPLACEMENTS -- statistic {s}")
        print("-" * 78)
        print("  h      " + "".join(f"{f}/{sc:<6} " for f in FAMILIES
                                    for sc in ("global", "local")))
        for T in tgrid:
            print(f"T = {T}")
            for h in H_QUERY:
                row = "".join(f"{mods[(s, f, sc)][T][h]:<13.4f}"
                              for f in FAMILIES
                              for sc in ("global", "local"))
                print(f"  {h:<6.2f} {row}")

    print("\n" + "-" * 78)
    print("MONTE CARLO NOISE FLOOR (same cbar, two independent halves)")
    print("-" * 78)
    print("M_T(h) is a MAXIMUM over thousands of pairs: MC error always enters "
          "it with\na positive sign. An M_T(h) of the order of the floor below "
          "does not measure\ncurvature of the power family -- it measures the "
          "simulation itself.")
    print("\nT      " + "".join(f"{s + '/' + f:<13}"
                                for s in STATS for f in FAMILIES))
    for T in tgrid:
        print(f"{T:<7}" + "".join(f"{noise[(s, f)][T]:<13.4f}"
                                  for s in STATS for f in FAMILIES))
    print(f"\nMC standard error of each power <= {se:.5f}; differences use "
          f"common random numbers and are considerably more precise.")
    print("NOTE. The cbar grid has step 0.25, so M_T(h) only changes at "
          "multiples of 0.25:\n      h = 0.26 reads the same pair as h = 0.25. "
          "Use a finer grid if resolution\n      below that is needed.")

    if args.csv_out:
        with open(args.csv_out, "w") as fh:
            fh.write("stat,family,scope,T,h,modulus,nrep,c_alt,kloc_lo,kloc_hi\n")
            for s in STATS:
                for f in FAMILIES:
                    for sc in ("global", "local"):
                        for T in tgrid:
                            for h in hs_all:
                                fh.write(f"{s},{f},{sc},{T},{h},"
                                         f"{mods[(s,f,sc)][T][h]:.6f},"
                                         f"{args.nrep},{args.c_alt},"
                                         f"{kloc[0]},{kloc[1]}\n")
            for s in STATS:
                for f in FAMILIES:
                    for T in tgrid:
                        fh.write(f"{s},{f},noisefloor,{T},0.0,"
                                 f"{noise[(s,f)][T]:.6f},{args.nrep},"
                                 f"{args.c_alt},{kloc[0]},{kloc[1]}\n")
        print(f"\ncsv written to {args.csv_out}")

    if args.curves_out:
        with open(args.curves_out, "w") as fh:
            fh.write("stat,family,T,m,lams,cbar,power\n")
            for s in STATS:
                for f in FAMILIES:
                    for (T, lams), p in curves[(s, f)].items():
                        tag = "|".join(f"{x}" for x in lams)
                        for cb, v in zip(CBAR_GRID, p):
                            fh.write(f"{s},{f},{T},{len(lams)},{tag},{cb},"
                                     f"{v:.6f}\n")
        print(f"curves written to {args.curves_out}")

    if args.latex:
        print("\n% ------------------------------------------------------------")
        print("% Replacement for stab:modulus -- FEASIBLE statistic, two scopes")
        print("% ------------------------------------------------------------")
        print(r"\begin{tabular}{@{}lcccccccc@{}}")
        print(r"\toprule")
        print(r"& \multicolumn{4}{c}{$K=[-20,-3]$}"
              r" & \multicolumn{4}{c}{$K_{\mathrm{loc}}=[-12,-5]$} \\")
        print(r"\cmidrule(lr){2-5}\cmidrule(l){6-9}")
        print(r"$T$ & $0.25$ & $0.5$ & $1$ & $2$ "
              r"& $0.25$ & $0.5$ & $1$ & $2$ \\")
        print(r"\midrule")
        for T in tgrid:
            g = " & ".join(f"{mods[('fea','diag','global')][T][h]:.3f}"
                           for h in H_GRID)
            l = " & ".join(f"{mods[('fea','diag','local')][T][h]:.3f}"
                           for h in H_GRID)
            print(f"{T} & {g} & {l} " + r"\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")

        print("\n% ------------------------------------------------------------")
        print("% Applied cost: MZ_t, fixed-alternative family, both scopes.")
        print("% This is the table the paragraph 'The applied span' needs: the")
        print("% statistic of tab:power/tab:ppp, the family in which the cost")
        print("% is measured, and the region in which cbar is calibrated.")
        print("% ------------------------------------------------------------")
        print(r"\begin{tabular}{@{}lcccccccc@{}}")
        print(r"\toprule")
        print(r"& \multicolumn{4}{c}{$K=[-20,-3]$}"
              r" & \multicolumn{4}{c}{$K_{\mathrm{loc}}=[-12,-5]$} \\")
        print(r"\cmidrule(lr){2-5}\cmidrule(l){6-9}")
        print(r"$T$ & $0.25$ & $0.5$ & $1$ & $2$ "
              r"& $0.25$ & $0.5$ & $1$ & $2$ \\")
        print(r"\midrule")
        for T in tgrid:
            g = " & ".join(f"{mods[('mzt','alt','global')][T][h]:.3f}"
                           for h in H_GRID)
            l = " & ".join(f"{mods[('mzt','alt','local')][T][h]:.3f}"
                           for h in H_GRID)
            print(f"{T} & {g} & {l} " + r"\\")
        print(r"\bottomrule")
        print(r"\end{tabular}")
    return 0


if __name__ == "__main__":
    sys.exit(main())