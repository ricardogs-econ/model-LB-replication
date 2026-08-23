"""
power_cbar_star_vs_c7.py
========================

PURPOSE
-------
Settle, by Monte Carlo, the one claim in the manuscript that is currently
asserted without evidence:

    "once the critical value is finite-sample, a calibrated c*(m,T) and the
     flat -7 deliver the same power"   (Section 1, the operational rule).

`tab:power` as it stands compares -7 against -13.5, -17 and -21.7.  The
SMALLEST over-detrending it tests is 6.5 units away from -7.  The values the
claim actually needs -- c*(2,60) = -8.67, c*(2,50) = -9.0, and the MAIC values
-10.58 / -11.10 / -12.79 that column III of `tab:ppp` uses -- appear in NO
power experiment anywhere in the paper.  Since size-corrected power is
decreasing in over-detrending, the paper currently shows that -21.7 is costly
and *infers* that -9.0 is free.  That is a non sequitur.  This script closes it.

It also supplies, numerically, the analytical argument the manuscript asserts
four times but never states: at the tangency dPi/dcbar = 0, so a displacement
delta costs O(delta^2) of size-corrected power, while the null quantile moves
at O(delta).  The asymmetry of ORDERS is the mathematical reason the same
Delta_T is immaterial on the power axis and material on the size axis.

DESIGN (identical to the kernel behind `tab:power`)
---------------------------------------------------
DGP (Model LB, Lemma 1 invariance => set theta = 0 and simulate u only):
    y_t = Z_t' theta + u_t ,   u_t = (1 + c/T) u_{t-1} + eps_t ,   u_0 = 0
    Z_t = (1, DU_{1,t}, ..., DU_{m,t})' ,  DU_{j,t} = 1{t > tau_j},
    tau_j = floor(lambda_j T),  lambda_j = j/(m+1)      [tab:cv configuration]
    eps_t ~ i.i.d. N(0,1)

Test: demeaned M-class MZ_t on the GLS-detrended level, exact-linear
detrending at abar = 1 + cbar/T, with the DIFFERENCE-BASED s^2 (the k = 0
member of the s^2_AR family).  Serial correlation is deliberately absent:
size correction plus i.i.d. innovations isolates the effect of cbar on
detrending, which is the object under dispute.

Size correction: every cbar is confronted with ITS OWN finite-sample 5%
critical value, obtained from the same null draws.  Differences across rows
are therefore power, not size.

Common random numbers: all cbar values see the SAME innovation draws, under
the null and under the alternative.  This matters.  The claim to be tested is
a NULL-LIKE claim ("no power difference"), so what must be reported is a tight
confidence interval on the DIFFERENCE, not two loose intervals on two levels.
CRN makes the paired standard error roughly an order of magnitude smaller than
the unpaired one; without it, N = 500,000 cannot distinguish a 0.2 pp loss
from zero.

OUTPUT
------
Panel A  (T,m) = (60,2)   -- the cell of `tab:power`
Panel B  (T,m) = (52,2)   -- the application span of `tab:ppp`
Panel C  fine sweep of cbar, size-corrected power + finite-sample 5% cv,
         with a local quadratic fit around the argmax:
             the curvature gives the O(delta^2) power cost,
             the cv slope gives the O(delta) quantile shift.

Everything is deterministic given SEED.  Pure numpy, no third-party deps.

USAGE
-----
    python power_cbar_star_vs_c7.py                 # full run  (N = 500,000)
    python power_cbar_star_vs_c7.py --quick         # smoke test (N = 20,000)
    python power_cbar_star_vs_c7.py --nrep 1000000  # tighter
    python power_cbar_star_vs_c7.py --csv out.csv   # also dump machine-readable

Expected wall time at the default N on a laptop: ~2-5 minutes.
Peak memory: < 1 GB (the simulation is chunked).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field

import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

SEED = 20260727          # deterministic; change only to check seed-sensitivity
NREP_DEFAULT = 500_000
CHUNK = 25_000           # replications per chunk; bounds peak memory
ALPHA = 0.05             # nominal size
C_ALT = -10.0            # local alternative (root ~ 0.83 at T = 60)
CV_ASY_MZT = -1.98       # asymptotic demeaned MZ_t 5% cv (Ng-Perron 2001)

# Panel A: the (T,m) = (60,2) cell of tab:power.
#   -7.00  Model LB, the paper's rule
#   -8.67  cbar*(2,60), the paper's OWN iid-calibrated surface (suppl. Table S1)
#   -9.00  cbar*(2,50), the nearest tabulated cell to the application span
#   -10.32 cbar*(2,30), the most adverse cell in tab:cbar
#   -13.5 / -17.0 / -21.7  the three over-detrending values already in tab:power
PANEL_A_CBARS = [-7.00, -8.67, -9.00, -10.32, -13.50, -17.00, -21.70]

# Panel B: the application span (T,m) = (52,2), with the tab:ppp column III
# values, so that Q3 (rule and application must not be decoupled) is settled
# by the same experiment.
PANEL_B_CBARS = [-7.00, -9.00, -10.58, -11.10, -12.79, -13.50, -21.70]

# Panel C: fine sweep for the tangency-curvature argument.
PANEL_C_CBARS = list(np.round(np.arange(-16.0, -3.99, 0.5), 2))

BASELINE = -7.00         # the reference row against which losses are measured


# ----------------------------------------------------------------------------
# Design matrices and statistic  (verbatim conventions of the tab:power kernel)
# ----------------------------------------------------------------------------

def build_Z(T: int, m: int) -> np.ndarray:
    """Model LB regressors: constant plus m level dummies at lambda_j=j/(m+1).

    DU_{j,t} = 1{t > tau_j} with tau_j = floor(lambda_j * T), t = 1..T.
    Column 0 is the constant; no trend, by construction of Model LB.
    """
    if m < 0:
        raise ValueError("m must be non-negative")
    cols = [np.ones(T)]
    for j in range(1, m + 1):
        lam = j / (m + 1)
        tau = int(np.floor(lam * T))
        if not (0 < tau < T):
            raise ValueError(f"degenerate break date tau={tau} for T={T}, m={m}")
        du = np.zeros(T)
        du[tau:] = 1.0                       # t > tau  (0-indexed: rows tau..T-1)
        cols.append(du)
    return np.column_stack(cols)


def Dmat(T: int, a: float) -> np.ndarray:
    """Quasi-differencing operator (1 - a L), first row left in levels."""
    D = np.eye(T)
    idx = np.arange(1, T)
    D[idx, idx - 1] = -a
    return D


def detrender(T: int, m: int, cbar: float) -> np.ndarray:
    """Fixed T x T matrix M with ytilde = M y (GLS-detrended LEVEL).

    Exact-linear detrending: theta is estimated on the quasi-differenced data
    at abar = 1 + cbar/T, then subtracted from the levels.  By Lemma 1
    (theta-invariance) M Z = 0 in exact arithmetic, which is asserted below.
    """
    a = 1.0 + cbar / T
    Z = build_Z(T, m)
    D = Dmat(T, a)
    Zt = D @ Z
    G = Zt.T @ Zt
    M = np.eye(T) - Z @ np.linalg.solve(G, Zt.T @ D)
    return M


def Lmat(T: int, c: float) -> np.ndarray:
    """Lower-triangular partial-sum operator: u = L(c) eps with u_0 = 0,
    u_t = (1 + c/T) u_{t-1} + eps_t.  L[t, s] = a^(t-s) for s <= t."""
    a = 1.0 + c / T
    L = np.zeros((T, T))
    pw = a ** np.arange(T)
    for t in range(T):
        L[t, : t + 1] = pw[: t + 1][::-1]
    return L


def mzt(Ytil: np.ndarray, T: int) -> np.ndarray:
    """Demeaned MZ_t for each column of the detrended-level array (T x N).

    MZ_t = MZ_alpha * MSB, with
        MZ_alpha = (T^{-1} yt_T^2 - s^2) / (2 T^{-2} sum_{t=1}^{T} yt_{t-1}^2)
        MSB      = ( T^{-2} sum yt_{t-1}^2 / s^2 )^{1/2}
        s^2      = (T-1)^{-1} sum_{t=2}^{T} (yt_t - yt_{t-1})^2   [k = 0]
    """
    ss = np.sum(Ytil * Ytil, axis=0)
    yT2 = Ytil[-1] ** 2
    Sy2 = (ss - yT2) / T ** 2                    # T^{-2} sum_{t=1}^{T} yt_{t-1}^2
    dy = np.diff(Ytil, axis=0)
    s2 = np.sum(dy * dy, axis=0) / (T - 1)
    MZa = (yT2 / T - s2) / (2.0 * Sy2)
    MSB = np.sqrt(Sy2 / s2)
    return MZa * MSB


def check_invariance(T: int, m: int, cbar: float, tol: float = 1e-8) -> float:
    """Lemma 1 check: the detrender must annihilate the Model LB design."""
    M = detrender(T, m, cbar)
    Z = build_Z(T, m)
    err = float(np.max(np.abs(M @ Z)))
    if err > tol:
        raise AssertionError(
            f"theta-invariance violated at (T={T}, m={m}, cbar={cbar}): "
            f"max|M Z| = {err:.3e} > {tol:.1e}"
        )
    return err


# ----------------------------------------------------------------------------
# Simulation engine
# ----------------------------------------------------------------------------

@dataclass
class CellResult:
    """One (T, m) cell: critical values, power, and paired losses vs BASELINE."""
    T: int
    m: int
    cbars: list
    nrep: int
    c_alt: float
    cv_fs: dict = field(default_factory=dict)     # finite-sample ALPHA-level cv
    size_asy: dict = field(default_factory=dict)  # size using CV_ASY_MZT
    power: dict = field(default_factory=dict)     # size-corrected power at c_alt
    loss: dict = field(default_factory=dict)      # power[BASELINE] - power[cb]
    loss_se: dict = field(default_factory=dict)   # PAIRED s.e. of that loss
    power_se: dict = field(default_factory=dict)  # unpaired s.e. of the level


def _null_pass(T, m, cbars, nrep, chunk, rng):
    """Simulate under c = 0 with common random numbers; return the full arrays
    of MZ_t statistics, one per cbar.  Needed in full because we take a
    quantile."""
    Ms = {cb: detrender(T, m, cb) for cb in cbars}
    L0 = Lmat(T, 0.0)
    out = {cb: np.empty(nrep) for cb in cbars}
    done = 0
    while done < nrep:
        n = min(chunk, nrep - done)
        E = rng.standard_normal((T, n))
        U = L0 @ E                                # same u for every cbar (CRN)
        for cb in cbars:
            out[cb][done:done + n] = mzt(Ms[cb] @ U, T)
        done += n
    return Ms, out


def _alt_pass(T, Ms, cbars, cv_fs, nrep, chunk, rng, c_alt, baseline):
    """Simulate under c = c_alt with common random numbers; accumulate paired
    rejection indicators so the DIFFERENCE has a small standard error."""
    L1 = Lmat(T, c_alt)
    rej_sum = {cb: 0.0 for cb in cbars}
    # paired difference moments vs the baseline rule
    d_sum = {cb: 0.0 for cb in cbars}
    d_sq = {cb: 0.0 for cb in cbars}
    done = 0
    while done < nrep:
        n = min(chunk, nrep - done)
        E = rng.standard_normal((T, n))
        U = L1 @ E
        rej = {}
        for cb in cbars:
            rej[cb] = (mzt(Ms[cb] @ U, T) < cv_fs[cb]).astype(np.float64)
            rej_sum[cb] += rej[cb].sum()
        base = rej[baseline]
        for cb in cbars:
            d = base - rej[cb]                    # in {-1, 0, +1}
            d_sum[cb] += d.sum()
            d_sq[cb] += np.dot(d, d)
        done += n
    return rej_sum, d_sum, d_sq


def run_cell(T, m, cbars, nrep, chunk, rng, c_alt=C_ALT,
             alpha=ALPHA, baseline=BASELINE, verbose=True) -> CellResult:
    if baseline not in cbars:
        raise ValueError(f"baseline {baseline} must be among cbars")
    for cb in cbars:
        check_invariance(T, m, cb)

    t0 = time.time()
    Ms, null_stats = _null_pass(T, m, cbars, nrep, chunk, rng)
    res = CellResult(T=T, m=m, cbars=list(cbars), nrep=nrep, c_alt=c_alt)
    for cb in cbars:
        res.cv_fs[cb] = float(np.quantile(null_stats[cb], alpha))
        res.size_asy[cb] = float(np.mean(null_stats[cb] < CV_ASY_MZT))
    del null_stats                                # free before the second pass

    rej_sum, d_sum, d_sq = _alt_pass(
        T, Ms, cbars, res.cv_fs, nrep, chunk, rng, c_alt, baseline)

    for cb in cbars:
        p = rej_sum[cb] / nrep
        res.power[cb] = p
        res.power_se[cb] = float(np.sqrt(max(p * (1 - p), 0.0) / nrep))
        dbar = d_sum[cb] / nrep
        var_d = max(d_sq[cb] / nrep - dbar ** 2, 0.0)
        res.loss[cb] = float(dbar)
        res.loss_se[cb] = float(np.sqrt(var_d / nrep))

    if verbose:
        print(f"    [cell (T={T}, m={m}) done in {time.time() - t0:.1f}s]",
              file=sys.stderr)
    return res


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

LABELS = {
    -7.00:  "Model LB, correct (paper's rule)",
    -8.67:  "cbar*(2,60), paper's own surface",
    -9.00:  "cbar*(2,50), nearest tabulated cell",
    -10.32: "cbar*(2,30), most adverse cell",
    -10.58: "tab:ppp col. III, AR(1) MAIC",
    -11.10: "tab:ppp col. III, AR(2) MAIC",
    -12.79: "tab:ppp col. III, i.i.d.",
    -13.50: "ERS linear-trend value",
    -17.00: "linear break-count scaling",
    -21.70: "reused trend-break surface",
}


def print_cell(res: CellResult, title: str) -> None:
    print()
    print("=" * 96)
    print(title)
    print(f"Model LB, demeaned MZ_t, difference-based s^2, i.i.d. N(0,1), "
          f"T={res.T}, m={res.m}, breaks at lambda_j=j/(m+1)")
    print(f"alternative c={res.c_alt:g}, nominal size {ALPHA:.0%}, "
          f"N={res.nrep:,} (common random numbers across rows)")
    print("-" * 96)
    print(f"{'cbar':>7} {'FS 5% cv':>9} {'size @ asy cv':>14} "
          f"{'power (SC)':>11} {'loss vs -7':>11} {'paired s.e.':>12}   label")
    print("-" * 96)
    for cb in res.cbars:
        tag = LABELS.get(cb, "")
        loss_pp = 100 * res.loss[cb]
        se_pp = 100 * res.loss_se[cb]
        loss_txt = "   ---" if cb == BASELINE else f"{loss_pp:8.2f} pp"
        se_txt = "     ---" if cb == BASELINE else f"{se_pp:9.3f} pp"
        print(f"{cb:7.2f} {res.cv_fs[cb]:9.3f} {100*res.size_asy[cb]:13.1f}% "
              f"{100*res.power[cb]:10.2f}% {loss_txt:>11} {se_txt:>12}   {tag}")
    print("-" * 96)
    print("'size @ asy cv' = null rejection rate when the ASYMPTOTIC cv "
          f"({CV_ASY_MZT}) is used: the size axis.")
    print("'power (SC)'    = rejection rate at the alternative against the row's "
          "OWN finite-sample cv: the power axis.")
    print("'paired s.e.'   = s.e. of the difference under common random numbers; "
          "the relevant precision for a no-difference claim.")

    # The decision the manuscript's rule depends on.
    print()
    for cb in res.cbars:
        if cb == BASELINE:
            continue
        lo = 100 * (res.loss[cb] - 1.96 * res.loss_se[cb])
        hi = 100 * (res.loss[cb] + 1.96 * res.loss_se[cb])
        verdict = ("FLAT   " if hi < 1.0 else
                   "MARGIN " if hi < 2.5 else "COSTLY ")
        print(f"  {verdict} cbar={cb:7.2f}: power loss 95% CI = "
              f"[{lo:+6.2f}, {hi:+6.2f}] pp")
    print("  FLAT   = upper CI bound below 1 pp  -> 'red herring' claim survives at this cbar")
    print("  COSTLY = upper CI bound above 2.5 pp -> the claim fails at this cbar")


def print_curvature(res: CellResult) -> None:
    """Panel C: the O(delta^2) vs O(delta) argument, made numerical."""
    cb = np.array(res.cbars, dtype=float)
    pw = np.array([res.power[c] for c in res.cbars], dtype=float)
    cv = np.array([res.cv_fs[c] for c in res.cbars], dtype=float)

    print()
    print("=" * 96)
    print("PANEL C.  Tangency curvature: why the same Delta_T is immaterial on "
          "the power axis and material on the size axis")
    print(f"Model LB, T={res.T}, m={res.m}, alternative c={res.c_alt:g}, "
          f"N={res.nrep:,}, every cbar size-corrected")
    print("-" * 96)
    print(f"{'cbar':>7} {'power (SC)':>11} {'FS 5% cv':>10}")
    for c, p, q in zip(cb, pw, cv):
        print(f"{c:7.2f} {100*p:10.2f}% {q:10.3f}")
    print("-" * 96)

    # (i) Power axis: local quadratic around the argmax  ->  O(delta^2).
    k = int(np.argmax(pw))
    lo, hi = max(k - 3, 0), min(k + 4, len(cb))
    A = np.vstack([np.ones(hi - lo), cb[lo:hi], cb[lo:hi] ** 2]).T
    coef, *_ = np.linalg.lstsq(A, pw[lo:hi], rcond=None)
    a0, a1, a2 = coef
    cbar_hat = -a1 / (2 * a2) if a2 != 0 else float("nan")
    curv = 2 * a2                                     # d^2 Pi / dcbar^2

    # (ii) Size axis: slope of the finite-sample quantile  ->  O(delta).
    B = np.vstack([np.ones(hi - lo), cb[lo:hi]]).T
    scoef, *_ = np.linalg.lstsq(B, cv[lo:hi], rcond=None)
    slope = scoef[1]                                  # d q_{0.05} / dcbar

    print(f"Local quadratic fit of size-corrected power on cbar over "
          f"[{cb[lo]:.1f}, {cb[hi-1]:.1f}]:")
    print(f"    argmax cbar_hat        = {cbar_hat:8.3f}")
    print(f"    dPi/dcbar at cbar_hat  = {0.0:8.3f}      (zero by construction "
          "of a tangency)")
    print(f"    d2Pi/dcbar^2           = {curv:8.5f}      "
          f"=> power cost of a displacement delta is {abs(curv)/2:.5f} * delta^2")
    print(f"Linear fit of the finite-sample 5% quantile on cbar over the same range:")
    print(f"    dq_(0.05)/dcbar        = {slope:8.5f}      "
          "=> quantile shift is linear in delta")
    print("-" * 96)
    for delta in (1.67, 2.0, 3.32):
        pw_cost = 100 * abs(curv) / 2 * delta ** 2
        q_shift = abs(slope) * delta
        print(f"  delta = {delta:4.2f}:  power cost ~ {pw_cost:5.2f} pp   |   "
              f"quantile shift ~ {q_shift:5.3f} (in MZ_t units)")
    print("  delta = 1.67 is |-8.67 - (-7)|; 3.32 is |-10.32 - (-7)|.")
    print("The orders differ: O(delta^2) against O(delta).  That, and not "
          "simulation alone, is why")
    print("the detrending surface can be discarded while the critical-value "
          "surface cannot.")


def write_csv(path: str, cells: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "T", "m", "nrep", "c_alt", "cbar", "cv_fs",
                    "size_asy", "power_sc", "power_se", "loss_vs_c7",
                    "loss_se_paired"])
        for panel, res in cells.items():
            for cb in res.cbars:
                w.writerow([panel, res.T, res.m, res.nrep, res.c_alt, cb,
                            f"{res.cv_fs[cb]:.6f}", f"{res.size_asy[cb]:.6f}",
                            f"{res.power[cb]:.6f}", f"{res.power_se[cb]:.6f}",
                            f"{res.loss[cb]:.6f}", f"{res.loss_se[cb]:.6f}"])
    print(f"\n[csv written to {path}]", file=sys.stderr)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--nrep", type=int, default=NREP_DEFAULT,
                    help=f"replications per panel (default {NREP_DEFAULT:,})")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test with 20,000 replications")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--csv", type=str, default=None,
                    help="also write a machine-readable copy of Panels A and B")
    ap.add_argument("--skip-curvature", action="store_true",
                    help="skip Panel C (the fine cbar sweep)")
    args = ap.parse_args(argv)

    nrep = 20_000 if args.quick else args.nrep
    if nrep < 1000:
        ap.error("--nrep must be at least 1000")

    rng = np.random.default_rng(args.seed)
    t_start = time.time()

    print("#" * 96)
    print("# Size-corrected power of cbar* against the flat -7 under Model LB")
    print(f"# seed = {args.seed}, N = {nrep:,}, chunk = {args.chunk}, "
          f"numpy {np.__version__}")
    print("#" * 96)

    cells = {}

    cells["A"] = run_cell(60, 2, PANEL_A_CBARS, nrep, args.chunk, rng)
    print_cell(cells["A"],
               "PANEL A.  The tab:power cell -- does the calibrated surface buy "
               "any power over -7?")

    cells["B"] = run_cell(52, 2, PANEL_B_CBARS, nrep, args.chunk, rng)
    print_cell(cells["B"],
               "PANEL B.  The application span (tab:ppp) -- is column III "
               "decoupled from the paper's own rule?")

    if not args.skip_curvature:
        cells["C"] = run_cell(60, 2, PANEL_C_CBARS, nrep, args.chunk, rng)
        print_curvature(cells["C"])

    if args.csv:
        write_csv(args.csv, {k: v for k, v in cells.items() if k in ("A", "B")})

    print()
    print("=" * 96)
    print(f"total wall time: {time.time() - t_start:.1f}s")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())