"""The power of the over-detrending shortcuts at high precision: the data of Figure 2
and of the power losses printed in Section 4.4.

The defaults are the run that produced reference/calibration/overdetrend_power.json
(--seed 520260724 --n-null 320000000 --n-printed 80000000 --n-figure 2000000):

    python mlb/overdetrend_power_precise.py --jobs 8

428,000,000 replications, about 1,110 CPU-minutes (divide by --jobs).

The cbars are -7, -13.5 and the CKP -21.7. The -21.7 is round(cbar_ckp([20, 40],
60), 1) from mlb/ckp_cbar_surface.py (the CKP 2009 response
surface), checked equal (SystemExit, not an assert) to the -21.7 of that run.
Every cbar is evaluated on the same draws, and no cbar enters the draws or another
cbar's statistic.

THE STANDARD ERRORS
-------------------
The losses are printed to 0.1 pp. Each power cell is size-corrected against a
critical value that is itself estimated from the null run, and the error of that
estimate propagates into every cell through the density of the alternative at the
critical value:

    Var(power) = p (1 - p) / N_alt + f_alt(cv)^2 Var(cv),
    Var(cv)    = alpha (1 - alpha) / (N_null f_null(cv)^2);

a loss (two cbars on the same draws) has the paired binomial term and the two
critical-value terms. The critical values carry most of the variance of a loss,
so the null run is the largest:

    null            320,000,000     the finite-sample 5% critical values
    c = -10          80,000,000     the printed alternative
    14 figure points  2,000,000     each, the curve of Figure 2

On this run, at c = -10, se(power) is 0.0071 to 0.0089 pp (its binomial part 0.0052
to 0.0056 pp) and se(loss) is 0.0100 pp for both losses, of which the critical
values give 72% and 82% of the variance (by_c['-10.0'] of overdetrend_power.json).

THE DESIGN
----------

  * Model LB DGP, demeaned MZ_t, i.i.d. N(0,1), T = 60, m = 2,
    breaks at lambda_j = j/(m+1)
  * cbars = [-7.0, -13.5, -21.7]
  * size correction: each cbar against its OWN finite-sample 5% critical value,
    estimated under the null c = 0
  * the printed alternative is c = -10
  * the kernel: mstats_nb (code 0), replication r of a stream seeded
    default_rng(seed_base + r); the streams of the alternatives are spaced by
    SEED_STRIDE (below)
  * the cbars are evaluated on ONE draw per replication. This equals one
    kernel_numba._stats_under call per cbar, which seeds replication r as
    default_rng(seed0 + r) independently of cbar; `--verify` asserts
    bit-identity. The shared draws make the paired standard error of the losses
    computable.
  * the printed alternative and the figure grid are one run, so the printed losses
    and the plotted curve describe the same draws.
  * workers return COUNTS, never samples (the null sample of this run, 320,000,000
    draws of three statistics, would take about 7.7 GB); the critical values are
    recovered from a fine histogram over a window around the quantile (bin width
    5e-5, a discretization of about 2.5e-5 against a sampling error se(cv) of 0.7e-4
    to 0.9e-4, so the binning is not the binding constraint).

    python mlb/overdetrend_power_precise.py --verify
    python mlb/overdetrend_power_precise.py --jobs 8 --out <file.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_pp = os.environ.get("PYTHONPATH", "")
if _ROOT not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _ROOT + (os.pathsep + _pp if _pp else "")

from mlb.ckp_cbar_surface import cbar_ckp  # noqa: E402

SEED = 520260724
T, M = 60, 2
# The reused trend-break value is CKP's own response surface at this cell
# (ckp_cbar_surface.py; msbur.src c_bar_rs), not a literal; the reference
# overdetrend_power.json was simulated at -21.7, and the check keeps the two tied.
CBAR_CKP = round(cbar_ckp([20, 40], T), 1)
if CBAR_CKP != -21.7:          # not an assert: python -O must not strip the tie
    raise SystemExit("overdetrend_power.json was simulated at cbar = -21.7, the CKP "
                     f"surface gives {CBAR_CKP}")
CBARS = [-7.0, -13.5, CBAR_CKP]
REF = -7.0
KMAX = 12  # the lag bound mstats_nb takes; the const arm (code 0) does not read it
ALPHA = 0.05
C_PRINTED = -10.0

# the alternatives of Figure 2: c = -2.5 k (k = 0..10) and -7, -13.5, -17, -21.7
C_GRID = [-0.0, -2.5, -5.0, -7.0, -7.5, -10.0, -12.5, -13.5, -15.0, -17.0, -17.5, -20.0,
          -21.7, -22.5, -25.0]

# The replications of the run of reference/calibration/overdetrend_power.json (see the
# module docstring).
N_NULL = 320_000_000
N_PRINTED = 80_000_000
N_FIGURE = 2_000_000

CHUNK = 250_000            # replications per task: bounds worker memory
HIST_HALFWIDTH = 0.15      # window around the approximate cv
HIST_BINS = 6000           # -> bin width 5e-5

# SEED STRIDE. Replication r of a stream draws default_rng((seed_base + r) %
# (2**63-1)), so two streams whose [seed_base, seed_base + N) ranges intersect share
# innovation draws, and the standard errors assume independent streams. A spacing of
# 10**9, above the replications of every stream, keeps the streams of the run
# disjoint (main() checks it).
SEED_STRIDE = 10 ** 9


def _draw_mzt(seed_base, r0, r1, c, break_pos):
    """mzt for replications [r0, r1) at alternative c, one row per cbar.

    The body mirrors kn._stats_under exactly -- same seeding, same `ok`
    handling -- but evaluates every cbar on the single (eps, y) draw of each
    replication.
    """
    from mlb import kernel_numba as kn
    n = r1 - r0
    out = np.full((len(CBARS), n), np.nan)
    for j in range(n):
        rng = np.random.default_rng((seed_base + r0 + j) % (2 ** 63 - 1))
        eps = rng.standard_normal(T)
        y = kn.gen_dgp_nb(T, c, eps)
        Z = kn.build_z_nb(T, break_pos)
        for i, cb in enumerate(CBARS):
            _, _, mzt, _, _, ok = kn.mstats_nb(y, Z, cb, 0, KMAX)
            if ok > 0.5:
                out[i, j] = mzt
    return out


def _task_hist(seed_base, r0, r1, c, break_pos, edges):
    """Null task: histogram of mzt per cbar over `edges`, plus tail counts."""
    s = _draw_mzt(seed_base, r0, r1, c, break_pos)
    hists, below, n_ok = [], [], []
    for i in range(len(CBARS)):
        v = s[i][~np.isnan(s[i])]
        n_ok.append(int(v.size))
        below.append(int(np.count_nonzero(v < edges[i][0])))
        hists.append(np.histogram(v, bins=edges[i])[0])
    return np.array(n_ok), np.array(below), np.stack(hists)


def _task_count(seed_base, r0, r1, c, break_pos, cv, h):
    """Alternative task: rejection counts, joint counts, density counts."""
    s = _draw_mzt(seed_base, r0, r1, c, break_pos)
    k = len(CBARS)
    n_ok = np.zeros(k, dtype=np.int64)
    rej = np.zeros(k, dtype=np.int64)
    lo = np.zeros(k, dtype=np.int64)
    hi = np.zeros(k, dtype=np.int64)
    joint = np.zeros(k, dtype=np.int64)
    iref = CBARS.index(REF)
    good = [~np.isnan(s[i]) for i in range(k)]
    ind = [(s[i] < cv[i]) & good[i] for i in range(k)]
    for i in range(k):
        n_ok[i] = int(np.count_nonzero(good[i]))
        rej[i] = int(np.count_nonzero(ind[i]))
        lo[i] = int(np.count_nonzero((s[i] < cv[i] - h[i]) & good[i]))
        hi[i] = int(np.count_nonzero((s[i] < cv[i] + h[i]) & good[i]))
        joint[i] = int(np.count_nonzero(ind[i] & ind[iref]))
    return n_ok, rej, lo, hi, joint


def _map(fn, seed_base, nrep, jobs, args):
    """Run fn over r-chunks of [0, nrep), returning the list of results."""
    spans = [(a, min(a + CHUNK, nrep)) for a in range(0, nrep, CHUNK)]
    if jobs == 1:
        return [fn(seed_base, a, b, *args) for a, b in spans]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
        delayed(fn)(seed_base, a, b, *args) for a, b in spans)


def _quantile_from_hist(edges, hist, below, n, alpha):
    """alpha-quantile by linear interpolation inside the containing bin."""
    target = alpha * n
    cum = below + np.cumsum(hist)
    j = int(np.searchsorted(cum, target))
    if j >= len(hist):
        raise SystemExit("quantile fell outside the histogram window")
    lo_count = below if j == 0 else cum[j - 1]
    frac = (target - lo_count) / hist[j] if hist[j] else 0.5
    return float(edges[j] + frac * (edges[j + 1] - edges[j]))


def _density_from_hist(edges, hist, n, x, halfwidth):
    """f(x) as the histogram mass within +/- halfwidth of x, over 2*halfwidth."""
    lo = np.searchsorted(edges, x - halfwidth) - 1
    hi = np.searchsorted(edges, x + halfwidth) - 1
    lo, hi = max(lo, 0), min(hi, len(hist))
    mass = float(hist[lo:hi].sum())
    span = float(edges[min(hi, len(hist))] - edges[lo])
    return mass / (span * n) if span > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--n-null", type=int, default=N_NULL)
    ap.add_argument("--n-printed", type=int, default=N_PRINTED)
    ap.add_argument("--n-figure", type=int, default=N_FIGURE)
    from mlb import paths as _paths
    ap.add_argument("--out", default=str(_paths.out_dir("critical_values")
                                         / "overdetrend_power.json"))
    ap.add_argument("--verify", action="store_true",
                    help="assert bit-parity against kn._stats_under, then exit")
    a = ap.parse_args()

    from mlb import kernel_numba as kn
    kn.warm_up_numba(kmax=KMAX)
    lam = [j / (M + 1) for j in range(1, M + 1)]
    bp = kn.break_pos_from_lambdas(T, lam)

    if a.verify:
        n = 2000
        mine = _draw_mzt(a.seed, 0, n, -10.0, bp)
        for i, cb in enumerate(CBARS):
            theirs = kn._stats_under(T, bp, -10.0, cb, 0, n,
                                     a.seed, KMAX)["mzt"]
            d = float(np.nanmax(np.abs(mine[i] - theirs)))
            same = np.array_equal(np.isnan(mine[i]), np.isnan(theirs))
            print(f"  cbar={cb:>6}: max|diff|={d:.3e}  nan-pattern-identical={same}")
            assert d == 0.0 and same, "NOT bit-identical to _stats_under"
        split = np.concatenate([_draw_mzt(a.seed, 0, 700, -10.0, bp),
                                _draw_mzt(a.seed, 700, n, -10.0, bp)], axis=1)
        assert np.array_equal(np.isnan(split), np.isnan(mine))
        assert float(np.nanmax(np.abs(split - mine))) == 0.0
        print("  [OK] bit-identical to kn._stats_under, and invariant to "
              "the chunk split the parallel map uses")
        return 0

    t0 = time.time()
    total = a.n_null + a.n_printed + a.n_figure * (len(C_GRID) - 1)
    print(f"overdetrend power  null={a.n_null:,}  c=-10 {a.n_printed:,}  "
          f"figure {a.n_figure:,} x {len(C_GRID)-1}  total {total:,} reps  "
          f"jobs={a.jobs}", flush=True)

    # --- the seed streams must be provably disjoint -----------------------
    streams = [("null", a.seed, a.n_null)]
    for idx, c in enumerate(C_GRID):
        streams.append((f"c={c}", a.seed + (idx + 1) * SEED_STRIDE,
                        a.n_printed if c == C_PRINTED else a.n_figure))
    streams.sort(key=lambda s: s[1])
    for (n1, s1, k1), (n2, s2, _) in zip(streams[:-1], streams[1:]):
        if s1 + k1 > s2:
            raise SystemExit(
                f"[streams] seed streams overlap: {n1} spans "
                f"[{s1}, {s1+k1}) and {n2} starts at {s2}. Raise SEED_STRIDE "
                f"above the largest per-stream replication count.")
    slack = min(s2 - (s1 + k1)
                for (_, s1, k1), (_, s2, _) in zip(streams[:-1], streams[1:]))
    print(f"  [streams] {len(streams)} seed streams disjoint, "
          f"stride={SEED_STRIDE:,}, minimum slack {slack:,} replications",
          flush=True)

    # --- locate the critical values approximately, cheaply -----------------
    pilot = _draw_mzt(a.seed, 0, 200_000, 0.0, bp)
    cv0 = [float(np.nanquantile(pilot[i], ALPHA)) for i in range(len(CBARS))]
    edges = [np.linspace(c - HIST_HALFWIDTH, c + HIST_HALFWIDTH, HIST_BINS + 1)
             for c in cv0]
    print(f"  pilot cv0 = {[round(c,4) for c in cv0]}  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # --- the null run ------------------------------------------------------
    parts = _map(_task_hist, a.seed, a.n_null, a.jobs, (0.0, bp, edges))
    n_ok = np.sum([p[0] for p in parts], axis=0)
    below = np.sum([p[1] for p in parts], axis=0)
    hists = np.sum([p[2] for p in parts], axis=0)
    cv, se_cv, f_null, hband = {}, {}, {}, {}
    for i, cb in enumerate(CBARS):
        q = _quantile_from_hist(edges[i], hists[i], below[i], n_ok[i], ALPHA)
        # bandwidth for the density: wide enough for a stable count, narrow
        # enough that the density is locally flat
        hb = 0.05
        f0 = _density_from_hist(edges[i], hists[i], n_ok[i], q, hb)
        cv[cb], f_null[cb], hband[cb] = q, f0, hb
        se_cv[cb] = math.sqrt(ALPHA * (1 - ALPHA) / (int(n_ok[i]) * f0 ** 2))
        print(f"  null cbar={cb:>6}: cv={q:.6f}  f_null={f0:.5f}  "
              f"se(cv)={se_cv[cb]:.3e}  n_ok={int(n_ok[i]):,}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
    var_cv = {cb: se_cv[cb] ** 2 for cb in CBARS}
    cvv = [cv[cb] for cb in CBARS]
    hv = [hband[cb] for cb in CBARS]

    # --- the alternatives --------------------------------------------------
    rows = {}
    for idx, c in enumerate(C_GRID):
        n_c = a.n_printed if c == C_PRINTED else a.n_figure
        seed_c = a.seed + (idx + 1) * SEED_STRIDE
        parts = _map(_task_count, seed_c, n_c, a.jobs, (c, bp, cvv, hv))
        nk = np.sum([p[0] for p in parts], axis=0)
        rj = np.sum([p[1] for p in parts], axis=0)
        lo = np.sum([p[2] for p in parts], axis=0)
        hi = np.sum([p[3] for p in parts], axis=0)
        jt = np.sum([p[4] for p in parts], axis=0)
        ent = {"c": c, "seed": seed_c, "nrep": n_c, "cells": {}}
        for i, cb in enumerate(CBARS):
            n = int(nk[i])
            p = rj[i] / n
            f1 = (hi[i] - lo[i]) / (2 * hband[cb] * n)
            v_binom = p * (1 - p) / n
            v = v_binom + f1 ** 2 * var_cv[cb]
            ent["cells"][str(cb)] = {
                "power": float(p), "n_ok": n, "f_alt_at_cv": float(f1),
                "se_binomial": math.sqrt(v_binom), "se_total": math.sqrt(v),
            }
        pr = ent["cells"][str(REF)]["power"]
        nr = ent["cells"][str(REF)]["n_ok"]
        fr = ent["cells"][str(REF)]["f_alt_at_cv"]
        for i, cb in enumerate(CBARS):
            if cb == REF:
                continue
            pc = ent["cells"][str(cb)]["power"]
            both = float(jt[i]) / nr
            cov = (both - pr * pc) / nr
            fc = ent["cells"][str(cb)]["f_alt_at_cv"]
            v = (pr * (1 - pr) / nr + pc * (1 - pc) / nr - 2 * cov
                 + fr ** 2 * var_cv[REF] + fc ** 2 * var_cv[cb])
            ent["cells"][str(cb)]["loss_vs_ref"] = float(pr - pc)
            ent["cells"][str(cb)]["p_both_reject"] = both
            ent["cells"][str(cb)]["se_loss"] = math.sqrt(max(v, 0.0))
        rows[str(c)] = ent
        line = "  ".join(f"{cb}:{100*ent['cells'][str(cb)]['power']:8.5f}"
                         for cb in CBARS)
        print(f"  c={c:>6} N={n_c:>10,}: {line}   [{time.time()-t0:.0f}s]",
              flush=True)

    res = {
        "producer": "mlb/overdetrend_power_precise.py",
        "kernel": "mlb/kernel_numba.py",
        "n_null": a.n_null, "n_printed": a.n_printed, "n_figure": a.n_figure,
        "total_replications": total,
        "seed": a.seed, "seed_stride": SEED_STRIDE,
        "T": T, "m": M, "alpha": ALPHA,
        "cbars": CBARS, "reference_cbar": REF, "grid": C_GRID,
        "printed_alternative": C_PRINTED,
        "cv": {str(k): v for k, v in cv.items()},
        "se_cv": {str(k): se_cv[k] for k in cv},
        "f_null_at_cv": {str(k): f_null[k] for k in cv},
        "by_c": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"\nwrote {a.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
