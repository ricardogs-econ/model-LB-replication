"""The Perron-Qu (2007) OLS-GLS convention contract (Phase-5 ruling s.3).

Two assertions, both on the SAME draws:

  (A) the corrected k=0 path equals  sum (dy_GLS)^2 / (T-1)  exactly, where
      dy_GLS is the first difference of the GLS-detrended level residual;
  (B) the corrected kernel reproduces the PRINTED tab:cv cells to the
      rounding envelope  |diff| <= 0.005 + 2*SE.

(B) runs at a reduced R by default so the suite stays inside the session
budget; --full runs it at the author's R = 200,000 over all 54 cells.
Cell values are read from refactor/01_claims_registry.csv (the printed
table, extracted mechanically in Phase 1) -- never from the manuscript.

    python tests\\test_pq_convention.py            # R = 20,000, 9 cells
    python tests\\test_pq_convention.py --full     # R = 200,000, 54 cells
"""

from _common import REPO, Report

import csv
import math
import os
import re
import sys

import numpy as np

CBAR = -7.0
QS = 0.05


def printed_tabcv():
    out = {}
    path = os.path.join(REPO, "refactor", "01_claims_registry.csv")
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"] == "tab:cv" and r["kind"] == "table_cell":
                try:
                    m = int(r["row_key"])
                    T = int(re.findall(r"\d+", r["col_key"])[-1])
                    out[(m, T)] = float(r["value"])
                except (ValueError, IndexError):
                    continue
    return out


def check_k0_identity(rep, mlb_core):
    """(A) k=0 == sum (dy_GLS)^2/(T-1) on the GLS-detrended residual."""
    for (m, T) in [(0, 60), (1, 60), (2, 52), (3, 100), (5, 300)]:
        lam = [j / (m + 1) for j in range(1, m + 1)]
        bp = mlb_core.break_pos_from_lambdas(T, lam)
        Z = mlb_core.build_z_nb(T, bp)
        rng = np.random.default_rng(50820803 + 100 * m + T)
        eps = rng.standard_normal(T)
        y = mlb_core.gen_dgp_nb(T, bp, 0.0, 0.0, eps)
        yt, _ = mlb_core.glsd_nb(y, Z, CBAR)
        d = np.diff(yt)
        # Exact test: the same formula in the same summation order as the
        # kernel (sequential). NumPy's pairwise summation is a different
        # order and legitimately differs in the last ulp, so it is checked
        # separately at 1e-12 rather than bitwise -- this is a statement
        # about float associativity, not a tolerance on the estimator.
        acc = 0.0
        for v in d:
            acc += float(v) * float(v)
        want_seq = acc / (T - 1)
        want_np = float(np.sum(d * d) / (T - 1))
        got = float(mlb_core.s2ar_diff_nb(yt))
        rep.check(
            f"k0_identity/m{m}_T{T}", got == want_seq,
            f"kernel={got!r} sequential-formula={want_seq!r}",
        )
        rep.check(
            f"k0_identity_np/m{m}_T{T}", abs(got - want_np) <= 1e-12,
            f"kernel={got!r} numpy-formula={want_np!r} "
            f"diff={abs(got - want_np):.3e} (float associativity)",
        )
        # and the statistic path must consume exactly that s2
        mza, msb, mzt, pt, mpt, ok = mlb_core.mstats_nb(y, Z, CBAR, 0, 12)
        # T^{-2}, per silva_levelbreaks_jtsa_v2.tex:435-441. (This line read
        # (T-1)^2 when it was written on 2026-08-03, mirroring the kernel
        # defect that the same day's ruling fixed; corrected here so
        # the test encodes the paper's definition, not the old bug.)
        sig = float(np.sum(yt[:-1] ** 2)) / (T ** 2)
        msb_want = math.sqrt(sig / want_np)
        rep.check(
            f"k0_on_path/m{m}_T{T}", abs(msb - msb_want) < 1e-12,
            f"MSB={msb!r} implied={msb_want!r}",
        )


def check_tabcv(rep, mlb_core, R, cells):
    """(B) reproduce the printed tab:cv cells to 0.005 + 2 SE."""
    printed = printed_tabcv()
    P = dict(mlb_core.DEFAULTS)
    worst, n_out = 0.0, 0
    for (m, T) in cells:
        old = printed.get((m, T))
        if old is None:
            continue
        lam = [j / (m + 1) for j in range(1, m + 1)]
        bp = mlb_core.break_pos_from_lambdas(T, lam)
        seed0 = mlb_core.config_seed(P, T, m, lam) + 2_000_000
        out = np.empty(R)
        done = 0
        while done < R:
            b = min(10_000, R - done)
            out[done:done + b] = mlb_core._stats_under(
                P, T, bp, 0.0, CBAR, 0, b, seed0 + done, 12)["mzt"]
            done += b
        a = out[np.isfinite(out)]
        q = float(np.percentile(a, 100 * QS))
        # SE(q) via the Bahadur form with a kernel density estimate at q
        h = 1.06 * float(np.std(a)) * len(a) ** (-0.2)
        f = float(np.mean(np.exp(-0.5 * ((a - q) / h) ** 2))
                  / (h * math.sqrt(2 * math.pi)))
        se = math.sqrt(QS * (1 - QS) / len(a)) / f if f > 0 else float("nan")
        env = 0.005 + 2 * se
        d = abs(q - old)
        worst = max(worst, d)
        n_out += d > env
        rep.check(f"tabcv/m{m}_T{T}", d <= env,
                  f"printed={old:.2f} corrected={q:.4f} diff={d:.4f} "
                  f"env={env:.4f} ({d / se:.1f} SE)")
    print(f"  [tabcv] R={R} cells={len(cells)} worst|diff|={worst:.4f} "
          f"outside envelope={n_out}")


def main(full=False):
    from src.mlb import kernel_numba as mlb_core
    rep = Report("test_pq_convention")
    mlb_core.warm_up_numba(kmax=12)
    check_k0_identity(rep, mlb_core)
    if full:
        R = 200_000
        cells = [(m, T) for m in range(6)
                 for T in (30, 45, 50, 60, 80, 100, 150, 200, 300)]
    else:
        R = 20_000
        cells = [(0, 30), (0, 100), (1, 30), (1, 60), (2, 30), (2, 60),
                 (3, 100), (4, 60), (5, 300)]
    check_tabcv(rep, mlb_core, R, cells)
    return rep.summary()


if __name__ == "__main__":
    _, n_fail, _ = main(full="--full" in sys.argv)
    sys.exit(1 if n_fail else 0)
