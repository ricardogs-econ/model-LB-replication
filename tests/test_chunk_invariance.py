"""Chunked vs unchunked accumulation must be BIT-IDENTICAL (brief s.3 step 7;
mandatory because tools/regen_tabcv.py chunks its replication loop).

Safe by construction - each replication draws default_rng(seed0 + r) - and
proved here: one cell at R=5000 computed in one block and in blocks of 700
(a non-divisor, so the last block is ragged) must agree bitwise.
"""

from _common import Report

import sys

import numpy as np


def main():
    from src.mlb import kernel_numba as mlb_core

    rep = Report("test_chunk_invariance")
    mlb_core.warm_up_numba(kmax=12)
    P = dict(mlb_core.DEFAULTS)
    T, m, lam = 60, 2, [1 / 3, 2 / 3]
    bp = mlb_core.break_pos_from_lambdas(T, lam)
    seed0 = mlb_core.config_seed(P, T, m, lam) + 2_000_000
    R, B = 5000, 700
    full = mlb_core._stats_under(P, T, bp, 0.0, -7.0, 0, R, seed0, 12)
    chunked = {k: np.empty(R) for k in full}
    done = 0
    while done < R:
        b = min(B, R - done)
        block = mlb_core._stats_under(P, T, bp, 0.0, -7.0, 0, b, seed0 + done, 12)
        for k in chunked:
            chunked[k][done : done + b] = block[k]
        done += b
    for k in full:
        same = np.array_equal(full[k], chunked[k]) or np.array_equal(
            np.nan_to_num(full[k], nan=-9e99), np.nan_to_num(chunked[k], nan=-9e99)
        )
        rep.check(f"bitwise/{k}", same, "chunked stream differs")
    return rep.summary()


if __name__ == "__main__":
    _, n_fail, _ = main()
    sys.exit(1 if n_fail else 0)
