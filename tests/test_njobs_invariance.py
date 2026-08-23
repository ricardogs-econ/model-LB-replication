"""n_jobs invariance (brief s.0.6 rule 5 / s.3 step 6).

One small grid run at n_jobs=1 and at n_jobs=3 must be BIT-IDENTICAL:
seeds are pure functions of the work item, so scheduling cannot touch the
output. Compares every checkpoint npz array bytewise and the aggregated
cbar_surface.csv textually. A failure here is a real bug in the existing
package - reported, not fixed (RULE N).
"""

from _common import REPO, Report, sha_arr  # noqa: F401

import os
import sys
import tempfile

import numpy as np


def small_P(mlb_core, ckp_dir):
    P = dict(mlb_core.DEFAULTS)
    P.update(
        T_grid=[60],
        m_grid=[1],
        n_grid=3,
        R_cv=150,
        R_pow=150,
        R_curve=0,
        compute_power_curves=False,
        save_raw_vectors=True,
        refine_cheap_configs=False,
        m0_seed_averaging=False,
        checkpoint_dir=ckp_dir,
    )
    return P


def fingerprint(mlb_core, ckp_dir):
    out = {}
    for f in sorted(os.listdir(ckp_dir)):
        p = os.path.join(ckp_dir, f)
        if f.endswith(".npz"):
            with np.load(p) as z:
                out[f] = {k: sha_arr(z[k]) for k in sorted(z.files)}
        elif f == "cbar_surface.csv":
            out[f] = open(p, encoding="utf-8").read()
    return out


def main():
    from src.mlb import kernel_numba as mlb_core

    rep = Report("test_njobs_invariance")
    mlb_core.warm_up_numba(kmax=12)
    work = tempfile.mkdtemp(prefix="mlb_njobs_")
    fps = {}
    for n_jobs in (1, 3):
        ckp = os.path.join(work, f"ckp_j{n_jobs}")
        P = small_P(mlb_core, ckp)
        mlb_core.run_grid(P, n_jobs=n_jobs)
        mlb_core.aggregate(P)
        fps[n_jobs] = fingerprint(mlb_core, ckp)
    a, b = fps[1], fps[3]
    rep.check("same file set", sorted(a) == sorted(b), f"{sorted(a)} vs {sorted(b)}")
    for f in sorted(set(a) & set(b)):
        rep.check(
            f"bit-identical/{f}",
            a[f] == b[f],
            "content differs between n_jobs=1 and n_jobs=3",
        )
    return rep.summary()


if __name__ == "__main__":
    _, n_fail, _ = main()
    sys.exit(1 if n_fail else 0)
