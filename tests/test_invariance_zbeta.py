"""Theory-required invariance (brief s.3 step 5): adding Z @ beta to y leaves
the GLS-detrended series and the M statistics unchanged.

The invariance is algebraic (Lemma "Exact invariance to break magnitudes"),
so the tolerance is tight: TOL_ZBETA = 1e-13 (brief says ~1e-14; one order of
margin for T=300 float accumulation). Fixed before the first run; if it
fails, the failure is reported, not the tolerance widened (RULE N).
"""

from _common import Report, TOL_ZBETA

import sys

import numpy as np

CONFIGS = [(0, 60), (1, 60), (2, 60), (1, 300)]
CBARS = [-7.0, -21.7]
SEED0 = 745001


def main():
    from src.mlb import kernel_numba as mlb_core

    rep = Report("test_invariance_zbeta")
    mlb_core.warm_up_numba(kmax=12)
    for i, (m, T) in enumerate(CONFIGS):
        lam = [j / (m + 1) for j in range(1, m + 1)]
        bp = mlb_core.break_pos_from_lambdas(T, lam)
        rng = np.random.default_rng(SEED0 + i)
        eps = rng.standard_normal(T)
        y = mlb_core.gen_dgp_nb(T, bp, 0.0, 0.0, eps)
        Z = mlb_core.build_z_nb(T, bp)
        beta = rng.standard_normal(Z.shape[1]) * 5.0
        y2 = y + Z @ beta
        for cbar in CBARS:
            yt2, ssr2 = mlb_core.glsd_nb(y2, Z, cbar)
            yt1, ssr1 = mlb_core.glsd_nb(y, Z, cbar)
            rep.check_close(f"m{m}_T{T}/cbar{cbar}/glsd_yt", yt2, yt1, TOL_ZBETA)
            rep.check_close(f"m{m}_T{T}/cbar{cbar}/glsd_ssr", [ssr2], [ssr1], TOL_ZBETA)
            for name, code in (("const", 0), ("maic", 1)):
                rep.check_close(
                    f"m{m}_T{T}/cbar{cbar}/mstats_{name}",
                    list(mlb_core.mstats_nb(y2, Z, cbar, code, 12)),
                    list(mlb_core.mstats_nb(y, Z, cbar, code, 12)),
                    TOL_ZBETA,
                )
    return rep.summary()


if __name__ == "__main__":
    _, n_fail, _ = main()
    sys.exit(1 if n_fail else 0)
