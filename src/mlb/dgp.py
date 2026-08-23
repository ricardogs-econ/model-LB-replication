"""The Model LB data-generating process (paper sec:setup; Lemma
lem:invariance).

Symbols: y_t = Z_t' theta + u_t, u_t = (1 + c/T) u_{t-1} + eps_t, u_0 = 0;
c = 0 is the null (driftless random walk), c < 0 the local alternative.
beta_scale sets ||theta|| for invariance checks — by Lemma 1 the statistics
are invariant to it (at the ridge-bounded precision).
Breaks are KNOWN and IMPOSED; no estimator exists or is provided.

Facade over the production kernel (numerical content unchanged):
  gen_dgp(nt, break_pos, c, beta_scale, eps) -> y   [kernel gen_dgp_nb]
"""

from src.mlb._compat import numba_kernel as _nk

_k = _nk()
gen_dgp = _k.gen_dgp_nb
