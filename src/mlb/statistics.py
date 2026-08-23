"""The M-class statistics for Model LB (paper eq:Mstats, eq:PT).

Symbols: MZa, MSB, MZt = MZa*MSB, PT = [S(abar) - abar*S(1)]/s2ar (NP2001
eq.7), MPT demeaned form (NP2001 eq.9, p=0). sigma2 (s2ar) comes from the
OLS-detrended series — invariant to theta — with the 'const'
(difference-based, demeaned, ddof) or 'maic' estimator; see lrv.py and the
kernel-convention table.

Rejection is for SMALL values of every statistic (lower-percentile critical
values).

Facade over the production kernel (numerical content unchanged):
  mstats(y, Z, cbar, sigma2_method, kmax) -> (mza, msb, mzt, pt, mpt, ok)
  METHOD_CODE = {'const': 0, 'maic': 1}
"""

from src.mlb._compat import numba_kernel as _nk

_k = _nk()
mstats = _k.mstats_nb
METHOD_CODE = _k._METHOD_CODE
critical_values_on_the_fly = _k.critical_values_on_the_fly
run_test = _k.run_test
