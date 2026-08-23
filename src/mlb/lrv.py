"""Long-run variance estimators (paper eq:s2ar; sec:lags).

Symbols: s2ar. Two members:
  'const' (code 0): difference-based, k=0 — sample variance of the demeaned
      first difference of the OLS-detrended series (ddof; ex-src/mlb/kernel_numba.py:
      253-272). Consistent under i.i.d.; INCONSISTENT under serial
      correlation (plim sigma^2/(1-rho^2) vs omega^2 = sigma^2/(1-rho)^2).
      The headline estimator of the paper's tables.
  'maic' (code 1): autoregressive spectral estimator with MAIC lag selection
      (Ng-Perron 2001), on the OLS-detrended series; consistent under serial
      correlation, unstable at very short T (ex-src/mlb/kernel_numba.py:275+).

Facade over the production kernel (numerical content unchanged):
  s2_const(yt_ols)        -> kernel s2ar_const_nb
  s2_maic(yt_ols, kmax)   -> kernel s2ar_maic_nb
"""

from src.mlb._compat import numba_kernel as _nk

_k = _nk()
s2_const = _k.s2ar_const_nb
s2_maic = _k.s2ar_maic_nb
