"""GLS quasi-difference detrending at cbar (paper eq:qd-step, eq:gls-obj).

Symbols: abar = 1 + cbar/T; quasi-differenced regression of y on Z; returns
the LEVEL residual ytilde = y - Z @ thetahat (used by the M statistics) and
the quasi-differenced SSR (used by PT/MPT). Note the deliberate 1e-10 ridge
in the normal equations (kernel source, ex-src/mlb/kernel_numba.py:199): theta-invariance
holds at ~1e-10 * ||beta||, not machine precision.

Facade over the production kernel (numerical content unchanged):
  gls_detrend(y, Z, cbar) -> (ytilde, ssr)   [kernel glsd_nb]
  ols_detrend(y, Z)       -> ytilde_ols      [kernel ols_detrend_nb]
"""

from src.mlb._compat import numba_kernel as _nk

_k = _nk()
gls_detrend = _k.glsd_nb
ols_detrend = _k.ols_detrend_nb
