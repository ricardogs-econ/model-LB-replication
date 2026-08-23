"""Tangency calibration cbar*(m,T[,p]) (paper sec:calibration; eq:decomp).

Symbols: cbar* is the detrending value at which size-corrected local power
crosses the target 0.50 (the ERS envelope tangency); the paper's operational
result is that cbar* is flat enough at -7 to fix (prop:flat), with the
configuration entering through the critical value instead.

The oracle-PT convention of the Section-6 bootstrap calibration is specified
by the EQUIVALENCE CONTRACT in boot_ppp_cbar.py's module docstring (kernels
imported from src.mlb.kernel_numba, never reimplemented; Option-B/ERS-envelope tangency
on the oracle PT; R_cv, R_pow, kmax conventions) — that contract is the
specification and boot_ppp_cbar.py remains its implementation.

Facade over the production kernel (numerical content unchanged):
  calibrate_config(T, m, lambdas, P), run_grid(P, n_jobs, chunk_size),
  aggregate(P), DEFAULTS
"""

from src.mlb._compat import numba_kernel as _nk

_k = _nk()
DEFAULTS = _k.DEFAULTS
calibrate_config = _k.calibrate_config
run_grid = _k.run_grid
aggregate = _k.aggregate
post_process = _k.post_process
