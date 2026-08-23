"""The single documented kernel binding — numba-only.

This package ships one kernel: the numba production kernel
(`src/mlb/kernel_numba.py`). There is NO pure-Python fallback, by design: a
NumPy fallback could return a silently different calibration number on the
MAIC k<=0 s^2 path (see README, "Monte Carlo design"). If numba is
unavailable, `_compat.numba_kernel()` aborts with a clear message rather than
degrading to an approximate kernel.
"""

from src.mlb import _compat

kernel = _compat.numba_kernel()
KERNEL_SOURCE = f"numba kernel ({kernel.__file__})"
