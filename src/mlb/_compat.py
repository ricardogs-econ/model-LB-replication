"""Locate the numba kernel. numba-only.

There is NO numpy fallback in this package, by design: a pure-Python fallback
could return a silently different calibration number on the MAIC k<=0 s^2 path
(see README, "Monte Carlo design"). `requirements.txt` pins numba.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def numba_kernel():
    try:
        from src.mlb import kernel_numba
    except ImportError as e:
        raise ImportError(
            "the numba kernel could not be imported and there is no numpy "
            "fallback in this package (by design -- a pure-Python fallback "
            "could return a silently different calibration number). Install "
            "the pinned dependencies: pip install -r requirements.txt"
        ) from e
    return kernel_numba
