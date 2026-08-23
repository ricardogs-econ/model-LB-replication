"""Shared plumbing for the parity/equivalence tests.

No pytest dependency (the package ships without it): every test module is a
plain script exposing main() -> (n_pass, n_fail, details) and printing a
PASS/FAIL line per assertion group. tests/run_all.py drives them.

Thread pinning and the numba cache location are set here, BEFORE numpy is
imported anywhere (brief s.0.6 rule 4).
"""

import os

for _v in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "mlb_numba_cache"
    ),
)

import hashlib
import json
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.path.join(REPO, "tests", "golden")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Tolerances (brief s.3 step 3 and step 5). Fixed before the first run and
# never widened after a failure (RULE N / F07).
TOL_PARITY_FLOAT = 1e-12  # golden vs recomputed, unrefactored/refactored
TOL_ZBETA = 1e-13  # algebraic invariance; brief says ~1e-14, one
# order of margin for T=300 accumulation


def load_golden(name):
    with open(os.path.join(GOLD, name), encoding="utf-8") as f:
        return json.load(f)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def max_abs_diff(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    both_nan = np.isnan(a) & np.isnan(b)
    d = np.abs(a - b)
    d[both_nan] = 0.0
    return float(np.nanmax(d)) if d.size else 0.0


class Report:
    def __init__(self, name):
        self.name = name
        self.n_pass = 0
        self.n_fail = 0
        self.details = {}

    def check(self, label, ok, info=""):
        self.details[label] = {"ok": bool(ok), "info": str(info)}
        if ok:
            self.n_pass += 1
        else:
            self.n_fail += 1
            print(f"  FAIL {label}: {info}")

    def check_close(self, label, got, want, tol):
        d = max_abs_diff(got, want)
        self.check(label, d <= tol, f"max_abs_diff={d:.3e} tol={tol:.0e}")
        return d

    def summary(self):
        print(f"[{self.name}] pass={self.n_pass} fail={self.n_fail}")
        return self.n_pass, self.n_fail, self.details
