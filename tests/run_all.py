"""Run the invariance/convention test suite (no pytest dependency).

    python tests\\run_all.py

numba-only package: the tests are the analytic invariances that anchor the
kernel without a second implementation -- theta-invariance, chunk-invariance,
--jobs bit-identity, and the Perron-Qu k=0 convention. The numpy-equivalence
tests (test_parity, test_kernel_equivalence) are NOT part of this package: it
ships one kernel by design.

Exit code 0 iff every test module reports zero failures.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_chunk_invariance  # noqa: E402
import test_invariance_zbeta  # noqa: E402
import test_njobs_invariance  # noqa: E402
import test_pq_convention  # noqa: E402


def main():
    total_fail = 0
    for mod in (
        test_chunk_invariance,
        test_invariance_zbeta,
        test_njobs_invariance,
        test_pq_convention,
    ):
        print(f"==== {mod.__name__} ====")
        _, n_fail, _ = mod.main()
        total_fail += n_fail
    print(f"==== TOTAL failures: {total_fail} ====")
    return total_fail


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
