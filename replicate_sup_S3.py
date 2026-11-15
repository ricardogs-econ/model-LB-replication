#!/usr/bin/env python3
"""Supplement S3: the proofs of Theorem 1 and Proposition 1.

Exhibits: Sections S3.1 to S3.5.

    python replicate_sup_S3.py                  # quick mode: about 1.7 min at 8 workers
    python replicate_sup_S3.py --full           # the warning and the plan; runs nothing
    python replicate_sup_S3.py --full --yes     # regenerates the simulations: about 1.7 min
    python replicate_sup_S3.py --jobs 4         # worker processes (default min(8, CPUs))

The quick mode recomputes the exact results and the small simulations and compares
them with reference/; it shows the long simulations from reference/. --full regenerates
every simulation of the section from the recorded seeds and compares it with
reference/. Each step is defined in mlb/steps.py; mlb/runner.py runs the steps and
compares their files. Everything a run writes goes to output/.
"""
import sys

from mlb import runner

TITLE = "Supplement S3: the proofs of Theorem 1 and Proposition 1"
EXHIBITS = (
    ("S3: (H1)-(H2) on the limit envelope",
     "recomputed exactly (reference/theory/limit_envelope/)"),
    ("S3: Proposition 1 on the exact envelope",
     "recomputed exactly (reference/calibration/propflat_exact/)"),
)
# the steps of each mode, in order (mlb/steps.py defines each step)
QUICK = ("limit_envelope", "propflat")
FULL = ("limit_envelope", "propflat")

if __name__ == "__main__":
    sys.exit(runner.main("replicate_sup_S3.py", TITLE, EXHIBITS, QUICK, FULL, doc=__doc__))
