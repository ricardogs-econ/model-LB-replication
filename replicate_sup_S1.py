#!/usr/bin/env python3
"""Supplement S1: notation, conventions and the limit envelope.

Exhibits: Sections S1.1 and S1.2.

    python replicate_sup_S1.py                  # quick mode: about 77 s at 8 workers
    python replicate_sup_S1.py --full           # the warning and the plan; runs nothing
    python replicate_sup_S1.py --full --yes     # regenerates the simulations: about 77 s
    python replicate_sup_S1.py --jobs 4         # worker processes (default min(8, CPUs))

The quick mode recomputes the exact results and the small simulations and compares
them with reference/; it shows the long simulations from reference/. --full regenerates
every simulation of the section from the recorded seeds and compares it with
reference/. Each step is defined in mlb/steps.py; mlb/runner.py runs the steps and
compares their files. Everything a run writes goes to output/.
"""
import sys

from mlb import runner

TITLE = "Supplement S1: notation, conventions and the limit envelope"
EXHIBITS = (
    ("S1.2: the limit envelope, c-bar_inf = -7.01",
     "recomputed exactly (reference/theory/limit_envelope/)"),
)
# the steps of each mode, in order (mlb/steps.py defines each step)
QUICK = ("limit_envelope",)
FULL = ("limit_envelope",)

if __name__ == "__main__":
    sys.exit(runner.main("replicate_sup_S1.py", TITLE, EXHIBITS, QUICK, FULL, doc=__doc__))
