#!/usr/bin/env python3
"""Supplement S2: invariance and the proof of Lemma 1.

Exhibits: Sections S2.1 and S2.2.

    python replicate_sup_S2.py                  # quick mode: about 3 s at 8 workers
    python replicate_sup_S2.py --full           # the warning and the plan; runs nothing
    python replicate_sup_S2.py --full --yes     # regenerates the simulations: about 3 s
    python replicate_sup_S2.py --jobs 4         # worker processes (default min(8, CPUs))

The quick mode recomputes the exact results and the small simulations and compares
them with reference/; it shows the long simulations from reference/. --full regenerates
every simulation of the section from the recorded seeds and compares it with
reference/. Each step is defined in mlb/steps.py; mlb/runner.py runs the steps and
compares their files. Everything a run writes goes to output/.
"""
import sys

from mlb import runner

TITLE = "Supplement S2: invariance and the proof of Lemma 1"
EXHIBITS = (
    ("S2.2: Lemma 1 on 3,004 designs",
     "the exact Gram entries and bounds recomputed in rational arithmetic "
     "(reference/theory/lemma1_bounds/)"),
)
# the steps of each mode, in order (mlb/steps.py defines each step)
QUICK = ("lemma1_bounds",)
FULL = ("lemma1_bounds",)

if __name__ == "__main__":
    sys.exit(runner.main("replicate_sup_S2.py", TITLE, EXHIBITS, QUICK, FULL, doc=__doc__))
