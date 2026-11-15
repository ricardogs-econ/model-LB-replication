#!/usr/bin/env python3
"""Sections 2-3 of the article: the model, the half-power point and the theory.

Exhibits: Sections 2 and 3, and their restatements in the introduction.

    python replicate_sec_2_3.py                  # quick mode: about 81 s at 8 workers
    python replicate_sec_2_3.py --full           # the warning and the plan; runs nothing
    python replicate_sec_2_3.py --full --yes     # regenerates the simulations: about 81 s
    python replicate_sec_2_3.py --jobs 4         # worker processes (default min(8, CPUs))

The quick mode recomputes the exact results and the small simulations and compares
them with reference/; it shows the long simulations from reference/. --full regenerates
every simulation of the section from the recorded seeds and compares it with
reference/. Each step is defined in mlb/steps.py; mlb/runner.py runs the steps and
compares their files. Everything a run writes goes to output/.
"""
import sys

from mlb import runner

TITLE = "Sections 2-3 of the article: the model, the half-power point and the theory"
EXHIBITS = (
    ("Section 2: c-bar_inf = -7.01, Pi_inf(-7) = 0.499, the slope -0.09",
     "the limit envelope, recomputed exactly "
     "(reference/theory/limit_envelope/)"),
    ("Section 2: strict decrease of the envelope on K = [-20, -3]",
     "the verdicts of reference/calibration/envelope_monotonicity/ "
     "(replicate_sec_4.py --full recomputes them)"),
    ("Section 2: the five M-class statistics, the critical values",
     "Table 2's block means rebuilt from "
     "reference/calibration/critical_values_blocks/ (replicate_sec_4.py "
     "--full regenerates the blocks)"),
    ("Lemma 1",
     "the exact Gram entries and bounds on 3,004 designs, recomputed in "
     "rational arithmetic (reference/theory/lemma1_bounds/)"),
)
# the steps of each mode, in order (mlb/steps.py defines each step)
QUICK = ("limit_envelope", "lemma1_bounds", "monotonicity_reference",
         "tabcv_from_reference")
FULL = ("limit_envelope", "lemma1_bounds", "monotonicity_reference",
        "tabcv_from_reference")

if __name__ == "__main__":
    sys.exit(runner.main("replicate_sec_2_3.py", TITLE, EXHIBITS, QUICK, FULL, doc=__doc__))
