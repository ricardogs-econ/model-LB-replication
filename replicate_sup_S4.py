#!/usr/bin/env python3
"""Supplement S4: numerical material and Monte Carlo errors.

Exhibits: Sections S4.1 to S4.3.

    python replicate_sup_S4.py                  # quick mode: about 2.5 min at 8 workers
    python replicate_sup_S4.py --full           # the warning and the plan; runs nothing
    python replicate_sup_S4.py --full --yes     # regenerates the simulations: about 2.6 h
    python replicate_sup_S4.py --jobs 4         # worker processes (default min(8, CPUs))

The quick mode recomputes the exact results and the small simulations and compares
them with reference/; it shows the long simulations from reference/. --full regenerates
every simulation of the section from the recorded seeds and compares it with
reference/. Each step is defined in mlb/steps.py; mlb/runner.py runs the steps and
compares their files. Everything a run writes goes to output/.
"""
import sys

from mlb import runner

TITLE = "Supplement S4: numerical material and Monte Carlo errors"
EXHIBITS = (
    ("S4.1: the calibration grid and its trimming",
     "recomputed exactly (reference/calibration/cbar_surface_exact/, "
     "reference/calibration/surface_summaries/)"),
    ("S4.2: trend-calibrated values of c-bar (CKP 2009)",
     "the response surface re-evaluated "
     "(reference/calibration/ckp_table1_grid.json)"),
    ("Table S1",
     "rendered from "
     "reference/serial_dependence/ar1_size_power/ar1_size_power.csv (--full "
     "regenerates it)"),
    ("Table S2",
     "recomputed exactly (reference/serial_dependence/ar1_recalibration/); "
     "the table rendered"),
)
# the steps of each mode, in order (mlb/steps.py defines each step)
QUICK = ("exact_surface", "surface_summaries", "tabcv_from_reference", "ckp_table1",
         "ar1_recalibration", "prose_grid", "fixed_index", "table_s1_reference")
FULL = ("exact_surface", "surface_summaries", "full_tabcv_blocks",
        "full_tabcv_table", "ckp_table1", "full_overdetrend", "full_ar1_size_power",
        "ar1_recalibration", "prose_grid", "fixed_index")

if __name__ == "__main__":
    sys.exit(runner.main("replicate_sup_S4.py", TITLE, EXHIBITS, QUICK, FULL, doc=__doc__))
