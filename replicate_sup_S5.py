#!/usr/bin/env python3
"""Supplement S5: the application.

Exhibits: Sections S5.1 to S5.8.

    python replicate_sup_S5.py                  # quick mode: about 4.5 min at 8 workers
    python replicate_sup_S5.py --full           # the warning and the plan; runs nothing
    python replicate_sup_S5.py --full --yes     # regenerates the simulations: about 43.6 min
    python replicate_sup_S5.py --jobs 4         # worker processes (default min(8, CPUs))

The quick mode recomputes the exact results and the small simulations and compares
them with reference/; it shows the long simulations from reference/. --full regenerates
every simulation of the section from the recorded seeds and compares it with
reference/. Each step is defined in mlb/steps.py; mlb/runner.py runs the steps and
compares their files. Everything a run writes goes to output/.
"""
import sys

from mlb import runner

TITLE = "Supplement S5: the application"
EXHIBITS = (
    ("S5.1-S5.3: registration, data, admission, multiplicity",
     "the event dates, the seed registry and the data side recomputed"),
    ("Table S3",
     "rendered from inputs/ppp_sweep/ and reference/application/admission/ "
     "(--full regenerates the simulated critical values and sizes)"),
    ("Tables S4-S6",
     "rendered from reference/application/main/ and "
     "reference/application/crisis_design/ (--full regenerates them)"),
    ("S5.6: the robust level-shift tests",
     "recomputed (reference/application/robust_tests/)"),
    ("Tables S7-S8",
     "the half-lives recomputed (B = 19,999); the tables rendered"),
    ("Table S9",
     "the variance profile recomputed (R = 200,000 per cell, B = 19,999) on "
     "the values of reference/ (reference/application/variance/)"),
)
# the steps of each mode, in order (mlb/steps.py defines each step)
QUICK = ("regime_events", "sj_events", "app_seeds", "app_design", "app_data",
         "sj_design", "sj_data", "half_lives", "robust_tests", "ppp_tables",
         "variance_scope")
FULL = ("regime_events", "sj_events", "app_seeds", "app_design", "app_data",
        "full_app_null", "full_app_bb", "full_app_power", "full_app_joint",
        "full_half_lives", "full_app_sets", "full_app_readings", "sj_design",
        "sj_data", "full_sj_null", "full_sj_hl", "full_sj_sets", "full_sj_readings",
        "full_variance_scope", "robust_tests", "ppp_tables", "full_c6",
        "full_gate_sizes")

if __name__ == "__main__":
    sys.exit(runner.main("replicate_sup_S5.py", TITLE, EXHIBITS, QUICK, FULL, doc=__doc__))
