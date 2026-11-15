#!/usr/bin/env python3
"""Section 6 of the article: the purchasing power parity puzzle.

Exhibits: Section 6 and its restatements in the abstract, the introduction, the
conclusion and the data statement.

    python replicate_sec_6.py                  # quick mode: about 2.4 min at 8 workers
    python replicate_sec_6.py --full           # the warning and the plan; runs nothing
    python replicate_sec_6.py --full --yes     # regenerates the simulations: about 39.5 min
    python replicate_sec_6.py --jobs 4         # worker processes (default min(8, CPUs))

The quick mode recomputes the exact results and the small simulations and compares
them with reference/; it shows the long simulations from reference/. --full regenerates
every simulation of the section from the recorded seeds and compares it with
reference/. Each step is defined in mlb/steps.py; mlb/runner.py runs the steps and
compares their files. Everything a run writes goes to output/.
"""
import sys

from mlb import runner

TITLE = "Section 6 of the article: the purchasing power parity puzzle"
EXHIBITS = (
    ("Figure 3",
     "the nine series with the break dates of design E and the regime means, "
     "and the facts of Section 6.1, recomputed from inputs/ppp_panel.csv "
     "(reference/figures/Figure_3_data.csv, "
     "reference/application/main/series_summary.json)"),
    ("Table 3",
     "the statistics of the nine series and c-bar_1/2 recomputed on the "
     "data; the simulated nulls read from reference/application/main/ "
     "(--full regenerates them); the table rendered"),
    ("Section 6: the count, the band's power, the sets",
     "read from reference/application/main/ (--full regenerates them)"),
    ("Section 6: the half-lives",
     "the grid bootstrap (B = 19,999) recomputed"),
    ("Section 6: the robust level-shift tests",
     "recomputed on the nine series (reference/application/robust_tests/)"),
    ("Section 6: the variance profile",
     "read from reference/application/variance/ (--full regenerates it; "
     "replicate_sup_S5.py recomputes it in the quick mode)"),
)
# the steps of each mode, in order (mlb/steps.py defines each step)
QUICK = ("figure_series", "regime_events", "sj_events", "app_seeds", "app_design",
         "app_data", "sj_design", "sj_data", "half_lives", "robust_tests",
         "ppp_tables")
FULL = ("figure_series", "regime_events", "sj_events", "app_seeds", "app_design",
        "app_data", "full_app_null", "full_app_bb", "full_app_power",
        "full_app_joint", "full_half_lives", "full_app_sets", "full_app_readings",
        "sj_design", "sj_data", "full_sj_null", "full_sj_hl", "full_sj_sets",
        "full_sj_readings", "full_variance_scope", "robust_tests", "ppp_tables")

if __name__ == "__main__":
    sys.exit(runner.main("replicate_sec_6.py", TITLE, EXHIBITS, QUICK, FULL, doc=__doc__))
