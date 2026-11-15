#!/usr/bin/env python3
"""Section 4 of the article: calibration, the detrending rule and critical values.

Exhibits: Sections 4.1 to 4.4, and their restatements in the abstract, the introduction
and the conclusion.

    python replicate_sec_4.py                  # quick mode: about 3.4 min at 8 workers
    python replicate_sec_4.py --full           # the warning and the plan; runs nothing
    python replicate_sec_4.py --full --yes     # regenerates the simulations: about 2.9 h
    python replicate_sec_4.py --jobs 4         # worker processes (default min(8, CPUs))

The quick mode recomputes the exact results and the small simulations and compares
them with reference/; it shows the long simulations from reference/. --full regenerates
every simulation of the section from the recorded seeds and compares it with
reference/. Each step is defined in mlb/steps.py; mlb/runner.py runs the steps and
compares their files. Everything a run writes goes to output/.
"""
import sys

from mlb import runner

TITLE = "Section 4 of the article: calibration, the detrending rule and critical values"
EXHIBITS = (
    ("Table 1",
     "the 404 exact half-power points, recomputed "
     "(reference/calibration/cbar_surface_exact/); the table rendered"),
    ("Figure 1",
     "drawn from reference/figures/Figure_1_data.csv (--full regenerates the "
     "18 cells)"),
    ("Table 2",
     "rebuilt from the 20 blocks of "
     "reference/calibration/critical_values_blocks/ (--full regenerates the "
     "blocks)"),
    ("Figure 2",
     "drawn from reference/calibration/overdetrend_power.json (--full "
     "regenerates its 428,000,000 draws)"),
    ("Section 4.1: displacements, Proposition 1, (m+1) regularity, ranges, trimming, strict decrease",
     "recomputed exactly; the grid summaries re-derived from reference/"),
    ("Section 4.2: fixing -7 against following the half-power point",
     "the fixed index recomputed; the power comparison read from "
     "reference/calibration/rule_comparison/ (--full regenerates it)"),
    ("Section 4.3: the oracle quantile, the MAIC critical values",
     "8 of the 12 oracle cells recomputed (--full: all 12)"),
    ("Section 4.4: the power cost of a trend-calibrated c-bar",
     "the CKP (2009) surface re-evaluated; the power read from "
     "reference/calibration/overdetrend_power.json"),
)
# the steps of each mode, in order (mlb/steps.py defines each step)
QUICK = ("exact_surface", "figure_size", "tabcv_from_reference",
         "figure_overdetrend", "propflat", "regularity", "surface_summaries",
         "monotonicity_reference", "prose_grid", "rule_agreement", "fixed_index",
         "oracle_cells", "ckp_table1")
FULL = ("exact_surface", "full_figure_size", "full_tabcv_blocks", "full_tabcv_table",
        "full_overdetrend", "full_figure_overdetrend", "propflat", "regularity",
        "surface_summaries", "full_monotonicity", "prose_grid",
        "full_rule_comparison", "fixed_index", "full_oracle", "full_ar1_size_power",
        "ckp_table1")

if __name__ == "__main__":
    sys.exit(runner.main("replicate_sec_4.py", TITLE, EXHIBITS, QUICK, FULL, doc=__doc__))
