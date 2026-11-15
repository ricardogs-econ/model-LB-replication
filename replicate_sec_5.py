#!/usr/bin/env python3
"""Section 5 of the article: serially dependent innovations.

Exhibits: Section 5 and the MAIC paragraph of Section 2.

    python replicate_sec_5.py                  # quick mode: about 15 s at 8 workers
    python replicate_sec_5.py --full           # the warning and the plan; runs nothing
    python replicate_sec_5.py --full --yes     # regenerates the simulations: about 10.3 min
    python replicate_sec_5.py --jobs 4         # worker processes (default min(8, CPUs))

The quick mode recomputes the exact results and the small simulations and compares
them with reference/; it shows the long simulations from reference/. --full regenerates
every simulation of the section from the recorded seeds and compares it with
reference/. Each step is defined in mlb/steps.py; mlb/runner.py runs the steps and
compares their files. Everything a run writes goes to output/.
"""
import sys

from mlb import runner

TITLE = "Section 5 of the article: serially dependent innovations"
EXHIBITS = (
    ("Section 5: the MAIC lag-zero share",
     "the two cells that bound the printed range recomputed "
     "(reference/calibration/maic_lag_zero/; --full: every cell)"),
    ("Table S1 of the supplement (Section 5 cites it)",
     "rendered from "
     "reference/serial_dependence/ar1_size_power/ar1_size_power.csv (--full "
     "regenerates it)"),
    ("Table S2 of the supplement (Section 5 cites it)",
     "the half-power point under AR(1) errors, recomputed exactly "
     "(reference/serial_dependence/ar1_recalibration/); the table rendered"),
    ("Section 5: the grid summaries",
     "re-derived from reference/"),
    ("Table 2 (the critical values that Table S1 and Section 5 use)",
     "rebuilt from the 20 blocks of "
     "reference/calibration/critical_values_blocks/"),
)
# the steps of each mode, in order (mlb/steps.py defines each step)
QUICK = ("prose_grid", "maic_cells", "ar1_recalibration", "tabcv_from_reference",
         "table_s1_reference")
FULL = ("prose_grid", "full_maic_T45_100", "full_maic_T150_300",
        "full_ar1_size_power", "ar1_recalibration", "tabcv_from_reference")

if __name__ == "__main__":
    sys.exit(runner.main("replicate_sec_5.py", TITLE, EXHIBITS, QUICK, FULL, doc=__doc__))
