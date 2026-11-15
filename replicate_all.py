#!/usr/bin/env python3
"""Every section of the article and the supplement, each step once.

    python replicate_all.py                  # quick mode: about 9.5 min at 8 workers
    python replicate_all.py --full           # the warning and the plan; runs nothing
    python replicate_all.py --full --yes     # every simulation: about 3.7 h

The steps of the nine section scripts (replicate_sec_2_3.py, replicate_sec_4.py,
replicate_sec_5.py, replicate_sec_6.py, replicate_sup_S1.py to replicate_sup_S5.py), in
their order, each run once: sections share steps (Table 1's surface, the stages of the
application), so the nine scripts one after the other take longer (17.4 min in the quick
mode, 7.2 h in the full mode) and run the shared steps again.
"""
import sys

from mlb import runner

TITLE = "every section of the article and the supplement"
EXHIBITS = (("Article", "Sections 2-6: Tables 1-3, Figures 1-3 and the printed numbers"),
            ("Supplement", "Sections S1-S5: Tables S1-S9 and the printed numbers"))
QUICK = tuple(runner.all_keys("quick"))
FULL = tuple(runner.all_keys("full"))

if __name__ == "__main__":
    sys.exit(runner.main("replicate_all.py", TITLE, EXHIBITS, QUICK, FULL, doc=__doc__))
