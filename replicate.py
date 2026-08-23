#!/usr/bin/env python3
"""Single entry point for the Model LB replication package.

    python replicate.py                 # door table + run every door (--quick)
    python replicate.py <door>          # run one door (--quick)
    python replicate.py <door> --full   # regenerate from the frozen surface

Doors: cv | robustness | ppp | supplement | figures  (aliases in DOORS).
--quick (default) reads inputs/ and output/golden/ and regenerates nothing
heavy. --full regenerates the regenerable-here objects; it never regenerates
the frozen calibration surface and never writes to inputs/ or output/golden/.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from section_replications import (
    critical_values, robustness, bootstrap_empirical, supplement, figures)

DOORS = {
    "cv": critical_values, "critical_values": critical_values,
    "robustness": robustness,
    "ppp": bootstrap_empirical, "bootstrap_empirical": bootstrap_empirical,
    "supplement": supplement,
    "figures": figures,
}
_ORDER = [critical_values, robustness, bootstrap_empirical, supplement, figures]


def _table():
    print(f"{'door':<20}{'objects'}")
    for d in _ORDER:
        print(f"{d.DOOR:<20}{', '.join(d.OBJECTS)}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    full = "--full" in argv
    argv = [a for a in argv if a != "--full"]
    if not argv:
        _table()
        for d in _ORDER:
            d.run(full=full)
        return 0
    door = DOORS.get(argv[0])
    if door is None:
        print(f"unknown door {argv[0]!r}; known: {sorted(DOORS)}"); return 2
    door.run(full=full)
    return 0


if __name__ == "__main__":
    sys.exit(main())
