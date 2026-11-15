"""The package root, the folders under it, and the paths of a few shared files.

REPO_ROOT is found by walking up from this module's own location to the first
directory that holds both `inputs/` and `mlb/`, so it does not depend on where a
script is invoked from. INPUTS, REFERENCE and OUTPUT are the folders inputs/ (the
data), reference/ (the results the paper prints, as shipped) and output/ (everything
a run writes) under it. reference_dir(section) is the folder of reference/ that holds
the results a module writes under output/<section>/. out_dir(section) is the folder of
a regenerated output (under output/, never under reference/ or inputs/).
resolve(name) gives the path of one of the files that several modules read (the keys
of _FILES below); the other modules build their paths from these folders.
"""
import os
from pathlib import Path


def _discover_root():
    here = Path(os.path.abspath(__file__)).resolve()
    for d in [here.parent, *here.parents]:
        if (d / "inputs").is_dir() and (d / "mlb").is_dir():
            return d
    raise RuntimeError(f"no package root (a folder with inputs/ and mlb/) above "
                       f"{here}")


REPO_ROOT = _discover_root()
INPUTS = REPO_ROOT / "inputs"
REFERENCE = REPO_ROOT / "reference"
OUTPUT = REPO_ROOT / "output"

# the folder of reference/ for each folder of output/ that the modules write
_REFERENCE_DIRS = {
    "supplement_appendix": "theory",
    "critical_values": "calibration",
    "calibration_robustness": "serial_dependence",
    "bootstrap_empirical": "application",
    "figures": "figures",
}

# The files read through resolve().
_FILES = {
    "regime_events": INPUTS / "regime_events.csv",
    "candidates": INPUTS / "ppp_sweep" / "candidates.csv",
    "sweep_funnel": INPUTS / "ppp_sweep" / "funnel.txt",
    "gate_sizes": REFERENCE / "application" / "admission" / "gate_sizes.csv",
    "c6_critical_values": REFERENCE / "application" / "admission"
    / "c6_critical_values.csv",
    "overdetrend_power": REFERENCE / "calibration" / "overdetrend_power.json",
    "tabcv_full": REFERENCE / "calibration" / "tabcv_full_percentiles.csv",
    "application": REFERENCE / "application" / "main",
    "crisis_design": REFERENCE / "application" / "crisis_design",
    "tables": REFERENCE / "application" / "tables",
}


def reference_dir(section: str) -> Path:
    """The folder of reference/ that holds the results a module writes under
    output/<section>/."""
    return REFERENCE / _REFERENCE_DIRS[section]


def resolve(name: str) -> Path:
    """Resolve a KNOWN key to its path in inputs/ or reference/ (a typo fails loudly,
    not by silently reading the wrong file)."""
    if name in _FILES:
        return _FILES[name]
    raise KeyError(f"unknown key {name!r}; known: {sorted(_FILES)}")


def out_dir(section: str) -> Path:
    """Directory for a regenerated output of a section's objects, under OUTPUT/ (never
    REFERENCE/, never INPUTS/). Created on demand."""
    d = OUTPUT / section
    d.mkdir(parents=True, exist_ok=True)
    return d
