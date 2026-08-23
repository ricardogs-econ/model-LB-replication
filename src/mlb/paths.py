"""The single path resolver for the public package — inputs AND outputs.

Every file location in this package resolves here. No door or producer computes
its own path from `__file__`, `os.getcwd()`, or a bare relative default; none
receives an artefact directory from a free `--outdir`. This is the one place
that answers "where is X".

Root discovery walks up from THIS module's own location until it finds the
public-tree signature (a directory holding both `inputs/` and `src/mlb/`), so
the answer does not depend on where a script is invoked from.
"""
import os
from pathlib import Path


def _discover_root():
    here = Path(os.path.abspath(__file__)).resolve()
    for d in [here.parent, *here.parents]:
        if (d / "inputs").is_dir() and (d / "src" / "mlb").is_dir():
            return d
    # fall back to two levels up (src/mlb/paths.py -> repo root)
    return here.parent.parent.parent


REPO_ROOT = _discover_root()
INPUTS = REPO_ROOT / "inputs"
OUTPUT = REPO_ROOT / "output"
GOLDEN = OUTPUT / "golden"

# Known INPUT artefacts (frozen; the package never regenerates the surface).
_INPUTS = {
    "cbar_surface": INPUTS / "cbar_surface.csv",
    "ppp_panel": INPUTS / "ppp_panel.csv",
    "exog_dates": INPUTS / "exog_dates.csv",
    "ppp_ar_diagnostic": INPUTS / "ppp_ar_diagnostic.csv",
    "data_terms": INPUTS / "DATA_TERMS.md",
    "limiting_density": INPUTS / "limiting_density.csv",
    "surface_ppp_boot": INPUTS / "boot_out" / "calib" / "surface_ppp_boot.csv",
    "manifest": INPUTS / "MANIFEST.json",
    "object_map": INPUTS / "OBJECT_MAP.csv",
}

# Known GOLDEN outputs (promoted, frozen references).
_GOLDEN = {
    "tab_cbar": GOLDEN / "printed_tables" / "tab_cbar_mT.tex",
    "tab_pppsurface": GOLDEN / "bootstrap_empirical" / "tab_pppsurface.tex",
    "tabcv": GOLDEN / "critical_values" / "tabcv_reconciliation.csv",
    "tabpower": GOLDEN / "critical_values" / "tabpower_runC.json",
    "tabcv_full": GOLDEN / "critical_values" / "tabcv_full_percentiles.csv",
    "ar1_size_power": GOLDEN / "calibration_robustness" / "ar1_size_power.csv",
    "ar1_recalibration": GOLDEN / "calibration_robustness" / "ar1_recalibration.csv",
    "ppp_empirical": GOLDEN / "bootstrap_empirical" / "ppp_empirical.csv",
    "modulus_applied": GOLDEN / "supplement_appendix" / "modulus_feasible_applied.csv",
    "modulus_curves": GOLDEN / "supplement_appendix" / "modulus_feasible_curves.csv",
    "g36_ladder": GOLDEN / "supplement_appendix" / "g36_ladder.json",
    "hl_results_wild": GOLDEN / "bootstrap_empirical" / "hl_results_wild.csv",
    "hl_results": GOLDEN / "bootstrap_empirical" / "hl_results.csv",
}


def resolve(name: str, *, fresh: bool = False) -> Path:
    """Resolve a KNOWN artefact key to its path (a typo fails loudly, not by
    silently reading the wrong file).

    fresh=False (default): serve the frozen copy in INPUTS/GOLDEN.
    fresh=True: return the corresponding OUTPUT path for a --full regeneration
    (under OUTPUT/, never under GOLDEN/ or INPUTS/); the caller creates it.
    """
    if name in _INPUTS:
        if fresh:
            raise ValueError(f"{name!r} is a frozen input; it is never regenerated")
        return _INPUTS[name]
    if name in _GOLDEN:
        if fresh:
            return OUTPUT / _GOLDEN[name].relative_to(GOLDEN)
        return _GOLDEN[name]
    raise KeyError(f"unknown artefact key {name!r}; known: "
                   f"{sorted(set(_INPUTS) | set(_GOLDEN))}")


def out_dir(section: str, *, fresh: bool = True) -> Path:
    """Directory for a --full regeneration of a section's objects, under
    OUTPUT/ (never GOLDEN/, never INPUTS/). Created on demand."""
    d = OUTPUT / section
    d.mkdir(parents=True, exist_ok=True)
    return d
