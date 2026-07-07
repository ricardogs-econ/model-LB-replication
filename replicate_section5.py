# -*- coding: utf-8 -*-
"""
================================================================================
replicate_section5.py -- Replicates Section 5 of the paper (robustness and the
    power comparison), as a single entry point over the tested components.
================================================================================

WHAT IT PRODUCES  (subcommands)
    ar1       Empirical size and power of the point-optimal statistic under
              AR(1) innovations, using the i.i.d.-calibrated c-bar with no
              recalibration; difference-based vs MAIC long-run variance.
              -> Table `ar1-sizepower` (and, with --recalibrate, `ar1-recal`).
    oracle    Optimal c-bar under AR(1) with the ORACLE long-run variance
              omega^2 = sigma^2/(1-rho)^2 (known, not estimated), isolating
              serial correlation from LRV estimation: the limiting tangency is
              (asymptotically) unmoved by rho.  -> Table `serial`.
    trimming  Sensitivity of the c-bar(m,T) surface to the break-fraction
              trimming epsilon in {0.10,0.15,0.20}: near-invariant per cell.
              -> Table `trimming`.
    power     Size and power of MZt under three c-bar specifications -- the
              calibrated Model LB tangency, the linear break-count value, and
              the reused trend-break surface -- all correctly sized, differing
              only in power (the shortcuts over-detrend and lose power).
              -> Table `power`, Figure `power`.  The trend-break comparison
              curve uses the CKP response surface c_bar_rs, INLINED in
              size_power_cbar_comparison.py so this package has no external dependency.
    all       Run ar1, oracle, trimming, power in sequence.

USAGE
    python replicate_section5.py all --outdir section5_out          # paper run
    python replicate_section5.py ar1 --recalibrate
    python replicate_section5.py power                              # Table/Fig power
    python replicate_section5.py --selftest        # pure-logic gates (no numba)
    python replicate_section5.py <sub> --quick     # fast smoke test

INPUT
    The power subcommand and the AR(1) anchor read the calibration surface
    produced by replicate_section3_4.py (cbar_surface.csv); pass --calib PATH
    if it is elsewhere. If absent, c-bar falls back to the ERS demeaned value.

REQUIRES
    mlb_core.py, robustness.py, size_power_cbar_comparison.py in the same directory;
    numpy, numba, matplotlib, joblib.
================================================================================
"""
import argparse
import runpy
import sys


def _run_module(modname, argv):
    """Invoke a component script's CLI with the given argv."""
    sys.argv = [modname + ".py"] + argv
    runpy.run_module(modname, run_name="__main__", alter_sys=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True)
    ap.add_argument("subcommand",
                    choices=["ar1", "oracle", "trimming", "power", "all", "selftest"],
                    nargs="?", default=None)
    ap.add_argument("--selftest", action="store_true",
                    help="pure-logic gates of the robustness suite (no numba)")
    args, rest = ap.parse_known_args()

    if args.selftest or args.subcommand == "selftest":
        _run_module("robustness", ["--selftest"])
        return

    if args.subcommand is None:
        ap.print_help(); sys.exit(0)

    if args.subcommand == "power":
        # Table/Figure `power` are produced by size_power_cbar_comparison (c_bar_rs inlined).
        _run_module("size_power_cbar_comparison", rest)
    elif args.subcommand == "all":
        for sub in ("ar1", "oracle", "trimming"):
            print(f"\n===== robustness: {sub} =====")
            _run_module("robustness", [sub] + rest)
        print("\n===== power (Table/Figure power) =====")
        _run_module("size_power_cbar_comparison", rest)
    else:
        _run_module("robustness", [args.subcommand] + rest)


if __name__ == "__main__":
    main()
