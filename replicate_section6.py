# -*- coding: utf-8 -*-
"""
================================================================================
replicate_section6.py -- Replicates Section 6 of the paper: the empirical PPP
    application (admissibility sweep, AR(p) bootstrap calibration, half-lives,
    and the half-life decomposition), as a single ordered entry point.
================================================================================

WHAT IT PRODUCES  (stages, run in order unless one is selected)
    sweep   Exhaustive admissibility sweep over the BIS currency universe
            (gates C1-C5): the funnel and the per-currency gate table.
            -> ppp_sweep/funnel.txt, ppp_sweep/sweep_gates.csv
    boot    AR(p) nuisance-family calibration of the applied pair
            (c-bar*(m,T,p), cv(m,T,p)) for the eight admissible currencies.
            -> boot_out/ (reads the calibration surface of Section 3-4)
    hl      Median-unbiased (Andrews-Chen) half-lives with grid-t and wild
            bootstrap intervals, and the level-break vs constant-mean
            decomposition.  -> hl_results.csv
    Figures are NOT produced here: all plotting lives in generate_figures.py,
    which reads hl_results.csv / hl_results_wild.csv (Fig 5) and
    ppp_panel.csv / exog_dates.csv (Fig 4). Run it after this module.

DATA (public; see "Data sources" in the README and each script header)
    ppp_panel.csv -- the eight admissible real exchange rates q_it, 1973-2024,
      built from BIS bilateral USD nominal rates and World Bank CPI (2010=100).
    exog_dates.csv -- the exogenous currency-regime break dates with sources.
    The sweep stage can (re)build the raw inputs from the public sources with
    `--fetch` (BIS bulk export + World Bank CPI API); see below.

USAGE
    python replicate_section6.py all                       # full application
    python replicate_section6.py sweep --fetch             # rebuild from source
    python replicate_section6.py hl
    python replicate_section6.py <stage> --quick           # fast smoke test
    # figures afterward:  python generate_figures.py --only fig4 --only fig5

INPUT DEPENDENCY
    The boot stage reads the calibration surface produced by
    replicate_section3_4.py (cbar_surface.csv). Run Section 3-4 first, or pass
    --calib PATH.

REQUIRES
    mlb_core.py, ppp_sweep_bis.py, boot_ppp_cbar.py,
    hl_median_unbiased.py in the same directory;
    numpy, scipy (sweep), numba. No matplotlib (figures live in
    generate_figures.py).
================================================================================
"""
import argparse
import os
import runpy
import sys


def _run(modname, argv):
    sys.argv = [modname + ".py"] + argv
    runpy.run_module(modname, run_name="__main__", alter_sys=True)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage",
                    choices=["sweep", "boot", "hl", "all"],
                    nargs="?", default="all")
    args, rest = ap.parse_known_args()

    def do_sweep():
        print("\n===== Section 6: admissibility sweep =====")
        argv = rest if rest else ["--exog", "exog_dates.csv", "--out", "ppp_sweep",
                                  "--c4-mode", "both"]
        _run("ppp_sweep_bis", argv)

    def do_boot():
        print("\n===== Section 6: sieve-AR(p) bootstrap calibration =====")
        if not (os.path.exists("cbar_surface.csv") or "--calib" in rest):
            print("[note] cbar_surface.csv not found; run replicate_section3_4.py "
                  "first, or pass --calib PATH. Proceeding with the script's "
                  "internal default c-bar path.")
        _run("boot_ppp_cbar", rest if rest else ["--empirical"])

    def do_hl():
        print("\n===== Section 6: median-unbiased half-lives =====")
        argv = rest if rest else ["--panel", "ppp_panel.csv",
                                  "--dates", "exog_dates.csv", "--out", "hl_results.csv"]
        _run("hl_median_unbiased", argv)

    if args.stage == "sweep":
        do_sweep()
    elif args.stage == "boot":
        do_boot()
    elif args.stage == "hl":
        do_hl()
    else:
        do_sweep(); do_boot(); do_hl()


if __name__ == "__main__":
    main()
