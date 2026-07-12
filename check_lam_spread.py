#!/usr/bin/env python3
"""
check_lam_spread.py -- diagnostic check on the (m,T) calibration surface.

Reports the R^2 of the (m,T) lookup, the within-cell |lambda| spread, the
R^2 and condition number of the lambda-only CKP surface at m=1, and the
number of distinct break configurations at m=5 -- the numbers the paper
cites in Section 4.3 to justify a (m,T) lookup table over a lambda
polynomial. Thin CLI wrapper around mlb_core.surface_diagnostics so there
is a single implementation of the diagnostic, not a second copy.

Usage:
    python check_lam_spread.py --csv cbar_surface.csv [--method const]
"""
from __future__ import annotations
import argparse
import sys

import pandas as pd

import mlb_core as M


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="cbar_surface.csv",
                     help="production calibration surface (from "
                          "replicate_section3_4.py --full)")
    ap.add_argument("--method", default="const", choices=["const", "maic"])
    args = ap.parse_args(argv)

    df = pd.read_csv(args.csv)
    n_configs = df["config_key"].nunique() if "config_key" in df.columns else len(df)
    n_cells = df.groupby(["m", "T"]).ngroups
    print(f"[check_lam_spread] {args.csv}: {n_configs} configs, {n_cells} (m,T) cells")
    M.surface_diagnostics({}, df, args.method)


if __name__ == "__main__":
    sys.exit(main())
