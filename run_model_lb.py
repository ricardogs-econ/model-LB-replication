# -*- coding: utf-8 -*-
"""
================================================================================
run_model_lb.py -- Stand-alone command-line tool: apply the Model LB
                   point-optimal unit-root test to YOUR OWN series.
================================================================================

WHAT IT DOES
    Runs the no-trend Model LB test (constant + known-date level dummies) of
    Carrion-i-Silvestre, Kim & Perron (2009) on a single user series, at break
    dates you supply and justify as EXOGENOUS (a policy/regime event dated
    independently of the data -- the known-date regime of Perron, 1989). It is
    NOT an endogenous break search: the dates are inputs. It uses the identical
    kernel (mlb_core.mstats_nb) and long-run-variance convention (OLS-detrended,
    Perron-Qu) that produce the paper's tables, so your numbers are on the same
    footing as the published results.

INPUTS
    --csv PATH        CSV with the series (rows in time order).
    --col NAME        Series column (default: last numeric column).
    --date-col NAME   Optional date column, for matching break dates / reporting.
    --breaks LIST     Comma-separated break DATES (matched against --date-col)
                      or 1-based observation indices, e.g. "1985,1998" or "13,26".
    --sigma2 {const,maic}
                      Long-run variance: difference-based (const; the calibration
                      baseline) or Ng-Perron MAIC (robust to serial correlation;
                      the applied default). Default: maic.
    --calib PATH      Calibration CSV (schema m,T,sigma2_method,cbar_otimo,
                      mzt_cv5). If given, c-bar and the tabulated MZt CV are read
                      by nearest (m,T); otherwise c-bar = -7 (ERS demeaned) and
                      ALL critical values are simulated on the fly in your exact
                      break configuration.
    --alpha FLOAT     Nominal level for the critical values (default 0.05).
    --nsim INT        Replications for on-the-fly critical values (default 9999).
    --json PATH       Also write the full result as JSON.

OUTPUT
    A report with the five M-class statistics, their critical values, the
    per-statistic verdict, and the resolved c-bar and its source. Rejection is
    the LOWER tail (small values reject the unit root).

EXAMPLE
    python run_model_lb.py --csv gbp_rer.csv --date-col year --col log_rer \\
        --breaks 1985,1992 --sigma2 maic --calib cbar_surface.csv

REQUIRES
    mlb_core.py in the same directory; numpy + numba.
================================================================================
"""
import argparse
import csv
import json
import sys

import numpy as np

try:
    import mlb_core as core
except ImportError:
    sys.exit("run_model_lb.py requires mlb_core.py in the same directory.")


def _isfloat(x):
    try:
        float(x); return True
    except (ValueError, TypeError):
        return False


def _read_series(path, col, date_col):
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"No data rows in {path}.")
    header = list(rows[0].keys())
    if col is None:
        numeric = [h for h in header if _isfloat(rows[0].get(h, ""))]
        if not numeric:
            sys.exit("No numeric column found; pass --col.")
        col = numeric[-1]
    elif col not in header:
        sys.exit(f"Column '{col}' not in {header}.")
    try:
        vals = np.array([float(r[col]) for r in rows], dtype=float)
    except ValueError as e:
        sys.exit(f"Non-numeric value in column '{col}': {e}")
    dates = [str(r[date_col]) for r in rows] if (date_col and date_col in header) else None
    if np.any(~np.isfinite(vals)):
        sys.exit("Series contains NaN/inf; clean it before testing.")
    return vals, dates, col


def _resolve_breaks(tokens, dates):
    if not tokens:
        return []
    out = []
    for tok in tokens.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if dates is not None and tok in dates:
            out.append(dates.index(tok))
        else:
            try:
                out.append(int(tok) - 1)
            except ValueError:
                sys.exit(f"Break '{tok}' is neither a date in the file nor an integer index.")
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description="Stand-alone Model LB unit-root test.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col", default=None)
    ap.add_argument("--date-col", default=None)
    ap.add_argument("--breaks", default="")
    ap.add_argument("--sigma2", choices=["const", "maic"], default="maic")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--kmax", type=int, default=12)
    ap.add_argument("--nsim", type=int, default=9999)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    vals, dates, col = _read_series(args.csv, args.col, args.date_col)
    breaks = _resolve_breaks(args.breaks, dates)
    T = len(vals)
    for b in breaks:
        if b <= 0 or b >= T - 1:
            sys.exit(f"Break at 0-based position {b} is not interior for T={T}.")

    core.warm_up_numba(args.kmax)
    res = core.run_test(vals, breaks, sigma2_method=args.sigma2, kmax=args.kmax,
                        cbar_csv=args.calib, alpha=args.alpha, R_null=args.nsim)

    shown = ([dates[b] for b in breaks] if dates is not None else [b + 1 for b in breaks])
    print(f"input series      : column '{col}' from {args.csv}  (T={T})")
    print(f"break dates/pos   : {shown if breaks else '(none: m=0)'}")
    core.print_report(res)

    if args.json:
        n = sum(1 for k in res["verdict"] if res["verdict"][k] == "reject")
        payload = {"input_csv": args.csv, "series_column": col, "T": T,
                   "breaks_pos0": breaks, "breaks_shown": shown, "m": len(breaks),
                   "sigma2_method": args.sigma2, "cbar": res["cbar"],
                   "cbar_source": res["cbar_source"], "cv_source": res["cv_source"],
                   "alpha": args.alpha,
                   "stats": {k: float(v) for k, v in res["stats"].items()},
                   "cvs": {k: float(v) for k, v in res["cvs"].items()},
                   "verdict": dict(res["verdict"]), "n_reject_of_5": n}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"[json] written -> {args.json}")


if __name__ == "__main__":
    main()
