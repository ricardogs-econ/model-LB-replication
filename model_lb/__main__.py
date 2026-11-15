"""python -m model_lb: the paper's Model LB unit-root test on a series in a CSV file.

    python -m model_lb --csv FILE --date-col year --col q --breaks 1986,1993

The test is the paper's: MZ_t with GLS detrending on (1, DU_1, ..., DU_m) at c-bar =
-7 and s^2_AR with the MAIC lag (k_max = 12), against (I) the asymptotic critical
value -1.98, (II) its null simulated at the series' configuration under the series'
BIC sieve (k_max = 10), with the p-value and the true size of -1.98, and (III) the
same at the exact half-power point c-bar_1/2. The break dates are inputs: known and
exogenous, not estimated. See model_lb/api.py for the definitions and the functions.

The rows are one series in time order: the dates of --date-col must be unique (select
one series of a long file with --where) and, when their form is recognized (years,
1990Q1, 1990M01, 1990-01, ISO dates), increasing. A constant series, or one constant
within every regime, is refused; a series whose within-regime first differences have
a root mean square outside [1e-3, 1e3] is divided by it before computing (the report
says so). Warnings are printed for T < 45, a BIC order or MAIC lag at its bound, and
a half-power point outside K = [-20, -3].
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
import sys
import time

import numpy as np


def _isfloat(x):
    try:
        float(x)
    except (ValueError, TypeError):
        return False
    return True


def _pair(m):
    return int(m.group(1)), int(m.group(2))


# Date forms whose order can be checked: (name, pattern, key of a match).
_DATE_FORMS = (
    ("number", re.compile(r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?"),
     lambda m: float(m.group(0))),
    ("quarter", re.compile(r"(\d{4})[-/ :]?[Qq]([1-4])"), _pair),
    ("month", re.compile(r"(\d{4})[-/ :]?[Mm](\d{1,2})"), _pair),
    ("year-month", re.compile(r"(\d{4})[-/](\d{1,2})"), _pair),
    ("ISO date", re.compile(r"\d{4}-\d{2}-\d{2}([T ].*)?"),
     lambda m: datetime.datetime.fromisoformat(m.group(0))),
)


def date_keys(dates):
    """Sort keys of the dates when they all have one recognized form (numbers such
    as years, 1990Q1, 1990M01, 1990-01, ISO dates), else None."""
    for _, rx, key in _DATE_FORMS:
        ms = [rx.fullmatch(d) for d in dates]
        if all(ms):
            try:
                return [key(m) for m in ms]
            except ValueError:
                continue
    return None


def check_dates(dates, date_col, header, rows, col):
    """The dates of one series are unique and, when their form is recognized,
    increasing; a repeated date usually means that the file holds several series."""
    seen = {}
    for i, d in enumerate(dates):
        if d in seen:
            n = dates.count(d)
            same = [r for r in rows if r[date_col].strip() == d]
            labels = [h for h in header if h not in (date_col, col)
                      and not _isfloat(same[0][h])
                      and len({r[h] for r in same}) == len(same)]
            hint = (f" (e.g. --where {labels[0]}={same[0][labels[0]]})"
                    if labels else "")
            raise SystemExit(
                f"the date {d} appears {n} times in column '{date_col}': the rows hold "
                f"more than one series; select one with --where COL=VALUE{hint}"
            )
        seen[d] = i
    keys = date_keys(dates)
    if keys is None:
        print(f"[model_lb] the dates of column '{date_col}' have no recognized form; "
              "their order is taken as given (rows in time order)", file=sys.stderr)
        return
    for i in range(1, len(keys)):
        try:
            ordered = keys[i] > keys[i - 1]
        except TypeError:
            return
        if not ordered:
            raise SystemExit(
                f"the dates of column '{date_col}' are not increasing: {dates[i]} "
                f"comes after {dates[i - 1]}; sort the rows in time order"
            )


def read_series(path, col=None, date_col=None, where=()):
    """(values, dates or None, column name) from a CSV file whose rows are in time
    order; where: [(column, value)] filters on equality. The dates must be unique
    and, when their form is recognized, increasing."""
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"no data rows in {path}")
    header = list(rows[0].keys())
    for c, v in where:
        if c not in header:
            raise SystemExit(f"--where: column '{c}' is not in {header}")
        rows = [r for r in rows if r[c] == v]
    if not rows:
        raise SystemExit("no row is left after --where")
    if col is None:
        numeric = [h for h in header if h != date_col and _isfloat(rows[0].get(h))]
        if not numeric:
            raise SystemExit("no numeric column found; pass --col")
        col = numeric[-1]
    elif col not in header:
        raise SystemExit(f"column '{col}' is not in {header}")
    if date_col is not None and date_col not in header:
        raise SystemExit(f"date column '{date_col}' is not in {header}")
    try:
        vals = np.array([float(r[col]) for r in rows], dtype=float)
    except ValueError as e:
        raise SystemExit(f"a non-numeric value in column '{col}': {e}") from None
    if not np.all(np.isfinite(vals)):
        raise SystemExit("the series contains NaN or inf; clean it before testing")
    dates = None
    if date_col:
        dates = [str(r[date_col]).strip() for r in rows]
        check_dates(dates, date_col, header, rows, col)
    return vals, dates, col


def resolve_breaks(breaks, break_pos, dates, T):
    """The break positions (observations before each break) from --breaks (dates of
    the first period of each new regime, or their 1-based observation numbers without
    --date-col) or from --break-pos. Errors name the breaks as the user gave them."""
    if breaks and break_pos:
        raise SystemExit("give --breaks or --break-pos, not both")
    out = []
    if break_pos:
        for tok in break_pos.split(","):
            if tok.strip():
                try:
                    out.append(int(tok))
                except ValueError:
                    msg = f"--break-pos: '{tok}' is not an integer"
                    raise SystemExit(msg) from None
        return out
    given = []
    for tok in (breaks or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in given:
            raise SystemExit(f"--breaks: {tok} is given more than once")
        given.append(tok)
        if dates is not None:
            if tok not in dates:
                raise SystemExit(f"--breaks: {tok} is not a date of the series "
                                 f"({dates[0]} to {dates[-1]})")
            i = dates.index(tok)
            if i == 0:
                later = f" ({dates[1]} to {dates[-1]})" if len(dates) > 1 else ""
                raise SystemExit(
                    f"--breaks: {tok} is the first date of the series; a break date is "
                    f"the first period of a new regime, so it is a later date{later}"
                )
            out.append(i)
        else:
            try:
                n = int(tok)
            except ValueError:
                raise SystemExit(
                    f"--breaks: '{tok}' is not an observation number (pass "
                    "--date-col to give dates)"
                ) from None
            if not 2 <= n <= T:
                raise SystemExit(
                    f"--breaks: observation {tok} is not in 2..{T}: a break is the "
                    "first observation (1-based) of a new regime"
                )
            out.append(n - 1)
    return out


def _f(x, d=4):
    return "nan" if not np.isfinite(x) else f"{x:.{d}f}"


def _decision(rej):
    return "reject the unit root at 5%" if rej else "do not reject at 5%"


def format_report(res, source, dates):
    """The printed report of model_lb_test's result."""
    st, sv = res["statistic"], res["sieve"]
    T, bp = st["T"], st["break_pos"]
    span = f", {dates[0]} to {dates[-1]}" if dates else ""
    lines = [
        "Model LB unit-root test: MZ_t, GLS detrending on (1, DU_1, ..., DU_m) at "
        f"c-bar = {st['cbar']:g}",
        f"  series     : {source}; T = {T}{span}",
    ]
    if bp:
        shown = [dates[b] for b in bp] if dates else [str(b + 1) for b in bp]
        what = "date" if dates else "observation"
        lines.append(f"  breaks     : {', '.join(shown)} (first {what} of each new "
                     f"regime); break_pos {', '.join(map(str, bp))}; m = {len(bp)}")
    else:
        lines.append("  breaks     : none (m = 0: a constant only)")
    sc = res["scale"]
    if sc["rescaled"]:
        lo, hi = sc["band"]
        lines.append(f"  scale      : the first differences within regimes have root "
                     f"mean square {sc['rms']:.3g}, outside [{lo:g}, {hi:g}]: computed "
                     "on the series divided by it (MZ_t, the p-values and the "
                     "decisions do not depend on the scale)")
    lines.append(f"  statistic  : MZ_t = {_f(st['mzt'])}; MAIC lag k = {st['maic_k']} "
                 "(k_max 12)")
    if sv["p"]:
        phis = ", ".join(f"{_f(a)} (se {_f(s)})" for a, s in zip(sv["phi"], sv["se"]))
        lines.append(f"  sieve      : BIC order p_hat = {sv['p']} (k_max 10); "
                     f"phi_hat = {phis}")
    else:
        lines.append("  sieve      : BIC order p_hat = 0 (k_max 10): the i.i.d. null")
    lines.append(f"               sum phi_hat = {_f(sv['sum_phi'])}; largest |root| "
                 f"{_f(sv['max_root'])}: "
                 + ("stationary" if sv["stationary"] else "not stationary"))
    warnings = [f"  warning    : {w}" for w in res.get("warnings", [])]
    c3 = res.get("iii")
    s3 = None if c3 is None else c3.get("simulated")
    iii = []
    if c3 is not None and c3["statistic"] is None:
        iii = [f"  (III) {c3['status']}"]
    elif c3 is not None:
        head = (f"  (III) c-bar_1/2 = {_f(c3['cbar_half'])}: MZ_t = "
                f"{_f(c3['statistic']['mzt'])}")
        if s3 is None:
            iii = [f"{head}; {c3['status']}"]
        else:
            iii = [f"{head}; critical value {_f(s3['cv'])} (se {_f(s3['cv_se'])});",
                   f"        p-value {_f(s3['p'])} (se {_f(s3['p_se'])}): "
                   f"{_decision(s3['reject'])}"]
    if res["status"] != "ok":
        lines.append(f"  (I)   -1.98: {_decision(res['reject_I'])}")
        lines.append(f"  (II)  {res['status']}")
        return "\n".join(lines + iii + warnings) + "\n"
    s2 = res["simulated"]
    lines += [
        f"  (I)   asymptotic critical value -1.98: {_decision(res['reject_I'])}",
        f"        true size of -1.98 under the sieve null {_f(s2['size_asy'])} "
        f"(se {_f(s2['size_asy_se'])})",
        f"  (II)  critical value {_f(s2['cv'])} (se {_f(s2['cv_se'])}); p-value "
        f"{_f(s2['p'])} (se {_f(s2['p_se'])}): {_decision(s2['reject'])}",
    ] + iii
    seeds = f"seed (II) {res['seed']}"
    nan = f"{s2['n_nan']}"
    if s3 is not None:
        seeds += f", seed (III) {c3['seed']}"
        nan += f" and {s3['n_nan']}"
    lines.append(f"  null draws : R = {s2['R']:,} per column; {seeds}; degenerate "
                 f"draws {nan}")
    return "\n".join(lines + warnings) + "\n"


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return x if np.isfinite(x) else repr(x)
    return x


def build_parser():
    ap = argparse.ArgumentParser(
        prog="python -m model_lb",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--csv", required=True,
                    help="CSV file with the series, one row per period, in time order")
    ap.add_argument("--col", default=None,
                    help="the series column (default: the last numeric column)")
    ap.add_argument("--date-col", default=None,
                    help="the date column; --breaks gives values of it")
    ap.add_argument("--where", action="append", default=[], metavar="COL=VALUE",
                    help="keep only the rows whose COL equals VALUE (repeatable), "
                         "e.g. one series of a long panel")
    ap.add_argument("--breaks", default=None,
                    help="comma-separated break dates: the first period of each new "
                         "regime (values of --date-col; without it, 1-based "
                         "observation numbers)")
    ap.add_argument("--break-pos", default=None,
                    help="comma-separated break positions instead: the number of "
                         "observations before each break")
    ap.add_argument("--R", type=int, default=200_000,
                    help="replications per simulated null (default 200,000)")
    ap.add_argument("--seed", type=int, default=20260701,
                    help="column (II): replication r draws from "
                         "numpy.random.default_rng(seed + r) (default 20260701)")
    ap.add_argument("--seed-iii", type=int, default=None,
                    help="the seed of column (III) (default seed + 10**9)")
    ap.add_argument("--no-iii", action="store_true",
                    help="skip column (III) and the half-power point")
    ap.add_argument("--jobs", type=int, default=1,
                    help="worker processes (default 1); the draws do not depend on it")
    ap.add_argument("--json", default=None,
                    help="also write every number at full precision to this file")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    where = []
    for w in a.where:
        if "=" not in w:
            raise SystemExit(f"--where '{w}': use COL=VALUE")
        where.append(tuple(w.split("=", 1)))
    vals, dates, col = read_series(a.csv, a.col, a.date_col, where)
    bp = resolve_breaks(a.breaks, a.break_pos, dates, len(vals))

    from model_lb import api

    try:
        bp = api.design_positions(len(vals), bp)
        api.prepare(vals, bp)  # refuses a degenerate series before any simulation
    except ValueError as e:
        raise SystemExit(str(e)) from None
    source = f"column '{col}' of {a.csv}"
    if where:
        source += " where " + ", ".join(f"{c}={v}" for c, v in where)
    cols = "(II)" if a.no_iii else "(II) and (III)"
    print(f"[model_lb] simulating {cols}: R = {a.R:,} per column, {a.jobs} "
          "worker(s); the first run compiles the kernels", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    try:
        res = api.model_lb_test(vals, bp, R=a.R, seed=a.seed, seed_iii=a.seed_iii,
                                jobs=a.jobs, iii=not a.no_iii)
    except (ValueError, RuntimeError) as e:
        raise SystemExit(str(e)) from None
    print(f"[model_lb] done in {time.perf_counter() - t0:.1f} s", file=sys.stderr,
          flush=True)
    sys.stdout.write(format_report(res, source, dates))
    if a.json:
        result = dict(res)
        result["sieve"] = {k: v for k, v in res["sieve"].items()
                           if k not in ("u", "e", "e_periods")}
        payload = dict(
            input_csv=a.csv, series_column=col, where=dict(where), T=len(vals),
            first_date=dates[0] if dates else None, break_pos=bp,
            break_dates=[dates[b] for b in bp] if dates else None, result=result,
        )
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(_jsonable(payload), fh, indent=1)
            fh.write("\n")
        print(f"[model_lb] JSON written to {a.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
