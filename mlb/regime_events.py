"""The regime events of the PPP application: dates -> tau -> break positions, per
design set.

inputs/regime_events.csv is curated by hand, one row per event, with its date and
primary source; the columns f and tau are generated here from the date (--write), and
checked (--check). tau is never typed.

The date is the day the change took effect (for a G-5/G-7 communique, its date); when
the announcement came on an earlier day, the event text gives it. Every row of a design
set gives the same tau under either day, and the rows of S1 and S5 the same event year.

The inclusion rule E. An event counts if and only if
  E1  it is an official announcement of the currency's monetary authority or government,
      or a G-5/G-7 communique, dated to the day;
  E2  it is (i) the adoption of a float by the currency: the end of a peg, a basket, a
      crawling peg, a band or snake/ERM membership, followed by a float (a switch from
      one peg or band to another adopts no float); or (ii) a multilateral agreement
      whose communique names the dollar and announces an intended change in its level
      (the Plaza Accord of 22 September 1985 qualifies; the Louvre Accord, which
      announces stabilization at current levels, does not; the G-7 statement of
      25 April 1995, which names no currency, does not);
  E3  it is relevant for the numeraire: the Plaza applies to every currency not pegged
      to the USD in September 1985; a currency's own float applies to it.

The date rule. f is the fraction of the calendar year Y after the event, the event day
included, f = (days in Y - day of year + 1) / days in Y;
tau = Y if f >= 1/2, else Y + 1. tau is the first predominantly post-event year:
DU_t = 1{t >= tau}, the convention of Perron (1989) in which T_B = tau - 1 is the last
pre-break observation. break_pos = tau - 1973 is the number of pre-break observations,
lambda = break_pos / 52.

Merging. Every regime has at least 2 observations. Events with the same tau give one
break (logged). A one-observation regime in the interior merges its two breaks into one,
at the EARLIER date; a one-observation regime at either end removes that break. The
leftmost one-observation regime is fixed first, and the scan is repeated until none
remains. An event with tau <= 1973 or tau > 2024 is outside the sample.

Design sets (a row lists its sets in the 'sets' column, ';'-separated; its tau column is
always the main rule's):
  E   float adoptions and the Plaza (rule E), tau by the date rule;
  S1  the dummy in the event year (tau = Y) for: AUD at its float (1983); the Plaza
      (1985) for CAD, CHF, GBP, JPY, NOK, NZD and SEK; the 1992 exits of GBP, NOK and
      SEK; no DKK design;
  S2  E plus five entries: sterling into the ERM (8 Oct 1990), the krone's ECU link
      (22 Oct 1990), the krona's ECU peg (17 May 1991), the CHF floor (6 Sep 2011), the
      krone into ERM II (1 Jan 1999);
  S3  E plus seven realignments within a peg: SEK 1977, 1981 and 1982; NOK 1986;
      AUD 1976; NZD 1975 and 1984;
  S4  the Plaza alone;
  S5  E with the dummy in the event year (tau = Y).
A row whose currency is ALL is a statement or act that applies to every currency; such
rows are all excluded and belong to no set.

    python mlb/regime_events.py --check
    python mlb/regime_events.py --write
    python mlb/regime_events.py --configurations E
"""
import argparse
import csv
import datetime
import os
import sys

WINDOW_START, WINDOW_END = 1973, 2024
SETS = ("E", "S1", "S2", "S3", "S4", "S5")
EVENT_YEAR_SETS = ("S1", "S5")  # the sets whose dummy is on in the event year
ALL = "ALL"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = os.path.join(_ROOT, "inputs", "regime_events.csv")
COLUMNS = ["currency", "event", "date", "clause", "primary_source", "f", "tau",
           "included", "exclusion_reason", "sets"]


def fraction_after(date):
    """Share of the calendar year after the event, the event day included."""
    d = datetime.date.fromisoformat(date)
    n = (datetime.date(d.year + 1, 1, 1) - datetime.date(d.year, 1, 1)).days
    return (n - d.timetuple().tm_yday + 1) / n


def tau_of(date, always_event_year=False):
    """First predominantly post-event year (tau = Y if f >= 1/2, else Y + 1); Y itself
    for S1 and S5."""
    y = datetime.date.fromisoformat(date).year
    if always_event_year:
        return y
    return y if fraction_after(date) >= 0.5 else y + 1


def load(path=EVENTS):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path}: no event rows")
    return rows


def sets_of(row):
    return [s for s in row["sets"].split(";") if s]


def currencies(rows):
    """The currencies with their own rows (ALL is not one)."""
    return sorted({r["currency"] for r in rows} - {ALL})


def merge_regimes(taus):
    """Apply the merging rules to in-sample tau; return (sorted distinct taus, log)."""
    log = [f"events share tau {t}: one break" for t in sorted(set(taus))
           if taus.count(t) > 1]
    taus = sorted(set(taus))
    changed = True
    while changed and taus:
        changed = False
        bounds = [WINDOW_START] + taus + [WINDOW_END + 1]
        lengths = [b - a for a, b in zip(bounds[:-1], bounds[1:])]
        for i, n in enumerate(lengths):
            if n >= 2:
                continue
            if i == 0:                   # first regime: remove the first break
                log.append(f"one-observation regime {WINDOW_START}: "
                           f"break {taus[0]} removed")
                taus = taus[1:]
            elif i == len(lengths) - 1:  # last regime: remove the last break
                log.append(f"one-observation regime {WINDOW_END}: "
                           f"break {taus[-1]} removed")
                taus = taus[:-1]
            else:                        # interior: keep the earlier break
                log.append(f"one-observation regime {bounds[i]}: "
                           f"breaks {taus[i - 1]} and {taus[i]} merged "
                           f"at {taus[i - 1]}")
                taus = taus[:i] + taus[i + 1:]
            changed = True
            break
    return taus, log


def configuration(rows, currency, design_set):
    """(tau list, break_pos list, merge log, the rows used) of one currency under one
    design set.

    An empty tau list means the set has no event for the currency (S1 for DKK).
    """
    assert design_set in SETS, design_set
    use = [r for r in rows if r["currency"] == currency and design_set in sets_of(r)]
    taus = []
    for r in use:
        t = tau_of(r["date"], always_event_year=(design_set in EVENT_YEAR_SETS))
        if WINDOW_START < t <= WINDOW_END:
            taus.append(t)
    taus, log = merge_regimes(taus)
    return taus, [t - WINDOW_START for t in taus], log, use


def generated(rows):
    """The f and tau columns as this module generates them from the date."""
    return [(repr(fraction_after(r["date"])), str(tau_of(r["date"]))) for r in rows]


def integrity(rows):
    """The curated columns agree with each other; returns the list of violations (empty
    when none)."""
    bad = []
    for r in rows:
        key = f"{r['currency']} {r['date']} {r['event'][:40]}"
        s = sets_of(r)
        if list(r.keys()) != COLUMNS:
            bad.append(f"{key}: columns")
        if any(x not in SETS for x in s) or len(set(s)) != len(s):
            bad.append(f"{key}: unknown or repeated set in {r['sets']!r}")
        if r["included"] not in ("Y", "N"):
            bad.append(f"{key}: included is not Y or N")
        if (r["included"] == "Y") != ("E" in s):
            bad.append(f"{key}: included = Y exactly for the rows of E")
        if (r["included"] == "Y") == bool(r["exclusion_reason"].strip()):
            bad.append(f"{key}: an exclusion reason exactly for the excluded rows")
        if ("E" in s) != ("S5" in s) or ("E" in s and not {"S2", "S3"} <= set(s)):
            bad.append(f"{key}: the rows of S5 are the rows of E, "
                       "which are also in S2 and S3")
        if ("S4" in s) != (r["included"] == "Y" and "E2(ii)" in r["clause"]):
            bad.append(f"{key}: the rows of S4 are the included E2(ii) rows "
                       "(the Plaza)")
        if r["currency"] == ALL and s:
            bad.append(f"{key}: an ALL row belongs to no set")
        if not r["primary_source"].strip() or not r["clause"].strip():
            bad.append(f"{key}: no clause or primary source")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true",
                   help="f and tau as generated, and the columns consistent")
    g.add_argument("--write", action="store_true",
                   help="write the generated f and tau into the file")
    g.add_argument("--configurations", choices=SETS,
                   help="print every currency's configuration in a set")
    a = ap.parse_args(argv)
    rows = load(EVENTS)
    if list(rows[0].keys()) != COLUMNS:
        sys.exit(f"{EVENTS}: the columns are not {COLUMNS}")
    gen = generated(rows)
    if a.check:
        bad = [f"{r['currency']} {r['date']}: f/tau" for r, g_ in zip(rows, gen)
               if (r["f"], r["tau"]) != g_]
        bad += integrity(rows)
        print("f and tau as generated; columns consistent" if not bad
              else "FAILED:\n  " + "\n  ".join(bad))
        return 0 if not bad else 1
    if a.write:
        for r, (f, t) in zip(rows, gen):
            r["f"], r["tau"] = f, t
        with open(EVENTS, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print(f"wrote f and tau for {len(rows)} rows -> {EVENTS}")
        return 0
    for cur in currencies(rows):
        taus, bp, log, _ = configuration(rows, cur, a.configurations)
        print(f"{cur}: m = {len(taus)}, tau = {taus}, break_pos = {bp}"
              + (f"  [{'; '.join(log)}]" if log else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
