"""Break dates of the sensitivity design SJ (inputs/sj_dates.csv).

SJ is the principal design E plus the systemic banking crises of each currency's
issuing country (inputs/jst_crises.csv, built by build_jst_crises.py). A crisis that
starts in year Y is dated at tau = Y + 1, the first full year after the onset; the
dates of E are those of regime_events.py with the same date rule. Episodes are merged
by the rule of the principal design: events whose dates are in the same or in
consecutive years are one episode dated at the earliest, and a one-observation regime
at either end of the sample drops its break (regime_events.merge_regimes). The crises
of the United States are not dated as breaks common to the panel, and New Zealand has
none in the database, so its design is E.

Every kept date records where it comes from (E, JST or both), the dates it absorbed,
and whether the event that dates it, or one it absorbed, is the adoption of a float
(clause E2(i) of the events file), the mark an analysis of variance regimes needs.

The file is generated, never typed:

    python mlb/sj_events.py --write
    python mlb/sj_events.py --check
    python mlb/sj_events.py --configurations

--write writes inputs/sj_dates.csv and its run record
output/inputs/sj_dates.manifest.json; --check rebuilds the file's bytes in memory and
compares them with inputs/sj_dates.csv.
"""

import argparse
import csv
import hashlib
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlb import build_jst_crises as JC  # noqa: E402
from mlb import paths  # noqa: E402
from mlb import regime_events as RE  # noqa: E402

SJ_DATES = os.path.join(_ROOT, "inputs", "sj_dates.csv")
# the run record of --write (output/, not part of the inputs)
SJ_MANIFEST = os.path.join(paths.OUTPUT, "inputs", "sj_dates.manifest.json")
COLUMNS = ["currency", "tau", "break_pos", "m", "sources", "merged", "float_mark"]
M_MAX = 5  # the model admits at most five level breaks


def is_float_adoption(row):
    """The event ends a peg, a band or a floor (clause E2(i), not the Plaza E2(ii))."""
    return "E2(i);" in row["clause"]


def principal_events(rows, currency):
    """[(tau, "E", float?)] of the included events of design E."""
    return [
        (RE.tau_of(r["date"]), "E", is_float_adoption(r))
        for r in rows
        if r["currency"] == currency and "E" in RE.sets_of(r)
    ]


def crisis_events(crises, currency):
    """[(tau, "JST", False)] for the first year of every crisis, tau = Y + 1."""
    return [
        (int(r["year"]) + 1, "JST", False)
        for r in crises
        if r["currency"] == currency and r["run_start"] == "1"
    ]


def configuration(rows, crises, currency):
    """(taus, break positions, merge log, kept) of one currency under SJ.

    `kept` lists, for every kept tau, the events it stands for: the one that dates it
    and those absorbed by the merging rule (the same date from another source, or the
    date one year later)."""
    ev = [
        e
        for e in principal_events(rows, currency) + crisis_events(crises, currency)
        if RE.WINDOW_START < e[0] <= RE.WINDOW_END
    ]
    taus, log = RE.merge_regimes([e[0] for e in ev])
    if len(taus) > M_MAX:
        sys.exit(f"{currency}: m = {len(taus)} > {M_MAX}")
    keep = []
    for t in taus:
        members = [
            e
            for e in sorted(ev)
            if t <= e[0] <= t + 1 and (e[0] == t or e[0] not in taus)
        ]
        keep.append((t, members))
    return taus, [t - RE.WINDOW_START for t in taus], log, keep


def generate(rows, crises):
    """The rows of inputs/sj_dates.csv, one per kept date, currencies in order."""
    out = []
    for cur in RE.currencies(rows):
        taus, bp, _, keep = configuration(rows, crises, cur)
        for (t, members), b in zip(keep, bp):
            out.append(
                dict(
                    currency=cur,
                    tau=str(t),
                    break_pos=str(b),
                    m=str(len(taus)),
                    sources="+".join(sorted({s for _, s, _ in members})),
                    merged=";".join(f"{u}({s})" for u, s, _ in members if u != t),
                    float_mark="Y" if any(f for _, _, f in members) else "N",
                )
            )
    return out


def render(gen):
    """The file's bytes: LF line endings."""
    lines = [",".join(COLUMNS)]
    lines += [",".join(r[c] for c in COLUMNS) for r in gen]
    return ("\n".join(lines) + "\n").encode("utf-8")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_text(data):
    man = {
        "output": "sj_dates.csv",
        "output_sha256": hashlib.sha256(data).hexdigest(),
        "rows": data.count(b"\n") - 1,
        "columns": COLUMNS,
        "inputs_sha256": {
            "regime_events.csv": sha256_file(RE.EVENTS),
            "jst_crises.csv": sha256_file(JC.CRISES),
        },
        "rule": "design E plus the first year after the onset of every systemic "
        "banking crisis of the issuer; events in the same or consecutive years are "
        "one episode dated at the earliest; a one-observation regime at either end "
        "drops its break (regime_events.merge_regimes)",
        "break_pos": "tau - 1973, the number of observations before the break",
        "float_mark": "Y when the event that dates the break, or one it absorbed, is "
        "the adoption of a float (clause E2(i))",
    }
    return json.dumps(man, indent=1) + "\n"


def load(path=SJ_DATES):
    """The committed file as {currency: [row, ...]} in file order."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or list(rows[0].keys()) != COLUMNS:
        sys.exit(f"{path}: the columns are not {COLUMNS}")
    out = {}
    for r in rows:
        out.setdefault(r["currency"], []).append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true",
                   help="write the file and its run record")
    g.add_argument("--check", action="store_true", help="regenerate and compare")
    g.add_argument("--configurations", action="store_true", help="print the designs")
    a = ap.parse_args(argv)
    rows, crises = RE.load(), JC.load()
    gen = generate(rows, crises)
    data = render(gen)
    if a.configurations:
        for cur in RE.currencies(rows):
            taus, bp, log, _ = configuration(rows, crises, cur)
            print(
                f"{cur}: m = {len(taus)}, tau = {taus}, break_pos = {bp}"
                + (f"  [{'; '.join(log)}]" if log else "")
            )
        return 0
    if a.write:
        with open(SJ_DATES, "wb") as fh:
            fh.write(data)
        os.makedirs(os.path.dirname(SJ_MANIFEST), exist_ok=True)
        with open(SJ_MANIFEST, "w", encoding="utf-8", newline="") as fh:
            fh.write(manifest_text(data))
        print(f"wrote {len(gen)} rows -> {SJ_DATES}")
        return 0
    with open(SJ_DATES, "rb") as fh:
        ok = fh.read() == data
    print(
        "sj_dates.csv regenerates byte for byte"
        if ok
        else "FAILED: sj_dates.csv does not regenerate byte for byte"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
