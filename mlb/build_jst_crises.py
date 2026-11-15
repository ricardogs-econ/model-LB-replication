"""Systemic banking crisis years of the currencies' issuers (inputs/jst_crises.csv).

Data building: this module needs the raw download named below (JSTdatasetR6.dta, the
Macrohistory Database), and neither the quick nor the full mode runs it. The file it
writes ships in inputs/.

The sensitivity design SJ adds to the principal break dates the systemic banking
crises of each currency's issuing country. The crises come from the Jordà-Schularick-
Taylor Macrohistory Database (release R6), variable crisisJST, a 0-1 dummy that takes
the value 1 in the first year of a crisis and 0 otherwise. This script lists the years
in which the dummy is 1 for the eight currencies whose issuer is in the database, and
marks the first year of every run of consecutive ones (a run is one crisis).

New Zealand is not in the database. The United States issues the numeraire, not a
currency of the panel, so its crises are not read. The older coding crisisJST_old is
not used.

The raw file is not redistributed (CC BY-NC-SA 4.0). It is read from a local cache
directory (--cache) that holds JSTdatasetR6.dta; its sha256 is checked. --write
writes inputs/jst_crises.csv and the run record output/inputs/jst_crises.manifest.json,
which records that sha256, the variable, the mapping from currencies to issuers and the
sha256 of the output. --check builds the file again and compares it byte for byte with
inputs/jst_crises.csv.

    python mlb/build_jst_crises.py --cache <dir> --write
    python mlb/build_jst_crises.py --cache <dir> --check
"""

import argparse
import csv
import hashlib
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRISES = os.path.join(_ROOT, "inputs", "jst_crises.csv")
MANIFEST = os.path.join(_ROOT, "output", "inputs", "jst_crises.manifest.json")
RAW_NAME = "JSTdatasetR6.dta"
RAW_SHA256 = "b0ebb74a8d1b5b1bc9033fc46a6dcc578736afff8ce1e1086b7840f2649e79b3"
VARIABLE = "crisisJST"
# currency -> ISO 3166 alpha-3 code of its issuer in the database
ISSUER = {
    "AUD": "AUS",
    "CAD": "CAN",
    "CHF": "CHE",
    "DKK": "DNK",
    "GBP": "GBR",
    "JPY": "JPN",
    "NOK": "NOR",
    "SEK": "SWE",
}
NOT_READ = {
    "NZD": "New Zealand is not in the database",
    "USD": "the numeraire's crises are not dated as breaks common to the panel",
}
# the first and last year kept: a crisis in year Y is dated at tau = Y + 1, and the
# sample of the application is 1973-2024
FIRST_YEAR, LAST_YEAR = 1972, 2024
COLUMNS = ["currency", "iso", "year", "run_start"]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_flags(cache):
    """{iso: {year: crisisJST}} for the issuers, from the raw file in `cache`."""
    path = os.path.join(cache, RAW_NAME)
    if not os.path.exists(path):
        sys.exit(f"{path}: not found (the raw file is read from --cache)")
    got = sha256_file(path)
    if got != RAW_SHA256:
        sys.exit(f"{path}: sha256 {got} is not the release R6 file {RAW_SHA256}")
    import pandas as pd

    df = pd.read_stata(
        path, convert_categoricals=False, columns=["iso", "year", VARIABLE]
    )
    out = {}
    for iso in ISSUER.values():
        sub = df[df["iso"] == iso]
        if sub.empty:
            sys.exit(f"{iso}: no rows in {RAW_NAME}")
        out[iso] = {int(y): v for y, v in zip(sub["year"], sub[VARIABLE])}
    return out


def build_rows(flags):
    """The crisis years of each currency's issuer, with the first year of each run."""
    rows = []
    for cur in sorted(ISSUER):
        iso = ISSUER[cur]
        years = sorted(y for y, v in flags[iso].items() if v == 1)
        for y in years:
            if FIRST_YEAR <= y <= LAST_YEAR:
                start = 0 if (y - 1) in years else 1
                rows.append(
                    dict(currency=cur, iso=iso, year=str(y), run_start=str(start))
                )
    return rows


def last_year(flags):
    """The last year present in the raw file for the issuers."""
    return max(y for by_year in flags.values() for y in by_year)


def render(rows):
    """The file's bytes: LF line endings, the columns above."""
    lines = [",".join(COLUMNS)]
    lines += [",".join(r[c] for c in COLUMNS) for r in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def manifest_for(rows, data, last_year_in_database):
    return {
        "output": "jst_crises.csv",
        "output_sha256": sha256_bytes(data),
        "rows": len(rows),
        "columns": COLUMNS,
        "source": "Jordà, Schularick and Taylor, Macrohistory Database, release R6 "
        "(https://www.macrohistory.net/database/)",
        "variable": VARIABLE,
        "raw_file": RAW_NAME,
        "raw_sha256": RAW_SHA256,
        "license_of_the_raw_file": "CC BY-NC-SA 4.0 (not redistributed)",
        "issuers": ISSUER,
        "not_read": NOT_READ,
        "years_kept": [FIRST_YEAR, LAST_YEAR],
        "last_year_in_database": last_year_in_database,
        "coverage": "no crisis can be flagged after last_year_in_database; the "
        "years after it are kept in the window but carry no JST information",
        "run_start": "1 for the first year of a run of consecutive crisis years, "
        "0 for the others",
    }


def load(path=CRISES):
    """The rows of the shipped file, as dictionaries of strings."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or list(rows[0].keys()) != COLUMNS:
        sys.exit(f"{path}: the columns are not {COLUMNS}")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--write", action="store_true", help="write the file and its run record"
    )
    g.add_argument("--check", action="store_true", help="regenerate and compare")
    ap.add_argument("--cache", required=True, help="directory that holds " + RAW_NAME)
    a = ap.parse_args(argv)
    flags = read_flags(a.cache)
    rows = build_rows(flags)
    data = render(rows)
    man = manifest_for(rows, data, last_year(flags))
    text = json.dumps(man, indent=1, ensure_ascii=False) + "\n"
    if a.write:
        with open(CRISES, "wb") as fh:
            fh.write(data)
        os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
        with open(MANIFEST, "w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        print(f"wrote {len(rows)} rows -> {CRISES} (sha256 {man['output_sha256']}); "
              f"run record {MANIFEST}")
        return 0
    with open(CRISES, "rb") as fh:
        same_file = fh.read() == data
    print(
        "jst_crises.csv regenerates byte for byte"
        if same_file
        else "FAILED: jst_crises.csv differs from the file built from the raw data"
    )
    return 0 if same_file else 1


if __name__ == "__main__":
    sys.exit(main())
