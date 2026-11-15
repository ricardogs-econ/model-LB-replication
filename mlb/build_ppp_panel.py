"""Build the real exchange rate panel of the PPP application (inputs/ppp_panel.csv).

Data building: this module needs the raw downloads named below (BIS WS_XRU, WDI
FP.CPI.TOTL), and neither the quick nor the full mode runs it. The panel it writes
ships in inputs/.

The real exchange rate of currency i against the US dollar (Section 6 of the article):

    q_it = ln X_it + ln P_US,t - ln P_it

    X_it   units of currency i per US dollar: BIS WS_XRU, annual frequency, period
           average (FREQ = A, COLLECTION = A);
    P      consumer price index, World Bank WDI FP.CPI.TOTL (2010 = 100).

q rises when currency i depreciates in real terms against the dollar. The window is
1973-2024 (T = 52 annual observations).

The raw files are not redistributed. They are read from a local cache directory
(--cache): the BIS bulk file WS_XRU_csv_flat.zip (or its annual subset
bis_xru_annual_A.csv, written on first use) and the WDI payload wb_cpi.json. --fetch
downloads both into the cache from the addresses below. The run record,
output/inputs/ppp_panel.manifest.json (named after --out), records the sha256 of the
panel and of every raw file read, the currencies, the window and the formula.

    python mlb/build_ppp_panel.py --cache <dir> \\
        --currencies AUD,CAD,... [--fetch] [--out inputs/ppp_panel.csv]
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(_ROOT, "inputs", "ppp_panel.csv")
RECORD_DIR = os.path.join(_ROOT, "output", "inputs")  # the run record
WINDOW_START, WINDOW_END = 1973, 2024
T_WINDOW = WINDOW_END - WINDOW_START + 1          # 52
BIS_BULK_URL = "https://data.bis.org/static/bulk/WS_XRU_csv_flat.zip"
WB_CPI_URL = (
    "https://api.worldbank.org/v2/country/all/indicator/FP.CPI.TOTL"
    "?format=json&per_page=20000&date={}:{}"
).format(WINDOW_START - 1, WINDOW_END)
BIS_ZIP, BIS_ANNUAL, WB_JSON = ("WS_XRU_csv_flat.zip", "bis_xru_annual_A.csv",
                                "wb_cpi.json")
# ISO 4217 currency -> ISO 3166 alpha-2 area of its issuer (the BIS REF_AREA and the WDI
# country id)
CURRENCY_AREA = {"AUD": "AU", "CAD": "CA", "CHF": "CH", "DKK": "DK", "GBP": "GB",
                 "JPY": "JP", "NOK": "NO", "NZD": "NZ", "SEK": "SE", "ISK": "IS",
                 "TRY": "TR"}
COLUMNS = ["currency", "iso2", "year", "xr", "cpi", "cpi_us", "ln_xr", "p_i", "p_us",
           "q"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(url, dest):
    """Fetch url into dest (used only with --fetch)."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "model-lb-replication/ppp-panel"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as fh:
        fh.write(r.read())


def load_bis_annual(cache, fetch=False):
    """{(area, currency): {year: X}} from the BIS annual period-average series.

    The annual subset of the bulk file is written to the cache on first use; X <= 0 is
    dropped. Returns the data and the raw files read.
    """
    sub = os.path.join(cache, BIS_ANNUAL)
    read = []
    if not os.path.exists(sub):
        zpath = os.path.join(cache, BIS_ZIP)
        if not os.path.exists(zpath):
            if not fetch:
                sys.exit(f"{zpath} not found: pass --fetch, "
                         "or put the BIS bulk file in the cache")
            download(BIS_BULK_URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(name) as fh, open(sub, "w", newline="",
                                          encoding="utf-8") as out:
                rdr = csv.reader(io.TextIOWrapper(fh, encoding="utf-8"))
                header = next(rdr)
                idx = {c.split(":")[0]: i for i, c in enumerate(header)}
                need = ["FREQ", "REF_AREA", "CURRENCY", "COLLECTION", "TIME_PERIOD",
                        "OBS_VALUE"]
                w = csv.writer(out)
                w.writerow(need)
                for row in rdr:
                    if (row and row[idx["FREQ"]].startswith("A")
                            and row[idx["COLLECTION"]].startswith("A")):
                        w.writerow([row[idx[c]] if c == "OBS_VALUE"
                                    else row[idx[c]].split(":")[0].strip()
                                    for c in need])
    zpath = os.path.join(cache, BIS_ZIP)
    if os.path.exists(zpath):                      # the bulk file the subset came from
        read.append(zpath)
    read.append(sub)
    data = defaultdict(dict)
    with open(sub, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                y, v = int(r["TIME_PERIOD"]), float(r["OBS_VALUE"])
            except (TypeError, ValueError):
                continue
            if v > 0:
                data[(r["REF_AREA"], r["CURRENCY"])][y] = v
    return data, read


def load_wb_cpi(cache, fetch=False):
    """{area: {year: CPI}} from the WDI FP.CPI.TOTL payload, and the raw file read."""
    path = os.path.join(cache, WB_JSON)
    if not os.path.exists(path):
        if not fetch:
            sys.exit(f"{path} not found: pass --fetch, "
                     "or put the WDI payload in the cache")
        download(WB_CPI_URL, path)
    payload = json.load(open(path, encoding="utf-8"))
    if not (isinstance(payload, list) and len(payload) == 2):
        sys.exit(f"{path}: not a WDI payload")
    cpi = defaultdict(dict)
    for e in payload[1]:
        if e.get("value") is None:
            continue
        area = (e.get("country") or {}).get("id", "")
        try:
            cpi[area][int(e["date"])] = float(e["value"])
        except (TypeError, ValueError):
            continue
    return cpi, [path]


def build_rows(bis, cpi, currencies):
    """The panel rows, 1973-2024, for each currency in order; every value required."""
    us = cpi.get("US", {})
    rows = []
    for cur in currencies:
        area = CURRENCY_AREA[cur]
        xr = bis.get((area, cur), {})
        ci = cpi.get(area, {})
        for y in range(WINDOW_START, WINDOW_END + 1):
            if y not in xr or y not in ci or y not in us:
                sys.exit(f"{cur} {y}: missing {'X' if y not in xr else 'CPI'}; "
                         f"the panel needs every year {WINDOW_START}-{WINDOW_END}")
            ln_xr, p_i, p_us = math.log(xr[y]), math.log(ci[y]), math.log(us[y])
            rows.append(dict(currency=cur, iso2=area, year=y, xr=xr[y], cpi=ci[y],
                             cpi_us=us[y], ln_xr=ln_xr, p_i=p_i, p_us=p_us,
                             q=ln_xr + p_us - p_i))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True,
                    help="directory holding (or receiving, with --fetch) the raw files")
    ap.add_argument("--currencies", required=True,
                    help="comma-separated ISO 4217 codes, in output order")
    ap.add_argument("--fetch", action="store_true",
                    help="download missing raw files into --cache")
    ap.add_argument("--out", default=OUT,
                    help="the panel (default: the package's inputs/ppp_panel.csv)")
    a = ap.parse_args(argv)
    os.makedirs(a.cache, exist_ok=True)
    currencies = [c.strip() for c in a.currencies.split(",") if c.strip()]
    unknown = [c for c in currencies if c not in CURRENCY_AREA]
    if unknown:
        sys.exit(f"no issuer area for {unknown}: add it to CURRENCY_AREA")
    bis, read_b = load_bis_annual(a.cache, a.fetch)
    cpi, read_w = load_wb_cpi(a.cache, a.fetch)
    rows = build_rows(bis, cpi, currencies)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (repr(v) if isinstance(v, float) else v)
                        for k, v in r.items()})
    man = dict(output=os.path.basename(a.out), output_sha256=sha256(a.out),
               currencies=currencies, window=[WINDOW_START, WINDOW_END], T=T_WINDOW,
               formula="q = ln(X) + ln(CPI_US) - ln(CPI_i); "
                       "X in units of currency i per US dollar",
               sources=dict(bis=BIS_BULK_URL + " (FREQ=A, COLLECTION=A)",
                            wdi=WB_CPI_URL),
               raw_sha256={os.path.basename(p): sha256(p) for p in read_b + read_w})
    os.makedirs(RECORD_DIR, exist_ok=True)
    record = os.path.join(RECORD_DIR, os.path.splitext(os.path.basename(a.out))[0]
                          + ".manifest.json")
    with open(record, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(man, fh, indent=1)
        fh.write("\n")
    print(f"wrote {a.out}: {len(currencies)} currencies x {T_WINDOW} years "
          f"(run record {record})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
