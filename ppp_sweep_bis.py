# =============================================================================
# ppp_sweep_bis.py -- Exhaustive admissibility sweep over the BIS currency
#                     universe for the Model LB PPP application.
#
# PURPOSE
#   Reproduces, as an audited artifact, the admissibility cascade of Section 6
#   ("An admissible universe, not a sample of convenience"): applies the gates
#   C1-C5 to every currency with a BIS annual USD series and reports the
#   per-currency gate table and the funnel counts. The printed funnel is the
#   provenance for the cascade sentence of the paper.
#
# DATA (fetched with --fetch, cached under <out>/cache/)
#   BIS: WS_XRU bulk flat CSV, official static export
#        https://data.bis.org/static/bulk/WS_XRU_csv_flat.zip
#        (US dollar exchange rates; FREQ=A annual, COLLECTION=A period average)
#   WB : CPI index FP.CPI.TOTL (2010=100), all countries,
#        https://api.worldbank.org/v2/country/all/indicator/FP.CPI.TOTL
#   Citation in the paper: "BIS bilateral nominal USD rates and World Bank CPI
#   (2010=100), accessed <date>".
#
# CONSTRUCTION (identical to the empirical pipeline)
#   q_t = ln(CPI_US,t) - ln(CPI_i,t) - ln(XR_i,t),  XR in local currency / USD.
#
# GATES (window 1973-2024, W = 52 years) -- thresholds are PRE-REGISTERED here
#   TRIAGE (C1) own currency, continuous:
#     C1a exactly one currency code within the window, XR present every year
#         1973-2024 (euro-legacy currencies fail: code switches or series ends);
#     C1b not a hard peg: sd(dln XR) >= PEG_SD_MIN and share of years with
#         |dln XR| < 1e-6 at most PEG_ZERO_MAX (currency-union members and hard
#         pegs fail here; micro-states typically fail CPI availability in C2).
#   C2  span: matched q series (XR and CPI and US CPI) with T >= T_MIN = 45.
#   C3  inflation ceiling: max annual CPI inflation over matched years < 0.30.
#   C4  conditional homoskedasticity of dq, three statistics:
#         S1 variance ratio  VR   = Var(dq, 2nd half) / Var(dq, 1st half),
#            two-sided score max(VR, 1/VR);
#         S2 Brown-Forsythe robust Levene across halves (median-centred),
#            p-value from F(1, T-2);
#         S3 Inclan-Tiao CUSUM-of-squares  IT = max_k sqrt(T/2)|C_k/C_T - k/T|.
#       Modes (--c4-mode, default both):
#         absolute : VR in [1/3, 3]  and  p_BF >= 0.05  and  IT < 1.358 (5%);
#         envelope : each score <= max over the ADMITTED_8 reference set
#                    (the operationalisation used in the original evaluation;
#                    mildly circular by construction -- reported for audit).
#       Reported, not gating: ROLL = max/min of rolling 10-year variances
#       (an additional stability diagnostic a referee may request).
#   C5  known-date overlay: survivors of C1-C4 must carry at least one
#       exogenously datable currency-regime event (exog_dates.csv). Survivors
#       whose only datable event is a twin (currency-banking) crisis -- a level
#       shift arriving jointly with a variance shift -- are excluded to the
#       variance-break extension (TWIN_CRISIS list, documented below).
#
# OUTPUTS (under --out, default ppp_sweep/)
#   sweep_gates.csv : one row per (area, currency) with every statistic + flag;
#   funnel.txt      : the cascade counts in the paper's order, both C4 modes;
#   admitted.txt    : final admitted list per C4 mode.
#   The run is deterministic (no RNG).
#
# USAGE
#   python ppp_sweep_bis.py --fetch --exog exog_dates.csv --out ppp_sweep
#   python ppp_sweep_bis.py --bis-zip WS_XRU_csv_flat.zip --cpi-json wb_cpi.json \
#                           --exog exog_dates.csv --out ppp_sweep
#
# DEPENDENCIES: numpy, scipy (Brown-Forsythe p-value). Pure standard library
# otherwise; Windows/PowerShell friendly.
# =============================================================================

import argparse
import csv
import io
import json
import math
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict

import numpy as np
from scipy import stats as st

# ----------------------------- pre-registered constants ---------------------
WINDOW_START, WINDOW_END = 1973, 2024
W = WINDOW_END - WINDOW_START + 1          # 52
T_MIN = 45                                  # C2
INFL_CEIL = 0.30                            # C3
PEG_SD_MIN = 0.005                          # C1b: 0.5% annual log change
PEG_ZERO_MAX = 0.50                         # C1b: at most half the years frozen
VR_LO, VR_HI = 1.0 / 3.0, 3.0               # C4 absolute
BF_ALPHA = 0.05                             # C4 absolute
IT_CV_5 = 1.358                             # C4 absolute (Inclan-Tiao, 5%)
ROLL_WIN = 10                               # diagnostic only
ADMITTED_8 = ["AUD", "CAD", "CHF", "GBP", "JPY", "NOK", "NZD", "SEK"]
# Twin-crisis exclusions (C5): the only datable regime event is a joint
# level+variance shift; documented in Section 6 and the conclusion.
TWIN_CRISIS = {"KRW": "1997 twin currency-banking crisis",
               "THB": "1997 twin currency-banking crisis",
               "ZAR": "crisis-driven identification (level+variance)"}

BIS_BULK_URL = "https://data.bis.org/static/bulk/WS_XRU_csv_flat.zip"
WB_CPI_URL = ("https://api.worldbank.org/v2/country/all/indicator/FP.CPI.TOTL"
              "?format=json&per_page=20000&date={}:{}").format(
                  WINDOW_START - 1, WINDOW_END)


# ----------------------------- fetch & cache --------------------------------
def _download(url, dest, label):
    print(f"[fetch] {label}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "ppp-sweep/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"[fetch] saved -> {dest} ({os.path.getsize(dest):,} bytes)")


def load_bis_annual(cache_dir, bis_zip=None, fetch=False):
    """Return dict[(iso2, cur)] -> {year: xr}, annual period-average USD rates.

    Streams the bulk flat CSV, keeping FREQ=A and COLLECTION=A only, and caches
    the filtered subset for instant re-runs.
    """
    sub = os.path.join(cache_dir, "bis_xru_annual_A.csv")
    if not os.path.exists(sub):
        zpath = bis_zip or os.path.join(cache_dir, "WS_XRU_csv_flat.zip")
        if not os.path.exists(zpath):
            if not fetch:
                sys.exit("BIS zip not found; pass --fetch or --bis-zip PATH")
            _download(BIS_BULK_URL, zpath, "BIS WS_XRU bulk")
        print("[bis] streaming annual/average subset out of the bulk file ...")
        with zipfile.ZipFile(zpath) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            with z.open(name) as fh, open(sub, "w", newline="",
                                          encoding="utf-8") as out:
                rdr = csv.reader(io.TextIOWrapper(fh, encoding="utf-8"))
                wtr = csv.writer(out)
                header = next(rdr)
                idx = {c.split(":")[0]: i for i, c in enumerate(header)}
                need = ["FREQ", "REF_AREA", "CURRENCY", "COLLECTION",
                        "TIME_PERIOD", "OBS_VALUE"]
                wtr.writerow(need)
                kept = 0
                for row in rdr:
                    if not row or not row[idx["FREQ"]].startswith("A"):
                        continue
                    if not row[idx["COLLECTION"]].startswith("A"):
                        continue
                    wtr.writerow([row[idx[c]].split(":")[0].strip()
                                  if c != "OBS_VALUE" else row[idx[c]]
                                  for c in need])
                    kept += 1
        print(f"[bis] annual subset cached ({kept:,} rows) -> {sub}")
    data = defaultdict(dict)
    with open(sub, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                y = int(r["TIME_PERIOD"]); v = float(r["OBS_VALUE"])
            except (ValueError, TypeError):
                continue
            if v > 0:
                data[(r["REF_AREA"], r["CURRENCY"])][y] = v
    print(f"[bis] series loaded: {len(data)} (area,currency) pairs")
    return data


def load_wb_cpi(cache_dir, cpi_json=None, fetch=False):
    """Return dict[iso2] -> {year: cpi} and the US series."""
    jpath = cpi_json or os.path.join(cache_dir, "wb_cpi.json")
    if not os.path.exists(jpath):
        if not fetch:
            sys.exit("WB CPI json not found; pass --fetch or --cpi-json PATH")
        _download(WB_CPI_URL, jpath, "World Bank CPI FP.CPI.TOTL")
    with open(jpath, encoding="utf-8") as fh:
        payload = json.load(fh)
    if not (isinstance(payload, list) and len(payload) == 2):
        sys.exit("WB payload unexpected; delete the cache and re-fetch")
    cpi = defaultdict(dict)
    for e in payload[1]:
        if e.get("value") is None:
            continue
        iso2 = (e.get("country") or {}).get("id", "")
        try:
            cpi[iso2][int(e["date"])] = float(e["value"])
        except (ValueError, TypeError):
            continue
    us = cpi.get("US", {})
    if len(us) < W:
        sys.exit("US CPI series incomplete in the WB payload")
    print(f"[wb] CPI loaded for {len(cpi)} country codes; US years: {len(us)}")
    return cpi, us


# ----------------------------- gate statistics ------------------------------
def variance_ratio(dq):
    h = len(dq) // 2
    v1, v2 = np.var(dq[:h], ddof=1), np.var(dq[h:], ddof=1)
    if min(v1, v2) <= 0:
        return np.inf
    return v2 / v1


def brown_forsythe_p(dq):
    h = len(dq) // 2
    g1, g2 = dq[:h], dq[h:]
    z1 = np.abs(g1 - np.median(g1)); z2 = np.abs(g2 - np.median(g2))
    n1, n2 = len(z1), len(z2); N = n1 + n2
    zb1, zb2 = z1.mean(), z2.mean(); zb = (z1.sum() + z2.sum()) / N
    num = (N - 2) * (n1 * (zb1 - zb) ** 2 + n2 * (zb2 - zb) ** 2)
    den = ((z1 - zb1) ** 2).sum() + ((z2 - zb2) ** 2).sum()
    if den <= 0:
        return 0.0
    F = num / den
    return float(st.f.sf(F, 1, N - 2))


def inclan_tiao(dq):
    e2 = dq ** 2
    C = np.cumsum(e2); CT = C[-1]
    if CT <= 0:
        return np.inf
    k = np.arange(1, len(dq) + 1)
    D = C / CT - k / len(dq)
    return float(np.sqrt(len(dq) / 2.0) * np.max(np.abs(D)))


def bb_sup_cv(alpha, K=200):
    """Critical value x solving P(sup|B°|>x)=2*sum_k (-1)^(k+1) exp(-2 k^2 x^2)=alpha."""
    def tail(x):
        s = 0.0
        for k in range(1, K + 1):
            s += (-1) ** (k + 1) * math.exp(-2.0 * k * k * x * x)
        return 2.0 * s
    lo, hi = 0.3, 4.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tail(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def vr_pvalue(dq):
    """Two-sided F p-value for the half-sample variance ratio."""
    h = len(dq) // 2
    g1, g2 = dq[:h], dq[h:]
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    if min(v1, v2) <= 0:
        return 0.0
    F = v2 / v1
    d2, d1 = len(g2) - 1, len(g1) - 1
    return float(2.0 * min(st.f.cdf(F, d2, d1), st.f.sf(F, d2, d1)))


def kappa2(dq):
    """Sanso-Arago-Carrion-i-Silvestre (2004) HAC-robust CUSUM-of-squares:
    kappa2 = T^{-1/2} max_k |C_k - (k/T) C_T| / sqrt(omega4_hat), with
    omega4_hat a Bartlett-kernel HAC long-run variance of e_t^2 - sigma2_hat,
    Newey-West bandwidth m = floor(4 (T/100)^{2/9}). Same Brownian-bridge
    limit as the raw Inclan-Tiao statistic, robust to non-Gaussian and
    serially dependent innovations."""
    e2 = dq ** 2
    T = len(e2)
    C = np.cumsum(e2)
    k = np.arange(1, T + 1)
    Dstar = np.max(np.abs(C - (k / T) * C[-1])) / math.sqrt(T)
    eta = e2 - e2.mean()
    m = int(np.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    g0 = float(np.mean(eta * eta))
    om4 = g0
    for j in range(1, m + 1):
        gj = float(np.mean(eta[j:] * eta[:-j]))
        om4 += 2.0 * (1.0 - j / (m + 1.0)) * gj
    om4 = max(om4, 1e-12 * max(g0, 1e-300))
    return float(Dstar / math.sqrt(om4))


def rolling_var_ratio(dq, win=ROLL_WIN):
    if len(dq) < 2 * win:
        return np.nan
    v = [np.var(dq[i:i + win], ddof=1) for i in range(len(dq) - win + 1)]
    v = [x for x in v if x > 0]
    return (max(v) / min(v)) if v else np.nan


# ----------------------------- the sweep -------------------------------------
def run_sweep(bis, cpi, us_cpi, exog_map, outdir, c4_mode, alpha_grid=None):
    AREAS_PER_CUR = defaultdict(set)
    for (a, c), s in bis.items():
        if any(WINDOW_START <= y <= WINDOW_END for y in s):
            AREAS_PER_CUR[c].add(a)
    rows = []
    for (iso2, cur), xr in sorted(bis.items()):
        rec = dict(iso2=iso2, currency=cur)
        yrs_xr = [y for y in range(WINDOW_START, WINDOW_END + 1) if y in xr]
        # C1a: continuity + single currency code + reaches the window end.
        # Own-currency rule: a currency shared by multiple reference areas is a
        # currency union or an externally issued unit (EUR, XOF, XAF, XCD, USD
        # in dollarised areas); the euro additionally postdates the window
        # start, so any pre-1999 EUR observation is back-calculated. Both fail
        # the "own currency, continuous" condition of the paper's C1.
        codes_in_window = {c for (a, c), s in bis.items()
                           if a == iso2 and any(WINDOW_START <= y <= WINDOW_END
                                                for y in s)}
        rec["n_xr"] = len(yrs_xr)
        # Issuer rule (ISO 4217): the pair is the currency's OWN series iff
        # the reference area is the issuer, identified by iso2 == currency[:2]
        # (AUD-AU, CHF-CH, GBP-GB, ...). Non-issuer users of a shared currency
        # (Kiribati-AUD, Liechtenstein-CHF, dollarised areas-USD), currency
        # unions (EUR, XOF, XAF, XCD: no issuing iso2) and the back-calculated
        # euro-legacy series all fail here, exactly the paper's C1 intent.
        rec["issuer"] = int(iso2 == cur[:2])
        rec["C1a"] = int(len(yrs_xr) == W and len(codes_in_window) == 1
                         and rec["issuer"] and cur != "EUR")
        # C1b: hard-peg screen on dln XR over available window years
        if len(yrs_xr) >= 10:
            lx = np.log([xr[y] for y in yrs_xr])
            dlx = np.diff(lx)
            rec["peg_sd"] = float(np.std(dlx, ddof=1))
            rec["peg_zero_share"] = float(np.mean(np.abs(dlx) < 1e-6))
            rec["C1b"] = int(rec["peg_sd"] >= PEG_SD_MIN
                             and rec["peg_zero_share"] <= PEG_ZERO_MAX)
        else:
            rec["peg_sd"] = np.nan; rec["peg_zero_share"] = np.nan
            rec["C1b"] = 0
        rec["TRIAGE"] = int(rec["C1a"] and rec["C1b"])
        # matched q series
        c = cpi.get(iso2, {})
        yrs = [y for y in yrs_xr if y in c and y in us_cpi]
        rec["T"] = len(yrs)
        rec["C2"] = int(len(yrs) >= T_MIN)
        if len(yrs) >= 10:
            q = np.array([math.log(us_cpi[y]) - math.log(c[y])
                          - math.log(xr[y]) for y in yrs])
            infl = np.array([c[y] / c[y - 1] - 1.0 for y in yrs
                             if (y - 1) in c])
            rec["max_infl"] = float(np.max(infl)) if len(infl) else np.nan
            rec["C3"] = int(len(infl) > 0 and rec["max_infl"] < INFL_CEIL)
            dq = np.diff(q)
            vr = variance_ratio(dq)
            rec["VR"] = float(vr)
            rec["VR_score"] = float(max(vr, 1.0 / vr)) if np.isfinite(vr) else np.inf
            rec["VR_p"] = vr_pvalue(dq)
            rec["BF_p"] = brown_forsythe_p(dq)
            rec["IT"] = inclan_tiao(dq)
            rec["K2"] = kappa2(dq)
            rec["ROLL10"] = rolling_var_ratio(dq)
            rec["C4_abs"] = int(VR_LO <= vr <= VR_HI
                                and rec["BF_p"] >= BF_ALPHA
                                and rec["IT"] < IT_CV_5)
        else:
            for k in ("max_infl", "VR", "VR_score", "VR_p", "BF_p", "IT",
                      "K2", "ROLL10"):
                rec[k] = np.nan
            rec["C3"] = 0; rec["C4_abs"] = 0
        rows.append(rec)

    # C4 envelope: reference = max score over the ADMITTED_8 (audited-circular)
    ref = [r for r in rows if r["currency"] in ADMITTED_8
           and r["TRIAGE"] and r["C2"]]
    env = {"VR_score": max(r["VR_score"] for r in ref),
           "IT": max(r["IT"] for r in ref),
           "BF_p": min(r["BF_p"] for r in ref)}
    for r in rows:
        ok = (np.isfinite(r.get("VR_score", np.nan))
              and r["VR_score"] <= env["VR_score"] + 1e-12
              and r["IT"] <= env["IT"] + 1e-12
              and r["BF_p"] >= env["BF_p"] - 1e-12)
        r["C4_env"] = int(bool(ok))

    # C5 overlay
    for r in rows:
        cur = r["currency"]
        if cur in TWIN_CRISIS:
            r["C5"] = 0; r["C5_note"] = "twin crisis: " + TWIN_CRISIS[cur]
        elif cur in exog_map:
            r["C5"] = 1; r["C5_note"] = exog_map[cur]
        else:
            r["C5"] = 0; r["C5_note"] = "no documented exogenous event (curate)"

    # funnel in the paper's order, for each C4 mode
    def funnel(c4key):
        u0 = rows
        u1 = [r for r in u0 if r["TRIAGE"] and r["C2"]]
        u2 = [r for r in u1 if r["C3"]]
        u3 = [r for r in u2 if r[c4key]]
        u4 = [r for r in u3 if r["C5"]]
        return u0, u1, u2, u3, u4

    lines = []
    n_cur = len({r["currency"] for r in rows})
    lines.append(f"BIS annual/average universe: {len(rows)} (area,currency) "
                 f"series, {n_cur} distinct currencies; "
                 f"window {WINDOW_START}-{WINDOW_END}")
    lines.append(f"C4 envelope reference (max over admitted 8): "
                 f"VR_score<= {env['VR_score']:.3f}, IT<= {env['IT']:.3f}, "
                 f"BF_p>= {env['BF_p']:.3f}")
    for key, label in (("C4_abs", "absolute"), ("C4_env", "envelope")):
        if c4_mode not in ("both", label):
            continue
        u0, u1, u2, u3, u4 = funnel(key)
        lines.append("")
        lines.append(f"--- funnel (C4 mode: {label}) ---")
        lines.append(f"  universe (BIS annual series)        : {len(u0)}")
        lines.append(f"  after triage C1 + span C2           : {len(u1)}  "
                     f"(removed {len(u0)-len(u1)})")
        lines.append(f"  after inflation ceiling C3          : {len(u2)}  "
                     f"(removed {len(u1)-len(u2)})")
        lines.append(f"  after variance gate C4              : {len(u3)}  "
                     f"(removed {len(u2)-len(u3)})")
        lines.append(f"  after known-date condition C5       : {len(u4)}  "
                     f"(removed {len(u3)-len(u4)})")
        adm = sorted(r["currency"] for r in u4)
        lines.append(f"  admitted: {adm}")
        surv = sorted((r["currency"], r["C5_note"]) for r in u3 if not r["C5"])
        lines.append("  C5 removals (curated notes):")
        for cur, note in surv:
            lines.append(f"    {cur}: {note}")
    # ---- alpha-grid survival report (C4 absolute; S3 in {IT, K2}) ----
    if alpha_grid:
        post_c3 = [r for r in rows if r["TRIAGE"] and r["C2"] and r["C3"]]
        cvs = {a: bb_sup_cv(a) for a in alpha_grid}
        lines.append("")
        lines.append("--- C4-absolute alpha grid (VR two-sided F p, BF p, "
                     "S3 vs Brownian-bridge cv) ---")
        lines.append("  cv(alpha): " + ", ".join(
            f"{a:.2f}->{cvs[a]:.3f}" for a in alpha_grid))
        hdr = (f"  {'cur':<5}{'VR_p':>7}{'BF_p':>7}{'IT':>7}{'K2':>7}  "
               + "  ".join(f"IT@{int(a*100)}%" for a in alpha_grid) + "  "
               + "  ".join(f"K2@{int(a*100)}%" for a in alpha_grid))
        lines.append(hdr)
        surv = {("IT", a): [] for a in alpha_grid}
        surv.update({("K2", a): [] for a in alpha_grid})
        for r in sorted(post_c3, key=lambda x: x["currency"]):
            cur = r["currency"]
            marks_it, marks_k2 = [], []
            for a in alpha_grid:
                ok_common = (r["VR_p"] >= a and r["BF_p"] >= a)
                ok_it = ok_common and r["IT"] < cvs[a]
                ok_k2 = ok_common and r["K2"] < cvs[a]
                marks_it.append("pass " if ok_it else " --  ")
                marks_k2.append("pass " if ok_k2 else " --  ")
                if ok_it:
                    surv[("IT", a)].append(cur)
                if ok_k2:
                    surv[("K2", a)].append(cur)
            lines.append(f"  {cur:<5}{r['VR_p']:>7.3f}{r['BF_p']:>7.3f}"
                         f"{r['IT']:>7.3f}{r['K2']:>7.3f}  "
                         + " ".join(marks_it) + "  " + " ".join(marks_k2))
        lines.append("")
        for s3 in ("IT", "K2"):
            for a in alpha_grid:
                adm = sorted(c for c in surv[(s3, a)]
                             if c in exog_map and c not in TWIN_CRISIS)
                lines.append(f"  [S3={s3}, alpha={a:.2f}] C4 pass: "
                             f"{len(surv[(s3, a)]):>2}; admitted after C5: "
                             f"{len(adm)} {adm}")
    report = "\n".join(lines)
    print("\n" + report + "\n")

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "funnel.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    cols = ["iso2", "currency", "n_xr", "issuer", "C1a", "peg_sd", "peg_zero_share",
            "C1b", "TRIAGE", "T", "C2", "max_infl", "C3", "VR", "VR_score",
            "VR_p", "BF_p", "IT", "K2", "ROLL10", "C4_abs", "C4_env", "C5", "C5_note"]
    with open(os.path.join(outdir, "sweep_gates.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"[out] {os.path.join(outdir, 'sweep_gates.csv')}")
    print(f"[out] {os.path.join(outdir, 'funnel.txt')}")


def load_exog(path):
    """exog_dates.csv -> {currency: 'dates (events)'} for the C5 overlay."""
    m = {}
    if not path or not os.path.exists(path):
        print("[warn] exog_dates.csv not found; C5 will mark every survivor "
              "as 'curate'")
        return m
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cur = (r.get("currency") or r.get("cur") or "").strip()
            if not cur:
                continue
            note = ", ".join(f"{k}={v}" for k, v in r.items()
                             if k != "currency" and v)
            m.setdefault(cur, note or "documented event")
    print(f"[exog] events loaded for: {sorted(m)}")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true",
                    help="download BIS bulk + WB CPI into the cache")
    ap.add_argument("--bis-zip", default=None)
    ap.add_argument("--cpi-json", default=None)
    ap.add_argument("--exog", default="exog_dates.csv")
    ap.add_argument("--out", default="ppp_sweep")
    ap.add_argument("--c4-mode", choices=["absolute", "envelope", "both"],
                    default="both")
    ap.add_argument("--c4-alpha-grid", default=None,
                    help="comma list, e.g. 0.05,0.04,0.03,0.02,0.01: report "
                         "C4-absolute survival at each level, S3 in {IT,K2}")
    args = ap.parse_args()
    alpha_grid = ([float(x) for x in args.c4_alpha_grid.split(",")]
                  if args.c4_alpha_grid else None)

    cache = os.path.join(args.out, "cache")
    os.makedirs(cache, exist_ok=True)
    bis = load_bis_annual(cache, args.bis_zip, args.fetch)
    cpi, us = load_wb_cpi(cache, args.cpi_json, args.fetch)
    exog = load_exog(args.exog)
    run_sweep(bis, cpi, us, exog, args.out, args.c4_mode, alpha_grid)


if __name__ == "__main__":
    main()
