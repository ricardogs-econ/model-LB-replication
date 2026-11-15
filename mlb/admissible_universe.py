"""The admissible universe of the PPP application: the conditions C0-C6 over the BIS
annual USD series.

Data building: this module needs the raw downloads of build_ppp_panel (BIS WS_XRU,
WDI FP.CPI.TOTL), and neither the quick nor the full mode runs it. The three files it
writes (under output/, see --out-dir) ship in inputs/ppp_sweep/.

A series (reference area, currency) belongs to the universe if and only if it meets
C0-C6 (window 1973-2024, T = 52):

  C0  industrialized at the start of the window: an OECD member by 31 December 1973
      (inputs/oecd_1973.csv, the date the Convention entered into force for it; the
      deposit dates give the same members), the numeraire (USD) excluded;
  C1  own currency, observed continuously in 1973-2024:
        C1a  the reference area issues the currency (ISO 4217: area == the currency
             code's first two letters), one currency code in the window, X in every
             year, not the euro;
        C1b  not a hard peg: sd(d ln X) >= 0.005 and at most half of the years with
             |d ln X| < 1e-6;
  C3  annual CPI inflation always below 30% (the maximum over the years with CPI in t
      and t - 1);
  C5  at least one datable event under rule E (inputs/regime_events.csv); evaluated for
      the series that pass C0, C1 and C3 (blank otherwise);
  C4  the unconditional variance of Delta q is stable: kappa2 of Sansó, Aragó and
      Carrion-i-Silvestre (2004) applied to eps_hat, the OLS residual of Delta q_t on a
      constant and the impulses 1{t = tau_j} at the event dates of rule E (the level
      shifts of the breaks are outliers in Delta q, not variance breaks); rejects at 5%
      if kappa2 > 1.358 (at 1%: 1.628), the quantiles of sup|Brownian bridge|;
  C6  no variance break at the event dates: W, the Wald statistic that the regime
      indicators 1{t >= tau_j}, j = 1..m, are jointly zero in the OLS regression of
      eps_hat_t^2 on a constant and them, with a Bartlett HAC covariance of lag
      L = floor(4 (T/100)^(2/9)); rejects at 5% (at 1%) when W exceeds the critical
      value simulated at T = 51 on the currency's own design
      (reference/application/admission/c6_critical_values.csv, produced and validated by
      mlb/c6_critical_values.py: the asymptotic chi2(m) does not give
      a test of the stated level at T = 51). The verdict is that rule; the p-value,
      #{draws >= W} / R from the same draws, agrees with it and is reported with its
      Monte Carlo se; the chi2(m) p-value is reported, and it defines the descriptive
      "as registered" block below.

The attrition is reported in the order BIS -> C0 -> C1 -> C3 -> C5 -> C4 -> C6, at 5%,
with two sensitivities: C4 and C6 at 1%, and the universe without C6. funnel.txt also
carries a descriptive counterfactual, the rule as registered before its correction:
C6 with the asymptotic chi2(m) reference at 5% (a currency fails iff its chi2(m)
p-value is not finite or below 0.05); no simulation uses that universe. C2 (T >= 45) is
not a condition: C1 implies it for X. C4 and C6 are computed on
q_t = ln X_t + ln P_US,t - ln P_t (build_ppp_panel) for the series that pass C0, C1, C3
and C5 and have X and both CPIs in every year of the window (q_complete); a series
without a complete q could not be tested and fails C4.

kappa2 (Sansó et al. 2004): with C_k = sum_{t<=k} eps_t^2 and sigma2 = C_T / T,
    kappa2 = max_k |C_k - (k/T) C_T| / sqrt(T omega4),
omega4 the long-run variance of v_t = eps_t^2 - sigma2 with Sansó et al.'s Bartlett
window, omega4 = g_0 + 2 sum_{l=1}^{m} (1 - l/(m + 1)) g_l, g_l = T^-1 sum_{t>l}
v_t v_{t-l}, the lag m = floor(S) chosen by the automatic procedure of Newey and West
(1994): preliminary lag n = floor(4 (T/100)^(2/9)), s0 = g_0 + 2 sum_{j<=n} g_j,
s1 = 2 sum_{j<=n} j g_j, gamma = 1.1447 ((s1/s0)^2)^(1/3), S = gamma T^(1/3); if
s0 <= 0, m = n. C6's covariance uses the same Bartlett window with its fixed lag.

The candidates of C4 and C6, with their statistics and the thresholds, are written to
candidates.csv, and sweep.manifest.json records the sha256 of the three outputs, of the
inputs they rest on and of the raw files, with the thresholds.

    python mlb/admissible_universe.py \\
        --cache <dir with the raw BIS and WDI files> [--out-dir <dir>] [--check]

The files are written to --out-dir (default output/replicate/steps/admissible_universe);
the shipped inputs/ppp_sweep/ is never written (--out-dir may not name it). --check
compares the three outputs with the shipped ones by sha256 and exits 1 on a difference.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np
from scipy import stats as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlb import build_ppp_panel as P  # noqa: E402
from mlb import paths  # noqa: E402
from mlb import regime_events as R  # noqa: E402

WINDOW_START, WINDOW_END = 1973, 2024
W = WINDOW_END - WINDOW_START + 1  # 52
NUMERAIRE = "US"
OECD_CUTOFF = "1973-12-31"
PEG_SD_MIN = 0.005  # C1b
PEG_ZERO_MAX = 0.50  # C1b
INFL_CEIL = 0.30  # C3
# C4: quantiles of sup|Brownian bridge| (see kolmogorov_quantile)
KAPPA2_CV = {0.05: 1.358, 0.01: 1.628}
LEVELS = (0.05, 0.01)
OECD = os.path.join(_ROOT, "inputs", "oecd_1973.csv")
C6_CV = str(paths.resolve("c6_critical_values"))
SHIPPED_DIR = os.path.join(_ROOT, "inputs", "ppp_sweep")  # read, never written
OUT_DIR = os.path.join(_ROOT, "output", "replicate", "steps", "admissible_universe")
OUTPUTS = ("sweep_gates.csv", "funnel.txt", "candidates.csv")
CONDITIONS = ["C0", "C1", "C3", "C5", "C4", "C6"]


# --------------------------------------------------------------------------- statistics
def kolmogorov_quantile(p, K=200):
    """x with P(sup|B| <= x) = p, B a Brownian bridge:
    P(sup|B| > x) = 2 sum (-1)^(k+1) exp(-2 k^2 x^2)."""
    def tail(x):
        return 2.0 * sum((-1) ** (k + 1) * math.exp(-2.0 * k * k * x * x)
                         for k in range(1, K + 1))
    lo, hi = 0.3, 4.0
    while hi - lo > 1e-12:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tail(mid) > 1.0 - p else (lo, mid)
    return 0.5 * (lo + hi)


def nw1994_bandwidth(v):
    """Newey-West (1994) automatic bandwidth S for the Bartlett kernel, on the demeaned
    series v; the preliminary lag n when s0 <= 0."""
    T = len(v)
    n = int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))
    g = [float(np.dot(v[j:], v[:T - j])) / T for j in range(n + 1)]
    s0 = g[0] + 2.0 * sum(g[1:])
    s1 = 2.0 * sum(j * g[j] for j in range(1, n + 1))
    if s0 <= 0.0:  # degenerate: fall back on the preliminary lag
        return float(n)
    gamma = 1.1447 * ((s1 / s0) ** 2) ** (1.0 / 3.0)
    return gamma * T ** (1.0 / 3.0)


def bartlett_lrv(v, m):
    """g_0 + 2 sum_{l=1}^{m} (1 - l/(m + 1)) g_l on the demeaned series v."""
    T = len(v)
    out = float(np.dot(v, v)) / T
    for lag in range(1, min(int(m), T - 1) + 1):
        out += 2.0 * (1.0 - lag / (m + 1.0)) * float(np.dot(v[lag:], v[:T - lag])) / T
    return out


def kappa2(eps):
    """(kappa2, the bandwidth S, the lag m = floor(S), omega4) of Sansó, Aragó and
    Carrion-i-Silvestre (2004) on eps."""
    e2 = np.asarray(eps, dtype=float) ** 2
    T = len(e2)
    C = np.cumsum(e2)
    k = np.arange(1, T + 1)
    v = e2 - C[-1] / T
    S = nw1994_bandwidth(v)
    m = int(math.floor(S))
    om4 = bartlett_lrv(v, m)
    if om4 <= 0.0:  # eps^2 constant: untestable (fails C4)
        return float("nan"), S, m, om4
    return (float(np.max(np.abs(C - (k / T) * C[-1])) / math.sqrt(T * om4)), S, m,
            om4)


def impulse_residual(dq, t_years, taus):
    """eps_hat: the OLS residual of dq on a constant and the impulses 1{t = tau_j}."""
    X = np.column_stack([np.ones(len(dq))]
                        + [(t_years == t).astype(float) for t in taus])
    b, *_ = np.linalg.lstsq(X, dq, rcond=None)
    return dq - X @ b


def hac_lag(T):
    return int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))


def wald_hac(y, X, cols, L):
    """Wald statistic of b[cols] = 0 in the OLS of y on X, Bartlett HAC covariance of
    lag L."""
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ b
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for lag in range(1, L + 1):
        G = Xu[lag:].T @ Xu[:-lag]
        S += (1.0 - lag / (L + 1.0)) * (G + G.T)
    XX = np.linalg.inv(X.T @ X)
    V = XX @ S @ XX
    bc = b[cols]
    return float(bc @ np.linalg.solve(V[np.ix_(cols, cols)], bc))


def c6_test(eps, t_years, taus):
    """(Wald, m, chi2(m) p) of the regime indicators 1{t >= tau_j} in the regression of
    eps^2 on them."""
    e2 = np.asarray(eps, dtype=float) ** 2
    X = np.column_stack([np.ones(len(e2))]
                        + [(t_years >= t).astype(float) for t in taus])
    m = len(taus)
    Wd = wald_hac(e2, X, list(range(1, m + 1)), hac_lag(len(e2)))
    return Wd, m, float(st.chi2.sf(Wd, m))


# ----------------------------------------------------------------------- the conditions
def load_oecd(path=OECD):
    """{iso2: (country, date in force)} of the members by the cutoff, the numeraire
    excluded."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {r["iso2"]: (r["country"], r["convention_in_force"]) for r in rows
            if r["convention_in_force"] <= OECD_CUTOFF and r["numeraire"] == "N"}


def triage(bis, iso2, cur):
    """C1 (C1a and C1b) of one BIS series."""
    xr = bis[(iso2, cur)]
    yrs = [y for y in range(WINDOW_START, WINDOW_END + 1) if y in xr]
    codes = {c for (a, c), s in bis.items()
             if a == iso2 and any(WINDOW_START <= y <= WINDOW_END for y in s)}
    rec = dict(n_xr=len(yrs), issuer=int(iso2 == cur[:2]))
    rec["C1a"] = int(len(yrs) == W and len(codes) == 1 and rec["issuer"] == 1
                     and cur != "EUR")
    if len(yrs) >= 10:
        dlx = np.diff(np.log([xr[y] for y in yrs]))
        rec["peg_sd"] = float(np.std(dlx, ddof=1))
        rec["peg_zero_share"] = float(np.mean(np.abs(dlx) < 1e-6))
        rec["C1b"] = int(rec["peg_sd"] >= PEG_SD_MIN
                         and rec["peg_zero_share"] <= PEG_ZERO_MAX)
    else:
        rec.update(peg_sd=float("nan"), peg_zero_share=float("nan"), C1b=0)
    rec["C1"] = int(rec["C1a"] and rec["C1b"])
    return rec


def max_inflation(xr, cpi_i, cpi_us):
    """C3's statistic: the maximum of CPI_t / CPI_{t-1} - 1 over the window years with
    X, CPI and the US CPI (and CPI in t - 1); undefined (fails C3) with fewer than 10
    such matched years."""
    yrs = [y for y in range(WINDOW_START, WINDOW_END + 1)
           if y in xr and y in cpi_i and y in cpi_us]
    infl = [cpi_i[y] / cpi_i[y - 1] - 1.0 for y in yrs if y - 1 in cpi_i]
    return max(infl) if len(yrs) >= 10 and infl else float("nan")


def gate_statistics(q_by_year, taus):
    """kappa2 and C6 on the impulse residual of Delta q (t = 1974..2024) at the event
    dates taus."""
    years = np.arange(WINDOW_START, WINDOW_END + 1)
    q = np.array([q_by_year[y] for y in years])
    dq, t = np.diff(q), years[1:]
    eps = impulse_residual(dq, t, taus)
    k2, S, m_lag, om4 = kappa2(eps)
    wd, m, p = c6_test(eps, t, taus)
    return dict(T_dq=len(dq), kappa2=k2, nw_bandwidth=S, nw_lag=m_lag, omega4=om4,
                c6_wald=wd, c6_df=m, c6_p_chi2=p)


def sweep(bis, cpi, oecd, events, c6_cv=None):
    """One record per BIS series: the conditions in the attrition order, with their
    statistics. c6_cv: the simulated critical values,
    {currency: row of reference/application/admission/c6_critical_values.csv}.
    """
    from mlb import c6_critical_values as C6
    assert c6_cv is not None, "C6 needs its simulated critical values"
    us = cpi.get(NUMERAIRE, {})
    ev_rows = R.load(events)
    ev_currencies = set(R.currencies(ev_rows))
    out = []
    for (iso2, cur) in sorted(bis):
        if not any(WINDOW_START <= y <= WINDOW_END for y in bis[(iso2, cur)]):
            continue
        rec = dict(iso2=iso2, currency=cur, C0=int(iso2 in oecd))
        rec.update(triage(bis, iso2, cur))
        c = cpi.get(iso2, {})
        rec["max_infl"] = max_inflation(bis[(iso2, cur)], c, us)
        rec["C3"] = int(math.isfinite(rec["max_infl"]) and rec["max_infl"] < INFL_CEIL)
        full = all(y in bis[(iso2, cur)] and y in c and y in us
                   for y in range(WINDOW_START, WINDOW_END + 1))
        rec["q_complete"] = int(full)
        if not (rec["C0"] and rec["C1"] and rec["C3"]):
            out.append(rec)  # C5, C4 and C6 not reached
            continue
        taus = R.configuration(ev_rows, cur, "E")[0] if cur in ev_currencies else []
        rec["events_E"] = ";".join(str(t) for t in taus)
        rec["C5"] = int(len(taus) >= 1)
        if rec["C5"] and full:
            q = {y: math.log(bis[(iso2, cur)][y]) + math.log(us[y]) - math.log(c[y])
                 for y in range(WINDOW_START, WINDOW_END + 1)}
            rec.update(gate_statistics(q, taus))
            cv = c6_cv[cur]
            assert cv["taus"] == ";".join(map(str, taus)), (cur, cv["taus"], taus)
            draws, same = C6.calibration(cur, cv)  # cv_5, cv_1 regenerated to 1e-9
            rec["c6_draws_sha_match"] = int(same)
            rec["c6_p"] = C6.p_value(rec["c6_wald"], draws)
            rec["c6_p_se"] = C6.p_value_se(rec["c6_p"], len(draws))
            for a in LEVELS:
                tag = f"{int(round(100 * a))}"
                rec[f"c6_cv_{tag}"] = float(cv[f"cv_{tag}"])
                rec[f"C4_{tag}"] = int(rec["kappa2"] <= KAPPA2_CV[a])
                rec[f"C6_{tag}"] = int(rec["c6_wald"] <= rec[f"c6_cv_{tag}"])
        out.append(rec)
    return out


COLUMNS = ["iso2", "currency", "C0", "n_xr", "issuer", "C1a", "peg_sd",
           "peg_zero_share", "C1b", "C1", "max_infl", "C3", "q_complete", "events_E",
           "C5", "T_dq", "kappa2", "nw_bandwidth", "nw_lag", "omega4", "C4_5", "C4_1",
           "c6_wald", "c6_df", "c6_cv_5", "c6_cv_1", "c6_p", "c6_p_se", "c6_p_chi2",
           "C6_5", "C6_1"]
CANDIDATE_COLUMNS = ["currency", "iso2", "events_E", "c6_df", "kappa2", "nw_bandwidth",
              "nw_lag", "c4_cv_5", "c4_cv_1", "C4_5", "C4_1", "c6_wald", "c6_cv_5",
              "c6_cv_1", "c6_p", "c6_p_se", "c6_p_chi2", "C6_5", "C6_1"]


def funnel(rows, level=0.05, with_c6=True, registered=False):
    """[(condition, survivors, removed)] in the attrition order at one level.

    registered=True replaces C6's simulated critical values by the rule as
    registered before its correction: the asymptotic chi2(m) reference, a currency
    failing C6 iff its chi2(m) p-value (c6_p_chi2) is not finite or is below the
    level. Descriptive only: nothing downstream uses this universe."""
    tag = f"{int(round(100 * level))}"
    keys = {"C0": "C0", "C1": "C1", "C3": "C3", "C5": "C5", "C4": f"C4_{tag}",
            "C6": f"C6_{tag}"}
    alive = list(rows)
    steps = [("BIS", alive, [])]
    for g in CONDITIONS:
        if g == "C6" and not with_c6:
            continue
        if g == "C6" and registered:
            keep = [r for r in alive
                    if math.isfinite(r["c6_p_chi2"]) and r["c6_p_chi2"] >= level]
        else:
            keep = [r for r in alive if r.get(keys[g]) == 1]
        kept = {id(r) for r in keep}
        steps.append((g, keep, [r for r in alive if id(r) not in kept]))
        alive = keep
    return steps


def _fmt(v):
    return repr(v) if isinstance(v, float) else str(v)


def write_candidates(rows, out_dir):
    """The candidates of C4 and C6 with their statistics."""
    cand = [r for r in rows if "kappa2" in r]
    with open(os.path.join(out_dir, "candidates.csv"), "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CANDIDATE_COLUMNS)
        for r in cand:
            r = dict(r, c4_cv_5=KAPPA2_CV[0.05], c4_cv_1=KAPPA2_CV[0.01])
            w.writerow([_fmt(r[c]) for c in CANDIDATE_COLUMNS])
    return cand


def _sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def write_manifest(out_dir, inputs, raw):
    """sweep.manifest.json: the outputs' sha256, the inputs' and the raw files'."""
    man = dict(
        outputs={n: _sha_lf(os.path.join(out_dir, n)) for n in OUTPUTS},
        inputs={os.path.basename(p): _sha_lf(p) for p in inputs},
        raw={os.path.basename(p): P.sha256(p) for p in raw},
        thresholds=dict(kappa2_5=KAPPA2_CV[0.05], kappa2_1=KAPPA2_CV[0.01],
                        c6="the simulated critical values of the c6 file",
                        c3_inflation_ceiling=INFL_CEIL, c1b_sd_min=PEG_SD_MIN,
                        c1b_zero_share_max=PEG_ZERO_MAX, c0_cutoff=OECD_CUTOFF))
    with open(os.path.join(out_dir, "sweep.manifest.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(man, fh, indent=1)
    return man


def write(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sweep_gates.csv"), "w", encoding="utf-8",
              newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([_fmt(r[c]) if c in r else "" for c in COLUMNS])
    lines = ["BIS annual period-average USD series with an observation in "
             f"{WINDOW_START}-{WINDOW_END}: {len(rows)} "
             f"(area, currency) pairs, {len({r['currency'] for r in rows})} currencies",
             f"C4 critical values: kappa2 > {KAPPA2_CV[0.05]} (5%), "
             f"> {KAPPA2_CV[0.01]} (1%); "
             f"C6: W against the critical values simulated at T = {W - 1} "
             "on each currency's design "
             f"(reference/application/admission/c6_critical_values.csv); HAC lag "
             f"{hac_lag(W - 1)}"]
    for label, level, c6, reg in (
            ("main: 5%", 0.05, True, False),
            ("sensitivity: C4 and C6 at 1%", 0.01, True, False),
            ("sensitivity: without C6 (C4 at 5%)", 0.05, False, False),
            ("as registered (C6 with the asymptotic chi2(m), 5%)", 0.05, True, True)):
        lines.append(f"\n--- attrition, {label} ---")
        if reg:
            lines.append("  (descriptive counterfactual: the C6 rule before its "
                         "correction to simulated critical values; no simulation "
                         "uses this universe)")
        for g, keep, removed in funnel(rows, level, c6, reg):
            lines.append(f"  {g:4s}: {len(keep):4d}"
                         + (f"  (removed {len(removed)})" if g != "BIS" else ""))
            if g in ("C5", "C4", "C6") and removed:
                lines.append("        removed: "
                             + ", ".join(
                                 f"{r['currency']} ({r['iso2']}"
                                 + (f"; chi2(m) p {r['c6_p_chi2']!r}"
                                    if reg and g == "C6" else "") + ")"
                                 for r in removed))
        final = funnel(rows, level, c6, reg)[-1][1]
        lines.append("  admitted: " + ", ".join(sorted(r["currency"] for r in final)))
    with open(os.path.join(out_dir, "funnel.txt"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return lines


def check_against_shipped(out_dir, shipped=SHIPPED_DIR):
    """[(file, sha256 of the output, sha256 of the shipped file)] for each of the three
    outputs that differs from the shipped one by sha256 (LF bytes); empty when equal."""
    diff = []
    for n in OUTPUTS:
        a, b = _sha_lf(os.path.join(out_dir, n)), _sha_lf(os.path.join(shipped, n))
        if a != b:
            diff.append((n, a, b))
    return diff


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--cache", required=True,
        help="directory with the raw BIS and WDI files (see build_ppp_panel)")
    ap.add_argument("--fetch", action="store_true",
                    help="download the raw files into the cache if missing")
    ap.add_argument("--out-dir", default=OUT_DIR,
                    help="where the files are written (default "
                    "output/replicate/steps/admissible_universe); not inputs/ppp_sweep")
    ap.add_argument("--check", action="store_true",
                    help="compare the three outputs with the shipped inputs/ppp_sweep "
                    "by sha256; exit 1 on a difference")
    a = ap.parse_args(argv)
    if os.path.realpath(a.out_dir) == os.path.realpath(SHIPPED_DIR):
        ap.error("--out-dir may not be inputs/ppp_sweep: the shipped files are read by "
                 "the application")
    from mlb import c6_critical_values as C6
    bis, bis_read = P.load_bis_annual(a.cache, a.fetch)
    cpi, cpi_read = P.load_wb_cpi(a.cache, a.fetch)
    rows = sweep(bis, cpi, load_oecd(OECD), R.EVENTS, c6_cv=C6.load(C6_CV))
    print("\n".join(write(rows, a.out_dir)))
    for r in write_candidates(rows, a.out_dir):
        # informative: bitwise identity of the draws across builds is not guaranteed
        print(f"c6 calibration draws match the file's sha256: {r['currency']} "
              f"{r['c6_draws_sha_match']}")
    write_manifest(a.out_dir, [R.EVENTS, C6_CV, OECD], bis_read + cpi_read)
    if a.check:
        diff = check_against_shipped(a.out_dir)
        for n, got, shipped in diff:
            print(f"--check: {n} differs from the shipped file "
                  f"(sha256 {got[:12]}... against {shipped[:12]}...)")
        if not diff:
            print("--check: the three outputs equal the shipped inputs/ppp_sweep files "
                  "by sha256")
        return 1 if diff else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
