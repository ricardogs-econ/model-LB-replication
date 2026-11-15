"""Robust level-shift tests on the nine real exchange rates (supplement, section
"Robust level-shift tests"; the article's paragraph on them).

The series are q = ln X + p_US - p_i, 1973-2024 (T = 52), read from
inputs/ppp_panel.csv. Each one goes through the port of the authors' MATLAB code in
mlb/level_shifts.py:

  * Harvey, Leybourne and Taylor (2010), the S_1, S_0 and union (U) tests at 5%,
    windows 0.10 (the value in the Carrion-i-Silvestre and Gadea application) and
    0.15 (= the trimming of the article: regimes of at least round(0.15 T) = 8
    observations); lag bound k_max = round(4 (T/100)^(1/4)) = 3, lag chosen by BIC;
  * Carrion-i-Silvestre and Gadea (2023), trimming 0.15, up to m_max = 5 breaks, both
    break-dating strategies (joint: the global SSR of the impulses in Delta q;
    one at a time): the UDmax and WDmax tests of zero breaks with the union
    decisions at 10, 5 and 1%, and the sequential F_1, F_1a, F_1b, F_0, F_Ua and F_Ub
    counts and dates at 10% and 5%.

All critical values are the authors' asymptotic ones (inputs/level_shift_cv/ and the
constants of the port). The scale reference of the same paragraph: the 1986 change of
q divided by the sample standard deviation of the 51 changes 1974-2024 (ddof = 1; the
1986 change is not centered), and the 95% quantile of the largest |z| among 51
independent standard normals, Phi^{-1}((1 + 0.95^(1/51)) / 2).

Outputs (output/bootstrap_empirical/robust_tests/ by default):
  level_shift_tests_long.csv  every output array of every call, long format
      (series, block, setting, name, i, j, value; a "<name>__size" row gives the
      array's size, an empty array one row with i = j = 0; trailing dimensions folded
      into j in column-major order, as MATLAB's reshape(A, size(A, 1), []));
  cisg_ftest0.csv             UDmax / WDmax statistics (BP_1: _1, BP_0: _0; WDmax at
                              the 5% weights), 5% critical values, union decisions at
                              10, 5 and 1%, the 5% kappa of columns 6 (UDmax, and the
                              BP_1 part of WDmax) and 7 (the BP_0 part of WDmax), the
                              long-run variances and the breaks they use; then the
                              components alone: the 10% critical values of UDmax, the
                              WDmax at the 10% weights and its 10% critical values,
                              and the decision of each component against its own
                              critical value (no kappa) at 10% and 5% (critical
                              values: rows 1 (10%) and 2 (5%), columns 6 (UDmax) and
                              7 (WDmax) of cv_BP_1_Ftest_0_m_5.dat (_1) and
                              cv_BP_0_Ftest_0_m_5.dat (_0));
  robust_tests_summary.csv    the numbers the paper prints from these tests, each with
                              its full value and its printed form;
  robust_level_shift.manifest.json  the run record: inputs' sha256 (LF bytes),
                              parameters, command.

Dates: T_B is the last observation of the old regime (the authors' convention);
tau = the year of T_B + 1, the first year of the new regime (the article's).

    python mlb/robust_level_shift.py [--out-dir D] [--check]

--check regenerates and compares the three CSV files with their copies in
reference/application/robust_tests/ (REFERENCE_OUTPUTS): every token that is not a
float must be identical, floats must agree to a relative 1e-9 (the statistics go
through LAPACK and BLAS reductions whose summation order can change the last digits
from one CPU to another; every count, date and decision is compared exactly). The run
takes well under a minute on one core.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from mlb import level_shifts as LS  # noqa: E402
from mlb import paths  # noqa: E402

SECTION, OBJ, NAME = "bootstrap_empirical", "robust_tests", "robust_level_shift"
Y0, Y1 = 1973, 2024
TRIM = 0.15
M_MAX = 5
WINDOWS = (0.10, 0.15)
SEL_K = 1
STRATEGIES = (("joint", 1), ("one_at_time", 0))
EVENT_YEAR = 1986
QUANTILE = 0.95
CV_FILES = ("cv_BP_1_Ftest_0_m_5.dat", "cv_BP_0_Ftest_0_m_5.dat",
            "kappa_Ftest_0_m_5.dat", "cv_F_0.dat", "cv_F_1_a.dat", "cv_F_1_b.dat",
            "mat_kappa_U_a.dat", "mat_kappa_U_b.dat")
REL_TOL = 1e-9
# the components of the union tests, as (test, noise): noise "1" is BP_1 (I(1)
# noise, column 0 of the statistics and tables), "0" is BP_0 (I(0) noise, column 1);
# and the levels at which their decisions are written, as rows of the 4-row tables
COMPONENTS = (("UDmax", "1"), ("UDmax", "0"), ("WDmax", "1"), ("WDmax", "0"))
COMPONENT_LEVELS = (("10", 0), ("05", 1))
OUTPUTS = ("level_shift_tests_long.csv", "cisg_ftest0.csv", "robust_tests_summary.csv")
# the outputs --check compares with reference/application/robust_tests/ (all three)
REFERENCE_OUTPUTS = OUTPUTS


# ---------------------------------------------------------------------- input
def load_panel():
    """(years, currencies, Y) with Y[t, i] = q of currency i in year t."""
    path = paths.INPUTS / "ppp_panel.csv"
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    curs = sorted({r["currency"] for r in rows})
    years = list(range(Y0, Y1 + 1))
    q = {(r["currency"], int(r["year"])): float(r["q"]) for r in rows}
    y = np.array([[q[(c, t)] for c in curs] for t in years])
    if not np.isfinite(y).all():
        raise SystemExit("ppp_panel.csv: missing q values in 1973-2024")
    return np.array(years, dtype=float), curs, y


def sha256_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


# --------------------------------------------------------------------- format
def fmt(v):
    """A number as written: integers (and integral floats) without a decimal point,
    other floats by repr, NaN and infinities as MATLAB prints them."""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "Inf" if v > 0 else "-Inf"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def printed(v, places):
    """v rounded half up to the given number of decimals, as a string."""
    q = Decimal(1).scaleb(-places)
    return str(Decimal(repr(float(v))).quantize(q, rounding=ROUND_HALF_UP))


class Long:
    """Rows of the long-format file, one per element of each output array."""

    def __init__(self):
        self.rows = []

    def put(self, series, block, setting, name, a, shape=None):
        a = np.asarray(a, dtype=float)
        if shape is not None:
            a = a.reshape(shape)
        elif a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a.reshape(-1, 1)
        size = "x".join(str(s) for s in a.shape)
        self.rows.append([series, block, setting, f"{name}__size", 1, 1, size])
        if a.size == 0:
            self.rows.append([series, block, setting, name, 0, 0, ""])
            return
        flat = a.reshape(a.shape[0], -1, order="F")
        for jj in range(flat.shape[1]):
            for ii in range(flat.shape[0]):
                self.rows.append([series, block, setting, name, ii + 1, jj + 1,
                                  fmt(flat[ii, jj])])


def _write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([x if isinstance(x, str) else fmt(x) for x in r])


def _taus(years, tb):
    """tau = year of T_B + 1 for 1-based break points T_B."""
    tb = [int(v) for v in tb]
    return [int(years[v]) for v in tb]


def _join(vals):
    return ";".join(str(v) for v in vals)


# ------------------------------------------------------------------------ run
def run(out):
    years, curs, y = load_panel()
    t, n = y.shape
    k_max = LS.k_max_rule(t)
    long = Long()
    meta = (("T", t), ("N", n), ("first_year", Y0), ("last_year", Y1),
            ("k_max", k_max), ("sel_k", SEL_K), ("eps_trimming", TRIM),
            ("m_max", M_MAX))
    for name, v in meta:
        long.put("ALL", "meta", "input", name, v)
    ftest_rows = []
    res_hlt, res_ftest, res_breaks = {}, {}, {}
    for i, cur in enumerate(curs):
        x = y[:, i]
        long.put(cur, "data", "input", "x", x)
        for w in WINDOWS:
            st = f"w={w:.2f}"
            h = LS.hlt_test(x, w, k_max, SEL_K)
            res_hlt[(cur, w)] = h
            long.put(cur, "HLT", st, "s1v", h["s1v"])
            long.put(cur, "HLT", st, "s0v", h["s0v"])
            for key in ("nb_S1", "nb_S0", "nb_U"):
                long.put(cur, "HLT", st, key, h[key])
            for key in ("brk_dates", "brk_dates_S1", "brk_dates_S0", "brk_dates_U"):
                long.put(cur, "HLT", st, key, h[key], shape=(len(h[key]), 1))
            for key in ("omega_u", "omega_e", "k_opt_u", "k_opt_e"):
                long.put(cur, "HLT", st, key, h[key])
            dates, pos = LS.locate_breaks_hlt(
                h["nb_S1"], h["nb_S0"], h["nb_U"], h["brk_dates_S1"],
                h["brk_dates_S0"], h["brk_dates_U"], M_MAX, years)
            long.put(cur, "HLT", st, "dates_HLT", dates)
            long.put(cur, "HLT", st, "position_HLT", pos)
        for st, gssr in STRATEGIES:
            f = LS.ftest_0_m_max(x, M_MAX, TRIM, gssr)
            rud, rwd = LS.ftest_union_decisions(f)
            res_ftest[(cur, st)] = (f, rud, rwd)
            long.put(cur, "CISG", st, "Ftest_mat_BP", f["mat_BP"])
            long.put(cur, "CISG", st, "Ftest_datevec", f["datevec"])
            long.put(cur, "CISG", st, "BP_UDmax", f["BP_UDmax"], shape=(1, 2))
            long.put(cur, "CISG", st, "BP_WDmax", f["BP_WDmax"])
            long.put(cur, "CISG", st, "mat_cv_BP", f["mat_cv_BP"])
            long.put(cur, "CISG", st, "mat_cv_BP_UDmax", f["mat_cv_BP_UDmax"])
            long.put(cur, "CISG", st, "mat_cv_BP_WDmax", f["mat_cv_BP_WDmax"])
            long.put(cur, "CISG", st, "mat_kappa", f["mat_kappa"])
            long.put(cur, "CISG", st, "UDmax_U__rej10_05_01", rud, shape=(1, 3))
            long.put(cur, "CISG", st, "WDmax_U__rej10_05_01", rwd, shape=(1, 3))
            cv_ud, cv_wd = f["mat_cv_BP_UDmax"], f["mat_cv_BP_WDmax"]
            kap = f["mat_kappa"]
            comp = component_decisions(f)
            ftest_rows.append(
                [cur, st, f["BP_UDmax"][0], f["BP_UDmax"][1], cv_ud[1, 0], cv_ud[1, 1],
                 *rud, f["BP_WDmax"][1, 0], f["BP_WDmax"][1, 1], cv_wd[1, 0],
                 cv_wd[1, 1], *rwd, kap[1, 5], kap[1, 6], f["omega_e"], f["omega_u"],
                 _join(int(v) for v in f["vec_Tb"]),
                 cv_ud[0, 0], cv_ud[0, 1],
                 comp[("UDmax", "1", "10")][2], comp[("UDmax", "0", "10")][2],
                 comp[("UDmax", "1", "05")][2], comp[("UDmax", "0", "05")][2],
                 f["BP_WDmax"][0, 0], f["BP_WDmax"][0, 1], cv_wd[0, 0], cv_wd[0, 1],
                 comp[("WDmax", "1", "10")][2], comp[("WDmax", "0", "10")][2],
                 comp[("WDmax", "1", "05")][2], comp[("WDmax", "0", "05")][2]])
            if gssr == 1:
                s = LS.multiple_breaks_joint(x, M_MAX, TRIM)
                held = s["datevec"]
            else:
                s = LS.multiple_breaks_one_at_a_time(x, M_MAX, TRIM)
                held = s["mat_Tbe"]
            counts, _ = LS.multiple_breaks_inference(
                s["mat_BP_1"], s["mat_BP_1_mod_a"], s["mat_BP_1_mod_b"],
                s["mat_BP_0"], held, t, TRIM)
            for key in ("mat_BP_1", "mat_Tbe", "mat_BP_1_mod_a", "mat_Tbe_mod_a",
                        "mat_BP_1_mod_b", "mat_Tbe_mod_b", "mat_BP_0"):
                long.put(cur, "CISG", st, key, s[key],
                         shape=(M_MAX, 1) if key.startswith("mat_BP") else None)
            long.put(cur, "CISG", st, "datevec",
                     s["datevec"] if gssr == 1 else np.zeros((0, 0)))
            for name, key in (("F_1", "num_brk_BP_1"), ("F_1a", "num_brk_BP_1_mod_a"),
                              ("F_1b", "num_brk_BP_1_mod_b"), ("F_0", "num_brk_BP_0"),
                              ("F_Ua", "num_brk_BP_U_a"), ("F_Ub", "num_brk_BP_U_b")):
                long.put(cur, "CISG", st, key, counts[name], shape=(2, 1))
            dates, pos = LS.locate_breaks_bp(counts, s["mat_Tbe"], s["mat_Tbe_mod_a"],
                                             s["mat_Tbe_mod_b"], M_MAX, years)
            long.put(cur, "CISG", st, "dates_BP", dates)
            long.put(cur, "CISG", st, "position_BP", pos)
            res_breaks[(cur, st)] = (counts, pos)
    z86 = delta_q(years, curs, y)
    numbers = summary_rows(curs, t, res_hlt, res_ftest, res_breaks, z86, years)
    _write_csv(out / "level_shift_tests_long.csv",
               ["series", "block", "setting", "name", "i", "j", "value"], long.rows)
    _write_csv(out / "cisg_ftest0.csv",
               ["currency", "strategy", "UDmax_1", "UDmax_0", "cv05_UDmax_1",
                "cv05_UDmax_0", "UDmax_U_rej10", "UDmax_U_rej05", "UDmax_U_rej01",
                "WDmax05_1", "WDmax05_0", "cv05_WDmax_1", "cv05_WDmax_0",
                "WDmax_U_rej10", "WDmax_U_rej05", "WDmax_U_rej01", "kappa05_c6",
                "kappa05_c7", "omega_e", "omega_u", "lrv_breaks_T_B", "cv10_UDmax_1",
                "cv10_UDmax_0", "UDmax_1_rej10", "UDmax_0_rej10", "UDmax_1_rej05",
                "UDmax_0_rej05", "WDmax10_1", "WDmax10_0", "cv10_WDmax_1",
                "cv10_WDmax_0", "WDmax10_1_rej10", "WDmax10_0_rej10",
                "WDmax05_1_rej05", "WDmax05_0_rej05"], ftest_rows)
    _write_csv(out / "robust_tests_summary.csv",
               ["label", "value", "printed", "where"], numbers)
    write_manifest(out, t, k_max)
    return numbers


def component_decisions(f):
    """The components of the union tests, each alone against its own critical value
    (no kappa): {(test, noise, level): (statistic, critical value, reject)} for the
    components of COMPONENTS and the levels of COMPONENT_LEVELS. UDmax is one
    statistic at every level; the WDmax of a level weights the F statistics by that
    level's critical values."""
    out = {}
    for test, noise in COMPONENTS:
        col = 0 if noise == "1" else 1
        for level, row in COMPONENT_LEVELS:
            if test == "UDmax":
                stat, cv = f["BP_UDmax"][col], f["mat_cv_BP_UDmax"][row, col]
            else:
                stat, cv = f["BP_WDmax"][row, col], f["mat_cv_BP_WDmax"][row, col]
            out[(test, noise, level)] = (float(stat), float(cv), int(stat > cv))
    return out


def delta_q(years, curs, y):
    """{currency: z}, the 1986 change of q over the sample standard deviation
    (ddof = 1) of the changes 1974-2024; the change itself is not centered."""
    dq = y[1:] - y[:-1]
    k = int(np.flatnonzero(years[1:] == EVENT_YEAR)[0])
    z = {}
    for i, cur in enumerate(curs):
        sd = float(np.std(dq[:, i], ddof=1))
        z[cur] = float(dq[k, i]) / sd
    return z


def max_abs_normal_quantile(n, p):
    """The p quantile of max |z_i| over n independent standard normals."""
    return float(stats.norm.ppf((1 + p ** (1 / n)) / 2))


def summary_rows(curs, t, res_hlt, res_ftest, res_breaks, z86, years):
    """The numbers the supplement section and the article sentence print from these
    tests: (label, full value, printed form, where). The decisions of the sequential
    tests and of the components of the HLT test are written with the statistics they
    rest on."""
    sup, art = "supplement ssub:app-robust", "article, robust level-shift sentence"
    n_changes = t - 1
    hlt_series = {w: [c for c in curs if res_hlt[(c, w)]["nb_U"] > 0]
                  for w in WINDOWS}
    dkk_tau = _taus(years, sorted(res_hlt[("DKK", 0.10)]["brk_dates_U"]))
    reject_ft = sum(any(res_ftest[(c, s)][1][:2]) or any(res_ftest[(c, s)][2][:2])
                    for c in curs for s, _ in STRATEGIES)
    comp_rej = sorted(
        (level, c, s, f"{test}_{noise}") for c in curs for s, _ in STRATEGIES
        for (test, noise, level), (_, _, rej)
        in component_decisions(res_ftest[(c, s)][0]).items() if rej)
    union_seq = sum(max(res_breaks[(c, s)][0][k]) > 0 for c in curs
                    for s, _ in STRATEGIES for k in ("F_Ua", "F_Ub"))
    seq_comp = []
    for c in curs:
        for s, _ in STRATEGIES:
            counts, pos = res_breaks[(c, s)]
            for si, name in enumerate(LS.CISG_STATISTICS[:4]):
                for lv, level in enumerate(LS.CISG_LEVELS):
                    k = counts[name][lv]
                    if k:
                        tau = _taus(years, [int(v) for v in pos[:k, si, lv]])
                        seq_comp.append((level, c, s, name, k, _join(tau)))
    seq_comp.sort()
    sep = {w: {res_hlt[(c, w)]["sep"] for c in curs} for w in WINDOWS}
    if any(len(v) != 1 for v in sep.values()):
        raise SystemExit("the HLT separation differs across series")
    s1_rej = [(c, _taus(years, [h["brk_dates"][r]])[0], float(h["s1v"][r]))
              for c in curs for h in [res_hlt[(c, 0.15)]] for r in range(len(h["s1v"]))
              if h["rej_S1"][r]]
    s0_rej = sum(int(res_hlt[(c, 0.15)]["rej_S0"].sum()) for c in curs)
    cv_s1 = {res_hlt[(c, 0.15)]["cv_S1"] for c in curs}
    min_regime = LS.matlab_round(TRIM * t)
    # the sequential step (MeanBreak_segment.m, level_shifts.mean_break_segment)
    # splits a segment into a first part of at least t_desc + 1 and a second part of
    # at least t_desc observations, t_desc = round(TRIM T) - 1
    seq_min_regime = LS.matlab_round(TRIM * t) - 1
    rows = [
        ["n_series", len(curs), "nine", f"{sup}; {art}"],
        ["level_percent", 5, "5", sup],
        ["cisg_m_max", M_MAX, "five", sup],
        ["trimming", TRIM, "0.15", f"{sup}; {art}"],
        ["hlt_window_authors", WINDOWS[0], "0.10", f"{sup}; {art}"],
        ["cisg_union_rejections_10_or_5_both_strategies", reject_ft, "none", sup],
        ["cisg_component_rejections_10_or_5_both_strategies", len(comp_rej),
         "only one", sup],
        ["cisg_component_rejections",
         _join(f"{c} {s} {name} {level}%" for level, c, s, name in comp_rej),
         "the UDmax under I(0), for the Norwegian krone at 10%", sup],
        ["cisg_sequential_union_detections_both_strategies", union_seq, "none", sup],
        ["cisg_sequential_component_breaks", sum(x[4] for x in seq_comp), "one", sup],
        ["cisg_sequential_component_detections",
         _join(f"{c} {s} {name} {level}% {tau}" for level, c, s, name, _, tau
               in seq_comp),
         ("the I(0) statistic detects one, for the Norwegian krone at 10% (2015, "
          "one-at-a-time strategy)"), sup],
        ["hlt_candidate_fractions", _join((LS.HLT_TAU_L, LS.HLT_TAU_U)), "[0.15,0.85]",
         sup],
        ["hlt015_separation_obs", sep[0.15].copy().pop(), "8", sup],
        ["hlt010_separation_obs", sep[0.10].copy().pop(), "5", f"{sup}; {art}"],
        ["hlt015_S1_rejections", _join(f"{c} {tau}" for c, tau, _ in s1_rej),
         "only its S_1 component rejects, for the krona at 1982", sup],
        ["hlt015_S1_rejections_count", len(s1_rej), "only", sup],
        ["hlt015_S0_rejections_count", s0_rej, "only its S_1 component", sup],
        ["hlt015_S1_statistics", _join(v for _, _, v in s1_rej),
         "not printed (the statistics of the rejecting S_1 components)", sup],
        ["hlt015_S1_critical_value", cv_s1.copy().pop(),
         "not printed (the 5% critical value of S_1 at the window 0.15)", sup],
        ["hlt015_series_with_U_break", len(hlt_series[0.15]), "no break",
         f"{sup}; {art}"],
        ["hlt010_series_with_U_break", len(hlt_series[0.10]), "five",
         f"{sup}; {art}"],
        ["hlt010_series_list", ", ".join(hlt_series[0.10]),
         ", ".join(hlt_series[0.10]), sup],
        ["hlt010_DKK_tau", _join(dkk_tau), " and ".join(str(v) for v in dkk_tau),
         sup],
        ["min_regime_obs", min_regime, str(min_regime), sup],
        ["cisg_sequential_min_regime_obs", seq_min_regime, str(seq_min_regime), sup],
    ]
    rows.append(["event_year", EVENT_YEAR, str(EVENT_YEAR), sup])
    for cur in ("JPY", "CHF", "DKK", "AUD"):
        rows.append([f"z_1986_{cur}", z86[cur], printed(z86[cur], 2), sup])
    big = max(curs, key=lambda c: abs(z86[c]))
    rows.append(["z_1986_largest_abs", abs(z86[big]), printed(abs(z86[big]), 0)
                 + f" ({big}; 'about three')", art])
    rows.append(["n_changes", n_changes, str(n_changes), sup])
    rows.append(["quantile_level", QUANTILE, "95%", sup])
    q = max_abs_normal_quantile(n_changes, QUANTILE)
    rows.append(["max_abs_normal_q95_n51", q, printed(q, 2), f"{sup}; {art}"])
    return rows


def write_manifest(out, t, k_max):
    inputs = {"inputs/ppp_panel.csv": sha256_lf(paths.INPUTS / "ppp_panel.csv")}
    for name in CV_FILES:
        inputs[f"inputs/level_shift_cv/{name}"] = sha256_lf(LS.CV_DIR / name)
    module = os.path.join(_ROOT, "mlb", "level_shifts.py")
    man = dict(
        producer="mlb/robust_level_shift.py",
        command="python mlb/robust_level_shift.py",
        outputs=list(OUTPUTS),
        inputs_sha256_lf=inputs,
        module_sha256_lf={"mlb/level_shifts.py": sha256_lf(module)},
        seeds="none (no simulation)",
        parameters=dict(
            window_years=[Y0, Y1], T=t, hlt_windows=list(WINDOWS), hlt_level="5%",
            k_max=k_max, k_max_rule="round(4 (T/100)^(1/4))", sel_k=SEL_K,
            cisg_trimming=TRIM, cisg_m_max=M_MAX,
            cisg_strategies=[s for s, _ in STRATEGIES],
            ftest_levels=[lv for lv, _ in LS.FTEST_LEVEL_ROWS],
            sequential_levels=list(LS.CISG_LEVELS),
            component_levels=[lv for lv, _ in COMPONENT_LEVELS],
            component_decision=("each component of the union tests (UDmax and WDmax "
                                "of BP_1 and of BP_0) against its own critical value, "
                                "without kappa"),
            component_critical_values={
                "cv10_UDmax_1": "cv_BP_1_Ftest_0_m_5.dat, row 1 (10%), column 6",
                "cv10_UDmax_0": "cv_BP_0_Ftest_0_m_5.dat, row 1 (10%), column 6",
                "cv10_WDmax_1": "cv_BP_1_Ftest_0_m_5.dat, row 1 (10%), column 7",
                "cv10_WDmax_0": "cv_BP_0_Ftest_0_m_5.dat, row 1 (10%), column 7",
                "cv05_UDmax_1": "cv_BP_1_Ftest_0_m_5.dat, row 2 (5%), column 6",
                "cv05_UDmax_0": "cv_BP_0_Ftest_0_m_5.dat, row 2 (5%), column 6",
                "cv05_WDmax_1": "cv_BP_1_Ftest_0_m_5.dat, row 2 (5%), column 7",
                "cv05_WDmax_0": "cv_BP_0_Ftest_0_m_5.dat, row 2 (5%), column 7"},
            sequential_min_regime="round(0.15 T) - 1 (MeanBreak_segment.m)",
            event_year=EVENT_YEAR,
            dq_sd="ddof = 1 over the 51 changes 1974-2024; the 1986 change not "
                  "centered",
            max_abs_normal_quantile=f"Phi^-1((1 + {QUANTILE}^(1/51)) / 2)"),
        check=f"tokens identical except floats, which agree to a relative {REL_TOL}",
    )
    with open(out / f"{NAME}.manifest.json", "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(man, fh, indent=1)
        fh.write("\n")


# ---------------------------------------------------------------------- check
def _is_float_token(s):
    try:
        float(s)
    except ValueError:
        return False
    return any(c in s for c in ".eEN") or s in ("Inf", "-Inf")


def _same_token(a, b):
    if a == b:
        return True
    if not (_is_float_token(a) or _is_float_token(b)):
        return False
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return abs(x - y) <= REL_TOL * max(abs(x), abs(y))


def compare_csv(new, ref):
    """(n_differences, first messages) between two CSV files."""
    a = list(csv.reader(open(new, encoding="utf-8", newline="")))
    b = list(csv.reader(open(ref, encoding="utf-8", newline="")))
    if len(a) != len(b):
        return 1, [f"{len(a)} rows against {len(b)}"]
    bad = []
    for k, (ra, rb) in enumerate(zip(a, b)):
        if len(ra) != len(rb) or not all(_same_token(x, y) for x, y in zip(ra, rb)):
            bad.append(f"row {k + 1}: {ra} != {rb}")
    return len(bad), bad[:5]


def check(out):
    ref = paths.reference_dir(SECTION) / OBJ
    total = 0
    for name in REFERENCE_OUTPUTS:
        g = ref / name
        if not g.exists():
            print(f"  MISSING reference {name}")
            total += 1
            continue
        same = (open(out / name, "rb").read()
                == open(g, "rb").read().replace(b"\r\n", b"\n"))
        n, msgs = (0, []) if same else compare_csv(out / name, g)
        word = "identical" if same else ("equal within tolerance" if not n
                                         else f"{n} DIFFERENCES")
        print(f"  {name}: {word}")
        for m in msgs:
            print(f"    {m}")
        total += n
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out_dir) if a.out_dir else paths.out_dir(SECTION) / OBJ
    out.mkdir(parents=True, exist_ok=True)
    numbers = run(out)
    for label, value, shown, _ in numbers:
        print(f"{label:52s} {fmt(value) if not isinstance(value, str) else value:>22s}"
              f"  printed {shown}")
    print(f"written to {out}")
    if a.check:
        n = check(out)
        print("--check: " + ("equal to the reference" if n == 0
                             else f"{n} difference(s) from the reference"))
        return 0 if n == 0 else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
