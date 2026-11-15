"""The PPP application's printed tables and its derived readings, written from the
application's reference files and inputs (no simulation, no series).

Reads the reference files of ppp_application.py and
ppp_half_lives.py (reference/application/main/), those of ppp_sj.py
(reference/application/crisis_design/{sj_null.csv, sj_readings.csv,
sj_sets.csv}), inputs/regime_events.csv, the admissibility sweep's
inputs/ppp_sweep/{candidates.csv, funnel.txt} and the admission tests'
reference/application/admission/{c6_critical_values.csv, gate_sizes.csv}. Writes, into
--out-dir (by default output/bootstrap_empirical/tables/):

    tab_ppp.tex               tab:ppp under E, one row per currency: tau, p_hat,
                              MZ_t at -7, (I) against -1.98, the size of (I), (II)
                              (cv, p with its se, verdict), cbar_half, (III) (MZ_t,
                              cv, p, verdict), Berger-Boos; the sensitivity S5 in a
                              block to the right (tau, MZ_t, p (II), the verdicts
                              of (I), (II), (III))
    stab_app_hl.tex           the half-lives of stab:app-hl in the package's layout:
                              E, recursive scheme, LB and MP
    stab_app_corr.tex         the supplement's stab:app-corr: per currency, the
                              median-unbiased correction alpha_MU - alpha_OLS of the
                              recursive scheme at the E order, with no level dummy
                              (MP), the first E date only, the second E date only and
                              all E dates (LB); -- for a one-date currency
    stab_app_sets.tex         the supplement's stab:app-sets: per currency, p (II) and
                              its 5% verdict (*) under E, S1-S5 and S-JST (the crisis
                              design), n.v. without a verdict (a sieve not
                              stationary), -- without an event; below, the break
                              years (two digits) where they differ from E
    stab_app_dates.tex        the supplement's stab:app-dates: per candidate
                              (candidates.csv), the E break years, m, the spacing
                              (tau2 - tau1) / T (T = 52; -- for one date), C4's
                              kappa2, C6's W, simulated 5% cv and p (* where C6
                              rejects at 5%), the chi2(m) p-value (< 0.001 below the
                              printed place) and that reference's size at 5%
                              (gate_sizes.csv; the cv's se from
                              c6_critical_values.csv)
    stab_app_power.tex        the supplement's stab:app-power: per currency the power
                              of (II) at h = 3, 4, 5 (u0 = 0), its band mean, and the
                              band mean with the stationary initial condition
    stab_app_count.tex        the supplement's stab:app-count: the law of K over N8 at
                              kappa = 0, 1, under the null and at h = 3, 4, 5: E[K],
                              P(K = 0), P(K <= K_obs) (u0 = 0) and P(K <= K_obs) with
                              the stationary initial condition (-- under the null)
    readings_tables.csv       the derived readings of the tables and the text that
                              the provenance or a check reads
                              (blocks as_registered, tests, hl_text, designs,
                              paper_tables), at full precision (below)
    tables_provenance.csv     one row per printed data token of every .tex
    tables.manifest.json      the sha256 of the inputs read and of the outputs, and
                              the printed decimals of each class

    python mlb/ppp_tables.py           # write
    python mlb/ppp_tables.py --check   # render and compare readings_tables.csv
                                       # with reference/application/tables/

RULES:
  * Every number is rounded once, from the full-precision string of its reference, half
    up (decimal ROUND_HALF_UP). Nothing is rounded in a CSV.
  * Printed decimals, one convention per class of quantity, over the members of the
    class: the numbers this producer prints and, for the power and size levels, also
    the design's power of (II) and (III) at each h and their band means
    (ppp_power.csv, ppp_power_band.csv; design_power_members), which no table prints
    in this class: those of (II) appear in stab:app-power at three decimals (class
    level3), those of (III) in no table.
    A digit in decimal place d is sustained if the Monte Carlo se of the number is
    <= 1/2 10^-d (d_sustained = the largest such d).
      - Monte Carlo classes with their se in the reference files (critical values,
        power and size levels, p-values) print at the smallest d_sustained of their
        members.
      - Exact classes (statistics on the data: MZ_t at 2, alpha_OLS at 3; cbar_half
        at 2) print at a fixed place.
      - Monte Carlo quantities whose se is not in the reference files: alpha_MU (3
        decimals) and the half-life at alpha_MU (1 decimal, the convention for
        half-lives); the provenance records their d_sustained as "not estimated".
      - The limits of the half-life intervals are grid nodes k/200 (alpha, 3 decimals,
        exact as printed; which node is a limit is a Monte Carlo outcome) and their
        half-lives (1 decimal).
      - The grid's cap alpha = 1.05 prints as named.
    A standard error prints with one significant figure. The same number prints the
    same wherever it appears in the tables of the package's own layout.
  * The stab_app_*.tex tables are the supplement's tables in its layout: rows in the
    order of the candidates (candidates.csv), a dagger on the currencies of N9
    outside N8, numbers outside math mode, and the supplement's printed decimals, one
    fixed class each (the correction alpha_MU - alpha_OLS at 3, p-values, power,
    size and probability levels at 3, E[K] at 2, the admission statistics kappa2 and W and
    the C6 critical value at 2, the chi2(m) p-value at 3 with a floor, the spacing at
    2); a 5% verdict prints as $^{*}$ after its p (a token of its flag, empty when the
    flag does not trigger it) and must show in the printed p; a positive number that
    rounds to zero prints as $<$0.001 (rule copy_floor); their notes are the
    supplement's captions, not rows of the table, and groups of rows are separated by
    \\addlinespace (not a body line).
  * A printed Monte Carlo number is "resolved" when its distance to the nearest
    rounding boundary at the printed place is at least 2 se; the provenance records
    it. Verdicts are printed as columns (R = reject at 5%, -- = not). Where a verdict
    does not show strictly in the printed numbers of its row (a printed statistic
    against its printed critical value or -1.98, a printed p against 0.05), the
    verdict carries * and a note under the table gives the full-precision values.
  * Half-lives: inf prints as infinity. The grid of the half-life intervals stops at
    alpha = 1.05: an accepted set reaching it prints its upper limit as ">= 1.05", an
    upper side open at the cap as "> 1.05", a set entirely above it as "> 1.05", and
    a median-unbiased point at the cap as ">= 1.05". An accepted set with rejected
    nodes inside it (gaps) carries a double dagger, and a note gives the gaps' alpha.
  * Rows are the currencies of N9 in the order of N8, then the others, marked with a
    dagger and a note (in N9 only: they fail C6 at 5%). N8 = the candidates passing
    C4 and C6 at 5%, N9 = those passing C4 (inputs/ppp_sweep/candidates.csv); they
    must equal the universes of readings.csv and of funnel.txt.
  * Half-life cells are selected by their specification list (e.g. "LB@E" in specs),
    never by (model, currency).
  * A row whose status is not ok prints its status instead of numbers.
  * Provenance: every data token of a .tex body (a number, a verdict, a label read
    from or derived from a file) has one row naming its reference, the row key, the
    column, the full-precision value and the rule; the literal text of a body line
    carries no digit (the headers are constant text). A regime_events.csv key
    "data_row=i" is the 0-based index of its data rows (the header excluded).

readings_tables.csv (block, reading, keys, value, se, currencies):
  as_registered  the C6 rule with the asymptotic chi2(m) reference at 5%: a candidate
                 is kept iff it passes C4 at 5% and its chi2(m) p-value is finite and
                 >= 0.05 (the rule of admissible_universe.funnel(registered=True));
                 N and the members, which the checks compare with funnel.txt.
                 Descriptive: no simulation uses this universe.
  tests          the Holm and Bonferroni adjusted p (II) under E with the se of their
                 binding term when below 1 (for Holm the term of the running maximum,
                 whose currency is in `currencies`)
  hl_text        the half-lives under E (hl_cells.csv, the cells of specification
                 LB@E and MP@E), per universe: the currencies whose alpha_OLS is lower
                 with the breaks (LB) than without (MP; the same in both schemes);
                 per universe and scheme (rec, wild): the currencies whose HL_MU rises
                 from MP to LB, falls, or is infinite in both models; the medians of
                 HL_MU; the unbounded LB and MP intervals (k_hi >= 200), the LB
                 intervals meeting [3, 5] and the strict collapses (k_hi >= 200 for
                 MP, <= 199 for LB); per E cell with gaps, the number of gaps and the
                 half-life of the last node below the first gap. The correction
                 alpha_MU - alpha_OLS (recursive scheme, E order) per currency with no
                 dummy (MP@E), the first E date only (LB_b1@E), the second only
                 (LB_b2@E) and all E dates (LB@E); per universe, the currencies whose
                 correction with every dummy model exceeds the one without, and among
                 the two-date currencies those whose all-dates correction exceeds both
                 single-date corrections and those whose does not
  designs        column II across the seven designs E, S1-S5 and S-JST (ppp_sets.csv,
                 and sj_null.csv for S-JST), per universe and design: N (with a
                 verdict), K_obs (II) and the rejecting currencies; per currency, the
                 number and the list of the designs where (II) rejects. Under S-JST
                 per universe: the currencies rejecting with (I), and the Holm and
                 Bonferroni adjusted p (II) over the currencies with a verdict (as in
                 `tests`)
  paper_tables   the spacing (tau2 - tau1) / T of each two-date candidate (T = 52)

CHECKS (a failure stops the run; nothing is written before they pass):
  * every provenance row: the reference cell equals value_full, and the printed token is
    re-derived from it (rounded, or the label or text the rule derives); every body
    line of every .tex is its registered tokens and digit-free literal text;
  * the universes equal readings.csv's and funnel.txt's (every block's admitted line);
    K_obs under E and the Holm and Bonferroni p-values equal readings.csv's;
  * ppp_sets.csv's E rows equal ppp_null.csv's (p_hat, cfg, status and the test
    columns); its half-life columns equal the hl_cells.csv cell of their
    specification (p and, for LB, the configuration too); every verdict flag is 0 or 1
    and equals its rules ((I): MZ_t < -1.98; (II): MZ_t < cv (II) and p (II) < 0.05;
    (III): MZ_t at cbar_half < cv (III) and p (III) < 0.05);
  * regime_events.csv passes its own integrity() and its f and tau are the ones
    generated from the dates; the E configuration of every currency
    (regime_events.configuration) has the break years of ppp_sets.csv's E tau;
  * internal identities: a band mean is the mean of its h powers; HL = ln 0.5 /
    ln alpha (0 for alpha <= 0, inf for alpha >= 1); an interval limit alpha is its
    node k / 200 (below the cap); n_gaps is the number of gaps; p_BB = p_max + 0.01;
  * hl_text: its medians and its counts of unbounded intervals, of LB intervals
    meeting [3, 5] and of collapses equal readings.csv's; an E cell's alpha_OLS is the
    same in both schemes; a single-date cell (LB_b1@E, LB_b2@E) exists iff the
    currency has two E dates, its configuration is that date, and the four cells of a
    correction share the E order; a correction is alpha_mu - alpha_ols of its cell;
  * designs: sj_null.csv equals sj_sets.csv's S-JST rows and sj_sets.csv's E rows
    equal ppp_sets.csv's (the shared columns); the S-JST verdict flags follow their
    rules; K_obs, the rejecting currencies of (I) and (II) and the Holm and Bonferroni
    p under S-JST equal sj_readings.csv's; in stab:app-sets a star shows in its
    printed p against 0.05, and the years print iff the design is E or its years
    differ from E's;
  * admission: candidates.csv's C6 cv, dates and m equal those of
    c6_critical_values.csv and of gate_sizes.csv, m is the number of E dates,
    the C6 simulations ran at T - 1 = 51, and the C6 flag at 5% follows W > cv and
    p < 0.05; the count law printed under P(K <= 1) has K_obs = 1 (readings.csv's) and
    no NaN draw.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
from decimal import ROUND_HALF_UP, Decimal

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mlb import regime_events as RE  # noqa: E402

SECTION = "bootstrap_empirical"
H = ("3", "4", "5")
CV_ASY = "-1.98"
CAP = "1.05"  # the half-life grid's cap, k = 210 of k / 200
# the reading each cap flag prints (the grid's cap reached, or the upper side beyond it)
CAP_PREFIX = {
    "upper_at_cap": r"\geq ",
    "mu_at_cap": r"\geq ",
    "upper_open": "> ",
    "ci_above_cap": "> ",
}
LEVEL = Decimal("0.05")
REFERENCE_FILES = (
    "ppp_null.csv",
    "ppp_berger_boos_summary.csv",
    "ppp_power.csv",
    "ppp_power_band.csv",
    "ppp_count_law.csv",
    "hl_cells.csv",
    "ppp_sets.csv",
    "readings.csv",
)
SJ_FILES = ("sj_null.csv", "sj_readings.csv", "sj_sets.csv")
OUTPUTS = (
    "tab_ppp.tex",
    "stab_app_hl.tex",
    "stab_app_corr.tex",
    "stab_app_sets.tex",
    "stab_app_dates.tex",
    "stab_app_power.tex",
    "stab_app_count.tex",
    "readings_tables.csv",
    "tables_provenance.csv",
    "tables.manifest.json",
)
# the outputs whose reference file is shipped (reference/application/tables/): --check
# compares exactly these
REFERENCE_OUTPUTS = ("readings_tables.csv",)
HL_FIELDS = (
    "alpha_mu",
    "mu_at_cap",
    "n_crossings",
    "hl_mu",
    "alpha_lo",
    "alpha_hi",
    "upper_at_cap",
    "upper_open",
    "ci_above_cap",
    "hl_lo",
    "hl_hi",
    "k_hi",
)
# the columns of stab:app-corr: (reading suffix, model, specification), recursive scheme
CORR_COLS = (
    ("MP", "MP", "MP@E"),
    ("LB_b1", "LB", "LB_b1@E"),
    ("LB_b2", "LB", "LB_b2@E"),
    ("LB", "LB", "LB@E"),
)
PAPER_DAGGER = r"$^{\dagger}$"
STAR = r"$^{*}$"
SPACE = r"\addlinespace"  # a separator line between groups of rows
T_LEVELS = 52  # the sample 1973-2024 in levels (the C6 simulations use T - 1 = 51)
# the designs of stab:app-sets, in its column order; S-JST is read from sj_null.csv
PAPER_SETS = ("E", "S1", "S2", "S3", "S4", "S5", "S-JST")

# class: (name, printed decimals or None = the smallest d_sustained of its members,
# kind). kind: exact (no Monte Carlo error); mc (Monte Carlo, se in the reference
# files); mc_no_se (Monte Carlo, se not in the reference files); node (a grid node
# k/200 or its half-life, exact as printed, selected by Monte Carlo); constant.
CLASSES = {
    "stat": ("statistics on the data (MZ_t)", 2, "exact"),
    "alpha_ols": ("alpha_OLS (on the data)", 3, "exact"),
    "cbar": ("cbar_half (exact)", 2, "exact"),
    "cv": ("critical values", None, "mc"),
    "level": ("power and size levels", None, "mc"),
    "p": ("p-values", None, "mc"),
    "alpha_mu": ("alpha_MU", 3, "mc_no_se"),
    "hl": ("half-lives at alpha_MU", 1, "mc_no_se"),
    "node": ("alpha, the interval limits (grid nodes k/200)", 3, "node"),
    "hl_node": ("half-lives of the interval limits", 1, "node"),
    "cap": ("the grid's cap alpha = 1.05", 2, "constant"),
    "int": ("integers", 0, "exact"),
    "corr": ("alpha_MU - alpha_OLS (stab:app-corr)", 3, "mc_no_se"),
    "p3": ("p-values of the supplement's tables", 3, "mc"),
    "level3": (
        "power, size and probability levels of the supplement's tables",
        3,
        "mc",
    ),
    "ek": ("E[K] (stab:app-count)", 2, "mc"),
    "gate_stat": ("the admission statistics kappa2 and W (on the data)", 2, "exact"),
    "gate_cv": ("the simulated 5% critical value of C6", 2, "mc"),
    "p_chi2": ("the chi2(m) p-value of C6 (asymptotic)", 3, "exact"),
    "spacing": ("the spacing (tau2 - tau1) / T of two E dates", 2, "exact"),
}
PROV_COLS = [
    "object",
    "table_file",
    "body_row",
    "token",
    "printed",
    "value_full",
    "reference",
    "reference_key",
    "reference_column",
    "rule",
    "class",
    "decimals_printed",
    "se",
    "d_sustained",
    "resolved_at_printed",
]
READ_COLS = [
    "block",
    "reading",
    "universe",
    "set",
    "scheme",
    "currency",
    "value",
    "se",
    "currencies",
]


# ================================================================== numbers
def fmt(x):
    """Full precision: repr of a float, ints and strings as they are."""
    if isinstance(x, bool):
        return str(int(x))
    if isinstance(x, float):
        return repr(x)
    return str(x)


def is_inf(s):
    return s in ("inf", "-inf")


def rnd(s, d):
    """s (a full-precision decimal string) rounded half up to d decimals."""
    if is_inf(s):
        return s
    q = Decimal(s).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    return f"{q:f}"


def floor_mark(s, printed, d):
    """A positive number that rounds to zero at d decimals prints as $<$10^-d."""
    if Decimal(printed) == 0 and Decimal(s) > 0:
        return "$<$" + f"{Decimal(1).scaleb(-d):f}"
    return printed


def sig1(s):
    """A standard error with one significant figure, half up."""
    x = Decimal(s)
    if x == 0:
        return "0"
    e = x.adjusted()
    q = x.quantize(Decimal(1).scaleb(e), rounding=ROUND_HALF_UP)
    if q.adjusted() > e:
        q = x.quantize(Decimal(1).scaleb(e + 1), rounding=ROUND_HALF_UP)
    return f"{q:f}"


def d_sustained(se):
    """The largest decimal place d with se <= 1/2 10^-d (exact decimal comparison)."""
    x = Decimal(se)
    if x == 0:
        return None
    for d in range(12, -4, -1):
        if x <= Decimal("0.5").scaleb(-d):
            return d
    return -4


def resolved(value, d, se):
    """Distance to the nearest rounding boundary at place d, in se, is >= 2."""
    if is_inf(value):
        return "yes"
    if value.lower() == "nan":
        return "n.a. (nan)"
    a = abs(Decimal(value)).scaleb(d)
    frac = a - a.to_integral_value(rounding="ROUND_FLOOR")
    dist = abs(frac - Decimal("0.5")).scaleb(-d)
    if se in (None, "", "0", "0.0"):
        return "no (on a boundary)" if dist == 0 else "yes"
    return "yes" if dist >= 2 * Decimal(se) else "no"


def tex_escape(s):
    out = []
    for ch in s:
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "<":
            out.append(r"\textless{}")
        elif ch == ">":
            out.append(r"\textgreater{}")
        else:
            out.append(ch)
    return "".join(out)


def half_life(alpha):
    """The AR(1) half-life ln 0.5 / ln alpha: 0 for alpha <= 0, inf for alpha >= 1."""
    a = float(alpha)
    if a <= 0:
        return 0.0
    if a >= 1:
        return math.inf
    return math.log(0.5) / math.log(a)


def sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ================================================================== derived labels
def label_gaps(gaps):
    """Grid indices k (|-separated) as alpha = k / 200 at 3 decimals."""
    return ", ".join(rnd(str(Decimal(k) / 200), 3) for k in gaps.split("|"))


def label_na(status):
    if not status.startswith("not applicable"):
        raise ValueError(f"a Berger-Boos status other than not applicable: {status!r}")
    return "n.a."


def label_n9_only(c6_5):
    if c6_5 != "0":
        raise ValueError("a marked currency passes C6 at 5%")
    return "in N9 only: fails C6 at 5%"


def label_years2(taus):
    """Break years (|-separated) with two digits: 1984|1986 -> '84, '86."""
    out = []
    for t in taus.split("|"):
        if not re.fullmatch(r"(19|20)\d\d", t):
            raise ValueError(f"not a year: {t!r}")
        out.append("'" + t[2:])
    return ", ".join(out)


def label_nv(status):
    if not status.startswith("not simulated"):
        raise ValueError(f"a no-verdict status other than not simulated: {status!r}")
    return "n.v."


def label_no_event(status):
    if not status.startswith("no event"):
        raise ValueError(f"a status other than no event: {status!r}")
    return "--"


DERIVE = {
    "years2": label_years2,
    "nv": label_nv,
    "no_event": label_no_event,
    "semi": lambda v: v.replace(";", ", "),
    "gaps": label_gaps,
    "na": label_na,
    "n9_only": label_n9_only,
    "bars": lambda v: v.replace("|", ", "),
    "text": lambda v: v,
}


# ================================================================== the sources
class Sources:
    """The reference files and inputs, each CSV as a list of rows, with a locator for a
    cell."""

    def __init__(
        self, main_dir, events, candidates, gate_sizes, funnel, crisis_dir, c6_cv
    ):
        self.paths = {n: os.path.join(main_dir, n) for n in REFERENCE_FILES}
        self.paths.update({n: os.path.join(crisis_dir, n) for n in SJ_FILES})
        self.paths.update(
            {
                "regime_events.csv": events,
                "candidates.csv": candidates,
                "gate_sizes.csv": gate_sizes,
                "funnel.txt": funnel,
                "c6_critical_values.csv": c6_cv,
            }
        )
        self.rows = {
            n: read_csv(p) for n, p in self.paths.items() if n.endswith(".csv")
        }
        self.funnel = open(funnel, encoding="utf-8").read()

    def cell(self, name, key, column):
        """The value of `column` in the unique row of `name` matching key (a dict).
        {"data_row": i} is the i-th data row (0-based, the header excluded)."""
        rows = self.rows[name]
        if "data_row" in key:
            return rows[int(key["data_row"])][column]
        hit = [r for r in rows if all(r.get(k) == v for k, v in key.items())]
        if len(hit) != 1:
            raise SystemExit(f"{name}: {len(hit)} rows match {key}")
        return hit[0][column]


def key_str(key):
    return ";".join(f"{k}={v}" for k, v in key.items())


def key_parse(s):
    return dict(kv.split("=", 1) for kv in s.split(";")) if s else {}


# ================================================================== tokens and tables
class Table:
    """A tabular fragment: a constant header and body lines built from registered
    tokens (data) and literal text (no digits)."""

    def __init__(self, obj, fname, colspec, header):
        self.obj, self.fname, self.colspec, self.header = obj, fname, colspec, header
        self.lines = []  # each line: list of cells; each cell: list of parts
        self.ncol = sum(ch in "lcrP" for ch in re.sub(r"p\{[^}]*\}", "P", colspec))

    def row(self, cells):
        self.lines.append(cells)

    def rule(self):
        self.lines.append(None)

    def space(self):
        """An \\addlinespace between groups of rows (not a body line)."""
        self.lines.append(SPACE)

    def note(self, parts):
        """A row spanning the table: literal text and tokens."""
        self.lines.append(
            [[("lit", rf"\multicolumn{{{self.ncol}}}{{l}}{{")] + parts + [("lit", "}")]]
        )


def lit(text):
    if any(ch.isdigit() for ch in text):
        raise ValueError(f"literal text with a digit: {text!r}")
    return ("lit", text)


def tok(cls, reference, key, column, rule="copy", se=None, math_=True, derive=None):
    """A data token. rule: copy (rounded to the class), full (the reference string),
    se (one significant figure), verdict (R or --), cap (the constant, when the
    reference flag is 1; derive holds its prefix), derived (DERIVE[derive] of the
    reference value, escaped)."""
    return (
        "tok",
        dict(
            cls=cls,
            reference=reference,
            key=key,
            column=column,
            rule=rule,
            se=se,
            math=math_,
            derive=derive,
        ),
    )


def cell(*parts):
    return [p if p[0] == "tok" else ("lit", p[1]) for p in parts]


class Renderer:
    def __init__(self, src):
        self.src = src
        self.tables = []
        self.members = []  # no table prints them in their class (design_power_members)
        self.digits = None

    def value_of(self, t):
        return self.src.cell(t["reference"], t["key"], t["column"])

    def se_of(self, t):
        s = t["se"]
        if isinstance(s, tuple):  # (reference, key, column)
            return self.src.cell(*s)
        return s

    def collect(self):
        """The printed decimals of each class: fixed, or the smallest d_sustained
        of the Monte Carlo members (their se in the reference files)."""
        members = {}
        for tb in self.tables:
            for line in tb.lines:
                if line is SPACE:
                    continue
                for cl in line or []:
                    for kind, p in cl:
                        if kind != "tok" or p["rule"] != "copy":
                            continue
                        if CLASSES[p["cls"]][1] is not None:
                            continue
                        v = self.value_of(p)
                        if is_inf(v):
                            continue
                        d = d_sustained(self.se_of(p))
                        where = (
                            f"{tb.fname}: {p['reference']} {key_str(p['key'])} "
                            f"{p['column']}"
                        )
                        if d is not None:
                            members.setdefault(p["cls"], []).append((d, where))
        for p in self.members:
            if is_inf(self.value_of(p)):
                continue
            d = d_sustained(self.se_of(p))
            where = (f"not printed in this class: {p['reference']} {key_str(p['key'])} "
                     f"{p['column']}")
            if d is not None:
                members.setdefault(p["cls"], []).append((d, where))
        digits = {}
        for cls, (name, d, kind) in CLASSES.items():
            if d is not None:
                digits[cls] = dict(name=name, digits=d, kind=kind, rule="fixed")
                continue
            m = members.get(cls, [])
            if not m:  # nothing of the class is printed
                digits[cls] = dict(name=name, digits=None, kind=kind, rule="no member")
                continue
            dmin = min(x for x, _ in m)
            digits[cls] = dict(
                name=name,
                digits=dmin,
                kind=kind,
                rule="smallest d_sustained",
                binding=[w for x, w in m if x == dmin],
            )
        self.digits = digits

    def printed_number(self, reference, key, column, cls):
        """The printed decimal string of a copy token (for the verdict notes)."""
        return rnd(self.src.cell(reference, key, column), self.digits[cls]["digits"])

    def render_token(self, t):
        """(printed text, provenance fields)."""
        rule, cls = t["rule"], t["cls"]
        v = self.value_of(t)
        se = self.se_of(t)
        info = {
            "value_full": v,
            "reference": t["reference"],
            "reference_key": key_str(t["key"]),
            "reference_column": t["column"],
            "rule": f"{rule}:{t['derive']}" if rule in ("derived", "star") else rule,
            "class": cls,
            "decimals_printed": "",
            "se": se or "",
            "d_sustained": "",
            "resolved_at_printed": "n.a.",
        }
        if rule == "cap":  # the flag must be set; the cap prints as named
            if v != "1":
                raise SystemExit(f"cap token on an unset flag: {t}")
            info.update(
                decimals_printed=2, d_sustained="constant", resolved_at_printed="n.a."
            )
            return f"${t['derive']}{CAP}$", info
        if rule in ("copy", "copy_floor"):
            d = self.digits[cls]["digits"]
            kind = CLASSES[cls][2]
            if is_inf(v):
                txt = r"\infty" if v == "inf" else r"-\infty"
            else:
                txt = rnd(v, d)
                info["decimals_printed"] = d
                if rule == "copy_floor":
                    txt = floor_mark(v, txt, d)
            if kind == "mc":
                info["d_sustained"] = d_sustained(se) if Decimal(se) > 0 else "exact"
                if not is_inf(v):
                    info["resolved_at_printed"] = resolved(v, d, se)
            elif kind == "mc_no_se":
                info["d_sustained"] = "not estimated"
                info["resolved_at_printed"] = "not estimated"
            elif kind == "node":
                info["d_sustained"] = "grid node"
                info["resolved_at_printed"] = "n.a. (a grid node)"
            else:
                info["d_sustained"] = "exact"
                if not is_inf(v):
                    info["resolved_at_printed"] = resolved(v, d, None)
            return (f"${txt}$" if t["math"] else txt), info
        if rule == "full":
            info["decimals_printed"] = "full precision"
            return f"${v}$", info
        if rule == "se":
            info.update(decimals_printed="1 significant figure")
            info["class"] = "se"
            return sig1(v), info
        if rule == "verdict":
            if v not in ("0", "1"):
                raise SystemExit(f"a verdict flag that is not 0 or 1: {t}")
            return ("R" if v == "1" else "--"), info
        if rule == "star":  # the 5% mark: $^{*}$ when the flag equals derive
            if v not in ("0", "1"):
                raise SystemExit(f"a verdict flag that is not 0 or 1: {t}")
            return (STAR if v == t["derive"] else ""), info
        if rule == "derived":
            return tex_escape(DERIVE[t["derive"]](v)), info
        raise ValueError(rule)

    def render(self):
        if self.digits is None:
            self.collect()
        texts, prov = {}, []
        for tb in self.tables:
            out = [
                f"% {tb.fname}: written by mlb/ppp_tables.py from "
                f"the application's reference files; do not edit.",
                f"\\begin{{tabular}}{{{tb.colspec}}}",
                r"\toprule",
                tb.header + r" \\",
                r"\midrule",
            ]
            n_line = 0
            for line in tb.lines:
                if line is None:
                    out.append(r"\midrule")
                    continue
                if line is SPACE:
                    out.append(SPACE)
                    continue
                n_line += 1
                cells, k = [], 0
                for cl in line:
                    s = ""
                    for kind, p in cl:
                        if kind == "lit":
                            s += p
                            continue
                        txt, info = self.render_token(p)
                        k += 1
                        prov.append(
                            dict(
                                object=tb.obj,
                                table_file=tb.fname,
                                body_row=n_line,
                                token=k,
                                printed=txt,
                                **info,
                            )
                        )
                        s += txt
                    cells.append(s)
                out.append(" & ".join(cells) + r" \\")
            out += [r"\bottomrule", r"\end{tabular}", ""]
            texts[tb.fname] = "\n".join(out)
        return texts, prov


# ================================================================== universes
def universes(src):
    cand = src.rows["candidates.csv"]
    n9 = [r["currency"] for r in cand if r["C4_5"] == "1"]
    n8 = [r["currency"] for r in cand if r["C4_5"] == "1" and r["C6_5"] == "1"]
    return n8, n9


def row_order(n8, n9):
    return list(n8) + [c for c in n9 if c not in n8]


# ================================================================== reading rules
def as_registered(cand):
    """The members under C6 with the asymptotic chi2(m) reference at 5%: a candidate
    is kept iff it passes C4 at 5% and its chi2(m) p-value is finite and >= 0.05
    (admissible_universe.funnel(registered=True))."""

    def kept(r):
        return (
            r["C4_5"] == "1"
            and math.isfinite(float(r["c6_p_chi2"]))
            and Decimal(r["c6_p_chi2"]) >= LEVEL
        )

    return [r["currency"] for r in cand if kept(r)]


def holm_with_se(ps):
    """{currency: (p, se)} -> {currency: (Holm-adjusted p, se of the binding term or
    None when the adjusted p is 1, the binding currency)}: the adjusted p is the running
    maximum of min(1, (m - j) p_(j)); its se is (m - j*) se_(j*) of the term j* that
    attains it (the earliest on a tie)."""
    m = len(ps)
    order = sorted(ps, key=lambda c: (ps[c][0], c))
    run, arg, out = 0.0, None, {}
    for j, c in enumerate(order):
        term = min(1.0, (m - j) * ps[c][0])
        if arg is None or term > run:
            run, arg = term, (j, c)
        out[c] = (run, (m - arg[0]) * ps[arg[1]][1] if run < 1.0 else None, arg[1])
    return out


def spec_cells(cells, scheme, model, spec=None):
    """{currency: the hl_cells row of scheme and model whose specs list spec (by
    default model@E)}; two rows for one currency stop the run."""
    spec = spec or f"{model}@E"
    out = {}
    for r in cells:
        if (
            r["scheme"] == scheme
            and r["model"] == model
            and spec in r["specs"].split("|")
        ):
            if r["currency"] in out:
                raise SystemExit(f"hl_cells: two cells for {scheme} {spec} {r}")
            out[r["currency"]] = r
    return out


def median_inf(values):
    """The median with the even-N rule (the mean of the two middle values); inf when a
    middle value is inf."""
    v = sorted(float(x) for x in values)
    n = len(v)
    if n % 2:
        return v[n // 2]
    a, b = v[n // 2 - 1], v[n // 2]
    return math.inf if (math.isinf(a) or math.isinf(b)) else (a + b) / 2


def hl_change(lb, mp):
    """How HL_MU moves from MP to LB: rises, falls, or infinite_both; two equal finite
    half-lives stop the run."""
    a, b = float(lb), float(mp)
    if math.isinf(a) and math.isinf(b):
        return "infinite_both"
    if a == b:
        raise SystemExit(f"readings: equal finite half-lives {lb} and {mp}")
    return "rises" if a > b else "falls"


def corrections(cells, univ):
    """{(currency, column): alpha_mu - alpha_ols} of the cells of CORR_COLS (recursive
    scheme) for the currencies of univ that have them."""
    out = {}
    for name, model, spec in CORR_COLS:
        sel = spec_cells(cells, "rec", model, spec)
        for c in univ:
            if c in sel:
                out[(c, name)] = float(sel[c]["alpha_mu"]) - float(sel[c]["alpha_ols"])
    return out


# ================================================================== readings
def readings(src, n8, n9):
    rows = []

    def add(block, reading, value, se="", **kw):
        r = {k: "" for k in READ_COLS}
        r.update(
            block=block,
            reading=reading,
            value=fmt(value),
            se=fmt(se),
            **{k: fmt(v) for k, v in kw.items()},
        )
        rows.append(r)

    null = {r["currency"]: r for r in src.rows["ppp_null.csv"] if r["E"] == "1"}
    cells = src.rows["hl_cells.csv"]

    # ---- the as-registered C6 (chi2(m) at 5%)
    members = as_registered(src.rows["candidates.csv"])
    add("as_registered", "N", len(members), currencies="|".join(members))

    # ---- the tests (E)
    for u, univ in (("N8", n8), ("N9", n9)):
        ok = [c for c in univ if null[c]["status"] == "ok"]
        m = len(ok)
        hb = holm_with_se(
            {c: (float(null[c]["p_II"]), float(null[c]["p_II_se"])) for c in ok}
        )
        for c in sorted(ok, key=lambda c: (float(null[c]["p_II"]), c)):
            p = float(null[c]["p_II"])
            adj, se_h, binding = hb[c]
            if se_h is not None:
                add(
                    "tests",
                    "p_II_holm",
                    adj,
                    se_h,
                    universe=u,
                    currency=c,
                    currencies=binding,
                )
            else:
                add("tests", "p_II_holm", adj, universe=u, currency=c)
            pb = min(1.0, m * p)
            add(
                "tests",
                "p_II_bonferroni",
                pb,
                m * float(null[c]["p_II_se"]) if pb < 1.0 else "",
                universe=u,
                currency=c,
            )

    # ---- the half-life text (E)
    readings_hl_text(cells, n8, n9, add)
    # ---- column II across the seven designs
    readings_designs(src, n8, n9, add)
    # ---- the spacing of the supplement's admission table
    readings_paper_tables(src, add)
    return rows


# ================================================================== the half-life text
def readings_hl_text(cells, n8, n9, add):
    """The hl_text block (module docstring) from the hl_cells rows, through
    add(block, reading, value, se="", **columns)."""
    e = {
        (sc, mo): spec_cells(cells, sc, mo)
        for sc in ("rec", "wild")
        for mo in ("LB", "MP")
    }
    for u, univ in (("N8", n8), ("N9", n9)):
        kw = dict(universe=u)
        low = [
            c
            for c in univ
            if float(e[("rec", "LB")][c]["alpha_ols"])
            < float(e[("rec", "MP")][c]["alpha_ols"])
        ]
        add(
            "hl_text",
            "alpha_ols_LB_below_MP",
            len(low),
            currencies="|".join(low),
            **kw,
        )
        for sc in ("rec", "wild"):
            lb, mp = e[(sc, "LB")], e[(sc, "MP")]
            kw = dict(universe=u, scheme=sc)
            move = {c: hl_change(lb[c]["hl_mu"], mp[c]["hl_mu"]) for c in univ}
            for how in ("rises", "falls", "infinite_both"):
                who = [c for c in univ if move[c] == how]
                add(
                    "hl_text",
                    f"hl_mu_LB_vs_MP_{how}",
                    len(who),
                    currencies="|".join(who),
                    **kw,
                )
            for mo, sel in (("MP", mp), ("LB", lb)):
                add(
                    "hl_text",
                    f"median_hl_mu_{mo}",
                    median_inf([sel[c]["hl_mu"] for c in univ]),
                    **kw,
                )
            for mo, sel in (("LB", lb), ("MP", mp)):
                ub = [c for c in univ if int(sel[c]["k_hi"]) >= 200]
                add(
                    "hl_text",
                    f"{mo}_unbounded",
                    len(ub),
                    currencies="|".join(ub),
                    **kw,
                )
            meet = [
                c
                for c in univ
                if float(lb[c]["hl_lo"]) <= 5 and float(lb[c]["hl_hi"]) >= 3
            ]
            add("hl_text", "LB_meets_3_5", len(meet), currencies="|".join(meet), **kw)
            col = [
                c
                for c in univ
                if int(mp[c]["k_hi"]) >= 200 and int(lb[c]["k_hi"]) <= 199
            ]
            add("hl_text", "collapses", len(col), currencies="|".join(col), **kw)
    for sc in ("rec", "wild"):
        for mo in ("LB", "MP"):
            for c in n9:
                r = e[(sc, mo)][c]
                if r["n_gaps"] in ("", "0"):
                    continue
                first = min(int(k) for k in r["gaps"].split("|"))
                kw = dict(scheme=sc, currency=c)
                add("hl_text", f"{mo}_n_gaps", int(r["n_gaps"]), **kw)
                add(
                    "hl_text",
                    f"{mo}_hl_hi_below_first_gap",
                    half_life(str(Decimal(first - 1) / 200)),
                    **kw,
                )
    corr = corrections(cells, n9)
    for c in n9:
        for name, _, _ in CORR_COLS:
            if (c, name) in corr:
                add("hl_text", f"corr_{name}", corr[(c, name)], currency=c)
    for u, univ in (("N8", n8), ("N9", n9)):
        above = [
            c
            for c in univ
            if all(
                corr[(c, n)] > corr[(c, "MP")]
                for n in ("LB_b1", "LB_b2", "LB")
                if (c, n) in corr
            )
        ]
        two = [c for c in univ if (c, "LB_b1") in corr]
        both = [
            c
            for c in two
            if corr[(c, "LB")] > corr[(c, "LB_b1")]
            and corr[(c, "LB")] > corr[(c, "LB_b2")]
        ]
        for rd, who in (
            ("corr_dummies_above_none", above),
            ("corr_all_above_both_single", both),
            ("corr_all_not_above_both_single", [c for c in two if c not in both]),
        ):
            add("hl_text", rd, len(who), universe=u, currencies="|".join(who))


# ================================================================== the designs
def design_row(src, c, st):
    """The row of currency c under design st: ppp_sets.csv, or sj_null.csv for S-JST."""
    if st == "S-JST":
        g, k = "sj_null.csv", dict(currency=c)
    else:
        g, k = "ppp_sets.csv", dict(set=st, currency=c)
    hit = [r for r in src.rows[g] if all(r[a] == b for a, b in k.items())]
    if len(hit) != 1:
        raise SystemExit(f"{g}: {len(hit)} rows for {k}")
    return g, k, hit[0]


def readings_designs(src, n8, n9, add):
    """The designs block (module docstring), through add(block, reading, value, se,
    **columns)."""

    def row(c, st):
        return design_row(src, c, st)[2]

    def rejects(c, st):
        return row(c, st)["status"] == "ok" and row(c, st)["reject_II"] == "1"

    for u, univ in (("N8", n8), ("N9", n9)):
        for st in PAPER_SETS:
            kw = dict(universe=u, set=st)
            ok = [c for c in univ if row(c, st)["status"] == "ok"]
            rej = [c for c in ok if rejects(c, st)]
            add("designs", "N", len(ok), currencies="|".join(ok), **kw)
            add("designs", "K_obs_II", len(rej), currencies="|".join(rej), **kw)
    for c in n9:
        ds = [st for st in PAPER_SETS if rejects(c, st)]
        add("designs", "n_designs_rejecting_II", len(ds), currency=c)
        add("designs", "designs_rejecting_II", "|".join(ds), currency=c)
    sj = {c: row(c, "S-JST") for c in n9}
    for u, univ in (("N8", n8), ("N9", n9)):
        kw = dict(universe=u, set="S-JST")
        rej1 = [c for c in univ if sj[c]["reject_I"] == "1"]
        add("designs", "rejecting_I", len(rej1), currencies="|".join(rej1), **kw)
        ok = [c for c in univ if sj[c]["status"] == "ok"]
        m = len(ok)
        hb = holm_with_se(
            {c: (float(sj[c]["p_II"]), float(sj[c]["p_II_se"])) for c in ok}
        )
        for c in sorted(ok, key=lambda c: (float(sj[c]["p_II"]), c)):
            p = float(sj[c]["p_II"])
            adj, se_h, binding = hb[c]
            add(
                "designs",
                "p_II_holm",
                adj,
                se_h if se_h is not None else "",
                currency=c,
                currencies=binding if se_h is not None else "",
                **kw,
            )
            pb = min(1.0, m * p)
            add(
                "designs",
                "p_II_bonferroni",
                pb,
                m * float(sj[c]["p_II_se"]) if pb < 1.0 else "",
                currency=c,
                **kw,
            )


# ================================================================== paper_tables
def readings_paper_tables(src, add):
    """The paper_tables block (module docstring), through add(block, reading, value,
    se, **columns)."""
    for r in src.rows["candidates.csv"]:
        taus = [int(t) for t in r["events_E"].split(";")]
        if len(taus) == 2:
            add(
                "paper_tables",
                "spacing",
                (taus[1] - taus[0]) / T_LEVELS,
                currency=r["currency"],
            )


# ================================================================== tables
def hl_cell(scheme, model, spec, cells, c):
    hit = [
        r
        for r in cells
        if r["scheme"] == scheme
        and r["model"] == model
        and r["currency"] == c
        and spec in r["specs"].split("|")
    ]
    if len(hit) != 1:
        raise SystemExit(f"hl_cells: {len(hit)} cells for {scheme} {model} {c} {spec}")
    return hit[0]


def hl_key(r):
    return dict(
        scheme=r["scheme"],
        model=r["model"],
        currency=r["currency"],
        cfg=r["cfg"],
        p=r["p"],
    )


def cap_tok(reference, key, column):
    return tok("cap", reference, key, column, rule="cap", derive=CAP_PREFIX[column])


def hl_interval(g, k, r):
    """The cells alpha_MU, HL_MU, the CI for alpha and the CI for the half-life of one
    half-life result, read from the reference file g at key k; r is that row. A gapped
    set's CI for alpha carries a double dagger."""
    if r["mu_at_cap"] == "1":
        a_mu = cell(cap_tok(g, k, "mu_at_cap"))
    else:
        a_mu = cell(tok("alpha_mu", g, k, "alpha_mu"))
    hl_mu = cell(tok("hl", g, k, "hl_mu"))
    if r["ci_above_cap"] == "1":
        ci_a = cell(cap_tok(g, k, "ci_above_cap"))
    else:
        if r["upper_open"] == "1":
            up = cap_tok(g, k, "upper_open")
        elif r["upper_at_cap"] == "1":
            up = cap_tok(g, k, "upper_at_cap")
        else:
            up = tok("node", g, k, "alpha_hi")
        ci_a = cell(
            lit("["), tok("node", g, k, "alpha_lo"), lit(", "), up, lit("]")
        )
    ci_h = cell(
        lit("["),
        tok("hl_node", g, k, "hl_lo"),
        lit(", "),
        tok("hl_node", g, k, "hl_hi"),
        lit("]"),
    )
    return a_mu, hl_mu, ci_a, ci_h


class Marks:
    """The dagger (in N9 only) and double-dagger (gaps) marks of one table, and their
    notes."""

    def __init__(self, n8):
        self.n8 = n8
        self.marked = []  # currencies outside N8, in order of appearance
        self.gaps = []  # the hl_cells keys of the rows that carry a double dagger

    def dagger(self, c):
        if c in self.n8:
            return lit("")
        if c not in self.marked:
            self.marked.append(c)
        return lit(r"$^\dagger$")

    def gap(self, r):
        if r["n_gaps"] in ("", "0"):
            return lit("")
        k = hl_key(r)
        if k not in self.gaps:
            self.gaps.append(k)
        return lit(r"$^\ddagger$")

    def notes(self, tb):
        if self.marked:
            parts = [lit(r"$^\dagger$ ")]
            for i, c in enumerate(self.marked):
                k = dict(currency=c)
                parts += ([lit("; ")] if i else []) + [
                    tok(
                        "label",
                        "candidates.csv",
                        k,
                        "currency",
                        "derived",
                        derive="text",
                    ),
                    lit(": "),
                    tok(
                        "label",
                        "candidates.csv",
                        k,
                        "C6_5",
                        "derived",
                        derive="n9_only",
                    ),
                ]
            tb.note(parts + [lit(".")])
        if self.gaps:
            parts = [
                lit(
                    r"$^\ddagger$ the accepted set has gaps (grid nodes rejected "
                    r"inside it): "
                )
            ]
            for i, k in enumerate(self.gaps):
                parts += ([lit("; ")] if i else []) + [
                    tok(
                        "label", "hl_cells.csv", k, "currency", "derived", derive="text"
                    ),
                    lit(" " + k["model"]),
                ]
                parts += [
                    lit(r" at $\alpha$ = "),
                    tok("node", "hl_cells.csv", k, "gaps", "derived", derive="gaps"),
                ]
            tb.note(parts + [lit(".")])


def verdict_shown(reject, stat, thr):
    """Whether a verdict shows strictly in printed numbers: reject iff stat < thr."""
    s, t = Decimal(stat), Decimal(thr)
    return (reject == "1" and s < t) or (reject == "0" and s > t)


class Stars:
    """The verdicts that do not show in the printed numbers of their row, and the note
    with their full-precision values."""

    def __init__(self):
        self.rows = []  # (label tokens, value tokens)

    def check(self, rend, g, k, flag, comps):
        """comps: [(stat column, stat class, threshold: a constant string or a
        (column, class) pair, se column or None)] for one verdict flag of row k of g.
        Returns the mark (a literal) and records the note. In the first pass (the class
        digits not yet known) nothing is marked."""
        if rend.digits is None:
            return lit("")
        reject = rend.src.cell(g, k, flag)
        bad = []
        for scol, scls, thr, secol in comps:
            s = rend.printed_number(g, k, scol, scls)
            t = (
                thr
                if isinstance(thr, str)
                else rend.printed_number(g, k, thr[0], thr[1])
            )
            if not verdict_shown(reject, s, t):
                bad.append((scol, thr, secol))
        if not bad:
            return lit("")
        vals = []
        for scol, thr, secol in bad:
            vals.append((scol, secol))
            if not isinstance(thr, str):
                vals.append((thr[0], None))
        self.rows.append((g, k, vals))
        return lit(r"$^{*}$")

    def notes(self, tb, label):
        for g, k, vals in self.rows:
            parts = [lit(r"$^{*}$ ")] + label(g, k) + [lit(": ")]
            uniq = []
            for v in vals:
                if v not in uniq:
                    uniq.append(v)
            for i, (col, secol) in enumerate(uniq):
                parts += ([lit(", ")] if i else []) + [
                    lit(COLUMN_NAMES[col] + " "),
                    tok("note", g, k, col, rule="full"),
                ]
                if secol:
                    parts += [
                        lit(" (se "),
                        tok("note", g, k, secol, rule="full"),
                        lit(")"),
                    ]
            tb.note(parts + [lit(".")])


COLUMN_NAMES = {
    "mzt_I": r"MZ$_t$",
    "cv_II": "cv (II)",
    "p_II": r"$p$ (II)",
    "mzt_III": r"MZ$_t$ (III)",
    "cv_III": "cv (III)",
    "p_III": r"$p$ (III)",
}


def p_se(reference, key, pcol, secol):
    return cell(
        tok("p", reference, key, pcol, se=(reference, key, secol)),
        lit(" ("),
        tok("se", reference, key, secol, rule="se", math_=False),
        lit(")"),
    )


def paper_cur(g, k, c, n8):
    """The currency cell of a supplement table: its code, with a dagger outside N8."""
    return cell(
        tok("label", g, k, "currency", "derived", derive="text"),
        lit("" if c in n8 else PAPER_DAGGER),
    )


def has_row(src, name, key):
    """Whether some row of `name` matches key."""
    return any(all(r.get(a) == b for a, b in key.items()) for r in src.rows[name])


def status_row(tb, first, g, k, span):
    """A row whose status is not ok: its status instead of its numbers."""
    tb.row(
        first
        + [
            [("lit", rf"\multicolumn{{{span}}}{{l}}{{")]
            + cell(tok("label", g, k, "status", "derived", derive="text"))
            + [("lit", "}")]
        ]
    )


def tables(src, n8, n9, rend):
    order = row_order(n8, n9)
    out = []
    null = {r["currency"]: r for r in src.rows["ppp_null.csv"] if r["E"] == "1"}

    def cur(g, k, c, marks):
        return cell(
            tok("label", g, k, "currency", "derived", derive="text"), marks.dagger(c)
        )

    # ---- tab:ppp
    tb = Table(
        "tab:ppp",
        "tab_ppp.tex",
        "l" + "c" * 20,
        r" & \multicolumn{14}{c}{E} & \multicolumn{6}{c}{S5} \\ "
        r"\cmidrule(lr){2-15}\cmidrule(lr){16-21}"
        "\n"
        r" & $\tau$ & $\hat p$ & MZ$_t$ & (I) & size (I) & cv (II) & $p$ (II) & (II)"
        r" & $\bar c_{\frac{1}{2}}$ & MZ$_t(\bar c_{\frac{1}{2}})$ & cv (III)"
        r" & $p$ (III) & (III) & $p_{\mathrm{BB}}$"
        r" & $\tau$ & MZ$_t$ & $p$ (II) & (I) & (II) & (III)",
    )
    marks, stars = Marks(n8), Stars()
    for c in order:
        g, k = "ppp_null.csv", dict(currency=c, E="1")
        s5g, s5 = "ppp_sets.csv", dict(set="S5", currency=c)
        if null[c]["status"] != "ok":
            status_row(tb, [cur(g, k, c, marks)], g, k, 20)
            continue
        bbk = dict(currency=c)
        bb = [r for r in src.rows["ppp_berger_boos_summary.csv"] if r["currency"] == c][
            0
        ]
        if bb["status"] == "ok":
            pbb = p_se("ppp_berger_boos_summary.csv", bbk, "p_BB", "p_max_se")
        else:
            pbb = cell(
                tok(
                    "label",
                    "ppp_berger_boos_summary.csv",
                    bbk,
                    "status",
                    "derived",
                    derive="na",
                )
            )
        m_i = stars.check(rend, g, k, "reject_I", [("mzt_I", "stat", CV_ASY, None)])
        m_ii = stars.check(
            rend,
            g,
            k,
            "reject_II",
            [
                ("mzt_I", "stat", ("cv_II", "cv"), None),
                ("p_II", "p", "0.05", "p_II_se"),
            ],
        )
        m_iii = stars.check(
            rend,
            g,
            k,
            "reject_III",
            [
                ("mzt_III", "stat", ("cv_III", "cv"), None),
                ("p_III", "p", "0.05", "p_III_se"),
            ],
        )
        row = [
            cur(g, k, c, marks),
            cell(
                tok(
                    "int",
                    "ppp_sets.csv",
                    dict(set="E", currency=c),
                    "tau",
                    "derived",
                    derive="bars",
                )
            ),
            cell(tok("int", g, k, "p_hat")),
            cell(tok("stat", g, k, "mzt_I")),
            cell(tok("label", g, k, "reject_I", rule="verdict"), m_i),
            cell(tok("level", g, k, "size_I", se=(g, k, "size_I_se"))),
            cell(tok("cv", g, k, "cv_II", se=(g, k, "cv_II_se"))),
            p_se(g, k, "p_II", "p_II_se"),
            cell(tok("label", g, k, "reject_II", rule="verdict"), m_ii),
            cell(tok("cbar", g, k, "cbar_half")),
            cell(tok("stat", g, k, "mzt_III")),
            cell(tok("cv", g, k, "cv_III", se=(g, k, "cv_III_se"))),
            p_se(g, k, "p_III", "p_III_se"),
            cell(tok("label", g, k, "reject_III", rule="verdict"), m_iii),
            pbb,
        ]
        if src.cell(s5g, s5, "status") != "ok":
            row += [
                [("lit", r"\multicolumn{6}{l}{")]
                + cell(tok("label", s5g, s5, "status", "derived", derive="text"))
                + [("lit", "}")]
            ]
        else:
            s_i = stars.check(
                rend, s5g, s5, "reject_I", [("mzt_I", "stat", CV_ASY, None)]
            )
            s_ii = stars.check(
                rend, s5g, s5, "reject_II", [("p_II", "p", "0.05", "p_II_se")]
            )
            row += [
                cell(tok("int", s5g, s5, "tau", "derived", derive="bars")),
                cell(tok("stat", s5g, s5, "mzt_I")),
                p_se(s5g, s5, "p_II", "p_II_se"),
                cell(tok("label", s5g, s5, "reject_I", rule="verdict"), s_i),
                cell(tok("label", s5g, s5, "reject_II", rule="verdict"), s_ii),
                cell(tok("label", s5g, s5, "reject_III", rule="verdict")),
            ]
        tb.row(row)
    marks.notes(tb)

    def lab_ppp(g, k):
        s = "E" if g == "ppp_null.csv" else k["set"]
        return [
            tok("label", g, k, "currency", "derived", derive="text"),
            lit(", "),
            tok(
                "label",
                "ppp_sets.csv",
                dict(set=s, currency=k["currency"]),
                "set",
                "derived",
                derive="text",
            ),
        ]

    stars.notes(tb, lab_ppp)
    out.append(tb)

    cells_ = src.rows["hl_cells.csv"]
    # ---- stab:app-hl
    tb = Table(
        "stab:app-hl",
        "stab_app_hl.tex",
        "llcccccc",
        r" & model & $\hat p$ & $\hat\alpha_{\mathrm{OLS}}$"
        r" & $\hat\alpha_{\mathrm{MU}}$ & HL$_{\mathrm{MU}}$ & CI for $\alpha$"
        r" & CI for the half-life",
    )
    marks = Marks(n8)
    for c in order:
        for j, mo in enumerate(("LB", "MP")):
            r = hl_cell("rec", mo, f"{mo}@E", cells_, c)
            g, k = "hl_cells.csv", hl_key(r)
            first = cur(g, k, c, marks) if j == 0 else cell(lit(""))
            a_mu, hl_mu, ci_a, ci_h = hl_interval(g, k, r)
            tb.row(
                [
                    first,
                    cell(lit(mo)),
                    cell(tok("int", g, k, "p")),
                    cell(tok("alpha_ols", g, k, "alpha_ols")),
                    a_mu,
                    hl_mu,
                    ci_a + [marks.gap(r)],
                    ci_h,
                ]
            )
    marks.notes(tb)
    out.append(tb)

    out.append(table_app_corr(src, n8, n9))
    out.append(table_app_sets(src, n8, n9))
    out.append(table_app_dates(src))
    out.append(table_app_power(src, n8, n9))
    out.append(table_app_count())
    return out


# ================================================================== supplement tables
def design_power_members(src, order):
    """The members of the class 'level' (power and size levels) that no table prints
    in this class (the power of (II) at each h and its band mean appear in
    stab:app-power at three decimals, class level3; those of (III) in no table):
    the power of (II) and (III) at each h = 3, 4, 5 of the design (ppp_power.csv; a
    currency and h whose rows are not ok have none) and their band means
    (ppp_power_band.csv), currency by currency in the order of the rows. Their se
    enter the printed decimals of the class (Renderer.collect)."""
    out = []
    for c in order:
        for h in H:
            km = dict(currency=c, h=h, arm="main")
            k3 = dict(currency=c, h=h, arm="III")
            status = [
                src.cell("ppp_power.csv", km, "status"),
                src.cell("ppp_power.csv", k3, "status"),
            ]
            if status != ["ok", "ok"]:
                continue
            for k in (km, k3):
                out.append(
                    tok(
                        "level",
                        "ppp_power.csv",
                        k,
                        "power",
                        se=("ppp_power.csv", k, "se"),
                    )[1]
                )
        for k in (dict(currency=c, arm="main"), dict(currency=c, arm="III")):
            out.append(
                tok(
                    "level",
                    "ppp_power_band.csv",
                    k,
                    "band_mean",
                    se=("ppp_power_band.csv", k, "se"),
                )[1]
            )
    return out


def table_app_corr(src, n8, n9):
    """stab:app-corr from the hl_text readings: the correction per dummy model, --
    where the currency has one E date."""
    rt = "readings_tables.csv"
    tb = Table(
        "stab:app-corr",
        "stab_app_corr.tex",
        "lcccc",
        "Currency & none & first date only & second date only & all E dates",
    )
    for c in n9:
        k0 = dict(block="hl_text", reading="corr_MP", currency=c)
        row = [paper_cur(rt, k0, c, n8)]
        for name, _, _ in CORR_COLS:
            k = dict(block="hl_text", reading=f"corr_{name}", currency=c)
            if has_row(src, rt, k):
                row.append(cell(tok("corr", rt, k, "value", math_=False)))
            else:
                row.append(cell(lit("--")))
        tb.row(row)
    return tb


def table_app_sets(src, n8, n9):
    """stab:app-sets: per currency a row of p (II) with its star (n.v. or -- without a
    verdict) and a row of the break years where they differ from E."""
    tb = Table(
        "stab:app-sets",
        "stab_app_sets.tex",
        "lccccccc",
        "Currency & " + " & ".join(PAPER_SETS),
    )
    for c in n9:
        eg, ek, e = design_row(src, c, "E")
        first, years = [paper_cur(eg, ek, c, n8)], [cell(lit(r"{\scriptsize}"))]
        for st in PAPER_SETS:
            g, k, r = design_row(src, c, st)
            if r["status"] == "ok":
                first.append(
                    cell(
                        tok("p3", g, k, "p_II", se=(g, k, "p_II_se"), math_=False),
                        tok("label", g, k, "reject_II", rule="star", derive="1"),
                    )
                )
            else:
                how = "no_event" if r["status"].startswith("no event") else "nv"
                first.append(cell(tok("label", g, k, "status", "derived", derive=how)))
            shown = st == "E" or (r["tau"] and r["tau"] != e["tau"])
            years.append(
                cell(
                    lit(r"{\scriptsize "),
                    *(
                        [tok("int", g, k, "tau", "derived", derive="years2")]
                        if shown
                        else []
                    ),
                    lit("}"),
                )
            )
        tb.row(first)
        tb.row(years)
    return tb


def table_app_dates(src):
    """stab:app-dates: per candidate (candidates.csv) the E break years, m, the
    spacing, C4's kappa2, C6's W, simulated cv and p (* where C6 rejects at 5%), the
    chi2(m) p-value and the size of that reference at 5% (gate_sizes.csv)."""
    cand, gs, cv6 = "candidates.csv", "gate_sizes.csv", "c6_critical_values.csv"
    rt = "readings_tables.csv"
    tb = Table(
        "stab:app-dates",
        "stab_app_dates.tex",
        "llcccrrccc",
        r"& & & & C4 & \multicolumn{3}{c}{C6, simulated}"
        r" & \multicolumn{2}{c}{C6, $\chi^2(m)$} \\"
        "\n"
        r"\cmidrule(lr){5-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}"
        "\n"
        r"Currency & $\tau$ (rule E) & $m$ & Spacing & $\kappa_2$ & $W$ & cv & $p$"
        r" & $p_{\chi^2}$ & size",
    )
    for r in src.rows[cand]:
        k = dict(currency=r["currency"])
        ks = dict(block="paper_tables", reading="spacing", currency=r["currency"])
        spacing = (
            cell(tok("spacing", rt, ks, "value", math_=False))
            if has_row(src, rt, ks)
            else cell(lit("--"))
        )
        tb.row(
            [
                cell(tok("label", cand, k, "currency", "derived", derive="text")),
                cell(tok("int", cand, k, "events_E", "derived", derive="semi")),
                cell(tok("int", cand, k, "c6_df", math_=False)),
                spacing,
                cell(tok("gate_stat", cand, k, "kappa2", math_=False)),
                cell(tok("gate_stat", cand, k, "c6_wald", math_=False)),
                cell(
                    tok(
                        "gate_cv",
                        cand,
                        k,
                        "c6_cv_5",
                        se=(cv6, k, "se_cv_5"),
                        math_=False,
                    )
                ),
                cell(
                    tok("p3", cand, k, "c6_p", se=(cand, k, "c6_p_se"), math_=False),
                    tok("label", cand, k, "C6_5", rule="star", derive="0"),
                ),
                cell(tok("p_chi2", cand, k, "c6_p_chi2", "copy_floor", math_=False)),
                cell(
                    tok(
                        "level3",
                        gs,
                        k,
                        "size_C6chi2_5",
                        se=(gs, k, "se_size_C6chi2_5"),
                        math_=False,
                    )
                ),
            ]
        )
    return tb


def table_app_power(src, n8, n9):
    """stab:app-power: per currency the power of (II) at h = 3, 4, 5 (main arm), its
    band mean, and the band mean with the stationary initial condition."""
    pw, bd = "ppp_power.csv", "ppp_power_band.csv"
    tb = Table(
        "stab:app-power",
        "stab_app_power.tex",
        "lccccc",
        r"& \multicolumn{3}{c}{Power} & & Band, \\"
        "\n"
        r"\cmidrule(lr){2-4}"
        "\n"
        r"Currency & $h=3$ & $h=4$ & $h=5$ & Band & stationary $u_0$",
    )
    for c in n9:
        row = [paper_cur(bd, dict(currency=c, arm="main"), c, n8)]
        keys = [(pw, dict(currency=c, h=h, arm="main"), "power") for h in H]
        keys += [
            (bd, dict(currency=c, arm=a), "band_mean") for a in ("main", "stationary")
        ]
        for g, k, col in keys:
            if src.cell(g, k, "status") != "ok":
                raise SystemExit(f"stab:app-power: {g} {k} has status not ok")
            row.append(cell(tok("level3", g, k, col, se=(g, k, "se"), math_=False)))
        tb.row(row)
    return tb


def table_app_count():
    """stab:app-count: the joint law of K over N8 (u0 = 0) at kappa = 0, 1 under the
    null and at h = 3, 4, 5: E[K], P(K = 0), P(K <= K_obs), and P(K <= K_obs) with the
    stationary initial condition (-- under the null)."""
    law = "ppp_count_law.csv"
    tb = Table(
        "stab:app-count",
        "stab_app_count.tex",
        "llcccc",
        r"& & \multicolumn{3}{c}{$u_0=0$} & stationary $u_0$ \\"
        "\n"
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-6}"
        "\n"
        r"$h$ & $\kappa$ & $E[K]$ & $P(K=0)$ & $P(K\le1)$ & $P(K\le1)$",
    )
    for i, h in enumerate(("H0",) + H):
        if i:
            tb.space()
        for kappa in ("0", "1"):
            k = dict(universe="N8", h=h, kappa=kappa, arm="u0zero")
            if kappa != "0":
                first = cell(lit(""))
            elif h == "H0":
                first = cell(lit("null"))
            else:
                first = cell(tok("int", law, k, "h", math_=False))
            ks = dict(k, arm="stationary")
            tb.row(
                [
                    first,
                    cell(tok("int", law, k, "kappa", math_=False)),
                    cell(tok("ek", law, k, "EK", se=(law, k, "EK_se"), math_=False)),
                    cell(
                        tok("level3", law, k, "P0", se=(law, k, "P0_se"), math_=False)
                    ),
                    cell(
                        tok(
                            "level3",
                            law,
                            k,
                            "P_le_Kobs",
                            se=(law, k, "P_le_Kobs_se"),
                            math_=False,
                        )
                    ),
                    cell(lit("--"))
                    if h == "H0"
                    else cell(
                        tok(
                            "level3",
                            law,
                            ks,
                            "P_le_Kobs",
                            se=(law, ks, "P_le_Kobs_se"),
                            math_=False,
                        )
                    ),
                ]
            )
    return tb


# ================================================================= checks
def check_tokens(src, texts, prov):
    """Every provenance row re-read and its token re-derived; every body line is its
    tokens and digit-free literal text."""
    probs = []
    for p in prov:
        rule = p["rule"]
        got = src.cell(
            p["reference"], key_parse(p["reference_key"]), p["reference_column"]
        )
        if got != p["value_full"]:
            probs.append(
                f"{p['table_file']} row {p['body_row']}: reference {got!r} != "
                f"value_full {p['value_full']!r}"
            )
            continue
        if rule == "cap":  # the flag set, and the prefix its column names
            want = f"${CAP_PREFIX[p['reference_column']]}{CAP}$"
            if got != "1" or p["printed"] != want:
                probs.append(
                    f"{p['table_file']} row {p['body_row']}: cap token "
                    f"{p['printed']!r} on {p['reference_column']} = {got!r}"
                )
            continue
        if rule in ("copy", "copy_floor"):
            d = p["decimals_printed"]
            want = (
                r"\infty"
                if got == "inf"
                else r"-\infty"
                if got == "-inf"
                else rnd(got, int(d))
            )
            if rule == "copy_floor" and not is_inf(got):
                want = floor_mark(got, want, int(d))
            ok = p["printed"] in (f"${want}$", want)
        elif rule == "full":
            ok = p["printed"] == f"${got}$"
        elif rule == "se":
            ok = p["printed"] == sig1(got)
        elif rule == "verdict":
            ok = got in ("0", "1") and p["printed"] == ("R" if got == "1" else "--")
        elif rule.startswith("star:"):
            trig = rule.split(":", 1)[1]
            ok = got in ("0", "1") and p["printed"] == (STAR if got == trig else "")
        elif rule.startswith("derived:"):
            ok = p["printed"] == tex_escape(DERIVE[rule.split(":", 1)[1]](got))
        else:
            ok = False
        if not ok:
            probs.append(
                f"{p['table_file']} row {p['body_row']}: printed {p['printed']!r} "
                f"is not rule {rule} of {got!r}"
            )
    by_file = {}
    for p in prov:
        by_file.setdefault(p["table_file"], []).append(p)
    for fname, text in texts.items():
        lines = text.split("\n")
        body = lines[lines.index(r"\midrule") + 1 : lines.index(r"\bottomrule")]
        body = [ln for ln in body if ln not in (r"\midrule", SPACE)]
        toks = by_file.get(fname, [])
        for n, ln in enumerate(body, start=1):
            rest, pos = "", 0
            for p in [p for p in toks if p["body_row"] == n]:
                i = ln.find(p["printed"], pos)
                if i < 0:
                    probs.append(f"{fname} row {n}: token {p['printed']!r} not found")
                    continue
                rest += ln[pos:i] + "\x00"
                pos = i + len(p["printed"])
            rest = re.sub(r"\\multicolumn\{\d+\}", "", rest + ln[pos:])
            if any(ch.isdigit() for ch in rest) or r"\infty" in rest:
                probs.append(f"{fname} row {n}: an unregistered digit in {rest!r}")
    return probs


def funnel_blocks(text):
    """{block label: [admitted currencies]} of funnel.txt, one admitted line per
    block."""
    out = {}
    for blk in text.split("--- attrition, ")[1:]:
        label = blk.split(" ---", 1)[0]
        adm = [ln for ln in blk.splitlines() if ln.strip().startswith("admitted:")]
        if len(adm) != 1:
            raise SystemExit(
                f"funnel.txt: block {label!r} has {len(adm)} admitted lines"
            )
        out[label] = [x.strip() for x in adm[0].split(":", 1)[1].split(",")]
    return out


def check_consistency(src, n8, n9, rd_rows):
    probs = []
    R = src.rows["readings.csv"]
    null = {r["currency"]: r for r in src.rows["ppp_null.csv"] if r["E"] == "1"}
    cand = src.rows["candidates.csv"]
    # the universes: readings.csv and every block of funnel.txt
    for u, univ in (("N8", n8), ("N9", n9)):
        k = [r for r in R if r["universe"] == u and r["reading"] == "K_obs"][0]
        if sorted(k["currencies"].split("|")) != sorted(univ):
            probs.append(f"{u}: the universe {univ} != readings.csv {k['currencies']}")
        mine = [
            r
            for r in rd_rows
            if r["block"] == "designs"
            and r["universe"] == u
            and r["set"] == "E"
            and r["reading"] == "K_obs_II"
        ][0]
        if mine["value"] != k["value"]:
            probs.append(f"{u}: K_obs {mine['value']} != readings.csv {k['value']}")
        for adj in ("holm", "bonferroni"):
            for r in R:
                if r["universe"] == u and r["reading"] == f"p_II_{adj}":
                    m = [
                        x
                        for x in rd_rows
                        if x["block"] == "tests"
                        and x["reading"] == f"p_II_{adj}"
                        and x["universe"] == u
                        and x["currency"] == r["arm"]
                    ][0]
                    if m["value"] != r["value"]:
                        probs.append(
                            f"{u} {adj} {r['arm']}: {m['value']} != {r['value']}"
                        )
    fb = funnel_blocks(src.funnel)
    reg = [r for r in rd_rows if r["block"] == "as_registered" and r["reading"] == "N"][
        0
    ]
    want = {
        "main: 5%": n8,
        "sensitivity: C4 and C6 at 1%": [
            r["currency"] for r in cand if r["C4_1"] == "1" and r["C6_1"] == "1"
        ],
        "sensitivity: without C6 (C4 at 5%)": n9,
        "as registered (C6 with the asymptotic chi2(m), 5%)": reg["currencies"].split(
            "|"
        ),
    }
    if sorted(fb) != sorted(want):
        probs.append(f"funnel.txt blocks {sorted(fb)} != {sorted(want)}")
    for label, univ in want.items():
        if sorted(fb.get(label, [])) != sorted(univ):
            probs.append(f"funnel.txt {label}: {fb.get(label)} != {univ}")
    # ppp_sets against ppp_null and hl_cells; the verdict flags
    shared = (
        "p_hat",
        "cfg",
        "status",
        "mzt_I",
        "reject_I",
        "cv_II",
        "p_II",
        "p_II_se",
        "reject_II",
        "size_I",
        "size_I_se",
        "cbar_half",
        "mzt_III",
        "cv_III",
        "p_III",
        "p_III_se",
        "reject_III",
    )
    cells = src.rows["hl_cells.csv"]
    for r in src.rows["ppp_sets.csv"]:
        c = r["currency"]
        if r["set"] == "E":
            for f in shared:
                if r[f] != null[c][f]:
                    probs.append(f"ppp_sets E {c} {f} differs from ppp_null")
        if not r["cfg"]:
            continue
        for m in ("LB", "MP"):
            h = hl_cell("rec", m, f"{m}@{r['set']}", cells, c)
            if h["p"] != r["p_hat"]:
                probs.append(
                    f"ppp_sets {r['set']} {c}: p_hat {r['p_hat']} != {m} cell p "
                    f"{h['p']}"
                )
            if m == "LB" and h["cfg"] != r["cfg"]:
                probs.append(f"ppp_sets {r['set']} {c}: cfg {r['cfg']} != LB cell cfg")
            for f in HL_FIELDS:
                if r[f"{m}_{f}"] != h[f]:
                    probs.append(
                        f"ppp_sets {r['set']} {c} {m}_{f} differs from hl_cells"
                    )
    for g, rows in (
        ("ppp_null.csv", src.rows["ppp_null.csv"]),
        ("ppp_sets.csv", [r for r in src.rows["ppp_sets.csv"] if r["cfg"]]),
    ):
        for r in rows:
            if r["status"] != "ok":
                continue
            rules = (
                ("reject_I", float(r["mzt_I"]) < float(CV_ASY)),
                ("reject_II", float(r["mzt_I"]) < float(r["cv_II"])),
                ("reject_II", float(r["p_II"]) < float(LEVEL)),
                ("reject_III", float(r["mzt_III"]) < float(r["cv_III"])),
                ("reject_III", float(r["p_III"]) < float(LEVEL)),
            )
            for f, v in rules:
                if r[f] not in ("0", "1") or (r[f] == "1") != v:
                    probs.append(
                        f"{g} {r.get('set', 'E')} {r['currency']}: {f} = {r[f]!r} "
                        f"is not its rule"
                    )
    # the events
    ev = src.rows["regime_events.csv"]
    probs += [f"regime_events.csv: {b}" for b in RE.integrity(ev)]
    if RE.generated(ev) != [(r["f"], r["tau"]) for r in ev]:
        probs.append(
            "regime_events.csv: f or tau is not the one generated from the date"
        )
    for c in n9:
        taus = RE.configuration(ev, c, "E")[0]
        e = [
            r
            for r in src.rows["ppp_sets.csv"]
            if r["set"] == "E" and r["currency"] == c
        ][0]
        if "|".join(map(str, taus)) != e["tau"]:
            probs.append(f"{c}: configuration E {taus} != ppp_sets tau {e['tau']}")
    # internal identities
    pw = {(r["currency"], r["h"], r["arm"]): r for r in src.rows["ppp_power.csv"]}
    for r in src.rows["ppp_power_band.csv"]:
        if r["status"] != "ok":
            continue
        hs = [pw[(r["currency"], h, r["arm"])] for h in r["h_used"].split("|")]
        mean = sum(float(x["power"]) for x in hs) / len(hs)
        if abs(mean - float(r["band_mean"])) > 1e-12:
            probs.append(
                f"band mean {r['currency']} {r['arm']} is not the mean of its h"
            )
    for r in cells:
        for a, hl in (
            ("alpha_mu", "hl_mu"),
            ("alpha_lo", "hl_lo"),
            ("alpha_hi", "hl_hi"),
        ):
            if r[a] in ("", "nan") or r[hl] == "":
                continue
            want = half_life(r[a])
            got = float(r[hl])
            if not (math.isinf(want) and math.isinf(got)) and abs(
                want - got
            ) > 1e-9 * max(1.0, abs(want)):
                probs.append(
                    f"hl_cells {hl_key(r)}: {hl} {got} != ln .5 / ln {a} = {want}"
                )
        if r["upper_open"] != "1" and r["ci_above_cap"] != "1":  # limits = nodes k/200
            for a, k in (("alpha_lo", "k_lo"), ("alpha_hi", "k_hi")):
                if Decimal(r[a]) * 200 != Decimal(r[k]):
                    probs.append(f"hl_cells {hl_key(r)}: {a} is not {k} / 200")
        n_gaps = len(r["gaps"].split("|")) if r["gaps"] else 0
        if int(r["n_gaps"] or 0) != n_gaps:
            probs.append(f"hl_cells {hl_key(r)}: n_gaps is not the number of gaps")
    for r in src.rows["ppp_berger_boos_summary.csv"]:
        if r["status"] == "ok" and float(r["p_BB"]) != float(r["p_max"]) + 0.01:
            probs.append(f"p_BB of {r['currency']} is not p_max + 0.01")
    probs += check_hl_text(src, n9, rd_rows)
    probs += check_designs(src, n9, rd_rows)
    probs += check_admission(src)
    return probs


# the hl_text readings that readings.csv also holds (its reading name)
HL_TEXT_IN_READINGS = {
    "median_hl_mu_LB": "median_hl_mu_LB",
    "median_hl_mu_MP": "median_hl_mu_MP",
    "LB_unbounded": "LB_unbounded",
    "MP_unbounded": "MP_unbounded",
    "LB_meets_3_5": "LB_meet_3_5",
    "collapses": "collapse",
}


def check_hl_text(src, n9, rd_rows):
    probs = []
    R = src.rows["readings.csv"]
    cells = src.rows["hl_cells.csv"]
    seen = 0
    for r in rd_rows:
        if r["block"] != "hl_text" or r["reading"] not in HL_TEXT_IN_READINGS:
            continue
        want = [
            x
            for x in R
            if x["universe"] == r["universe"]
            and x["scheme"] == r["scheme"]
            and x["reading"] == HL_TEXT_IN_READINGS[r["reading"]]
        ]
        seen += 1
        if len(want) != 1 or want[0]["value"] != r["value"]:
            probs.append(
                f"hl_text {r['universe']} {r['scheme']} {r['reading']} {r['value']} "
                f"!= readings.csv {[x['value'] for x in want]}"
            )
    if seen != 2 * 2 * len(HL_TEXT_IN_READINGS):
        probs.append(f"hl_text: {seen} readings compared with readings.csv")
    for mo in ("LB", "MP"):
        a, b = spec_cells(cells, "rec", mo), spec_cells(cells, "wild", mo)
        for c in n9:
            if a[c]["alpha_ols"] != b[c]["alpha_ols"]:
                probs.append(f"hl_cells {mo}@E {c}: alpha_ols differs between schemes")
    lb, mp = spec_cells(cells, "rec", "LB"), spec_cells(cells, "rec", "MP")
    b1 = spec_cells(cells, "rec", "LB", "LB_b1@E")
    b2 = spec_cells(cells, "rec", "LB", "LB_b2@E")
    for c in n9:
        dates = lb[c]["cfg"][1:].split("-")
        two = len(dates) == 2
        if len(dates) > 2:
            probs.append(f"{c}: {len(dates)} E dates (stab:app-corr has two)")
        if (c in b1) != two or (c in b2) != two:
            probs.append(f"{c}: single-date cells against {len(dates)} E dates")
            continue
        if two and (b1[c]["cfg"], b2[c]["cfg"]) != ("b" + dates[0], "b" + dates[1]):
            probs.append(f"{c}: single-date configurations differ from the E dates")
        for r in [mp[c]] + ([b1[c], b2[c]] if two else []):
            if r["p"] != lb[c]["p"]:
                probs.append(f"{c}: {r['specs']} at order {r['p']}, not the E order")
    model = {name: (mo, spec) for name, mo, spec in CORR_COLS}
    for r in rd_rows:
        if r["block"] != "hl_text" or not r["currency"]:
            continue
        if r["reading"].startswith("corr_"):
            mo, spec = model[r["reading"][len("corr_") :]]
            h = hl_cell("rec", mo, spec, cells, r["currency"])
            if r["value"] != fmt(float(h["alpha_mu"]) - float(h["alpha_ols"])):
                probs.append(f"hl_text {r['reading']} {r['currency']}: not its cell's")
        elif r["reading"].endswith("_hl_hi_below_first_gap"):
            mo = r["reading"].split("_")[0]
            h = hl_cell(r["scheme"], mo, f"{mo}@E", cells, r["currency"])
            gaps = [int(k) for k in h["gaps"].split("|")]
            node = min(gaps) - 1
            if not int(h["k_lo"]) <= node < int(h["k_hi"]) or node in gaps:
                probs.append(f"hl_text {r['reading']} {r['currency']}: node {node}")
    return probs


def check_designs(src, n9, rd_rows):
    probs = []
    sj = {r["currency"]: r for r in src.rows["sj_null.csv"]}
    sjs = {(r["set"], r["currency"]): r for r in src.rows["sj_sets.csv"]}
    e_sets = {(r["set"], r["currency"]): r for r in src.rows["ppp_sets.csv"]}
    if sorted(sj) != sorted(n9) or sorted(sjs) != sorted(
        (s, c) for s in ("E", "SJ") for c in n9
    ):
        probs.append("sj_null.csv / sj_sets.csv: not the currencies of N9")
        return probs
    cols_sj = (set(src.rows["sj_null.csv"][0]) & set(src.rows["sj_sets.csv"][0])) - {
        "set"
    }
    cols_e = set(src.rows["ppp_sets.csv"][0]) - {"set"}
    for c in n9:
        for f in sorted(cols_sj):
            if sjs[("SJ", c)][f] != sj[c][f]:
                probs.append(f"sj_sets S-JST {c} {f} differs from sj_null")
        for f in sorted(cols_e):
            if sjs[("E", c)][f] != e_sets[("E", c)][f]:
                probs.append(f"sj_sets E {c} {f} differs from ppp_sets")
        r = sj[c]
        if r["mzt_I"] and (r["reject_I"] == "1") != (float(r["mzt_I"]) < float(CV_ASY)):
            probs.append(f"sj_null {c}: reject_I is not its rule")
        if r["status"] != "ok":
            if r["reject_II"] != "":
                probs.append(f"sj_null {c}: a verdict without status ok")
            continue
        for v in (
            float(r["mzt_I"]) < float(r["cv_II"]),
            float(r["p_II"]) < float(LEVEL),
        ):
            if r["reject_II"] not in ("0", "1") or (r["reject_II"] == "1") != v:
                probs.append(f"sj_null {c}: reject_II is not its rule")
    R = src.rows["sj_readings.csv"]

    def mine(reading, u, **kw):
        hit = [
            x
            for x in rd_rows
            if x["block"] == "designs"
            and x["set"] == "S-JST"
            and x["reading"] == reading
            and x["universe"] == u
            and all(x[a] == b for a, b in kw.items())
        ]
        return hit[0] if len(hit) == 1 else None

    for u in ("N8", "N9"):
        for r in [x for x in R if x["universe"] == u]:
            rd = r["reading"]
            if rd == "K_obs":
                got = mine("K_obs_II", u)
                ok = got is not None and got["value"] == r["value"]
            elif rd == "rejecting_II":
                got = mine("K_obs_II", u)
                ok = got is not None and got["currencies"] == r["value"]
            elif rd == "rejecting_I":
                got = mine("rejecting_I", u)
                ok = got is not None and got["currencies"] == r["value"]
            elif rd in ("p_II_holm", "p_II_bonferroni"):
                got = mine(rd, u, currency=r["arm"])
                ok = got is not None and got["value"] == r["value"]
            else:
                continue
            if not ok:
                probs.append(f"designs S-JST {u} {rd} {r['arm']}: != sj_readings.csv")
    return probs


def check_paper(src, n9, prov):
    """The supplement tables' marks: a star follows its p and shows in the printed p
    against 0.05; the years of stab:app-sets print iff the design is E or its years
    differ from E's."""
    probs = []
    for i, p in enumerate(prov):
        if not p["rule"].startswith("star:"):
            continue
        q = prov[i - 1]
        where = f"{p['table_file']} row {p['body_row']}"
        if (q["table_file"], q["body_row"], q["class"]) != (
            p["table_file"],
            p["body_row"],
            "p3",
        ):
            probs.append(f"{where}: a star without its p")
            continue
        flag = "1" if p["printed"] else "0"
        if not verdict_shown(flag, q["printed"], str(LEVEL)):
            probs.append(f"{where}: a star not shown")
    toks = [p for p in prov if p["table_file"] == "stab_app_sets.tex"]
    if not toks:  # the table is not rendered (a partial render in a test)
        return probs
    got = sorted(
        (p["reference"], p["reference_key"])
        for p in toks
        if p["reference_column"] == "tau"
    )
    want = []
    for c in n9:
        e = design_row(src, c, "E")[2]
        for st in PAPER_SETS:
            g, k, r = design_row(src, c, st)
            if st == "E" or (r["tau"] and r["tau"] != e["tau"]):
                want.append((g, key_str(k)))
    if got != sorted(want):
        probs.append("stab_app_sets.tex: the years printed are not the rule's")
    return probs


def check_admission(src):
    probs = []
    cv6 = {r["currency"]: r for r in src.rows["c6_critical_values.csv"]}
    gs = {r["currency"]: r for r in src.rows["gate_sizes.csv"]}
    for r in src.rows["candidates.csv"]:
        c = r["currency"]
        a, g = cv6.get(c), gs.get(c)
        if a is None or g is None:
            probs.append(f"{c}: no C6 critical value or admission-test size")
            continue
        same = (
            a["cv_5"] == r["c6_cv_5"]
            and a["taus"] == g["taus"] == r["events_E"]
            and a["m"] == g["m"] == r["c6_df"]
            and len(r["events_E"].split(";")) == int(r["c6_df"])
        )
        if not same:
            probs.append(f"{c}: C6 cv, dates or m differ across the admission files")
        if int(a["T"]) + 1 != T_LEVELS:
            probs.append(f"{c}: the C6 simulations ran at T = {a['T']}")
        rej = r["C6_5"] == "0"
        if (
            r["C6_5"] not in ("0", "1")
            or rej != (float(r["c6_wald"]) > float(r["c6_cv_5"]))
            or rej != (float(r["c6_p"]) < float(LEVEL))
        ):
            probs.append(f"{c}: C6_5 is not its rule")
    kobs = [
        r["value"]
        for r in src.rows["readings.csv"]
        if r["universe"] == "N8" and r["reading"] == "K_obs"
    ]
    for r in src.rows["ppp_count_law.csv"]:
        if r["universe"] != "N8" or r["kappa"] not in ("0", "1"):
            continue
        if [r["K_obs"]] != kobs or r["K_obs"] != "1" or r["n_nan"] != "0":
            probs.append(f"ppp_count_law N8 {r['h']} {r['kappa']}: K_obs or NaN")
    return probs


# ================================================================== driver
def csv_text(cols, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for r in rows:
        w.writerow([fmt(r.get(c, "")) for c in cols])
    return buf.getvalue()


def build(src):
    """Everything in memory: {file name: text}, and the list of problems."""
    n8, n9 = universes(src)
    rd_rows = readings(src, n8, n9)
    rd_text = csv_text(READ_COLS, rd_rows)
    src.rows["readings_tables.csv"] = list(csv.DictReader(io.StringIO(rd_text)))
    # the class digits come from the table cells (the notes print full precision) and
    # from the members that no table prints in their class
    pre = Renderer(src)
    pre.tables = tables(src, n8, n9, pre)
    pre.members = design_power_members(src, row_order(n8, n9))
    pre.collect()
    rend = Renderer(src)
    rend.digits = pre.digits
    rend.tables = tables(src, n8, n9, rend)
    texts, prov = rend.render()
    probs = check_tokens(src, texts, prov) + check_consistency(
        src, n8, n9, rd_rows
    )
    probs += check_paper(src, n9, prov)
    files = dict(texts)
    files["readings_tables.csv"] = rd_text
    files["tables_provenance.csv"] = csv_text(PROV_COLS, prov)
    man = dict(
        producer="mlb/ppp_tables.py",
        inputs_sha256_lf={n: sha_lf(p) for n, p in sorted(src.paths.items())},
        universes=dict(N8=n8, N9=n9),
        class_digits=rend.digits,
        provenance_keys=(
            "data_row=i is the 0-based index of a file's data rows"
        ),
        outputs_sha256={
            n: hashlib.sha256(t.encode("utf-8")).hexdigest()
            for n, t in sorted(files.items())
        },
    )
    files["tables.manifest.json"] = json.dumps(man, indent=1, ensure_ascii=False) + "\n"
    for n, t in files.items():
        if "\r" in t:
            probs.append(f"{n}: a carriage return")
        if any(ln != ln.rstrip() for ln in t.split("\n")):
            probs.append(f"{n}: trailing space")
    return files, probs


def default_paths():
    from mlb import paths

    return dict(
        main=str(paths.resolve("application")),
        crisis_design=str(paths.resolve("crisis_design")),
        c6_cv=str(paths.resolve("c6_critical_values")),
        events=str(paths.resolve("regime_events")),
        candidates=str(paths.resolve("candidates")),
        gate_sizes=str(paths.resolve("gate_sizes")),
        funnel=str(paths.resolve("sweep_funnel")),
        reference_tables=str(paths.resolve("tables")),
        out=str(paths.out_dir(SECTION) / "tables"),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="render in memory and compare readings_tables.csv with "
        "reference/application/tables/",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output folder (default output/bootstrap_empirical/tables/)",
    )
    a = ap.parse_args(argv)
    given = default_paths()
    if a.out_dir is not None:
        given["out"] = a.out_dir
    src = Sources(
        given["main"],
        given["events"],
        given["candidates"],
        given["gate_sizes"],
        given["funnel"],
        given["crisis_design"],
        given["c6_cv"],
    )
    files, probs = build(src)
    if probs:
        print("\n".join(probs[:40]), file=sys.stderr)
        print(
            f"ppp_tables: {len(probs)} check failures; nothing written", file=sys.stderr
        )
        return 1
    if a.check:
        bad = []
        for n in REFERENCE_OUTPUTS:
            p = os.path.join(given["reference_tables"], n)
            if not os.path.exists(p) or open(p, "rb").read() != files[n].encode(
                "utf-8"
            ):
                bad.append(n)
        print(
            json.dumps(
                dict(
                    check="tables",
                    identical=not bad,
                    compared=list(REFERENCE_OUTPUTS),
                    differ=bad,
                ),
                indent=1,
            )
        )
        return 1 if bad else 0
    os.makedirs(given["out"], exist_ok=True)
    for n in OUTPUTS:
        with open(
            os.path.join(given["out"], n), "w", encoding="utf-8", newline="\n"
        ) as fh:
            fh.write(files[n])
    print(json.dumps(dict(written=given["out"], files=list(OUTPUTS)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
