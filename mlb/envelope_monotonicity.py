"""The finite-sample envelope decreases strictly from node to node on the grid of step
0.5 over K in every configuration computed in Section 4 of the article (article
section 2: "The finite-sample envelope decreases strictly from node to node on the
grid of step 0.5 over K in every configuration computed in Section 4, and every
half-power point computed there lies between the two adjacent nodes that bracket one
half; this is verified numerically on that grid, not proved").

Three sets of configurations (T, lambda) are scanned.

Table 1. The 404 configurations of the grid T in {45, 50, 60, 80, 100, 150, 200, 300},
m = 0, ..., 5: the 404 rows of
reference/calibration/cbar_surface_exact/cbar_surface_exact.csv, with their
break fractions and their exact half-power points cbar_1/2.

The lattices. The other configurations of Section 4.1 at which the envelope is
computed, those not in Table 1:
  * the trimming lattices of the four cells (m, T) = (1, 60), (2, 60), (3, 60),
    (2, 100) (Section 4.1, "varying the trimming"; supplement Section S4,
    "Trimming"): every distinct (T, lambda) of
    reference/calibration/surface_summaries/surface_trimming_configs.csv,
    with the half-power point of that file. For each trimming eps in
    {0.10, 0.15, 0.20} the file holds two lattices (the two the supplement
    describes): "theory", the lattice of Lambda_eps (minimum spacing eps), and
    "generator", the endpoint trimming varied with the spacing rule held fixed. Both
    are scanned; the column computed_for names the lattices a configuration belongs
    to;
  * Proposition 1 at T = 52, 100, 200, 400 (Section 4.1): the one- and two-break
    lattices on which the scaled displacements are computed (part iii, with the
    half-power points of
    reference/calibration/propflat_exact/propflat_exact_configs.csv), and
    the designs at which the power gap at cbar = -7 is computed (part i: the
    two-break lattice and the design without breaks). Section 4 computes no
    half-power point for the design without breaks at T = 52 and 400; there the
    grid alone is scanned.
A configuration that these files share with Table 1 is scanned once, in Table 1;
the half-power points the files give for it are checked against its Table 1
bracket.

The rule comparison. Section 4.2 ("Fixing cbar = -7 against following the
half-power point") compares 78 configurations: the equispaced placement
lambda_j = j/(m+1) of every cell of Table 1 and the placement whose half-power point
lies farthest from -7. They are the 78 rows with X = cbar_half(config) of
reference/calibration/rule_comparison/rule_configs.csv. 51 of them are
configurations of Table 1 (the placement without breaks, the equispaced placement
with one break, and the 35 farthest placements); their half-power points in that file
must equal those of Table 1. The other 27, the equispaced placements with m = 2, ...,
5 (8 + 8 + 6 + 5 cells), are not on the lattice of Table 1; they are scanned with the
half-power points that file computes for them (break dates checked against the
file's).

Designs. The envelope depends on lambda only through the break dates
tau_j = floor(lambda_j T): two configurations with the same T and the same break
dates have the same regressors and the same envelope, and are one design. Three
trimming configurations are, in this sense, designs of Table 1 (at T = 60 the
fractions 0.367 and 0.383 both give tau = 22, and 0.633 and 0.617 both give 37).
The totals over Section 4 count designs.

For each configuration the power of the 5% envelope test against c = cbar,

    Pi^T_lambda(cbar) = P_{c = cbar}[L_T(cbar) < b^T_lambda(cbar)],

is computed exactly (Imhof's formula; mlb/envelope_exact.power_at, the function
from which the half-power points of those files were computed) at the 35 nodes
cbar = -20, -19.5, ..., -3 of K = [-20, -3], the grid on which the strict decrease of
the limit envelope is verified (mlb/limit_envelope.py), and at
cbar_1/2. A configuration passes if the 34 successive differences
Pi(cbar_{k+1}) - Pi(cbar_k) are all negative. Then Pi^T_lambda - 1/2 changes sign
exactly once on the grid; the script reports the two nodes that bracket the change,
checks that the half-power point lies between them, and reports Pi^T_lambda at that
point (one half up to the root search's tolerance). Strict decrease between the
nodes is not examined: the statement is verified on the grid, not proved.
Cross-checks: Pi at -7 on the grid equals the power at -7 of the Proposition 1 file
(part power_m7, T = 52, ..., 400; the same function and call) to 1e-9 (1 + |b|); the
half-power points of the trimming and Proposition 1 files for configurations of
Table 1 lie in the Table 1 brackets; those of the rule file for configurations of
Table 1 equal Table 1's; and configurations of one design have the same Pi on the
grid to 1e-9 (1 + |b|).

Outputs (output/critical_values/envelope_monotonicity/, or --out-dir):
  envelope_monotonicity_grid.csv       Table 1: Pi^T_lambda at the 35 nodes and at
                                       cbar_1/2, one row per configuration, full
                                       precision;
  envelope_monotonicity.csv            Table 1: one row per configuration: the
                                       verdict, the largest (least negative)
                                       successive difference and its node, Pi at
                                       -20 and at -3, the bracketing nodes, Pi at
                                       cbar_1/2;
  envelope_monotonicity.json           Table 1: the counts and the extremes over
                                       the 404;
  envelope_monotonicity_lattices_grid.csv, envelope_monotonicity_lattices.csv
                                       the same for the lattices (Pi at cbar_1/2
                                       empty where no half-power point is
                                       computed), the verdicts with the source of
                                       cbar_1/2 and the lattices of each
                                       configuration;
  envelope_monotonicity_lattices.json  the counts and the extremes over the
                                       lattices and their two cross-checks;
  envelope_monotonicity_rule_grid.csv, envelope_monotonicity_rule.csv,
  envelope_monotonicity_rule.json      the same for the 27 configurations of the
                                       rule comparison that are not in Table 1
                                       (the verdicts with the break dates), and
                                       the check of the 51 that are;
  envelope_monotonicity_section_4.json the totals over every design computed in
                                       Section 4 and the configurations that are one
                                       design;
  envelope_monotonicity.manifest.json  the parameters, the inputs' sha256 and the
                                       outputs' sha256; nothing time- or
                                       machine-dependent.
All but the three grid files are functions of the grid files and of the inputs
alone.

    python mlb/envelope_monotonicity.py [--out-dir D] [--jobs N]
        [--check]                       compute everything (637 configurations;
                                        about 61 CPU-minutes on an idle machine,
                                        about 15 minutes at --jobs 4; two to three
                                        times longer on a loaded one)
    ... --part NAME[,NAME...] [--out-dir D] [--jobs N]
                                        compute only these parts, into
                                        D/envelope_monotonicity_grid_T<T>.csv
                                        (parts T45, T50, T60, T80, T100, T150,
                                        T200, T300 of Table 1),
                                        D/envelope_monotonicity_lattices_grid_T<T>.csv
                                        (parts lattices_T52, lattices_T60,
                                        lattices_T100, lattices_T400) and
                                        D/envelope_monotonicity_rule_grid_part.csv
                                        (part rule: the 27 configurations of the
                                        rule comparison, T = 45, ..., 300)
    ... --T 300,200 [...]               the Table 1 and lattice parts of these T
                                        (the part rule is selected by its name)
    ... --aggregate [--out-dir D] [--check]
                                        the outputs from the thirteen part files
                                        in D
    ... --aggregate --grids-dir G [--out-dir D] [--check]
                                        the outputs from the three grid files
                                        envelope_monotonicity_grid.csv,
                                        envelope_monotonicity_lattices_grid.csv
                                        and envelope_monotonicity_rule_grid.csv
                                        of an earlier run, in G (seconds)
    ... --show-reference                the verdict and the totals of each of the
                                        four summaries in
                                        reference/calibration/envelope_monotonicity/
                                        (reads nothing else; exit 0 only if every
                                        one reports strict decrease)

Table 1 is about 39 CPU-minutes, the lattices about 20 and the part rule about 3;
no part exceeds about 10 CPU-minutes (T300, lattices_T60). Each run prints its
estimate first.

--check compares the four summaries (envelope_monotonicity.json,
envelope_monotonicity_lattices.json, envelope_monotonicity_rule.json,
envelope_monotonicity_section_4.json) with their reference copies in
reference/calibration/envelope_monotonicity/: byte for byte (LF), failing that every
number to |a - b| <= 1e-9 (1 + |b|), because the eigenvalues and the integrals can
move the last digits between platforms; every field that is not a number (the
verdicts among them) must be equal. The other outputs have no reference copy and are
only written. The exit code is 1 if a configuration fails, a cross-check disagrees or
on any difference, and 2 if an input or a part file is missing or malformed, the
options do not go together, or the output folder cannot be created or written.
--jobs and --out-dir do not change the outputs.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_v, "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_pp = os.environ.get("PYTHONPATH", "")
if _ROOT not in _pp.split(os.pathsep):
    # the worker processes of --jobs import mlb by name
    os.environ["PYTHONPATH"] = _ROOT + (os.pathsep + _pp if _pp else "")

from mlb import envelope_exact as EX  # noqa: E402
from mlb import paths  # noqa: E402

SECTION, OBJ = "critical_values", "envelope_monotonicity"
T_GRID = (45, 50, 60, 80, 100, 150, 200, 300)
LATTICE_T = (52, 60, 100, 400)
PROP1_T = (52, 100, 200, 400)
GRID = tuple(-20.0 + 0.5 * k for k in range(35))
I_MINUS7 = GRID.index(-7.0)
TOL = 1e-9
# seconds per exact evaluation of Pi on one core of an idle machine: a + b T
SECONDS_PER_EVALUATION = (0.07, 0.0007)
SURFACE_REL = "reference/calibration/cbar_surface_exact/cbar_surface_exact.csv"
SURFACE = paths.reference_dir("critical_values") / "cbar_surface_exact" / (
    "cbar_surface_exact.csv"
)
TRIMMING_REL = (
    "reference/calibration/surface_summaries/surface_trimming_configs.csv"
)
TRIMMING = paths.reference_dir("critical_values") / "surface_summaries" / (
    "surface_trimming_configs.csv"
)
PROP1_REL = "reference/calibration/propflat_exact/propflat_exact_configs.csv"
PROP1 = paths.reference_dir("critical_values") / "propflat_exact" / (
    "propflat_exact_configs.csv"
)
RULE_REL = "reference/calibration/rule_comparison/rule_configs.csv"
RULE = paths.reference_dir("critical_values") / "rule_comparison" / "rule_configs.csv"
RULE_X = "cbar_half(config)"
INPUTS = (
    (SURFACE_REL, SURFACE),
    (TRIMMING_REL, TRIMMING),
    (PROP1_REL, PROP1),
    (RULE_REL, RULE),
)
LAMBDAS = ("lambda1", "lambda2", "lambda3", "lambda4", "lambda5")
GRID_FILE = f"{OBJ}_grid.csv"
CSV_FILE, JSON_FILE, MANIFEST = f"{OBJ}.csv", f"{OBJ}.json", f"{OBJ}.manifest.json"
LAT_GRID_FILE = f"{OBJ}_lattices_grid.csv"
LAT_CSV_FILE, LAT_JSON_FILE = f"{OBJ}_lattices.csv", f"{OBJ}_lattices.json"
RULE_GRID_FILE = f"{OBJ}_rule_grid.csv"
RULE_CSV_FILE, RULE_JSON_FILE = f"{OBJ}_rule.csv", f"{OBJ}_rule.json"
SEC4_JSON_FILE = f"{OBJ}_section_4.json"
# the outputs whose reference copy is in reference/calibration/envelope_monotonicity/
REFERENCE_OUTPUTS = (JSON_FILE, LAT_JSON_FILE, RULE_JSON_FILE, SEC4_JSON_FILE)
GRID_FILES = dict(table1=GRID_FILE, lattices=LAT_GRID_FILE, rule=RULE_GRID_FILE)
GRID_HEADER = (
    "config_key",
    "T",
    "m",
    *(f"pi_at_{c!r}" for c in GRID),
    "pi_at_cbar_half",
)
HEADER = (
    "config_key",
    "T",
    "m",
    *LAMBDAS,
    "cbar_half_exact",
    "strictly_decreasing",
    "largest_difference",
    "largest_difference_from",
    "pi_at_minus20",
    "pi_at_minus3",
    "bracket_lo",
    "bracket_hi",
    "cbar_half_in_bracket",
    "pi_at_cbar_half",
)
LAT_HEADER = HEADER + ("cbar_half_source", "computed_for")
RULE_HEADER = HEADER + ("break_dates", "cbar_half_source")
# the sets of configurations; a part is (set, T), and the part rule is the whole set
PARTS = {
    **{f"T{T}": ("table1", T) for T in T_GRID},
    **{f"lattices_T{T}": ("lattices", T) for T in LATTICE_T},
    "rule": ("rule", None),
}


def part_file(name):
    kind, T = PARTS[name]
    if kind == "table1":
        return f"{OBJ}_grid_T{T}.csv"
    if kind == "lattices":
        return f"{OBJ}_lattices_grid_T{T}.csv"
    return f"{OBJ}_rule_grid_part.csv"


class InputError(Exception):
    """A missing or malformed input or part file."""


class OutputError(Exception):
    """The output folder cannot be created or written."""


def _read(path):
    if not os.path.isfile(path):
        raise InputError(f"missing input {path}")
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def configurations(path=SURFACE):
    """The 404 rows of the exact surface, in the file's order."""
    rows = _read(path)
    return [
        dict(
            key=r["config_key"],
            T=int(r["T"]),
            m=int(r["m"]),
            lambda_text=[r[k] for k in LAMBDAS],
            lambdas=tuple(float(r[k]) for k in LAMBDAS if r[k]),
            cbar_half_text=r["cbar_half_exact"],
            cbar_half=float(r["cbar_half_exact"]),
        )
        for r in rows
    ]


def _trimming_rows():
    """(T, lambda text, cbar_1/2 text, config_key, lattice tag) of the trimming file."""
    return [
        (
            int(r["T"]),
            r["lambdas"].split(),
            r["cbar_half_exact"],
            r["config_key"],
            f"trimming:{r['convention']}:{r['eps']}",
        )
        for r in _read(TRIMMING)
    ]


def _prop1_rows():
    """(T, lambda text, cbar_1/2 text or '', None, tag, value) of the Proposition 1
    file at the T of Section 4."""
    out = []
    for r in _read(PROP1):
        T = int(r["T"])
        if T not in PROP1_T:
            continue
        lam = [r[k] for k in ("lambda1", "lambda2") if r[k]]
        if r["part"] == "cbar_half":
            out.append((T, lam, r["value"], None, "proposition_1_iii", r["value"]))
        elif r["part"] == "power_m7":
            out.append((T, lam, "", None, "proposition_1_i", r["value"]))
        else:
            raise InputError(f"{PROP1}: unexpected part {r['part']!r}")
    return out


def _key(T, lam_text):
    return f"T{T}_m{len(lam_text)}_lambdas_" + "_".join(lam_text)


def lattice_configurations(table1=None):
    """The configurations of the trimming and Proposition 1 lattices that are not in
    Table 1, sorted by (T, m, lambda)."""
    t1 = {(c["T"], c["lambdas"]) for c in (table1 or configurations())}
    found = {}
    rows = [(r, TRIMMING.name) for r in _trimming_rows()]
    rows += [(r[:5], PROP1.name) for r in _prop1_rows()]
    for (T, lam_text, cbar_text, given_key, tag), source in rows:
        lam = tuple(float(x) for x in lam_text)
        key = _key(T, lam_text)
        if given_key is not None and given_key != key:
            raise InputError(f"{source}: config_key {given_key} is not {key}")
        if (T, lam) in t1:
            continue
        c = found.setdefault(
            (T, lam),
            dict(
                key=key,
                T=T,
                m=len(lam),
                lambda_text=list(lam_text) + [""] * (len(LAMBDAS) - len(lam)),
                lambdas=lam,
                cbar_half_text="",
                cbar_half=None,
                cbar_half_source="",
                tags=[],
            ),
        )
        if c["key"] != key:
            raise InputError(f"{source}: {key} and {c['key']} are one configuration")
        if cbar_text:
            if c["cbar_half_text"] and c["cbar_half_text"] != cbar_text:
                raise InputError(f"{key}: two different half-power points")
            c.update(
                cbar_half_text=cbar_text,
                cbar_half=float(cbar_text),
                cbar_half_source=source,
            )
        if tag not in c["tags"]:
            c["tags"].append(tag)
    out = sorted(found.values(), key=lambda c: (c["T"], c["m"], c["lambdas"]))
    for c in out:
        c["computed_for"] = ";".join(c.pop("tags"))
    return out


def shared_half_power_points(table1):
    """(config_key, cbar_1/2 text, file) for each half-power point that the trimming
    and Proposition 1 files give for a configuration of Table 1."""
    t1 = {(c["T"], c["lambdas"]): c["key"] for c in table1}
    out = set()
    rows = [(r, TRIMMING.name) for r in _trimming_rows()]
    rows += [(r[:5], PROP1.name) for r in _prop1_rows()]
    for (T, lam_text, cbar_text, _, _), source in rows:
        k = t1.get((T, tuple(float(x) for x in lam_text)))
        if k is not None and cbar_text:
            out.add((k, cbar_text, source))
    return sorted(out)


def _break_dates(T, lambdas):
    return " ".join(str(b) for b in EX.break_positions(T, lambdas))


def _rule_key(T, lambdas):
    """The rule file's config_key: the fractions rounded to three decimals."""
    return f"T{T}_m{len(lambdas)}_lambdas_" + "_".join(
        str(round(x, 3)) for x in lambdas
    )


def rule_comparison(table1):
    """The 78 configurations of the rule comparison (the rows with
    X = cbar_half(config) of the rule file), as two lists: [(config_key, cbar_1/2
    text)] of those in Table 1, and the configurations of the others (the
    equispaced placements lambda_j = j/(m+1)), sorted by (T, m). The break dates of
    every row must be the file's."""
    t1 = {c["key"]: c for c in table1}
    t1_lam = {(c["T"], c["lambdas"]) for c in table1}
    shared, not_in_table_1 = [], []
    for r in _read(RULE):
        T, m, key = int(r["T"]), int(r["m"]), r["config_key"]
        if r["X"] != RULE_X:
            continue
        if key in t1:
            lam = t1[key]["lambdas"]
            shared.append((key, r["X_value"]))
        else:
            lam = tuple(j / (m + 1) for j in range(1, m + 1))
            if r["kind"] != "equispaced" or _rule_key(T, lam) != key:
                raise InputError(
                    f"{RULE.name}: {key} is neither in Table 1 nor the equispaced "
                    f"placement of ({m}, {T})"
                )
            if (T, lam) in t1_lam:
                raise InputError(f"{RULE.name}: {key} is in Table 1 under another key")
            not_in_table_1.append(
                dict(
                    key=key,
                    T=T,
                    m=m,
                    lambda_text=[repr(x) for x in lam] + [""] * (len(LAMBDAS) - m),
                    lambdas=lam,
                    cbar_half_text=r["X_value"],
                    cbar_half=float(r["X_value"]),
                    cbar_half_source=RULE.name,
                    break_dates=_break_dates(T, lam),
                )
            )
        if _break_dates(T, lam) != r["break_pos"]:
            raise InputError(
                f"{RULE.name}: {key}: break dates {r['break_pos']!r}, here "
                f"{_break_dates(T, lam)!r}"
            )
    keys = [k for k, _ in shared] + [c["key"] for c in not_in_table_1]
    if len(set(keys)) != len(keys):
        raise InputError(f"{RULE.name}: a configuration appears twice")
    return shared, sorted(not_in_table_1, key=lambda c: (c["T"], c["m"]))


def rule_configurations(table1=None):
    """The configurations of the rule comparison that are not in Table 1."""
    return rule_comparison(table1 or configurations())[1]


def all_sets():
    """{'table1': ..., 'lattices': ..., 'rule': ...}: the three sets scanned."""
    configs = configurations()
    return dict(
        table1=configs,
        lattices=lattice_configurations(configs),
        rule=rule_configurations(configs),
    )


def power_at_minus7():
    """{(T, lambda): Pi at -7} of the Proposition 1 file at the T of Section 4."""
    return {
        (T, tuple(float(x) for x in lam)): float(value)
        for T, lam, _, _, tag, value in _prop1_rows()
        if tag == "proposition_1_i"
    }


def part_configs(name, sets):
    kind, T = PARTS[name]
    return [c for c in sets[kind] if T is None or c["T"] == T]


# ------------------------------------------------------------------- computation
def envelope(T, lambdas, points):
    """Pi^T_lambda at each point, exactly."""
    Z = EX.design_Z(T, lambdas)
    return [float(EX.power_at(c, T, Z)[0]) for c in points]


def _one(T, lambdas, cbar_half):
    if cbar_half is None:
        return envelope(T, lambdas, GRID) + [None]
    return envelope(T, lambdas, GRID + (cbar_half,))


def compute(configs, jobs):
    """{config_key: [Pi at the 35 nodes, Pi at cbar_1/2]}; costliest first."""
    todo = sorted(configs, key=lambda c: (-c["T"], -c["m"], c["key"]))
    if jobs > 1:
        from joblib import Parallel, delayed

        vals = Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
            delayed(_one)(c["T"], c["lambdas"], c["cbar_half"]) for c in todo
        )
    else:
        vals = [_one(c["T"], c["lambdas"], c["cbar_half"]) for c in todo]
    return {c["key"]: v for c, v in zip(todo, vals)}


def cpu_minutes(configs):
    """The idle-machine estimate of compute(configs) in CPU-minutes."""
    a, b = SECONDS_PER_EVALUATION
    return sum((len(GRID) + 1) * (a + b * c["T"]) for c in configs) / 60.0


# ------------------------------------------------------------------- the grid files
def grid_bytes(configs, vals):
    rows = [
        [c["key"], str(c["T"]), str(c["m"])]
        + ["" if x is None else repr(x) for x in vals[c["key"]]]
        for c in configs
    ]
    return _csv_bytes(GRID_HEADER, rows)


def read_grid(paths_):
    """{config_key: values} from one or more grid files (the header is checked; an
    empty Pi at cbar_1/2 is None)."""
    vals = {}
    for p in paths_:
        if not os.path.isfile(p):
            raise InputError(f"missing grid file {p}")
        with open(p, encoding="utf-8", newline="") as fh:
            rd = csv.reader(fh)
            if tuple(next(rd, ())) != GRID_HEADER:
                raise InputError(f"{p}: unexpected header")
            for n, r in enumerate(rd, start=2):
                if len(r) != len(GRID_HEADER) or r[0] in vals:
                    raise InputError(f"{p}: line {n}: malformed or repeated row")
                try:
                    v = [float(x) for x in r[3 : 3 + len(GRID)]]
                    v.append(float(r[-1]) if r[-1] else None)
                except ValueError:
                    raise InputError(f"{p}: line {n}: not a number") from None
                vals[r[0]] = v
    return vals


# ------------------------------------------------------------------- the verdicts
def verdict(cfg, values):
    """The verdict and the reported numbers of one configuration (inside is None
    where the configuration has no half-power point)."""
    pi, at_root = values[: len(GRID)], values[len(GRID)]
    diffs = [b - a for a, b in zip(pi, pi[1:])]
    k = max(range(len(diffs)), key=lambda i: diffs[i])
    above = [i for i, p in enumerate(pi) if p > 0.5]
    lo = hi = None
    if above and above[-1] + 1 < len(GRID):
        lo, hi = GRID[above[-1]], GRID[above[-1] + 1]
    if cfg["cbar_half"] is None:
        inside = None
    else:
        inside = lo is not None and lo <= cfg["cbar_half"] <= hi
    return dict(
        decreasing=all(d < 0 for d in diffs),
        largest_difference=diffs[k],
        largest_difference_from=GRID[k],
        pi_lo=pi[0],
        pi_hi=pi[-1],
        bracket_lo=lo,
        bracket_hi=hi,
        inside=inside,
        pi_at_root=at_root,
    )


def _r(x):
    return "" if x is None else repr(float(x))


def _verdict_row(c, v):
    return [
        c["key"],
        str(c["T"]),
        str(c["m"]),
        *c["lambda_text"],
        c["cbar_half_text"],
        str(v["decreasing"]),
        _r(v["largest_difference"]),
        _r(v["largest_difference_from"]),
        _r(v["pi_lo"]),
        _r(v["pi_hi"]),
        _r(v["bracket_lo"]),
        _r(v["bracket_hi"]),
        "" if v["inside"] is None else str(v["inside"]),
        _r(v["pi_at_root"]),
    ]


def verdict_bytes(configs, vals):
    rows = [_verdict_row(c, verdict(c, vals[c["key"]])) for c in configs]
    return _csv_bytes(HEADER, rows)


def lattice_verdict_bytes(lattices, vals):
    rows = [
        _verdict_row(c, verdict(c, vals[c["key"]]))
        + [c["cbar_half_source"], c["computed_for"]]
        for c in lattices
    ]
    return _csv_bytes(LAT_HEADER, rows)


def rule_verdict_bytes(rule, vals):
    rows = [
        _verdict_row(c, verdict(c, vals[c["key"]]))
        + [c["break_dates"], c["cbar_half_source"]]
        for c in rule
    ]
    return _csv_bytes(RULE_HEADER, rows)


def summary(configs, vals):
    v = {c["key"]: verdict(c, vals[c["key"]]) for c in configs}
    keys = [c["key"] for c in configs]
    worst = max(keys, key=lambda k: v[k]["largest_difference"])
    far = max(keys, key=lambda k: abs(v[k]["pi_at_root"] - 0.5))
    by_T = {}
    for c in configs:
        by_T[str(c["T"])] = by_T.get(str(c["T"]), 0) + 1
    n_dec = sum(v[k]["decreasing"] for k in keys)
    return dict(
        statement="Pi^T_lambda decreases strictly from node to node on the grid of "
        "step 0.5 over K = [-20, -3] in "
        "every configuration of the grid of Table 1, so the half-power point is the "
        "unique root of Pi^T_lambda = 1/2 on the grid",
        grid=dict(lo=GRID[0], hi=GRID[-1], step=0.5, n_nodes=len(GRID)),
        n_configurations=len(keys),
        configurations_by_T=by_T,
        n_strictly_decreasing=n_dec,
        all_strictly_decreasing=n_dec == len(keys),
        n_cbar_half_in_bracket=sum(v[k]["inside"] for k in keys),
        largest_difference=dict(
            value=v[worst]["largest_difference"],
            from_node=v[worst]["largest_difference_from"],
            config_key=worst,
        ),
        largest_abs_pi_at_cbar_half_minus_half=dict(
            value=abs(v[far]["pi_at_root"] - 0.5), config_key=far
        ),
        smallest_pi_at_minus20=min(v[k]["pi_lo"] for k in keys),
        largest_pi_at_minus3=max(v[k]["pi_hi"] for k in keys),
    )


def _minus7_agreement(configs, vals, lattices, lat_vals):
    """Pi at -7 on the grid against the power at -7 of the Proposition 1 file."""
    ref = power_at_minus7()
    grid = {}
    for cs, vs in ((configs, vals), (lattices, lat_vals)):
        for c in cs:
            grid[(c["T"], c["lambdas"])] = vs[c["key"]][I_MINUS7]
    common = [k for k in ref if k in grid]
    diffs = [abs(grid[k] - ref[k]) for k in common]
    return dict(
        n=len(ref),
        n_not_scanned=len(ref) - len(common),
        max_abs_difference=max(diffs) if diffs else None,
        agree=len(common) == len(ref)
        and all(abs(grid[k] - ref[k]) <= TOL * (1.0 + abs(ref[k])) for k in common),
    )


def lattice_summary(lattices, lat_vals, configs, vals):
    v = {c["key"]: verdict(c, lat_vals[c["key"]]) for c in lattices}
    keys = [c["key"] for c in lattices]
    worst = max(keys, key=lambda k: v[k]["largest_difference"])
    rooted = [k for k in keys if v[k]["inside"] is not None]
    far = max(rooted, key=lambda k: abs(v[k]["pi_at_root"] - 0.5))
    by_T, by_lattice = {}, {}
    for c in lattices:
        by_T[str(c["T"])] = by_T.get(str(c["T"]), 0) + 1
        tags = c["computed_for"].split(";")
        for t in tags:
            by_lattice[t] = by_lattice.get(t, 0) + 1
    n_dec = sum(v[k]["decreasing"] for k in keys)
    n_in = sum(v[k]["inside"] is True for k in keys)
    t1 = {c["key"]: verdict(c, vals[c["key"]]) for c in configs}
    shared = shared_half_power_points(configs)
    n_shared_in = sum(
        t1[k]["bracket_lo"] is not None
        and t1[k]["bracket_lo"] <= float(x) <= t1[k]["bracket_hi"]
        for k, x, _ in shared
    )
    return dict(
        statement="Pi^T_lambda decreases strictly from node to node on the grid of "
        "step 0.5 over K = [-20, -3] in "
        "every configuration of the trimming and Proposition 1 lattices that is not "
        "in Table 1; where a half-power point is computed it is the unique root of "
        "Pi^T_lambda = 1/2 on the grid (the totals over Section 4 are in "
        f"{SEC4_JSON_FILE})",
        grid=dict(lo=GRID[0], hi=GRID[-1], step=0.5, n_nodes=len(GRID)),
        n_configurations=len(keys),
        configurations_by_T=by_T,
        configurations_by_lattice=dict(sorted(by_lattice.items())),
        n_strictly_decreasing=n_dec,
        all_strictly_decreasing=n_dec == len(keys),
        n_with_cbar_half=len(rooted),
        n_cbar_half_in_bracket=n_in,
        largest_difference=dict(
            value=v[worst]["largest_difference"],
            from_node=v[worst]["largest_difference_from"],
            config_key=worst,
        ),
        largest_abs_pi_at_cbar_half_minus_half=dict(
            value=abs(v[far]["pi_at_root"] - 0.5), config_key=far
        ),
        smallest_pi_at_minus20=min(v[k]["pi_lo"] for k in keys),
        largest_pi_at_minus3=max(v[k]["pi_hi"] for k in keys),
        shared_with_table_1=dict(
            n_configurations=len({k for k, _, _ in shared}),
            n_half_power_points=len(shared),
            n_in_table_1_bracket=n_shared_in,
        ),
        pi_at_minus7_against_proposition_1=_minus7_agreement(
            configs, vals, lattices, lat_vals
        ),
    )


def rule_summary(rule, rule_vals, configs):
    v = {c["key"]: verdict(c, rule_vals[c["key"]]) for c in rule}
    keys = [c["key"] for c in rule]
    worst = max(keys, key=lambda k: v[k]["largest_difference"])
    far = max(keys, key=lambda k: abs(v[k]["pi_at_root"] - 0.5))
    by_T, by_m = {}, {}
    for c in rule:
        by_T[str(c["T"])] = by_T.get(str(c["T"]), 0) + 1
        by_m[str(c["m"])] = by_m.get(str(c["m"]), 0) + 1
    n_dec = sum(v[k]["decreasing"] for k in keys)
    shared, _ = rule_comparison(configs)
    t1 = {c["key"]: c["cbar_half"] for c in configs}
    return dict(
        statement="Pi^T_lambda decreases strictly from node to node on the grid of "
        "step 0.5 over K = [-20, -3] in "
        "every configuration of the comparison of Section 4.2 that is not in Table 1 "
        "(the equispaced placements lambda_j = j/(m+1), m = 2, ..., 5), so the "
        "half-power point is the unique root of Pi^T_lambda = 1/2 on the grid",
        grid=dict(lo=GRID[0], hi=GRID[-1], step=0.5, n_nodes=len(GRID)),
        n_configurations=len(keys),
        configurations_by_T=by_T,
        configurations_by_m=dict(sorted(by_m.items())),
        n_strictly_decreasing=n_dec,
        all_strictly_decreasing=n_dec == len(keys),
        n_with_cbar_half=len(keys),
        n_cbar_half_in_bracket=sum(v[k]["inside"] is True for k in keys),
        largest_difference=dict(
            value=v[worst]["largest_difference"],
            from_node=v[worst]["largest_difference_from"],
            config_key=worst,
        ),
        largest_abs_pi_at_cbar_half_minus_half=dict(
            value=abs(v[far]["pi_at_root"] - 0.5), config_key=far
        ),
        smallest_pi_at_minus20=min(v[k]["pi_lo"] for k in keys),
        largest_pi_at_minus3=max(v[k]["pi_hi"] for k in keys),
        rule_comparison=dict(
            n_configurations=len(shared) + len(keys),
            n_in_table_1=len(shared),
            n_not_in_table_1=len(keys),
            n_cbar_half_equal_to_table_1=sum(float(x) == t1[k] for k, x in shared),
        ),
    )


def _design(c):
    """A design: T and the break dates (the envelope depends on nothing else)."""
    return (c["T"], tuple(EX.break_positions(c["T"], c["lambdas"])))


def section_4_summary(sets, vals):
    """The totals over the designs at which Section 4 computes the envelope: Table 1,
    the lattices described in the text and the rule comparison."""
    named = dict(
        table1="table_1",
        lattices="trimming_and_proposition_1_lattices",
        rule="rule_comparison_not_in_table_1",
    )
    text = []
    for kind in ("table1", "lattices", "rule"):
        for c in sets[kind]:
            entry = (named[kind], c, vals[kind][c["key"]])
            text.append(entry)
    first, same, pi_diff, cbar_diff = {}, [], [], []
    for group, c, v in text:
        d = _design(c)
        if d not in first:
            first[d] = (group, c, v)
            continue
        g0, c0, v0 = first[d]
        same.append(
            dict(
                config_key=c["key"],
                set=group,
                same_design_as=c0["key"],
                same_design_as_set=g0,
                T=c["T"],
                break_dates=" ".join(str(b) for b in d[1]),
            )
        )
        pi_diff += [
            (abs(a - b), abs(a - b) <= TOL * (1.0 + abs(b)))
            for a, b in zip(v[: len(GRID)], v0[: len(GRID)])
        ]
        if c["cbar_half"] is not None and c0["cbar_half"] is not None:
            cbar_diff.append(abs(c["cbar_half"] - c0["cbar_half"]))

    def count(entries):
        verdicts = [(_design(c), verdict(c, v), c) for _, c, v in entries]
        designs = {d for d, _, _ in verdicts}
        failing = {d for d, w, _ in verdicts if not w["decreasing"]}
        rooted = [w for _, w, c in verdicts if c["cbar_half"] is not None]
        return dict(
            n_configurations=len(entries),
            n_designs=len(designs),
            n_designs_strictly_decreasing=len(designs - failing),
            n_designs_with_cbar_half=len(
                {d for d, _, c in verdicts if c["cbar_half"] is not None}
            ),
            n_half_power_points=len(rooted),
            n_cbar_half_in_bracket=sum(w["inside"] is True for w in rooted),
        )

    tot = count(text)
    tags = {t for c in sets["lattices"] for t in c["computed_for"].split(";")}
    sources = {
        g: dict(n_configurations=sum(1 for x, _, _ in text if x == g))
        for g in named.values()
    }
    sources["table_1"]["file"] = JSON_FILE
    sources["trimming_and_proposition_1_lattices"].update(
        file=LAT_JSON_FILE,
        lattices=sorted(tags),
    )
    sources["rule_comparison_not_in_table_1"]["file"] = RULE_JSON_FILE
    return dict(
        statement="Pi^T_lambda decreases strictly from node to node on the grid of "
        "step 0.5 over K = [-20, -3] in "
        "every design at which Section 4 of the article computes the envelope: Table "
        f"1 ({JSON_FILE}), the trimming and Proposition 1 lattices of Section 4.1 "
        f"({LAT_JSON_FILE}) and the equispaced placements of the comparison of "
        f"Section 4.2 that are not in Table 1 ({RULE_JSON_FILE}); every half-power "
        "point computed there lies between the two nodes that bracket Pi = 1/2, so "
        "it is the unique root of Pi^T_lambda = 1/2 on the grid",
        design="T and the break dates tau_j = floor(lambda_j T): configurations with "
        "the same design have the same regressors and the same envelope, and the "
        "design is counted once",
        grid=dict(lo=GRID[0], hi=GRID[-1], step=0.5, n_nodes=len(GRID)),
        sources=sources,
        **tot,
        all_strictly_decreasing=tot["n_designs_strictly_decreasing"]
        == tot["n_designs"],
        same_design=same,
        same_design_max_abs_pi_difference=max((x for x, _ in pi_diff), default=None),
        same_design_pi_agree=all(ok for _, ok in pi_diff),
        same_design_max_abs_cbar_half_difference=max(cbar_diff, default=None),
    )


def statement_holds(files):
    """Every design of Section 4 passes and every cross-check agrees."""
    lat = json.loads(files[LAT_JSON_FILE])
    rule = json.loads(files[RULE_JSON_FILE])["rule_comparison"]
    sec = json.loads(files[SEC4_JSON_FILE])
    sh = lat["shared_with_table_1"]
    return (
        sec["all_strictly_decreasing"]
        and sec["n_cbar_half_in_bracket"] == sec["n_half_power_points"]
        and sec["same_design_pi_agree"]
        and sh["n_in_table_1_bracket"] == sh["n_half_power_points"]
        and lat["pi_at_minus7_against_proposition_1"]["agree"]
        and rule["n_cbar_half_equal_to_table_1"] == rule["n_in_table_1"]
    )


# ------------------------------------------------------------------- files and check
def _lf(data):
    return data.replace(b"\r\n", b"\n")


def _sha(data):
    return hashlib.sha256(_lf(data)).hexdigest()


def _csv_bytes(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _json_bytes(obj):
    return (json.dumps(obj, indent=1) + "\n").encode("utf-8")


def _complete(configs, vals, what):
    missing = [c["key"] for c in configs if c["key"] not in vals]
    if missing or len(vals) != len(configs):
        raise InputError(
            f"the {what} grid values do not match the configurations: "
            f"{len(vals)} values, {len(configs)} configurations, missing {missing}"
        )


def build(sets, vals):
    """The eleven output files as bytes, from the grid values of the three sets
    (sets and vals are keyed 'table1', 'lattices', 'rule')."""
    configs, lattices, rule = sets["table1"], sets["lattices"], sets["rule"]
    _complete(configs, vals["table1"], "Table 1")
    _complete(lattices, vals["lattices"], "lattice")
    _complete(rule, vals["rule"], "rule comparison")
    t1v, latv, rulev = vals["table1"], vals["lattices"], vals["rule"]
    files = {
        GRID_FILE: grid_bytes(configs, t1v),
        CSV_FILE: verdict_bytes(configs, t1v),
        JSON_FILE: _json_bytes(summary(configs, t1v)),
        LAT_GRID_FILE: grid_bytes(lattices, latv),
        LAT_CSV_FILE: lattice_verdict_bytes(lattices, latv),
        LAT_JSON_FILE: _json_bytes(lattice_summary(lattices, latv, configs, t1v)),
        RULE_GRID_FILE: grid_bytes(rule, rulev),
        RULE_CSV_FILE: rule_verdict_bytes(rule, rulev),
        RULE_JSON_FILE: _json_bytes(rule_summary(rule, rulev, configs)),
        SEC4_JSON_FILE: _json_bytes(section_4_summary(sets, vals)),
    }
    man = dict(
        object=OBJ,
        command="python mlb/envelope_monotonicity.py",
        command_note="--jobs, --out-dir and the split into parts (--part, --T) do "
        "not change the outputs",
        outputs_sha256={k: _sha(d) for k, d in files.items()},
        inputs_sha256_lf={rel: _sha(open(p, "rb").read()) for rel, p in INPUTS},
        seeds="none: exact computation, no random numbers",
        parameters=dict(
            grid=dict(lo=GRID[0], hi=GRID[-1], step=0.5, n_nodes=len(GRID)),
            function="mlb/envelope_exact.power_at(cbar, T, design_Z(T, lambdas))",
            lattices=dict(
                trimming="every distinct (T, lambda) of the trimming file that is "
                "not in Table 1",
                proposition_1="every (T, lambda) of the parts cbar_half and power_m7 "
                "of the Proposition 1 file with T in "
                f"{list(PROP1_T)} that is not in Table 1",
                T=list(LATTICE_T),
            ),
            rule_comparison=f"every row of the rule file with X = {RULE_X} that is not "
            "in Table 1 (the equispaced placements lambda_j = "
            "j/(m+1)), with that file's half-power point",
            design="(T, tau_j = floor(lambda_j T)); the totals over Section 4 count "
            "designs",
        ),
        conventions=EX.conventions(),
        check=dict(
            mode="LF bytes; otherwise every number to |a - b| <= tol (1 + |b|)",
            tol=TOL,
        ),
    )
    files[MANIFEST] = _json_bytes(man)
    return files


def _num(s):
    try:
        return float(s)
    except ValueError:
        return None


def _close_csv(new, ref, tol=TOL):
    a = list(csv.reader(io.StringIO(_lf(new).decode("utf-8"))))
    b = list(csv.reader(io.StringIO(_lf(ref).decode("utf-8"))))
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            fx, fy = _num(x), _num(y)
            if fx is None or fy is None:
                if x != y:
                    return False
            elif abs(fx - fy) > tol * (1.0 + abs(fy)):
                return False
    return True


def _close(a, b, tol=TOL):
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(_close(a[k], b[k], tol) for k in a)
        )
    if isinstance(a, list):
        return (
            isinstance(b, list)
            and len(a) == len(b)
            and all(_close(x, y, tol) for x, y in zip(a, b))
        )
    if isinstance(a, float) or isinstance(b, float):
        return (
            not isinstance(a, bool)
            and not isinstance(b, bool)
            and a is not None
            and b is not None
            and abs(a - b) <= tol * (1.0 + abs(b))
        )
    return a == b


def compare(files, ref_dir):
    """{file: 'identical' | 'within tolerance' | 'DIFFERENT' | 'MISSING'} for each
    of REFERENCE_OUTPUTS against its copy in ref_dir."""
    res = {}
    for name in REFERENCE_OUTPUTS:
        data = files[name]
        g = os.path.join(ref_dir, name)
        if not os.path.isfile(g):
            res[name] = "MISSING"
            continue
        ref = _lf(open(g, "rb").read())
        if _lf(data) == ref:
            res[name] = "identical"
            continue
        if name.endswith(".json"):
            same = _close(json.loads(data), json.loads(ref))
        else:
            same = _close_csv(data, ref)
        res[name] = "within tolerance" if same else "DIFFERENT"
    return res


def show_reference(ref_dir):
    """Print the verdict and the totals of each of the four summaries in ref_dir;
    0 if every one reports strict decrease, 1 if one does not, 2 if one is missing
    or malformed."""
    ok = True
    for name in REFERENCE_OUTPUTS:
        p = os.path.join(ref_dir, name)
        try:
            with open(p, encoding="utf-8") as fh:
                w = json.load(fh)
            if name == SEC4_JSON_FILE:
                n, n_dec = w["n_designs"], w["n_designs_strictly_decreasing"]
                unit = "designs"
                extra = (
                    f" ({w['n_configurations']} configurations); cbar_1/2 between "
                    f"the bracketing nodes {w['n_cbar_half_in_bracket']} of "
                    f"{w['n_half_power_points']}"
                )
            else:
                n, n_dec = w["n_configurations"], w["n_strictly_decreasing"]
                unit = "configurations"
                n_root = w.get("n_with_cbar_half", n)
                extra = (
                    f"; cbar_1/2 between the bracketing nodes "
                    f"{w['n_cbar_half_in_bracket']} of {n_root}"
                )
            holds = w["all_strictly_decreasing"] is True and n_dec == n
        except OSError:
            print(f"[{OBJ}] error: missing {p}", file=sys.stderr)
            return 2
        except (ValueError, KeyError, TypeError) as e:
            print(f"[{OBJ}] error: {p}: malformed ({e!r})", file=sys.stderr)
            return 2
        word = "strict decrease" if holds else "NOT strict decrease"
        print(f"{name}: {word}: {n_dec} of {n} {unit}{extra}")
        ok = ok and holds
    print(f"[{OBJ}] {'PASS' if ok else 'FAIL'}: {ref_dir}")
    return 0 if ok else 1


# ------------------------------------------------------------------- command line
def _parts(ap, a):
    """The part names that --part or --T select, in the order given."""
    if a.part is not None:
        names = [x.strip() for x in a.part.split(",") if x.strip()]
        bad = [x for x in names if x not in PARTS]
        if bad or not names:
            ap.error(f"--part: unknown part {bad}; the parts are {', '.join(PARTS)}")
        return list(dict.fromkeys(names))
    try:
        Ts = [int(x) for x in a.T.split(",") if x.strip()]
    except ValueError:
        ap.error(f"--T: not a list of integers: {a.T!r}")
    sizes = sorted({T for _, T in PARTS.values() if T is not None})
    if not Ts or not set(Ts) <= set(sizes):
        ap.error(f"--T must be among {sizes}")
    return list(dict.fromkeys(n for T in Ts for n in PARTS if PARTS[n][1] == T))


def _estimate(configs, jobs):
    cpu = cpu_minutes(configs)
    return (
        f"about {cpu:.0f} CPU-minutes on an idle machine (about "
        f"{max(1, round(cpu / jobs))} min at --jobs {jobs}; two to three times "
        "longer on a loaded one)"
    )


def _write(out, name, data):
    try:
        with open(os.path.join(out, name), "wb") as fh:
            fh.write(data)
    except OSError as e:
        raise OutputError(f"cannot write {os.path.join(out, name)}: {e}") from None


def _run_parts(names, sets, out, jobs):
    todo = [(n, part_configs(n, sets)) for n in names]
    every = [c for _, sub in todo for c in sub]
    print(
        f"[{OBJ}] parts {', '.join(names)}: {len(every)} configurations x "
        f"{len(GRID) + 1} exact evaluations, {_estimate(every, jobs)}",
        flush=True,
    )
    t0 = time.time()
    vals = compute(every, jobs)
    ok = True
    for name, sub in todo:
        data = grid_bytes(sub, vals)
        _write(out, part_file(name), data)
        vs = [verdict(c, vals[c["key"]]) for c in sub]
        n_dec = sum(v["decreasing"] for v in vs)
        n_root = sum(v["inside"] is not None for v in vs)
        n_in = sum(v["inside"] is True for v in vs)
        ok = ok and n_dec == len(sub) and n_in == n_root
        print(
            f"{part_file(name)}: strictly decreasing on the grid {n_dec} of "
            f"{len(sub)}; cbar_1/2 between the bracketing nodes {n_in} of {n_root}"
        )
    print(f"[{OBJ}] wrote {len(names)} part files to {out} in {time.time() - t0:.0f} s")
    return 0 if ok else 1


def _aggregate_values(out, grids_dir):
    """{'table1': ..., 'lattices': ..., 'rule': ...} grid values, from the three grid
    files in grids_dir (when given) or from the part files in out."""
    if grids_dir is not None:
        return {
            k: read_grid([os.path.join(grids_dir, f)]) for k, f in GRID_FILES.items()
        }
    missing = [n for n in PARTS if not os.path.isfile(os.path.join(out, part_file(n)))]
    if missing:
        raise InputError(
            f"--aggregate needs the {len(PARTS)} part files in {out}; missing: "
            f"{', '.join(part_file(n) for n in missing)}. Compute them with --part "
            f"{','.join(missing)} --out-dir {out}, or give the grid files of an "
            "earlier run with --grids-dir"
        )
    files = {k: [] for k in GRID_FILES}
    for n in PARTS:
        files[PARTS[n][0]].append(os.path.join(out, part_file(n)))
    return {k: read_grid(f) for k, f in files.items()}


def _prepare_out(out):
    """Create the output folder; an OutputError if it is not a folder or cannot be
    created."""
    if os.path.exists(out) and not os.path.isdir(out):
        raise OutputError(f"--out-dir {out} exists and is not a folder")
    try:
        os.makedirs(out, exist_ok=True)
    except OSError as e:
        raise OutputError(f"cannot create the output folder {out}: {e}") from None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default=None, help="output folder")
    ap.add_argument("--jobs", type=int, default=4, help="worker processes")
    ap.add_argument(
        "--part", default=None, help="compute only these parts (comma-separated)"
    )
    ap.add_argument(
        "--T",
        default=None,
        help="compute only the Table 1 and lattice parts of these T (comma-separated)",
    )
    ap.add_argument(
        "--aggregate", action="store_true", help="build the outputs from the parts"
    )
    ap.add_argument(
        "--grids-dir",
        default=None,
        help="with --aggregate: read the three grid files of an earlier run from this "
        "folder instead of the parts",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare the four summaries with the reference",
    )
    ap.add_argument(
        "--show-reference",
        action="store_true",
        help="print the verdict and the totals of the four reference summaries",
    )
    a = ap.parse_args(argv)
    ref_dir = str(paths.reference_dir(SECTION) / OBJ)
    if a.show_reference:
        if (
            a.part is not None
            or a.T is not None
            or a.aggregate
            or a.grids_dir is not None
            or a.check
            or a.out_dir is not None
        ):
            ap.error("--show-reference takes no other option")
        return show_reference(ref_dir)
    if a.part is not None and a.T is not None:
        ap.error("give --part or --T, not both")
    if a.aggregate and (a.part is not None or a.T is not None):
        ap.error("--aggregate builds the outputs from the parts; drop --part and --T")
    if a.grids_dir is not None and not a.aggregate:
        ap.error("--grids-dir goes with --aggregate")
    if a.check and (a.part is not None or a.T is not None):
        ap.error(
            "--check compares the four summaries, which the parts do not build; "
            "check with --aggregate --check"
        )
    names = _parts(ap, a) if (a.part is not None or a.T is not None) else None
    jobs = max(1, a.jobs)
    t0 = time.time()
    try:
        out = a.out_dir
        if out is None:
            out = str(paths.out_dir(SECTION) / OBJ)
        _prepare_out(out)
        sets = all_sets()
        if names is not None:
            return _run_parts(names, sets, out, jobs)
        if a.aggregate:
            vals = _aggregate_values(out, a.grids_dir)
        else:
            every = sets["table1"] + sets["lattices"] + sets["rule"]
            print(
                f"[{OBJ}] {len(sets['table1'])} + {len(sets['lattices'])} + "
                f"{len(sets['rule'])} configurations x {len(GRID) + 1} exact "
                f"evaluations, {_estimate(every, jobs)}",
                flush=True,
            )
            allv = compute(every, jobs)
            vals = {k: {c["key"]: allv[c["key"]] for c in cs} for k, cs in sets.items()}
        files = build(sets, vals)
        for name, data in files.items():
            _write(out, name, data)
    except (InputError, OutputError) as e:
        print(f"[{OBJ}] error: {e}", file=sys.stderr)
        return 2
    s = json.loads(files[JSON_FILE])
    lat = json.loads(files[LAT_JSON_FILE])
    rule = json.loads(files[RULE_JSON_FILE])
    sec = json.loads(files[SEC4_JSON_FILE])
    print(
        f"Table 1: strictly decreasing on the grid {s['n_strictly_decreasing']} of "
        f"{s['n_configurations']}; cbar_1/2 between the bracketing nodes "
        f"{s['n_cbar_half_in_bracket']} of {s['n_configurations']}"
    )
    for name, w in (("lattices", lat), ("rule comparison", rule)):
        print(
            f"{name}: strictly decreasing on the grid {w['n_strictly_decreasing']} "
            f"of {w['n_configurations']}; cbar_1/2 between the bracketing nodes "
            f"{w['n_cbar_half_in_bracket']} of {w['n_with_cbar_half']}"
        )
    print(
        f"Section 4: {sec['n_configurations']} configurations, {sec['n_designs']} "
        f"designs ({len(sec['same_design'])} configurations are designs counted "
        f"before); strictly decreasing {sec['n_designs_strictly_decreasing']} of "
        f"{sec['n_designs']} designs; cbar_1/2 in bracket "
        f"{sec['n_cbar_half_in_bracket']} of {sec['n_half_power_points']}"
    )
    print(
        f"cross-checks: shared half-power points in the Table 1 bracket "
        f"{lat['shared_with_table_1']['n_in_table_1_bracket']} of "
        f"{lat['shared_with_table_1']['n_half_power_points']}; Pi(-7) against "
        f"Proposition 1: max |difference| "
        f"{lat['pi_at_minus7_against_proposition_1']['max_abs_difference']!r}; rule "
        f"file's half-power points equal to Table 1's "
        f"{rule['rule_comparison']['n_cbar_half_equal_to_table_1']} of "
        f"{rule['rule_comparison']['n_in_table_1']}; one design, max |Pi difference| "
        f"{sec['same_design_max_abs_pi_difference']!r}"
    )
    for name, w in (("Table 1", s), ("lattices", lat), ("rule comparison", rule)):
        d = w["largest_difference"]
        print(
            f"{name}: largest successive difference {d['value']!r} (from "
            f"{d['from_node']}, {d['config_key']}); largest |Pi(cbar_1/2) - 1/2| "
            f"{w['largest_abs_pi_at_cbar_half_minus_half']['value']!r}"
        )
    print(f"[{OBJ}] wrote {out} in {time.time() - t0:.0f} s")
    holds = statement_holds(files)
    if not a.check:
        return 0 if holds else 1
    res = compare(files, ref_dir)
    for name, word in res.items():
        print(f"--check: {name}: {word}")
    ok = holds and all(w in ("identical", "within tolerance") for w in res.values())
    print(f"--check: {'PASS' if ok else 'FAIL'} against {ref_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
