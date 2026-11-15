"""The fixed index that maximizes power, full sample and blocks (article, sec:rule).

The article's subsection on fixing cbar = -7 (sec:rule) says that at (m, T) = (2, 60)
and c = -10 the best fixed index on a grid of step 0.5 improves on -7 by less than one
percentage point and that, across independent blocks of the simulation, it ranges
over [-7.0, -4.5]. This producer computes both.

Design: the (2, 60) configuration with equispaced breaks (lambda = 1/3, 2/3; break
positions 20 and 40) of rule_comparison.py, with its nodes, its seed and its
R = 200,000 draws per stream, recomputed with that module's configurations(),
simulate() and summarize(); no random number is drawn here that it does not draw.
Its reference files hold per-configuration summaries only, not per-block powers, so the draws
are recomputed: 2 to 5 minutes on one core, of which configurations() alone (the
exact half-power points of the configurations off the surface grid) takes one to two.

Agreement check (equals_reference_summary in the JSON): the recomputed summary of the
configuration equals its summary in rule_comparison's reference files, every number to a
relative 1e-9 (the critical values can move in their last bits with the BLAS kernel;
nothing printed can):
rule_levels.csv (the critical value, the power and its se at every node against every
alternative), rule_configs.csv
(W4, the differences against -7 and their se for cbar_half and the cell mean) and the
c = -10 row of rule_fixed_index.csv (the argmax, its power, the power of -7, the two
ratios). If it fails, the JSON records the differences instead of the results and the
exit code is 3.

Grid G: the nodes of the configuration in the fixed-index grid of rule_comparison
(-2.0, -2.5, ..., -12.0, -13.5, -21.7), with -7 and the configuration's cbar_half.
At c = -10, with Pi(x) the size-corrected power at node x:
  full_sample  j* = argmax over G of Pi; d = Pi(j*) - Pi(-7); the ratios
               (Pi(j*) - Pi(-13.5)) / d and (Pi(j*) - Pi(-21.7)) / d;
  blocks       B = 8 and B = 20 consecutive blocks of the draws, the critical value
               and the power recomputed within each block:
               fixed  the same three quantities at j*;
               block  j_b = argmax over G of Pi_b and the same three at j_b
               (a ratio is NaN when its denominator is not positive).
For each quantity and B: the mean, se = std(ddof=1) / sqrt(n), n, min and max of the
finite values, and the number of non-positive denominators; argmax_distribution counts
the blocks per argmax node. se_block_rule = max(se_8, se_20), the block rule for
standard errors of rule_comparison; se_fixed_d_equals_state_se_delta checks that,
for the fixed argmax, it equals that module's se of the same difference; d1_places is the
largest decimal place d with se <= 0.5 * 10^-d.

Output (output/critical_values/fixed_index/, or --out-dir): fixed_index.json (values
only, full precision) and the run record fixed_index.manifest.json (the sha256 of the
inputs, the seeds, the parameters; no time). The powers are counts over fixed draws and
every other number is arithmetic on them: on the BLAS kernel that wrote the reference
the output is identical byte for byte (LF); elsewhere the critical values can differ in
their last bits, so --check compares fixed_index.json with its reference, accepts every
number equal to a relative 1e-9 and every node, count and key exactly, and says which
of the two it found. The run is one configuration in one process.

    python mlb/fixed_index.py [--out-dir D] [--check]
"""

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

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

import numpy as np  # noqa: E402

from mlb import rule_comparison as rule  # noqa: E402
from mlb import paths  # noqa: E402

SECTION, OBJ = "critical_values", "fixed_index"
CELL = (2, 60, "equispaced")
C_ALT = -10.0
KEYS = (
    "fixed_d",
    "fixed_ratio_135",
    "fixed_ratio_217",
    "block_d",
    "block_ratio_135",
    "block_ratio_217",
)
RULE_REFERENCE = paths.reference_dir(SECTION) / rule.OBJ
INPUTS = dict(
    surface=Path(rule.SURFACE),
    rule_levels=RULE_REFERENCE / "rule_levels.csv",
    rule_configs=RULE_REFERENCE / "rule_configs.csv",
    rule_fixed=RULE_REFERENCE / "rule_fixed_index.csv",
)
REFERENCE_DIR = paths.reference_dir(SECTION) / OBJ
# the outputs --check compares: those with a reference file in reference/
REFERENCE_OUTPUTS = (f"{OBJ}.json",)
COMMAND = "python mlb/fixed_index.py"


# ------------------------------------------------------------------ configuration
def configuration():
    """The (2, 60) equispaced configuration of rule_comparison."""
    cfgs = rule.configurations(rule.load_surface())
    return next(x for x in cfgs if (x["m"], x["T"], x["kind"]) == CELL)


def simulate(c, nrep=rule.R):
    """rule_comparison's draws of the configuration (the first nrep per stream)."""
    return rule.simulate(c["T"], c["bp"], c["nodes"], c["seed"], nrep=nrep)


def grid(nodes):
    """Positions in `nodes` of the grid G: the fixed-index grid, -7 and cbar_half."""
    G = [j for j, x in enumerate(nodes) if x in rule.GRID_G or j in (0, 1)]
    if len({nodes[j] for j in G}) != len(G) or nodes[0] != -7.0:
        raise ValueError("the grid G has repeated nodes or does not start at -7")
    return G


def _anchors(nodes):
    return tuple(nodes.index(x) for x in (-7.0, -13.5, -21.7))


# -------------------------------------------------------------------- full sample
def full_sample(s, nodes, G):
    i10 = rule.ALTS.index(C_ALT)
    j7, j135, j217 = _anchors(nodes)
    pw = np.array(s["power"][i10])
    jstar = max(G, key=lambda j: pw[j])
    full = dict(
        argmax=nodes[jstar],
        power_max=float(pw[jstar]),
        power_minus7=float(pw[j7]),
        power_m135=float(pw[j135]),
        power_m217=float(pw[j217]),
        d=float(pw[jstar] - pw[j7]),
        ratio_135=float((pw[jstar] - pw[j135]) / (pw[jstar] - pw[j7])),
        ratio_217=float((pw[jstar] - pw[j217]) / (pw[jstar] - pw[j7])),
        se_d_from_state=float(s["se_delta"][i10][jstar]),
    )
    return full, jstar


# ---------------------------------------------------------------- agreement check
RTOL = 1e-9


def _close(a, b):
    """Equal, or both NaN, or within a relative RTOL (absolute for values near 0)."""
    if a == b or (math.isnan(a) and math.isnan(b)):
        return True
    return abs(a - b) <= RTOL * max(1.0, abs(a), abs(b))


def _same(x, y):
    """Structure, keys, strings, booleans and integers equal; floats to RTOL."""
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(_same(a, b) for a, b in zip(x, y))
    if isinstance(x, float) or isinstance(y, float):
        both = (int, float)
        if isinstance(x, both) and isinstance(y, both) and not isinstance(x, bool):
            return _close(float(x), float(y))
        return False
    return x == y


def _rows(path, **match):
    with open(path, encoding="utf-8", newline="") as fh:
        return [
            r for r in csv.DictReader(fh) if all(r[k] == v for k, v in match.items())
        ]


def agreement_check(c, s, full):
    """The recomputed summary against the reference files of rule_comparison.

    Returns the list of differences (empty when every number is equal)."""
    bad = []

    def eq(what, got, want):
        if not _close(float(got), float(want)):
            bad.append(f"{what}: {got!r} != {want!r}")

    nodes = s["nodes"]
    lv = _rows(INPUTS["rule_levels"], config=c["id"])
    seen = set()
    for r in lv:
        j, i = int(r["node_index"]), rule.ALTS.index(float(r["c"]))
        seen.add((j, i))
        tag = f"rule_levels node {r['node']} c {r['c']}"
        if j >= len(nodes):
            bad.append(f"{tag}: node_index {j} out of range")
            continue
        eq(tag + " node", nodes[j], r["node"])
        eq(tag + " cv", s["cv"][j], r["cv"])
        eq(tag + " power", s["power"][i][j], r["power"])
        eq(tag + " se_power", s["se_power"][i][j], r["se_power"])
    want = {(j, i) for j in range(len(nodes)) for i in range(len(rule.ALTS))}
    if seen != want or len(lv) != len(want):
        bad.append(f"rule_levels.csv: {len(lv)} rows, not one per node and alternative")

    cf = _rows(INPUTS["rule_configs"], m=str(c["m"]), T=str(c["T"]), kind=c["kind"])
    roles = {"cbar_half(config)": 1, "cbarbar_half(m,T)": 2}
    if sorted(r["X"] for r in cf) != sorted(roles):
        bad.append(f"rule_configs.csv: rows {[r['X'] for r in cf]}")
    for r in cf:
        k = roles.get(r["X"])
        if k is None:
            continue
        tag = f"rule_configs {r['X']}"
        eq(tag + " X_value", nodes[k], r["X_value"])
        eq(tag + " power_w4_minus7", s["w4"][0], r["power_w4_minus7"])
        eq(tag + " power_w4_X", s["w4"][k], r["power_w4_X"])
        eq(tag + " delta_w4", s["delta_w4"][k], r["delta_w4"])
        eq(tag + " se_delta_w4", s["se_delta_w4"][k], r["se_delta_w4"])
        eq(tag + " cv_minus7", s["cv"][0], r["cv_minus7"])
        eq(tag + " cv_X", s["cv"][k], r["cv_X"])
        for i, a in enumerate(rule.ALTS):
            eq(f"{tag} delta_c{int(a)}", s["delta"][i][k], r[f"delta_c{int(a)}"])
            eq(
                f"{tag} se_delta_c{int(a)}",
                s["se_delta"][i][k],
                r[f"se_delta_c{int(a)}"],
            )
            eq(
                f"{tag} power_minus7_c{int(a)}",
                s["power"][i][0],
                r[f"power_minus7_c{int(a)}"],
            )

    fixed_rows = _rows(INPUTS["rule_fixed"], m=str(c["m"]), T=str(c["T"]), c=str(C_ALT))
    if len(fixed_rows) != 1:
        bad.append(f"rule_fixed_index.csv: {len(fixed_rows)} rows at c = {C_ALT}")
    for r in fixed_rows:
        for col, key in (
            ("argmax_cbar", "argmax"),
            ("power_max", "power_max"),
            ("power_minus7", "power_minus7"),
            ("dist_minus7_to_max", "d"),
            ("ratio_135", "ratio_135"),
            ("ratio_217", "ratio_217"),
        ):
            eq(f"rule_fixed {col}", full[key], r[col])
    return bad


# ------------------------------------------------------------------------- blocks
def block_row(b, null_b, alts_b, nodes, G, jstar):
    """One block: critical value and power recomputed within it, at c = -10."""
    _, PB = rule._levels(null_b, alts_b)
    p = PB[rule.ALTS.index(C_ALT)]
    j7, j135, j217 = _anchors(nodes)
    jb = max(G, key=lambda j: p[j])
    row = dict(block=b, argmax_block=nodes[jb])
    for tag, j in (("fixed", jstar), ("block", jb)):
        d = float(p[j] - p[j7])
        row[f"{tag}_d"] = d
        row[f"{tag}_ratio_135"] = float((p[j] - p[j135]) / d) if d > 0 else math.nan
        row[f"{tag}_ratio_217"] = float((p[j] - p[j217]) / d) if d > 0 else math.nan
    return row


def edges(n, B):
    return np.linspace(0, n, B + 1).astype(int)


def block_summary(rows, B):
    summ = {}
    for k in KEYS:
        v = np.array([r[k] for r in rows], dtype=float)
        fin = v[np.isfinite(v)]
        n = int(fin.size)
        summ[k] = dict(
            mean=float(fin.mean()) if n else math.nan,
            se=float(fin.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan,
            n_finite=n,
            min=float(fin.min()) if n else math.nan,
            max=float(fin.max()) if n else math.nan,
            n_nonpositive_denominator=int(B - n) if "ratio" in k else 0,
        )
    counts = collections.Counter(r["argmax_block"] for r in rows)
    summ["argmax_distribution"] = dict(sorted(counts.items()))
    return summ


def blocks(sim, nodes, G, jstar):
    null = sim[0.0]
    alts = [sim[x] for x in rule.ALTS]
    out = {}
    for B in rule.BLOCKS:
        e = edges(null.shape[0], B)
        rows = [
            block_row(
                b,
                null[e[b] : e[b + 1]],
                [x[e[b] : e[b + 1]] for x in alts],
                nodes,
                G,
                jstar,
            )
            for b in range(B)
        ]
        if [r["block"] for r in rows] != list(range(B)):
            raise ValueError(f"B = {B}: the blocks are not 0..{B - 1}")
        out[B] = dict(rows=rows, summary=block_summary(rows, B))
    return out


def d1(se):
    """The largest decimal place d with se <= 0.5 * 10^-d (None if se is not > 0)."""
    if not (se > 0) or not math.isfinite(se):
        return None
    return math.floor(-math.log10(2 * se))


# ------------------------------------------------------------------------ results
def compute(c):
    """(results, differences of the agreement check); the results hold its verdict."""
    nodes = c["nodes"]
    G = grid(nodes)
    sim = simulate(c)
    s = rule.summarize(sim, nodes)
    full, jstar = full_sample(s, nodes, G)
    bad = agreement_check(c, s, full)
    res = dict(
        config=c["id"],
        seed=c["seed"],
        bp=list(c["bp"]),
        nodes=nodes,
        R=rule.R,
        equals_reference_summary=not bad,
    )
    if bad:
        res["agreement_differences"] = bad
        return res, bad
    blk = blocks(sim, nodes, G, jstar)
    se = {k: max(blk[B]["summary"][k]["se"] for B in rule.BLOCKS) for k in KEYS}
    res.update(
        full_sample=full,
        blocks={str(B): v for B, v in blk.items()},
        se_block_rule=se,
        se_fixed_d_equals_state_se_delta=abs(se["fixed_d"] - full["se_d_from_state"])
        <= 1e-15,
        d1_places={k: d1(v) for k, v in se.items()},
    )
    return res, bad


def to_bytes(obj):
    return (json.dumps(obj, indent=1) + "\n").encode("utf-8")


def sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def manifest(c, G, data):
    """The manifest of an output whose bytes are `data`: inputs, seeds, parameters."""
    streams = (0.0,) + tuple(rule.ALTS)
    return dict(
        output=f"{OBJ}.json",
        output_sha256=hashlib.sha256(data).hexdigest(),
        command=COMMAND,
        draws="mlb/rule_comparison.py: configurations, "
        "simulate, summarize, _levels",
        configuration=c["id"],
        m=c["m"],
        T=c["T"],
        bp=list(c["bp"]),
        seed=c["seed"],
        stream_seeds={
            str(x): (c["seed"] + p * rule.STRIDE) % (2**63 - 1)
            for p, x in enumerate(streams)
        },
        R=rule.R,
        chunk=rule.CHUNK,
        alpha=rule.ALPHA,
        alternatives=list(rule.ALTS),
        c=C_ALT,
        blocks=list(rule.BLOCKS),
        grid=[float(c["nodes"][j]) for j in G],
        inputs_sha256_lf={k: sha_lf(p) for k, p in INPUTS.items()},
    )


def _compare(out, names):
    ok = True
    for name in names:
        new, gold = out / name, REFERENCE_DIR / name
        if not gold.is_file():
            print(f"--check: {name}: no reference at reference/{SECTION}/{OBJ}/")
            ok = False
            continue
        a = open(new, "rb").read().replace(b"\r\n", b"\n")
        b = open(gold, "rb").read().replace(b"\r\n", b"\n")
        same = a == b
        close = same or (
            name.endswith(".json")
            and _same(json.loads(a), json.loads(b))
        )
        ok = ok and close
        word = "identical to" if same else (
            f"equal (floats to {RTOL:g}) to" if close else "DIFFERS from"
        )
        print(f"--check: {name} {word} the reference")
        if not close and name.endswith(".json"):
            ga, gb = json.loads(a), json.loads(b)
            diff = [
                k
                for k in sorted(set(ga) | set(gb))
                if json.dumps(ga.get(k)) != json.dumps(gb.get(k))
            ]
            print(f"         top-level keys that differ: {diff}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default=None, help="output folder")
    ap.add_argument(
        "--check", action="store_true", help="compare with the reference (LF bytes)"
    )
    a = ap.parse_args(argv)
    if sys.flags.optimize:
        raise SystemExit("refusing to run under -O: the draws' checks are asserts")
    out = Path(a.out_dir) if a.out_dir else paths.out_dir(SECTION) / OBJ
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[{OBJ}] (m, T) = (2, 60), R = {rule.R:,} draws x {1 + len(rule.ALTS)} "
        "streams: 2 to 5 minutes on one core",
        flush=True,
    )
    t0 = time.time()
    c = configuration()
    res, bad = compute(c)
    data = to_bytes(res)
    with open(out / f"{OBJ}.json", "wb") as fh:
        fh.write(data)
    man = manifest(c, grid(c["nodes"]), data)
    with open(out / f"{OBJ}.manifest.json", "wb") as fh:
        fh.write(to_bytes(man))
    print(f"[{OBJ}] agreement check (the recomputed summary equals the reference one): "
          f"{not bad}")
    if bad:
        for line in bad[:20]:
            print("   ", line)
        print(f"[{OBJ}] the recomputed draws are not those of {rule.OBJ}: stopping")
        return 3
    f = res["full_sample"]
    print(
        f"[{OBJ}] full sample, c = {C_ALT}: argmax {f['argmax']}, "
        f"Pi(max) - Pi(-7) = {f['d']!r} (se {f['se_d_from_state']!r})"
    )
    found = []
    for B in rule.BLOCKS:
        dist = res["blocks"][str(B)]["summary"]["argmax_distribution"]
        found += [float(x) for x in dist]
        print(f"[{OBJ}] B = {B}: block argmax distribution {json.dumps(dist)}")
    print(f"[{OBJ}] block argmax range [{min(found)}, {max(found)}]")
    print(f"[{OBJ}] done in {time.time() - t0:.1f} s -> {out}")
    if a.check:
        return 0 if _compare(out, REFERENCE_OUTPUTS) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
