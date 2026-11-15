"""The (m+1) regularity of the half-power points (article section 4.1, after
Table 1).

The text: the scaled displacement of the cell means, T |cbar_1/2 - cbar_inf| / (m+1),
"differs across the m of each column of Table 1 by at most 11% of its largest value,
and by at most 3% at T=80 and 100".

The definition the text states (the range across m relative to its largest value).
For each T of Table 1 and each m
of its column (m <= 3 at T = 45 and 50, m <= 4 at T = 60, m <= 5 from T = 80), with
cbar_1/2(m, T) the cell mean of the exact half-power points over the break
configurations of the cell,

    s(m, T)  = T |cbar_1/2(m, T) - cbar_inf| / (m + 1),
    d(T)     = [max_m s(m, T) - min_m s(m, T)] / max_m s(m, T),

the range across m relative to its largest value, and the statement is
d(T) <= 0.11 at every T and d(T) <= 0.03 at T = 80 and 100. cbar_inf is the root of
the limit envelope at one half (-7.0064..., mlb/limit_envelope.py).
Under this definition both statements hold without rounding.

Inputs (reference/): calibration/cbar_surface_exact/cbar_surface_exact.csv
(the exact half-power point of each of the 404 configurations) and
theory/limit_envelope/limit_envelope.json (cbar_inf). No simulation;
the cell means are exactly rounded sums (math.fsum), so the output is the same on
every platform.

Output (output/critical_values/regularity_m_plus_1/, or --out-dir):
regularity_m_plus_1.json (the definition, s(m, T) and d(T) at full precision, the two
statements) and regularity_m_plus_1.manifest.json (the inputs' sha256, the output's
sha256; nothing time- or machine-dependent). --check compares the JSON with
reference/calibration/regularity_m_plus_1/ byte for byte (LF) and fails if
either statement does not hold. Under a second.

    python mlb/regularity_m_plus_1.py [--out-dir D] [--check]
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlb import paths  # noqa: E402

SECTION, OBJ = "critical_values", "regularity_m_plus_1"
T_TABLE = (45, 50, 60, 80, 100, 150, 200, 300)
BOUND_ALL, BOUND_80_100, T_TIGHT = 0.11, 0.03, (80, 100)
SURFACE_REL = "reference/calibration/cbar_surface_exact/cbar_surface_exact.csv"
LIMIT_REL = "reference/theory/limit_envelope/limit_envelope.json"
SURFACE = paths.reference_dir("critical_values") / "cbar_surface_exact" / (
    "cbar_surface_exact.csv"
)
LIMIT = paths.reference_dir("supplement_appendix") / "limit_envelope" / (
    "limit_envelope.json"
)
DEFINITION = (
    "s(m,T) = T |cbar_1/2(m,T) - cbar_inf| / (m+1), cbar_1/2(m,T) the cell mean over "
    "the break configurations; d(T) = (max_m s - min_m s) / max_m s over the m of "
    "the column T of Table 1"
)


def cell_means(path=SURFACE):
    """{(m, T): (mean, n)} over the configurations with T in Table 1."""
    cells = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            T, m = int(r["T"]), int(r["m"])
            if T in T_TABLE:
                cells.setdefault((m, T), []).append(float(r["cbar_half_exact"]))
    return {k: (math.fsum(v) / len(v), len(v)) for k, v in sorted(cells.items())}


def limit_root(path=LIMIT):
    with open(path, encoding="utf-8") as fh:
        return float(json.load(fh)["limit"]["cbar_inf"])


def numbers(means, cbar_inf):
    by_T = {}
    for T in T_TABLE:
        ms = sorted(m for (m, t) in means if t == T)
        s = {m: T * abs(means[(m, T)][0] - cbar_inf) / (m + 1) for m in ms}
        hi, lo = max(s.values()), min(s.values())
        d = (hi - lo) / hi
        by_T[str(T)] = dict(
            m=ms,
            n_configurations={str(m): means[(m, T)][1] for m in ms},
            cell_mean={str(m): means[(m, T)][0] for m in ms},
            s={str(m): s[m] for m in ms},
            argmax_m=max(ms, key=lambda m: s[m]),
            argmin_m=min(ms, key=lambda m: s[m]),
            spread=d,
            spread_pct=100.0 * d,
        )
    worst = max(T_TABLE, key=lambda T: by_T[str(T)]["spread"])
    tight = max(by_T[str(T)]["spread"] for T in T_TIGHT)
    return dict(
        statement="T |cbar_1/2 - cbar_inf| / (m+1) differs across m by at most 11% "
        "at each T of Table 1, and by at most 3% at T = 80 and 100",
        definition=DEFINITION,
        cbar_inf=cbar_inf,
        T=by_T,
        largest_spread=dict(value=by_T[str(worst)]["spread"], T=worst),
        largest_spread_at_T80_T100=tight,
        holds_at_most_11pct_at_each_T=all(
            by_T[str(T)]["spread"] <= BOUND_ALL for T in T_TABLE
        ),
        holds_at_most_3pct_at_T80_T100=tight <= BOUND_80_100,
    )


RTOL = 1e-9


def _same_json(x, y):
    """Structure, strings and integers equal; floats to a relative RTOL (a root or an
    input regenerated on another platform moves in its last digits)."""
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same_json(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(map(_same_json, x, y))
    if isinstance(x, float) or isinstance(y, float):
        ok = isinstance(x, (int, float)) and isinstance(y, (int, float))
        return ok and abs(x - y) <= RTOL * max(1.0, abs(x), abs(y))
    return type(x) is type(y) and x == y


def _lf(data):
    return data.replace(b"\r\n", b"\n")


def _sha(path):
    return hashlib.sha256(_lf(open(path, "rb").read())).hexdigest()


def _dump(obj):
    return (json.dumps(obj, indent=1) + "\n").encode("utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default=None, help="output folder")
    ap.add_argument(
        "--check", action="store_true", help="compare the output with the reference"
    )
    a = ap.parse_args(argv)
    out = a.out_dir or str(paths.out_dir(SECTION) / OBJ)
    os.makedirs(out, exist_ok=True)
    res = numbers(cell_means(), limit_root())
    data = _dump(res)
    man = dict(
        object=OBJ,
        command="python mlb/regularity_m_plus_1.py",
        command_note="--out-dir does not change the outputs",
        outputs_sha256={f"{OBJ}.json": hashlib.sha256(data).hexdigest()},
        inputs_sha256_lf={SURFACE_REL: _sha(SURFACE), LIMIT_REL: _sha(LIMIT)},
        seeds="none: arithmetic on the reference files",
        parameters=dict(
            T=list(T_TABLE),
            bound_each_T=BOUND_ALL,
            bound_T80_T100=BOUND_80_100,
        ),
        check="LF bytes",
    )
    with open(os.path.join(out, f"{OBJ}.json"), "wb") as fh:
        fh.write(data)
    with open(os.path.join(out, f"{OBJ}.manifest.json"), "wb") as fh:
        fh.write(_dump(man))
    print("definition:", DEFINITION)
    print(f"cbar_inf = {res['cbar_inf']!r}")
    for T in T_TABLE:
        r = res["T"][str(T)]
        cells = " ".join(f"{r['s'][str(m)]:.2f}" for m in r["m"])
        print(f"  T={T:<3} s(m=0..{r['m'][-1]}): {cells}   d = {r['spread_pct']:.2f}%")
    print(
        f"largest d = {100 * res['largest_spread']['value']:.2f}% "
        f"(T = {res['largest_spread']['T']}): at most 11% at each T: "
        f"{res['holds_at_most_11pct_at_each_T']}"
    )
    print(
        f"largest d at T = 80, 100 = {100 * res['largest_spread_at_T80_T100']:.2f}%: "
        f"at most 3%: {res['holds_at_most_3pct_at_T80_T100']}"
    )
    holds = (
        res["holds_at_most_11pct_at_each_T"] and res["holds_at_most_3pct_at_T80_T100"]
    )
    if not a.check:
        return 0 if holds else 1
    ref = paths.reference_dir(SECTION) / OBJ / f"{OBJ}.json"
    if not ref.is_file():
        print(f"--check: MISSING {ref}")
        return 1
    raw = _lf(ref.read_bytes())
    if raw == data:
        word = "identical"
    elif _same_json(json.loads(raw), json.loads(data)):
        word = "equal within 1e-9"
    else:
        word = "DIFFERENT"
    print(f"--check: {OBJ}.json: {word}")
    ok = word in ("identical", "equal within 1e-9") and holds
    print(f"--check: {'PASS' if ok else 'FAIL'} against {ref.parent}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
