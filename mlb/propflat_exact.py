"""Proposition 1 (prop:flat) on the exact finite-sample envelope.

The numbers printed in the article after Table 1 (Section 4.1, "Exact half-power
points") and in its footnote:

  * part (iii), the rate: the largest displacement of the half-power point from
    -7 over the break configurations, sup_lambda |cbar_1/2(m, T, lambda) + 7|,
    multiplied by T/(m+1), for one and two breaks at T = 52, 100, 200, 400
    (printed 40.8, 53.9, 62.5, 67.0 and 41.1, 52.8, 60.4, 64.2);
  * the footnote: the same displacements measured from the limit root cbar_inf
    instead of -7, and the change of the scaled values, which is
    (-7 - cbar_inf) T/(m+1) because every configuration lies below cbar_inf
    (printed: 0.0064 T/(m+1), at most 1.3), with the largest cbar_1/2 over the
    configurations of each (m, T), which shows that every configuration lies below
    cbar_inf. cbar_inf is read from the result of
    mlb/limit_envelope.py
    (reference/theory/limit_envelope/limit_envelope.json);
  * part (i), the power gap: the power of the 5% envelope test at the fixed
    cbar = -7 on the diagonal c = -7, two breaks against the constant-only
    design (m = 0), at T = 52, 100, 200, 400; the largest loss over the
    configurations is printed as 8 percentage points at T = 52 and 3 at T = 400,
    and the smallest loss shows that two breaks lower the power in every
    configuration.

The break configurations are the lattices of the calibration grid,
kernel_numba.make_lambdas(m, 0.15, 0.15, 7): 7 for m = 1 and 15 for m = 2, with
tau_j = floor(lambda_j T). T = 52 and T = 400 are not on the grid of Table 1.

Computed exactly by the quadratic-form representation of the statistic,
following Imhof (1961) as in Dufour and King (1991) (mlb/envelope_exact.py):
no simulation and no random numbers. The half-power point is found by Brent's
method on (-25, -1) with xtol = 1e-9.

Outputs (in output/critical_values/propflat_exact/ unless --out-dir is given):
  propflat_exact_configs.csv   one row per computed configuration: cbar_1/2, or
                               the power at cbar = -7, at full precision;
  propflat_exact.csv           the summaries above, in long format
                               (part, m, T, quantity, value);
  propflat_exact.manifest.json the parameters, the input's sha256, the outputs'
                               sha256; nothing time- or machine-dependent.

--check compares the two CSV files with reference/calibration/propflat_exact/:
byte for byte (LF bytes); failing that, every number to |a - b| <= 1e-7 (1 + |b|),
because the eigenvalues, the integrals and the root search can move the last digits
between platforms (the root is located to 1e-9); fields that are not numbers must be
equal. The exit code is non-zero on any difference.

    python mlb/propflat_exact.py [--jobs N] [--out-dir D] [--check]

About 5 to 9 CPU-minutes (divide by --jobs).
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
from mlb import kernel_numba as KN  # noqa: E402
from mlb import paths  # noqa: E402

SECTION, OBJ = "critical_values", "propflat_exact"
CBAR = -7.0
M_SCALING = (1, 2)
T_SCALING = (52, 100, 200, 400)
M_POWER = 2
T_POWER = (52, 100, 200, 400)
TRIM, SPACING, N_GRID = 0.15, 0.15, 7
XTOL = 1e-9
TOL = 1e-7
LIMIT_REL = "reference/theory/limit_envelope/limit_envelope.json"
LIMIT = paths.reference_dir("supplement_appendix") / "limit_envelope" / (
    "limit_envelope.json"
)
CONFIGS_CSV = f"{OBJ}_configs.csv"
SUMMARY_CSV = f"{OBJ}.csv"
MANIFEST = f"{OBJ}.manifest.json"
REFERENCE_OUTPUTS = (SUMMARY_CSV, CONFIGS_CSV)  # the outputs --check compares
CONFIG_HEADER = ("part", "m", "T", "lambda1", "lambda2", "tau1", "tau2", "value")
SUMMARY_HEADER = ("part", "m", "T", "quantity", "value")


def lattice(m):
    """The break configurations of the calibration grid for m breaks."""
    return [
        tuple(float(x) for x in lam)
        for lam in KN.make_lambdas(m, TRIM, SPACING, N_GRID)
    ]


def cbar_half(T, lam):
    """The exact half-power point cbar_1/2(m, T, lambda)."""
    return EX.half_power_point(T, lam, xtol=XTOL)[0]


def power_m7(T, lam):
    """The exact power at cbar = -7 against the alternative c = -7."""
    return float(EX.power_at(CBAR, T, EX.design_Z(T, lam))[0])


def tasks():
    """Every exact computation, as (part, m, T, lambda)."""
    out = [
        ("cbar_half", m, T, lam)
        for m in M_SCALING
        for T in T_SCALING
        for lam in lattice(m)
    ]
    for T in T_POWER:
        out.append(("power_m7", 0, T, ()))
        out += [("power_m7", M_POWER, T, lam) for lam in lattice(M_POWER)]
    return out


def _one(part, T, lam):
    return cbar_half(T, lam) if part == "cbar_half" else power_m7(T, lam)


def compute(jobs):
    """{(part, m, T, lambda): value}; the costliest computations start first."""
    todo = sorted(tasks(), key=lambda k: (-k[2], k[0], k[1], k[3]))
    if jobs > 1:
        from joblib import Parallel, delayed

        vals = Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
            delayed(_one)(p, T, lam) for p, _, T, lam in todo
        )
    else:
        vals = [_one(p, T, lam) for p, _, T, lam in todo]
    return dict(zip(todo, vals))


def config_rows(vals):
    """The rows of propflat_exact_configs.csv, in the order of tasks()."""
    rows = []
    for key in tasks():
        part, m, T, lam = key
        taus = EX.break_positions(T, lam)
        pad = [""] * (2 - len(lam))
        rows.append(
            [part, m, T]
            + [repr(x) for x in lam]
            + pad
            + [str(t) for t in taus]
            + pad
            + [repr(vals[key])]
        )
    return rows


def read_configs(path):
    """propflat_exact_configs.csv back into {(part, m, T, lambda): value}."""
    vals = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            lam = tuple(float(r[k]) for k in ("lambda1", "lambda2") if r[k])
            vals[(r["part"], int(r["m"]), int(r["T"]), lam)] = float(r["value"])
    return vals


def limit_root():
    """cbar_inf, the root of the limit envelope at one half."""
    with open(LIMIT, encoding="utf-8") as fh:
        return float(json.load(fh)["limit"]["cbar_inf"])


def summary_rows(vals, cbar_inf):
    """The rows of propflat_exact.csv from the configuration values."""
    rows = [
        ["limit", "", "", "cbar_inf", cbar_inf],
        ["limit", "", "", "minus7_minus_cbar_inf", CBAR - cbar_inf],
    ]
    for m in M_SCALING:
        lams = lattice(m)
        for T in T_SCALING:
            c = [vals[("cbar_half", m, T, lam)] for lam in lams]
            s7 = max(abs(x - CBAR) for x in c)
            sinf = max(abs(x - cbar_inf) for x in c)
            scaled7 = T * s7 / (m + 1)
            scaled_inf = T * sinf / (m + 1)
            key = ["iii", m, T]
            rows += [
                key + ["scaled_from_m7", scaled7],
                key + ["scaled_from_cbar_inf", scaled_inf],
                key + ["scaled_change", scaled7 - scaled_inf],
                key + ["max_cbar_half", max(c)],
            ]
    lams = lattice(M_POWER)
    for T in T_POWER:
        p0 = vals[("power_m7", 0, T, ())]
        p = [vals[("power_m7", M_POWER, T, lam)] for lam in lams]
        key = ["i", M_POWER, T]
        rows += [
            key + ["gap_max", p0 - min(p)],
            key + ["gap_min", p0 - max(p)],
        ]
    return [r[:4] + [repr(float(r[4]))] for r in rows]


def _csv_bytes(header, rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _lf(data):
    return data.replace(b"\r\n", b"\n")


def _sha(data):
    return hashlib.sha256(_lf(data)).hexdigest()


def manifest(outputs):
    lams = {str(m): len(lattice(m)) for m in sorted(set(M_SCALING + (M_POWER,)))}
    man = dict(
        object=OBJ,
        command="python mlb/propflat_exact.py",
        command_note="--jobs and --out-dir do not change the outputs",
        outputs_sha256=outputs,
        inputs_sha256_lf={LIMIT_REL: _sha(open(LIMIT, "rb").read())},
        seeds="none: exact computation, no random numbers",
        parameters=dict(
            cbar_fixed=CBAR,
            m_scaling=list(M_SCALING),
            T_scaling=list(T_SCALING),
            m_power=[0, M_POWER],
            T_power=list(T_POWER),
            lattice=f"kernel_numba.make_lambdas(m, {TRIM}, {SPACING}, {N_GRID})",
            n_lambda=lams,
            half_power_bracket=[-25.0, -1.0],
            half_power_xtol=XTOL,
        ),
        conventions=EX.conventions(),
        check=dict(
            mode="LF bytes; otherwise every number to |a - b| <= tol (1 + |b|)",
            tol=TOL,
        ),
    )
    return (json.dumps(man, indent=1) + "\n").encode("utf-8")


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


def compare(new, ref_dir):
    """{file: 'identical' | 'within tolerance' | 'DIFFERENT' | 'MISSING'} over
    REFERENCE_OUTPUTS."""
    res = {}
    for name in REFERENCE_OUTPUTS:
        data = new[name]
        g = os.path.join(ref_dir, name)
        if not os.path.isfile(g):
            res[name] = "MISSING"
            continue
        ref = _lf(open(g, "rb").read())
        if _lf(data) == ref:
            res[name] = "identical"
            continue
        res[name] = "within tolerance" if _close_csv(data, ref) else "DIFFERENT"
    return res


def build(vals, cbar_inf):
    """The three output files as bytes, from the configuration values."""
    files = {
        CONFIGS_CSV: _csv_bytes(CONFIG_HEADER, config_rows(vals)),
        SUMMARY_CSV: _csv_bytes(SUMMARY_HEADER, summary_rows(vals, cbar_inf)),
    }
    files[MANIFEST] = manifest({k: _sha(v) for k, v in files.items()})
    return files


def _report(rows):
    """Print the printed quantities from the summary rows (CSV strings)."""
    get = {(r[0], r[1], r[2], r[3]): float(r[4]) for r in rows}
    print("part (iii): T sup|cbar_1/2 + 7| / (m+1)  [from cbar_inf]")
    for m in M_SCALING:
        cells = "  ".join(
            f"T={T}: {get[('iii', str(m), str(T), 'scaled_from_m7')]:.4f} "
            f"[{get[('iii', str(m), str(T), 'scaled_from_cbar_inf')]:.4f}]"
            for T in T_SCALING
        )
        print(f"  m={m}  {cells}")
    print("part (i): the loss of power at cbar = -7, c = -7, m = 0 against m = 2")
    for T in T_POWER:
        k = ("i", str(M_POWER), str(T))
        print(
            f"  T={T}: largest {get[k + ('gap_max',)]:.5f}, "
            f"smallest {get[k + ('gap_min',)]:.5f}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default=None, help="output folder")
    ap.add_argument("--jobs", type=int, default=4, help="worker processes")
    ap.add_argument(
        "--check", action="store_true", help="compare the outputs with the reference"
    )
    a = ap.parse_args(argv)
    out = a.out_dir or str(paths.out_dir(SECTION) / OBJ)
    os.makedirs(out, exist_ok=True)
    n = len(tasks())
    jobs = max(1, a.jobs)
    print(
        f"[{OBJ}] {n} exact computations: about 5 to 9 min on one core, "
        f"1.5 to 2.5 min at --jobs 4; running with --jobs {jobs}",
        flush=True,
    )
    t0 = time.time()
    vals = compute(jobs)
    files = build(vals, limit_root())
    for name, data in files.items():
        with open(os.path.join(out, name), "wb") as fh:
            fh.write(data)
    rows = list(csv.reader(io.StringIO(files[SUMMARY_CSV].decode("utf-8"))))[1:]
    _report(rows)
    print(f"[{OBJ}] wrote {out} in {time.time() - t0:.0f} s")
    if not a.check:
        return 0
    ref_dir = str(paths.reference_dir(SECTION) / OBJ)
    res = compare(files, ref_dir)
    for name, word in res.items():
        print(f"--check: {name}: {word}")
    ok = all(w in ("identical", "within tolerance") for w in res.values())
    print(f"--check: {'PASS' if ok else 'FAIL'} against {ref_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
