"""Table 1 of the article (tab:cbar): the exact half-power points of Model LB.

For each break configuration (m, T, lambda), cbar_half is the root of pi_T(cbar) = 1/2,
where pi_T(cbar) is the power, on the diagonal c = cbar, of the 5% point-optimal test
L_T(cbar) = [S(abar) - S(1)] / sigma^2. Under Gaussian innovations with u_0 = 0 and
sigma^2 known, L_T is a quadratic form in the innovations with a known matrix, so its
distribution function is Imhof's (1961) integral over the eigenvalues of that form, as
in Dufour and King (1991); see mlb/envelope_exact.py. No simulation enters, and
there is no seed.

The configurations are kernel_numba.enumerate_configs(kernel_numba.DEFAULTS), the
grid of the article (Table 1 and Section 4.1): T in {45, 50, 60, 80, 100, 150, 200,
300}, m = 0, ..., 5 with T at least DEFAULTS["m_min_T"][m] (45, 45, 45, 45, 60, 80),
and break fractions on the equispaced lattices of make_lambdas; 404 configurations.

Outputs (default output/critical_values/cbar_surface_exact/):
  cbar_surface_exact.csv   one row per configuration:
      config_key, T, m, lambda1..lambda5  the configuration (kernel_numba.config_key)
      cbar_half_exact                     the half-power point
      b_T                                 the 5% null quantile of L_T at cbar_half_exact
      imhof_tol, break_convention         the integration tolerance and the break dates
      The values are exact: they carry no simulation error.
  tab_cbar_mT_exact.tex    Table 1: the mean of cbar_half_exact over the
                           configurations of each (m, T) cell, in the layout of the
                           article
  cbar_surface_exact.manifest.json   the grid, the parameters, the conventions, the
                           sha256 of the source modules (of their LF bytes) and of
                           the outputs; nothing time- or machine-dependent.
Every printed number is rounded half up (away from zero) from the full-precision
value.

--check regenerates everything and compares cbar_surface_exact.csv with the reference
(reference/calibration/cbar_surface_exact/), after normalizing line endings to
LF. The CSV is reported identical when its bytes are; otherwise every text field
must be equal and every number close:
cbar_half_exact and b_T to 1e-6 absolute (the root of pi_T = 1/2 is bracketed to
xtol = 1e-6, and the last digits of the symmetric eigensolver and of the adaptive
quadrature can differ between platforms and library versions).

A full run takes at most about 404 x 3.5 / N seconds with --jobs N (about 24
CPU-minutes, an upper bound); that estimate is printed first.

    python mlb/exact_surface_producer.py [--out-dir D] [--jobs N]
        [--check]
"""

import argparse
import csv
import hashlib
import inspect
import io
import json
import math
import os
import sys
import time
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# One BLAS thread per worker, set before numpy is imported: every configuration is a
# dense T x T eigenproblem, and multi-threaded BLAS inside concurrent workers
# oversubscribes the machine.
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
    os.environ["PYTHONPATH"] = _ROOT + (os.pathsep + _pp if _pp else "")

from mlb import envelope_exact as EX  # noqa: E402
from mlb import kernel_numba as kn  # noqa: E402  (the grid only; no simulation)
from mlb import paths  # noqa: E402

SECTION, OBJ = "critical_values", "cbar_surface_exact"
CSV_NAME = f"{OBJ}.csv"
TAB_NAME = "tab_cbar_mT_exact.tex"
MANIFEST_NAME = f"{OBJ}.manifest.json"
REFERENCE_DIR = paths.reference_dir(SECTION) / OBJ
REFERENCE_OUTPUTS = (CSV_NAME,)  # the outputs --check compares with the reference

GRID_KEYS = ("T_grid", "m_grid", "m_min_T", "trim", "min_spacing", "n_grid")
SECONDS_PER_CONFIG = 3.5  # an upper bound of the time per configuration and worker

COLUMNS = [
    "config_key",
    "T",
    "m",
    "lambda1",
    "lambda2",
    "lambda3",
    "lambda4",
    "lambda5",
    "cbar_half_exact",
    "b_T",
    "imhof_tol",
    "break_convention",
]
IMHOF_TOL = (
    f"epsabs={EX.IMHOF_EPSABS:g},epsrel={EX.IMHOF_EPSREL:g},limit={EX.IMHOF_LIMIT}"
)
_ROOT_DEFAULTS = inspect.signature(EX.half_power_point).parameters  # bracket, xtol
_QUANTILE_DEFAULTS = inspect.signature(EX.quantile).parameters  # xtol of the null cv
ABS_COLUMNS = ("cbar_half_exact", "b_T")
TOL = 1e-6


# ------------------------------------------------------------------- the grid
def grid_parameters():
    """The entries of kernel_numba.DEFAULTS that define the configurations."""
    return {k: kn.DEFAULTS[k] for k in GRID_KEYS}


def configurations():
    """[(T, m, lambdas, config_key)] in the order of kernel_numba.enumerate_configs."""
    out = []
    for T, m, lam in kn.enumerate_configs(dict(kn.DEFAULTS)):
        lam = [float(x) for x in lam]
        out.append((int(T), int(m), lam, kn.config_key(T, m, lam)))
    return out


# --------------------------------------------------------- one configuration
def one(T, m, lambdas, key):
    """The CSV row of one configuration: the exact half-power point."""
    lam = tuple(lambdas)
    cbar, b_T = EX.half_power_point(T, lam)
    row = dict(config_key=key, T=T, m=m)
    for i in range(1, 6):
        row[f"lambda{i}"] = lam[i - 1] if i <= len(lam) else ""
    row.update(
        cbar_half_exact=cbar,
        b_T=b_T,
        imhof_tol=IMHOF_TOL,
        break_convention=EX.BREAK_CONVENTION,
    )
    return row


def csv_text(rows):
    """The CSV as text: LF line endings, every float at full precision (repr)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def read_rows(path):
    """The rows of a cbar_surface_exact.csv with T, m as int and cbar as float."""
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["T"], r["m"] = int(r["T"]), int(r["m"])
        r["cbar_half_exact"] = float(r["cbar_half_exact"])
    return rows


# --------------------------------------------------------------- the tables
def fmt2(x):
    """x at two decimals, rounded half up (away from zero) from its repr."""
    d = Decimal(repr(float(x))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return "0.00" if d == 0 else f"{d}"


def num(x):
    s = fmt2(x)
    return f"${s}$" if s.startswith("-") else s


def cells(rows):
    """{(m, T): [cbar_half_exact, ...]} over the configurations."""
    out = {}
    for r in rows:
        out.setdefault((int(r["m"]), int(r["T"])), []).append(
            float(r["cbar_half_exact"])
        )
    return out


def cell_summary(rows):
    """Per (m, T) cell: n, mean, min, max, range and sd (ddof 1; None if n = 1)."""
    out = {}
    for k, v in sorted(cells(rows).items()):
        n = len(v)
        mean = math.fsum(v) / n
        sd = None
        if n > 1:
            sd = math.sqrt(math.fsum((x - mean) ** 2 for x in v) / (n - 1))
        lo, hi = min(v), max(v)
        out[k] = dict(n=n, mean=mean, min=lo, max=hi, range=hi - lo, sd=sd)
    return out


def _spread(summ):
    multi = {k: s for k, s in summ.items() if s["n"] > 1}
    lo = min(multi, key=lambda k: multi[k]["range"])
    hi = max(multi, key=lambda k: multi[k]["range"])
    mean_range = math.fsum(s["range"] for s in multi.values()) / len(multi)
    mean_sd = math.fsum(s["sd"] for s in multi.values()) / len(multi)
    return multi, lo, hi, mean_range, mean_sd


def table1_tex(rows):
    """Table 1 (tab:cbar) of the article: the (m, T) cell means."""
    summ = cell_summary(rows)
    ms = sorted({m for m, _ in summ})
    ts = sorted({t for _, t in summ})
    multi, lo, hi, _, _ = _spread(summ)
    r_lo, r_hi = fmt2(multi[lo]["range"]), fmt2(multi[hi]["range"])
    first = ",".join(str(min(t for mm, t in summ if mm == m)) for m in ms)
    nt = len(ts)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Half-power point $\\bar c_{1/2}(m,T)$ of the Gaussian power"
        " envelope",
        "for Model~LB, averaged over the break configurations of each $(m,T)$ cell.",
        "Exact computation; the entries carry no simulation error. Within a cell the",
        f"break placement moves $\\bar c_{{1/2}}$ by between ${r_lo}$ and ${r_hi}$."
        " The test",
        "fixes $\\bar c=-7$.}",
        "\\label{tab:cbar}",
        "\\begin{tabular}{l" + "r" * nt + "}",
        "\\toprule",
        f"& \\multicolumn{{{nt}}}{{c}}{{Sample size $T$}}\\\\",
        f"\\cmidrule(lr){{2-{nt + 1}}}",
        "$m$ & " + " & ".join(str(t) for t in ts) + " \\\\",
        "\\midrule",
    ]
    for m in ms:
        vals = [num(summ[(m, t)]["mean"]) if (m, t) in summ else "---" for t in ts]
        lines.append(f"{m} & " + " & ".join(vals) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\par\\medskip",
        f"\\footnotesize The grid starts at $T={first}$ for"
        f" $m={ms[0]},\\ldots,{ms[-1]}$, a",
        "design choice.",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ the check
def _lf(path):
    return open(path, "rb").read().replace(b"\r\n", b"\n")


def compare_csv(new_text, ref_path):
    """(verdict, details): 'identical', 'close' (within the tolerances) or 'differs'."""
    ref = _lf(ref_path)
    new = new_text.encode("utf-8")
    if new == ref:
        return "identical", {}
    a = list(csv.DictReader(io.StringIO(new_text)))
    b = list(csv.DictReader(io.StringIO(ref.decode("utf-8"))))
    if len(a) != len(b) or (a and list(a[0]) != list(b[0])):
        return "differs", {"shape": [len(a), len(b)]}
    worst = {c: 0.0 for c in ABS_COLUMNS}
    bad = []
    for ra, rb in zip(a, b):
        for c in COLUMNS:
            if c in worst:
                x, y = float(ra[c]), float(rb[c])
                d = abs(x - y)
                worst[c] = max(worst[c], d)
                if d > TOL:
                    bad.append((ra["config_key"], c, x, y))
            elif ra[c] != rb[c]:
                bad.append((ra["config_key"], c, ra[c], rb[c]))
    return ("differs" if bad else "close"), {"largest": worst, "failures": bad[:20]}


# ---------------------------------------------------------------------- main
def _sha_lf(path):
    return hashlib.sha256(_lf(path)).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default=None, help="default output/critical_values/"
                    "cbar_surface_exact")
    ap.add_argument("--jobs", type=int, default=4, help="worker processes")
    ap.add_argument("--check", action="store_true",
                    help="compare the outputs with the reference")
    a = ap.parse_args(argv)
    out = paths.out_dir(SECTION) / OBJ if a.out_dir is None else Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfgs = configurations()
    est = len(cfgs) * SECONDS_PER_CONFIG / max(a.jobs, 1)
    print(
        f"[{OBJ}] {len(cfgs)} configurations, --jobs {a.jobs}; estimated "
        f"{est / 60:.0f} min ({SECONDS_PER_CONFIG} s per configuration and worker)",
        flush=True,
    )
    t0 = time.time()
    from joblib import Parallel, delayed

    rows = []
    gen = Parallel(n_jobs=a.jobs, backend="loky", batch_size=1, return_as="generator")(
        delayed(one)(*c) for c in cfgs
    )
    for r in gen:
        rows.append(r)
        if len(rows) % 25 == 0 or len(rows) == len(cfgs):
            el = time.time() - t0
            eta = el / len(rows) * (len(cfgs) - len(rows))
            print(f"  [{len(rows)}/{len(cfgs)}] {el:6.0f} s, eta {eta:6.0f} s",
                  flush=True)
    keys = [r["config_key"] for r in rows]
    assert len(set(keys)) == len(rows), "config_key is not unique"

    texts = {CSV_NAME: csv_text(rows), TAB_NAME: table1_tex(rows)}
    for name, text in texts.items():
        with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    mods = {
        m: _sha_lf(os.path.join(_ROOT, "mlb", m))
        for m in ("envelope_exact.py", "kernel_numba.py")
    }
    manifest = dict(
        producer="mlb/exact_surface_producer.py",
        command="python mlb/exact_surface_producer.py",
        method="exact (Imhof), no simulation",
        seeds="none (no simulation)",
        grid=dict(source="mlb/kernel_numba.py enumerate_configs(DEFAULTS)",
                  **{k: v for k, v in grid_parameters().items() if k != "m_min_T"},
                  m_min_T={str(k): v for k, v in kn.DEFAULTS["m_min_T"].items()}),
        n_configurations=len(rows),
        table=dict(cell_statistic="mean over the configurations of each (m, T) cell",
                   rounding="two decimals, half up (away from zero)"),
        root=dict(bracket=list(_ROOT_DEFAULTS["bracket"].default),
                  xtol=_ROOT_DEFAULTS["xtol"].default, target_power=EX.TARGET_POWER,
                  null_quantile_xtol=_QUANTILE_DEFAULTS["xtol"].default),
        conventions=EX.conventions(),
        modules_sha256_lf=mods,
        outputs_sha256={n: hashlib.sha256(t.encode("utf-8")).hexdigest()
                        for n, t in texts.items()},
    )
    with open(out / MANIFEST_NAME, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    print(f"[{OBJ}] {len(rows)} configurations in {time.time() - t0:.0f} s -> {out}")

    if not a.check:
        return 0
    status = 0
    for name in REFERENCE_OUTPUTS:
        verdict, det = compare_csv(texts[name], REFERENCE_DIR / name)
        if verdict == "identical":
            print(f"--check {name}: identical to the reference")
        elif verdict == "close":
            print(f"--check {name}: equal within the tolerances; largest "
                  f"differences {det['largest']}")
        else:
            status = 1
            print(f"--check {name}: DIFFERS from the reference {det}")
    return status


if __name__ == "__main__":
    sys.exit(main())
