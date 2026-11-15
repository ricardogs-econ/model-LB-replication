#!/usr/bin/env python3
"""Figure 1 (fig:size): the true size of the nominal-5% MZ_t test at cbar = -7
with the asymptotic critical value -1.98, against T, for m = 0, 1, 2 breaks:
Figure_1.pdf, the size against T = 50, 60, 80, 100, 150, 200 for m = 0, 1, 2
(18 cells), and its data file, with the finite-sample 5% quantile of MZ_t of each
cell.

Design: Model LB under the unit-root null (c = 0) with u_0 = 0 and i.i.d.
N(0, 1) innovations; breaks at lambda_j = j/(m+1), tau_j = floor(lambda_j T);
GLS detrending at cbar = -7; the difference-based long-run variance s2_Delta;
N = 200,000 replications per cell. Every draw goes through the package's kernel
(mlb/kernel_numba.py, _stats_under with sigma2 code 0), which draws
replication r of a cell from numpy's default_rng(seed0 + r).

Seeds: the cells are in run order, m-major and T-minor; cell (m, T) has the seed
index k = 12 + 9 m + i, i the position of T in the T grid, and seed0 = seed +
k * 10**6: the indices 12-17 (m = 0), 21-26 (m = 1) and 30-35 (m = 2) of the draws
of the printed figure; no other index is used. The 18 seed intervals [seed0,
seed0 + nrep) are pairwise disjoint, and none contains the seed of the script's only
other stream, the kernel's warm-up (warm_up_numba draws once from default_rng(0));
cell_seeds() asserts both, so no
replication is shared. The first n replications of a cell do not depend on nrep.

Outputs (in output/critical_values/size_figure/ unless --out-dir is given):
  Figure_1_data.csv            one row per cell, in run order: k, m, T,
                               n, seed0, n_reject (draws below -1.98), n_nan,
                               size_pct = n_reject / n * 100 and cv_q05 (the
                               5% quantile of the n draws), floats at full
                               precision (repr), LF line ends;
  Figure_1_data.manifest.json  the design, the seeds, the parameters, the
                               command and the data file's sha256; nothing
                               time- or machine-dependent;
  Figure_1.pdf                 drawn from the rows of the data file after it is
                               written (without a creation date).
The data file does not depend on --jobs or --out-dir. The PDF is not compared:
its bytes depend on the matplotlib version.

--check regenerates the 18 cells and compares the data file with
reference/figures/Figure_1_data.csv (the only output with a reference file):
byte for byte (LF bytes); failing that, every count, seed and size exactly and
cv_q05 to |a - b| <= 1e-9 (1 + |b|), because the draws come from compiled
floating-point code whose last digits a different compiler or processor can move
(a count would change only if a draw fell within such a distance of -1.98). It
also checks that the points drawn are the sizes of the data file. The exit code
is non-zero on any difference.

--from-reference simulates nothing: it reads reference/figures/Figure_1_data.csv,
checks its structure (18 cells in run order, the seed rule, n,
size_pct = n_reject / n * 100), prints the sizes, draws Figure_1.pdf from it
and checks that the drawn points are the reference's sizes (with or without
--check). A few seconds.

Time: about 4 minutes on one core at N = 200,000 (an estimate is printed
first); --jobs runs the cells in parallel processes without changing a byte.

    python mlb/size_figure.py               # Figure 1
    python mlb/size_figure.py --jobs 4 --check
    python mlb/size_figure.py --from-reference --check
    python mlb/size_figure.py --nrep 50000   # fewer draws
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# the palette and the house rcParams are defined once, in _figstyle.py; the
# rcParams are applied just before the figure is drawn (draw()).
from mlb._figstyle import (  # noqa: E402
    ORDINAL_RAMP,
    apply_house_rcparams,
)
from mlb.mzt_exact_linear import CV_ASY  # noqa: E402
from mlb import paths  # noqa: E402

SECTION, OBJ = "critical_values", "size_figure"
SEED = 20260720
NREP = 200_000
SEED_STRIDE = 10**6
# kernel_numba.warm_up_numba draws its test series from default_rng(0)
WARMUP_SEED = 0
CBAR = -7.0
KMAX = 12
TGRID = [50, 60, 80, 100, 150, 200]
FIGURE_CELLS = [(m, T) for m in (0, 1, 2) for T in TGRID]
# the seed index of each cell, seed0 = seed + k * SEED_STRIDE: 12-17 (m = 0), 21-26
# (m = 1) and 30-35 (m = 2), the indices of the printed figure's draws
SEED_INDEX = {(m, T): 12 + 9 * m + i for m in (0, 1, 2) for i, T in enumerate(TGRID)}
FIGURE = "Figure_1.pdf"
DATA = "Figure_1_data.csv"
MANIFEST = "Figure_1_data.manifest.json"
REFERENCE_DIR = paths.reference_dir("figures")
# the outputs compared by --check: those with a file in reference/figures/
REFERENCE_OUTPUTS = (DATA,)
DATA_COLUMNS = [
    "k",
    "m",
    "T",
    "n",
    "seed0",
    "n_reject",
    "n_nan",
    "size_pct",
    "cv_q05",
]
INT_COLUMNS = ("k", "m", "T", "n", "seed0", "n_reject", "n_nan")
TOL = 1e-9
# one-core cost of a replication, a + b T microseconds (measured), for the estimate
COST_US = (44.0, 0.155)
# the size ranges printed with Figure 1 (Section 4, sec:cv), as figure cells
PRINTED = {
    "one or two breaks, T = 50-60": [(m, T) for m in (1, 2) for T in (50, 60)],
    "no break, T = 50-60": [(0, 50), (0, 60)],
    "one or two breaks, T = 200": [(1, 200), (2, 200)],
}


def run_order():
    """The 18 cells as (m, T), in run order."""
    return [(m, T) for m, T in FIGURE_CELLS]


def cell_seeds(seed, nrep):
    """seed0 of every cell, keyed (m, T).

    Cell (m, T) gets seed0 = seed + SEED_INDEX[(m, T)] * SEED_STRIDE and consumes
    the replication seeds [seed0, seed0 + nrep). The checks below are the
    disjointness itself, pair by pair (it fails when nrep exceeds SEED_STRIDE), and
    that no interval contains WARMUP_SEED, the seed of the only other stream.
    """
    seed0 = {cell: seed + SEED_INDEX[cell] * SEED_STRIDE for cell in run_order()}
    spans = [(s, s + nrep) for s in seed0.values()]
    for i, (a0, a1) in enumerate(spans):
        for b0, b1 in spans[i + 1 :]:
            if not (a1 <= b0 or b1 <= a0):
                raise AssertionError(
                    f"seed intervals [{a0}, {a1}) and [{b0}, {b1}) overlap "
                    f"(nrep {nrep:,} > stride {SEED_STRIDE:,})"
                )
    for a0, a1 in spans:
        if a0 <= WARMUP_SEED < a1:
            raise AssertionError(
                f"seed interval [{a0}, {a1}) contains the warm-up seed {WARMUP_SEED}"
            )
    return seed0


def null_mzt_kernel(T, m, cbar, nrep, seed0):
    """Null MZ_t draws through the package's kernel (kernel_numba.mstats_nb),
    replication r from default_rng(seed0 + r); seed0 from cell_seeds()."""
    from mlb import kernel_numba as kn

    kn.warm_up_numba(kmax=KMAX)
    lam = [j / (m + 1) for j in range(1, m + 1)]
    bp = kn.break_pos_from_lambdas(T, lam)
    return kn._stats_under(T, bp, 0.0, cbar, 0, nrep, seed0, KMAX)["mzt"]


def cell_row(k, m, T, seed0, mz):
    """One cell's row of the data file, from its MZ_t draws."""
    n = len(mz)
    n_reject = int(np.count_nonzero(mz < CV_ASY))
    return dict(
        k=k,
        m=m,
        T=T,
        n=n,
        seed0=seed0,
        n_reject=n_reject,
        n_nan=int(np.count_nonzero(np.isnan(mz))),
        size_pct=n_reject / n * 100,
        cv_q05=float(np.quantile(mz, 0.05)),
    )


def run_cell(k, m, T, nrep, seed0):
    """Simulate cell k and return its row."""
    mz = null_mzt_kernel(T, m, CBAR, nrep, seed0)
    return cell_row(k, m, T, seed0, mz)


def compute(nrep, seed, jobs=1):
    """The rows of the cells, in run order; --jobs does not change them (every cell
    has its own seeds)."""
    seed0 = cell_seeds(seed, nrep)
    todo = [
        (SEED_INDEX[(m, T)], m, T, nrep, seed0[(m, T)])
        for m, T in run_order()
    ]
    if jobs > 1 and len(todo) > 1:
        from joblib import Parallel, delayed

        # the costliest cells (largest T) start first
        order = sorted(todo, key=lambda t: (-t[2], t[0]))
        rows = Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
            delayed(run_cell)(*t) for t in order
        )
        return sorted(rows, key=lambda r: r["k"])
    return [run_cell(*t) for t in todo]


def estimate_s(nrep, jobs):
    """Rough wall time in seconds: one core, and at `jobs` processes."""
    cells = [T for _, T in run_order()]
    one = sum(nrep * (COST_US[0] + COST_US[1] * T) for T in cells) / 1e6
    return one, one / max(1, min(jobs, len(cells)))


def data_bytes(rows):
    """The data file: DATA_COLUMNS, one row per cell, floats as repr, LF."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(DATA_COLUMNS)
    for r in rows:
        w.writerow(
            [
                repr(float(r[c])) if c in ("size_pct", "cv_q05") else str(r[c])
                for c in DATA_COLUMNS
            ]
        )
    return buf.getvalue().encode("utf-8")


def read_data(data):
    """The rows of a data file (bytes), typed; ValueError on a wrong header."""
    rd = csv.reader(io.StringIO(_lf(data).decode("utf-8")))
    header = next(rd)
    if header != DATA_COLUMNS:
        raise ValueError(f"header {header} != {DATA_COLUMNS}")
    rows = []
    for rec in rd:
        if len(rec) != len(DATA_COLUMNS):
            raise ValueError(f"row of {len(rec)} fields: {rec}")
        r = dict(zip(DATA_COLUMNS, rec))
        for c in INT_COLUMNS:
            r[c] = int(r[c])
        r["size_pct"] = float(r["size_pct"])
        r["cv_q05"] = float(r["cv_q05"])
        rows.append(r)
    return rows


def structure_problems(rows, nrep=NREP, seed=SEED):
    """What is wrong with the rows of a complete data file (empty if nothing):
    the 18 cells in run order, the seed rule, n, the counts and
    size_pct = n_reject / n * 100."""
    out = []
    want = run_order()
    got = [(r["m"], r["T"]) for r in rows]
    if got != want:
        return [f"cells {got} are not the run order {want}"]
    seed0 = cell_seeds(seed, nrep)
    for r in rows:
        cell = (r["m"], r["T"])
        if r["k"] != SEED_INDEX[cell]:
            out.append(f"{cell}: k {r['k']} != {SEED_INDEX[cell]}")
        if r["seed0"] != seed0[cell]:
            out.append(f"{cell}: seed0 {r['seed0']} != {seed0[cell]}")
        if r["n"] != nrep:
            out.append(f"{cell}: n {r['n']} != {nrep}")
        if not (0 <= r["n_reject"] <= r["n"] and 0 <= r["n_nan"] <= r["n"]):
            out.append(f"{cell}: counts out of range")
        if r["n"] > 0 and r["size_pct"] != r["n_reject"] / r["n"] * 100:
            out.append(f"{cell}: size_pct {r['size_pct']!r} != n_reject / n * 100")
    return out


def figure_points(rows):
    """{m: (T values, sizes)} that the figure draws."""
    pts = {}
    for m in (0, 1, 2):
        cells = [r for r in rows if r["m"] == m]
        shown = [(r["T"], r["size_pct"]) for r in cells]
        pts[m] = ([T for T, _ in shown], [sz for _, sz in shown])
    return pts


def draw(rows, out_path):
    """Draw Figure 1 from the rows of the data file; return {m: (x, y)} as read
    back from the drawn curves."""
    apply_house_rcparams()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    # m is ORDINAL (break count), same axis as Fig. 2's |c-bar| -- so it indexes
    # the same sequential ORDINAL_RAMP (black -> lighter blue) rather than a
    # qualitative palette, and the SAME m gets the SAME color in every figure
    # that indexes by it.
    drawn = {}
    for m, (xs, ys) in figure_points(rows).items():
        st = ORDINAL_RAMP[m]
        (line,) = ax.plot(
            xs,
            ys,
            marker=st["marker"],
            ls=st["ls"],
            color=st["color"],
            lw=1.8,
            ms=st["ms"],
            label=f"$m={m}$",
        )
        drawn[m] = (
            [float(x) for x in line.get_xdata()],
            [float(y) for y in line.get_ydata()],
        )
    ax.axhline(5.0, color="gray", ls="--", lw=1, zorder=0)
    # the label at the left end of the dashed line, 2 pt above it, clear of the
    # curves
    ax.annotate(
        "nominal 5%",
        xy=(0.01, 5.0),
        xycoords=ax.get_yaxis_transform(),
        xytext=(0, 2),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        color="gray",
    )
    # the x ticks are the T drawn
    ax.set_xticks(TGRID)
    ax.set_xlabel("sample size $T$")
    ax.set_ylabel(r"empirical size of the $5\%$ test (%)")
    ax.set_title(
        r"Over-rejection of $\mathrm{MZ}_t$ at $\bar c=-7$ with the "
        r"asymptotic critical value ($-1.98$)",
        fontsize=9,
    )
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    # a PDF without a creation date: the same bytes on every run
    pdf_kw = (
        {"metadata": {"CreationDate": None}}
        if str(out_path).lower().endswith(".pdf")
        else {}
    )
    fig.savefig(out_path, **pdf_kw)
    plt.close(fig)
    return drawn


def drawn_problems(drawn, rows):
    """Differences between the drawn points and the data file's sizes."""
    out = []
    for m, (xs, ys) in figure_points(rows).items():
        if drawn.get(m) != ([float(x) for x in xs], [float(y) for y in ys]):
            out.append(f"m={m}: drawn {drawn.get(m)} != data {(xs, ys)}")
    return out


def print_table(rows):
    """The figure's sizes, as the script prints them."""
    for m in (0, 1, 2):
        sizes = [r["size_pct"] for r in rows if r["m"] == m]
        if sizes:
            print(f"m={m}: sizes={[round(float(s), 1) for s in sizes]}")


def printed_ranges(rows):
    """{statement: (min, max) of size_pct} over the cells of PRINTED."""
    fig = {(r["m"], r["T"]): r["size_pct"] for r in rows}
    return {
        key: (min(fig[c] for c in cells), max(fig[c] for c in cells))
        for key, cells in PRINTED.items()
    }


def _lf(data):
    return data.replace(b"\r\n", b"\n")


def _sha(data):
    return hashlib.sha256(_lf(data)).hexdigest()


def _command(nrep, seed):
    cmd = "python mlb/size_figure.py"
    if nrep != NREP:
        cmd += f" --nrep {nrep}"
    if seed != SEED:
        cmd += f" --seed {seed}"
    return cmd


def manifest_bytes(nrep, seed, data_sha):
    """The manifest of the data file."""
    man = dict(
        object=OBJ,
        figure=f"{FIGURE} (fig:size)",
        command=_command(nrep, seed),
        command_note="--jobs and --out-dir do not change the data file",
        what=(
            "the Monte Carlo record of Figure 1: the size of the nominal-5% MZ_t test "
            "at cbar = -7 with the asymptotic critical value -1.98 under the "
            "unit-root null"
        ),
        design=dict(
            null="Model LB, c = 0, u_0 = 0, i.i.d. N(0, 1) innovations",
            breaks="lambda_j = j/(m+1), tau_j = floor(lambda_j T)",
            detrending_cbar=CBAR,
            long_run_variance="difference-based s2_Delta (kernel_numba sigma2 code 0)",
            critical_value=CV_ASY,
            kernel="mlb/kernel_numba.py (_stats_under, mstats_nb)",
        ),
        cells=(
            f"{len(FIGURE_CELLS)} cells: m in {{0, 1, 2}}, "
            f"T in {{{', '.join(str(T) for T in TGRID)}}}"
        ),
        columns=DATA_COLUMNS,
        n_rows=len(run_order()),
        n_replications_per_cell=nrep,
        seeds=dict(
            seed=seed,
            stride=SEED_STRIDE,
            rule=(
                "cell (m, T) has the seed index k = 12 + 9 m + i, i the position of T "
                "in the T grid (the indices 12-17, 21-26 and 30-35), and seed0 = "
                "seed + k * stride; replication r of a cell is drawn from numpy "
                "default_rng(seed0 + r)"
            ),
            warm_up_seed=WARMUP_SEED,
        ),
        size_pct="n_reject / n * 100 in floating point, written with repr",
        cv_q05="numpy.quantile of the n draws at 0.05 (linear), written with repr",
        inputs="none: a self-contained simulation",
        outputs_sha256_lf={DATA: data_sha},
        check=dict(
            mode=(
                "LF bytes; otherwise every count, seed and size exactly and cv_q05 "
                "to |a - b| <= tol (1 + |b|)"
            ),
            tol=TOL,
        ),
    )
    return (json.dumps(man, indent=1) + "\n").encode("utf-8")


def compare_data(new, ref):
    """('identical' | 'within tolerance' | 'DIFFERENT', [differences])."""
    if _lf(new) == _lf(ref):
        return "identical", []
    try:
        a, b = read_data(new), read_data(ref)
    except ValueError as e:
        return "DIFFERENT", [str(e)]
    if len(a) != len(b):
        return "DIFFERENT", [f"{len(a)} rows against {len(b)}"]
    diffs = []
    for ra, rb in zip(a, b):
        for c in DATA_COLUMNS:
            x, y = ra[c], rb[c]
            same = abs(x - y) <= TOL * (1.0 + abs(y)) if c == "cv_q05" else x == y
            if not same:
                diffs.append(f"k={rb['k']} {c}: {x!r} against {y!r}")
    return ("within tolerance" if not diffs else "DIFFERENT"), diffs


def _paths(a):
    """(folder, figure, data file, manifest) of a run."""
    out_dir = a.out_dir or str(paths.out_dir(SECTION) / OBJ)
    fig = os.path.join(out_dir, FIGURE)
    data = os.path.join(out_dir, DATA)
    man = os.path.join(out_dir, MANIFEST)
    return out_dir, fig, data, man


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _read_reference(name):
    """The reference file's bytes, or None if it is missing."""
    path = REFERENCE_DIR / name
    return _read(path) if path.is_file() else None


def _write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)


def _report_printed(rows):
    print("sizes printed with Figure 1 (min to max of size_pct):")
    for key, (lo, hi) in printed_ranges(rows).items():
        print(f"  {key}: {lo!r} to {hi!r} %")


def _from_reference(a, fig):
    ref = _read_reference(DATA)
    if ref is None:
        print(f"--from-reference: {REFERENCE_DIR / DATA}: MISSING")
        return 1
    rows = read_data(ref)
    problems = structure_problems(rows)
    print_table(rows)
    _report_printed(rows)
    drawn = draw(rows, fig)
    print(f"saved {fig} (drawn from {REFERENCE_DIR / DATA})")
    problems += drawn_problems(drawn, rows)
    n_pts = sum(len(x) for x, _ in drawn.values())
    for p in problems:
        print(f"--from-reference: {p}")
    ok = not problems
    print(
        f"--from-reference: {len(rows)} cells, {n_pts} points drawn: "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--nrep", type=int, default=NREP, help="Monte Carlo replications per cell"
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"RNG seed (default {SEED}, the seed of the printed figure; changing "
        "it changes the printed numbers)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output folder (default output/critical_values/size_figure/)",
    )
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    ap.add_argument(
        "--from-reference",
        action="store_true",
        help="draw the figure from the reference data file, without simulating",
    )
    ap.add_argument(
        "--check", action="store_true", help="compare the outputs with the reference"
    )
    a = ap.parse_args(argv)
    if (a.check or a.from_reference) and (a.nrep, a.seed) != (NREP, SEED):
        ap.error(f"the reference is at --nrep {NREP} --seed {SEED}")
    jobs = max(1, a.jobs)
    out_dir, fig, data_path, man_path = _paths(a)
    os.makedirs(out_dir, exist_ok=True)
    if a.from_reference:
        return _from_reference(a, fig)

    one, par = estimate_s(a.nrep, jobs)
    print(
        f"[{OBJ}] {len(run_order())} cells x "
        f"{a.nrep:,} replications through the kernel: about "
        f"{one / 60:.1f} min on one core"
        + (f", about {par / 60:.1f} min at --jobs {jobs}" if jobs > 1 else "")
        + " (estimate)",
        flush=True,
    )
    print(
        f"kernel: mlb.kernel_numba, cell k seed0="
        f"{a.seed}+k*{SEED_STRIDE}",
        flush=True,
    )
    t0 = time.time()
    rows = compute(a.nrep, a.seed, jobs)

    data = data_bytes(rows)
    man = manifest_bytes(a.nrep, a.seed, _sha(data))
    _write(data_path, data)
    _write(man_path, man)
    # the figure is drawn from the data file as written
    back = read_data(_read(data_path))
    print_table(back)
    _report_printed(back)
    drawn = draw(back, fig)
    print(f"saved {fig}")
    problems = drawn_problems(drawn, back)
    problems += structure_problems(back, a.nrep, a.seed)
    for p in problems:
        print(f"[{OBJ}] {p}")
    print(f"[{OBJ}] wrote {data_path} and {man_path} in {time.time() - t0:.0f} s")
    if not a.check:
        return 0 if not problems else 1
    new = {DATA: data}
    ok = not problems
    for name in REFERENCE_OUTPUTS:
        ref = _read_reference(name)
        word, diffs = ("MISSING", []) if ref is None else compare_data(new[name], ref)
        print(f"--check: {name}: {word}")
        for d in diffs[:20]:
            print(f"  {d}")
        ok = ok and word in ("identical", "within tolerance")
    print(f"--check: {'PASS' if ok else 'FAIL'} against {REFERENCE_DIR}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
