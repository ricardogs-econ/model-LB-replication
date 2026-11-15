"""The null quantile of MZ_t with the true long-run variance (article section 4.3).

The footnote on the value -1.98 of section 4.3 reads: simulating the null at
T = 9,600 with the true long-run variance gives -1.94 to -1.95 across m <= 3
(standard errors 0.004). This producer computes that simulation.

Design: Model LB under the null, m in {0, 1, 2, 3} level shifts at
lambda_j = j / (m + 1), i.i.d. N(0, 1) innovations, u_0 = 0, GLS detrending at
cbar = -7, R = 200,000 draws per cell; the demeaned MZ_t with s^2 = 1, the true
long-run variance of the innovations (the oracle). T = 9,600 is the footnote's;
T = 300 and 1,200 are the cells that the quick mode of the replication scripts
recomputes (the step oracle_cells of mlb/steps.py), at a fraction of the
cost.

For each cell the CSV holds the 5% quantile q_oracle and its standard error
se_oracle, sqrt(p (1 - p) / R) / f with the density f estimated by the quantile
spacing 2h / (q_{p+h} - q_{p-h}), h = 0.01, and the two quantiles of the spacing,
q_lo = q_{p-h} and q_hi = q_{p+h}. The run record holds the value
tabulated by Ng and Perron (2001), -1.98, which the footnote compares with. The
footnote reads the rows at T = 9,600: the quantiles are -1.9433, -1.9423, -1.9459
and -1.9446 for m = 0, 1, 2, 3 (mean -1.9440), with standard errors 0.0042 to
0.0044.

Engine: ytilde = M y with M = I - Z (Zt'Zt)^{-1} Zt' D, Zt = D Z, D the
quasi-difference at abar = 1 + cbar / T, and the random walk y = cumsum(e). At
T = 9,600 the dense T x T matrix M would take 737 MB, so M is applied in factored
form. Before any cell runs, the factored path is checked against the dense
`detrender`, `Lmat` and `mzt` of mlb/mzt_exact_linear.py
(T = 300, m = 2, 64 draws): the run stops unless all agree to 1e-10.

Seeds: cell (m, T) draws from numpy.random.default_rng(54020802 + 1000 T + m),
in blocks of 2,000 replications (one array of shape (T, 2000) per block), so the
output does not depend on --jobs.

Output (output/critical_values/oracle_sigma2_ladder/): oracle_sigma2_ladder.csv
and the run record oracle_sigma2_ladder.manifest.json (parameters, seeds, the sha256
of the shared core module and of the output; no time). --check compares only the CSV.

Cost on an idle machine: about 6 minutes in one process (each T = 9,600 cell about
80 seconds); with --jobs 4 about 1.5 minutes, with about 0.8 GB of memory per
worker at T = 9,600. Each cell runs in one process, so the four T = 9,600 cells bound
the wall time from below.

    python mlb/oracle_sigma2_ladder.py --jobs 4
    python mlb/oracle_sigma2_ladder.py --jobs 4 --check
    python mlb/oracle_sigma2_ladder.py --cells 0:300 3:300 --check

--cells runs only the cells named (m:T); the CSV then holds only their rows, and
--check compares them with the same rows of the reference.

--check compares the CSV with the reference byte for byte (LF); if the bytes differ
it accepts every number equal to 1e-9 (absolute and relative), because the
matrix products and the linear solves go through BLAS and LAPACK, whose order of
summation can move the last digits between builds. Exit code 1 on a difference.

BLAS threads: the reference was computed with OpenBLAS (numpy 2.3) running two or
more threads per process. With a single thread OpenBLAS takes another path for
the products Zt' D y (a sum over T), which moves the last one or two digits of
some entries; the producer therefore asks for two BLAS threads per process
unless the environment already sets OMP_NUM_THREADS, MKL_NUM_THREADS or
OPENBLAS_NUM_THREADS.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BLAS_THREADS = "2"  # see the module docstring
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, BLAS_THREADS)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

from mlb import mzt_exact_linear as core  # noqa: E402
from mlb import paths  # noqa: E402

SECTION, OBJ = "critical_values", "oracle_sigma2_ladder"
P = 0.05
H = 0.01
TABULATED = core.CV_ASY  # -1.98, Ng and Perron (2001)
R = 200_000
BLOCK = 2_000
CBAR = -7.0
M_GRID = (0, 1, 2, 3)
T_GRID = (300, 1200, 9600)
SEED0 = 54020802
VALIDATION = dict(seed=20260802, T=300, m=2, cbar=-7.0, draws=64, tolerance=1e-10)
TOL_CHECK = 1e-9
SECONDS_PER_DRAW_AND_PERIOD = 4.0e-8  # one process; a T = 9,600 cell takes ~77 s
COLUMNS = ("m", "T", "q_oracle", "se_oracle", "q_lo", "q_hi")
COMMAND = "python mlb/oracle_sigma2_ladder.py [--jobs N] [--cells m:T]"
# the output --check compares: the one with a reference file in reference/
REFERENCE_OUTPUTS = (f"{OBJ}.csv",)


def cell_seed(m, T):
    return SEED0 + 1000 * T + m


def quasi_diff(Y, a):
    """D_a Y: the first row unchanged, then Y_t - a Y_{t-1} (core.Dmat(T, a) @ Y)."""
    out = np.empty_like(Y)
    out[0] = Y[0]
    out[1:] = Y[1:] - a * Y[:-1]
    return out


def gls_detrend(Y, Z, cbar, T):
    """M @ Y in factored form: Y - Z (Zt'Zt)^{-1} Zt' D Y."""
    a = 1.0 + cbar / T
    Zt = quasi_diff(Z, a)
    G = Zt.T @ Zt
    b = np.linalg.solve(G, Zt.T @ quasi_diff(Y, a))
    return Y - Z @ b


def mzt_given_s2(Ytil, s2, T):
    """MZ_t of each column of the detrended levels Ytil (T x n) with the long-run
    variance s2 (one value per column)."""
    ss = np.sum(Ytil * Ytil, axis=0)
    yT2 = Ytil[-1] ** 2
    Sy2 = (ss - yT2) / T**2
    MZa = (yT2 / T - s2) / (2.0 * Sy2)
    return MZa * np.sqrt(Sy2 / s2)


def diff_s2(A, T):
    d = np.diff(A, axis=0)
    return np.sum(d * d, axis=0) / (T - 1)


def simulate_cell(T, m):
    """MZ_t with s^2 = 1 (the oracle) on R draws."""
    Z = core.build_Z(T, m)
    rng = np.random.default_rng(cell_seed(m, T))
    mz = np.empty(R)
    done = 0
    while done < R:
        n = min(BLOCK, R - done)
        Y = np.cumsum(rng.standard_normal((T, n)), axis=0)  # = core.Lmat(T, 0) @ E
        Ytil = gls_detrend(Y, Z, CBAR, T)
        one = np.ones(n)
        mz[done : done + n] = mzt_given_s2(Ytil, one, T)
        done += n
    return mz


def quantile_and_se(x, p=P, h=H):
    """The p-quantile, its order-statistic standard error, and the two quantiles
    q_{p-h} and q_{p+h} whose spacing estimates the density."""
    x = x[np.isfinite(x)]
    q = float(np.quantile(x, p))
    lo, hi = np.quantile(x, [p - h, p + h])
    f = (2.0 * h) / (hi - lo) if hi > lo else np.nan
    se = float(np.sqrt(p * (1 - p) / x.size) / f) if np.isfinite(f) else np.nan
    return q, se, float(lo), float(hi)


def cell_row(m, T):
    """One row of the CSV: cell (m, T) on its own seed."""
    q, se, lo, hi = quantile_and_se(simulate_cell(T, m))
    return dict(m=m, T=T, q_oracle=q, se_oracle=se, q_lo=lo, q_hi=hi)


def validate():
    """The factored path against the dense core: the random walk, the detrended
    series and the statistic with the difference-based s^2. Returns (ok, diffs)."""
    T, m, cbar = VALIDATION["T"], VALIDATION["m"], VALIDATION["cbar"]
    rng = np.random.default_rng(VALIDATION["seed"])
    E = rng.standard_normal((T, VALIDATION["draws"]))
    U_dense = core.Lmat(T, 0.0) @ E
    U = np.cumsum(E, axis=0)
    Yt_dense = core.detrender(T, m, cbar) @ U_dense
    Yt = gls_detrend(U, core.build_Z(T, m), cbar, T)
    mz_dense = core.mzt(Yt_dense, T)
    mz = mzt_given_s2(Yt, diff_s2(Yt, T), T)
    diffs = dict(
        random_walk=float(np.abs(U_dense - U).max()),
        detrended=float(np.abs(Yt_dense - Yt).max()),
        mzt=float(np.abs(mz_dense - mz).max()),
    )
    return max(diffs.values()) < VALIDATION["tolerance"], diffs


def csv_text(rows):
    lines = [",".join(COLUMNS)]
    lines += [",".join(str(r[c]) for c in COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


def read_csv_text(text):
    head, *body = text.strip("\n").split("\n")
    return head.split(","), [line.split(",") for line in body]


def same_within(a, b, tol=TOL_CHECK):
    """Two CSV texts with the same header and rows, every number equal to tol."""
    ha, ra = read_csv_text(a)
    hb, rb = read_csv_text(b)
    if ha != hb or len(ra) != len(rb):
        return False
    for x, y in zip(ra, rb):
        if len(x) != len(y):
            return False
        for u, v in zip(x, y):
            fu, fv = float(u), float(v)
            if not abs(fu - fv) <= tol * (1.0 + abs(fv)):
                return False
    return True


def sha256_lf(path):
    raw = open(path, "rb").read().replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def reference_rows(text, cells):
    """The header and the rows of the CSV text that belong to `cells`, as text."""
    lines = text.strip("\n").split("\n")
    keep = [ln for ln in lines[1:] if tuple(map(int, ln.split(",")[:2])) in cells]
    return "\n".join([lines[0]] + keep) + "\n"


def parse_cells(values):
    """'m:T' strings to grid cells in the order of the CSV (m, then T)."""
    grid = [(m, T) for m in M_GRID for T in T_GRID]
    if not values:
        return grid
    asked = set()
    for v in values:
        m, _, T = v.partition(":")
        cell = (int(m), int(T))
        if cell not in grid:
            raise ValueError(f"{v!r} is not a cell of the grid m {M_GRID}, T {T_GRID}")
        asked.add(cell)
    return [c for c in grid if c in asked]


def estimate(cells, jobs):
    cost = sorted((SECONDS_PER_DRAW_AND_PERIOD * R * T for _, T in cells), reverse=True)
    load = [0.0] * jobs
    for c in cost:
        load[load.index(min(load))] += c
    return sum(cost), max(load)


def _n_cells(cells):
    return f"{len(cells)} cell" + ("s" if len(cells) != 1 else "")


def _row_job(mt):
    m, T = mt
    t0 = time.time()
    return m, T, cell_row(m, T), time.time() - t0


def _report(m, T, r, secs):
    print(
        f"  m={m} T={T:>5}  q_oracle {r['q_oracle']:.4f} (se {r['se_oracle']:.4f})"
        f"  [{secs:.1f} s]",
        flush=True,
    )


def run(cells, jobs):
    """The rows of `cells` (in the order given); the costliest cells start first."""
    order = sorted(cells, key=lambda c: -c[1])
    rows = {}
    if jobs == 1:
        for mt in order:
            m, T, r, secs = _row_job(mt)
            rows[(m, T)] = r
            _report(m, T, r, secs)
    else:
        with ProcessPoolExecutor(max_workers=min(jobs, len(order))) as ex:
            futs = [ex.submit(_row_job, mt) for mt in order]
            for fut in as_completed(futs):
                m, T, r, secs = fut.result()
                rows[(m, T)] = r
                _report(m, T, r, secs)
    return [rows[c] for c in cells]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out-dir", default=None,
        help="default output/critical_values/oracle_sigma2_ladder/",
    )
    ap.add_argument("--jobs", type=int, default=1, help="worker processes")
    ap.add_argument(
        "--cells",
        nargs="+",
        metavar="m:T",
        default=None,
        help="only these cells of the grid (the CSV then holds only their rows)",
    )
    ap.add_argument(
        "--check", action="store_true", help="compare the CSV with the reference"
    )
    a = ap.parse_args(argv)
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")
    try:
        cells = parse_cells(a.cells)
    except ValueError as e:
        ap.error(str(e))
    out = Path(a.out_dir) if a.out_dir else paths.out_dir(SECTION) / OBJ
    out.mkdir(parents=True, exist_ok=True)

    seq, par = estimate(cells, a.jobs)
    print(
        f"[{OBJ}] {_n_cells(cells)}, R = {R:,}: about "
        f"{seq / 60:.1f} min on one core, about {par / 60:.1f} min at "
        f"--jobs {a.jobs} (on an idle machine)",
        flush=True,
    )
    ok, diffs = validate()
    print(
        "  factored path against the dense core (T = 300, m = 2): "
        + ", ".join(f"{k} {v:.1e}" for k, v in diffs.items()),
        flush=True,
    )
    if not ok:
        print(f"[{OBJ}] the factored path does not reproduce the dense core; stop")
        return 3

    rows = run(cells, a.jobs)
    text = csv_text(rows)
    path = out / f"{OBJ}.csv"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    write_manifest(path, cells)

    for r in rows:
        if r["T"] == 9600:
            print(
                f"  footnote row (oracle, T = 9,600): m={r['m']}  "
                f"q {r['q_oracle']!r}  se {r['se_oracle']!r}"
            )
    print(f"[{OBJ}] wrote {path}")

    if a.check:
        (name,) = REFERENCE_OUTPUTS
        gold = paths.reference_dir(SECTION) / OBJ / name
        if not gold.exists():
            print(f"--check: no reference at {gold}")
            return 1
        ref = open(gold, "rb").read().replace(b"\r\n", b"\n").decode("utf-8")
        ref = reference_rows(ref, cells)
        same = ref == text
        close = same or same_within(text, ref)
        word = (
            "identical to"
            if same
            else f"equal within {TOL_CHECK:g} to"
            if close
            else "DIFFERS from"
        )
        part = len(cells) < len(M_GRID) * len(T_GRID)
        scope = f" (its rows of the {_n_cells(cells)} run)" if part else ""
        print(f"--check: {word} {gold}{scope}")
        return 0 if close else 1
    return 0


def write_manifest(path, cells):
    """The manifest beside the CSV at `path`: parameters, seeds, the validation,
    the sha256 of the shared core and of the CSV (LF); nothing time-dependent."""
    core_file = Path(core.__file__)
    manifest = dict(
        output=path.name,
        output_sha256=sha256_lf(path),
        command=COMMAND,
        parameters=dict(
            R=R,
            block=BLOCK,
            cbar=CBAR,
            m=list(M_GRID),
            T=list(T_GRID),
            cells=[f"{m}:{T}" for m, T in cells],
            quantile=P,
            density_step=H,
            tabulated=TABULATED,
            break_fractions="lambda_j = j / (m + 1)",
            innovations="i.i.d. N(0, 1), u_0 = 0",
        ),
        seeds=dict(
            seed0=SEED0,
            cell="numpy.random.default_rng(seed0 + 1000 T + m)",
            draws="blocks of 2,000 replications, one (T, 2000) array per block",
        ),
        validation=dict(VALIDATION, passed=True),
        modules_sha256_lf={core_file.name: sha256_lf(core_file)},
    )
    mpath = path.with_name(f"{OBJ}.manifest.json")
    with open(mpath, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    return mpath


if __name__ == "__main__":
    sys.exit(main())
