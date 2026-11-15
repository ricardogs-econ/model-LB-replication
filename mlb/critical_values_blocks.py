"""Block critical values of the M-class statistics (tab:cv and its full table).

Produces, for the 46 cells (m, T) of the reference, the 1, 2.5, 5 and 10 percent
points of the five statistics MZa, MSB, MZt, PT and MPT at cbar = -7 with the
difference-based long-run variance (the kernel's const arm), in K = 20 blocks of
R_block = 165,000 draws (3,300,000 draws per cell); lambda_j = j/(m+1). The cells
are the 43 that Table 2 (tab:cv) prints, T in {45, 50, 60, 80, 100, 150, 200, 300}
for m = 0..3, T >= 60 for m = 4 and T >= 80 for m = 5, and T = 30 for m = 0, 1, 2,
the cells of Table S1's agreement checks that Table 2 does not print. Each label
'<stat>_<q>%' of each cell carries the 20 block quantiles, their mean and the
standard error of that mean between blocks. The table of all 5 x 46 x 4 means is
the file tabcv_full_percentiles.csv.

RESULT AND COST. The reference

    reference/calibration/critical_values_blocks/critical_values_blocks.json

is the result of the full run (K = 20, R_block = 165,000): about 80 CPU-minutes,
about 4 per block (divide by --jobs). Nothing in the package needs that run
repeated; the sub-runs below recompute any block of any cell, and --check
compares it with the reference bit for bit. The arithmetic checks that start from
the reference (--aggregate, --table) take under 1 s. The per-block caches are not
shipped: they equal the 'blocks' arrays of the reference.

THE STREAM, stated exactly so that a reader can restate it.

  T_max = 300, the largest sample size of the table.
  Block b, replication r:  eps = default_rng((22240601 + b * 10**9 + r)
                                             % (2**63 - 1)).standard_normal(300)
  A cell of sample size T uses the PREFIX eps[:T], never a subsample, so that a
  T = 30 path is the first thirty steps of the T = 300 path. Inside a block every
  cell (every m, every T) consumes the SAME stream: complete
  common random numbers, drawn once per replication. Cells with the same T and
  different m receive literally the same innovations; this is exact, because the
  statistics are exactly invariant to the break magnitudes (supplement S2.1,
  rem:invariance): the draws are those of zero magnitudes, and only the design
  matrix changes with m. Between blocks the seeds are 10**9 apart, so
  the blocks are disjoint streams. A cell's block quantile is np.percentile of
  its rows with a nondegenerate draw. Per label, the mean over the blocks is
  float(np.mean(blocks)) and the standard error between blocks is
  float(np.std(blocks, ddof=1) / K ** 0.5). The arithmetic is numpy's; Python's
  statistics module does not give the same bits.

  The seed is the seed of the replication, so the result of a cell-block does not
  depend on --jobs, on --chunk, or on which other cells are computed with it.

USAGE (from the package root; output goes to output/critical_values/<name>/)

    python mlb/critical_values_blocks.py --dry-run
    python mlb/critical_values_blocks.py --blocks 20 \\
        --r-block 165000 --jobs 8                   # the full run, about 10 minutes
    python mlb/critical_values_blocks.py --only-blocks 19 \\
        --r-block 165000 --jobs 8 --check           # one block
    python mlb/critical_values_blocks.py --aggregate \\
        --from-reference --table --check            # arithmetic only, seconds

OPTIONS
  --blocks K        number of blocks (default 20, at most 1000)
  --r-block R       draws per cell and block (default 165000)
  --jobs N          worker processes (default: cpu count - 1); 1 runs in-process
  --chunk C         replications per task (default 2500; no effect on results)
  --only-blocks L   compute only these blocks: '19', '0,19', '0-3' (default all)
  --cells L         comma-separated cell keys, e.g. m0_T300_const,m2_T150_const
  --name NAME       output folder name under output/critical_values/
  --aggregate       no simulation: rebuild the summary document from the K block
                    files of the output folder, or with --from-reference from the
                    'blocks' arrays of the reference, bit for bit
  --table           no simulation: write tabcv_full_percentiles.csv
                    (round(mean, 6)) from the summary of the output folder, or
                    with --from-reference from the reference
  --from-reference  read the reference instead of the output folder
  --check           compare with the reference: a run or a sub-run, block quantile by
                    block quantile (equality, as numbers); an aggregate, the
                    values and, for the whole document, the bytes; a table, the
                    bytes of the shipped table
  --dry-run         the stream guard only; nothing is drawn

A run writes block_NNN.json (the quantiles of each cell in the block, with n_ok,
the number of nondegenerate draws), cache_meta.json (the parameters the blocks
belong to), the manifest with the library versions, and, when all K blocks of
the selected cells exist, critical_values_blocks.json. A block file already in
the folder is reused when its parameters agree, so a stopped run resumes; blocks
of other parameters are refused. No timing is written to any JSON document.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import re
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_pp = os.environ.get("PYTHONPATH", "")
if _ROOT not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _ROOT + (os.pathsep + _pp if _pp else "")

import numpy as np  # noqa: E402

from mlb import kernel_numba as kn  # noqa: E402
from mlb import paths  # noqa: E402

SECTION = "critical_values"
OBJ = "critical_values_blocks"
REFERENCE_JSON = paths.reference_dir(SECTION) / OBJ / f"{OBJ}.json"

STRIDE = 10 ** 9
SEED_BASE = 22_240_601  # the seed base of the stream
SEED_MOD = 2 ** 63 - 1
T_GRID = [30, 45, 50, 60, 80, 100, 150, 200, 300]
M_GRID = [0, 1, 2, 3, 4, 5]
T_FROM = {0: 45, 1: 45, 2: 45, 3: 45, 4: 60, 5: 80}  # the smallest T Table 2 prints
T_CHECK, M_CHECK = 30, (0, 1, 2)  # Table S1's agreement cells not printed in Table 2
T_MAX = max(T_GRID)
CBAR = -7.0
KMAX = 12  # the lag bound mstats_nb takes; the const arm (code 0) does not read it
STATS = ["mza", "msb", "mzt", "pt", "mpt"]
QS = [1.0, 2.5, 5.0, 10.0]
K_BLOCKS = 20
R_BLOCK = 165_000
CHUNK = 2_500
MAX_BLOCKS = 1000

# The three descriptive strings of the stored document.
DESIGN = "K blocks, full CRN within each block"
TRUNCATION = "prefix eps[:T], not a subsample"
SAME_T = ("the identical stream (the statistics are invariant to the break magnitudes, "
          "supplement S2.1)")

# The extraction of the table: four comment lines, then the rows, the statistics
# in this order, m ascending, T ascending.
TABLE_FILE = "tabcv_full_percentiles.csv"
TABLE_STATS = ("mpt", "msb", "mza", "mzt", "pt")
TABLE_COLS = (("cv1", "1.0%"), ("cv2_5", "2.5%"), ("cv5", "5.0%"), ("cv10", "10.0%"))
TABLE_HEAD = [
    "# Table 2 (tab:cv) in full: the 1%, 2.5%, 5% and 10% points of P_T, MZa, MSB,",
    "# MZt and MPT at c-bar = -7, difference-based s^2, by (m, T): the 43 cells",
    "# Table 2 prints and T = 30 for m = 0, 1, 2. The means over the 20 blocks of",
    "# 165,000 replications of critical_values_blocks.py; Table 2 prints mzt cv5.",
]


# ================================================================ the cells
def cells():
    """(m, T, lambdas) for every cell of the reference, equal-lambda mode.

    lambda_j = j / (m + 1), as the caption of tab:cv states. The 43 cells that
    Table 2 prints (T >= T_FROM[m]) and T = 30 for m = 0, 1, 2 (Table S1's
    agreement checks).
    """
    return [(m, T, tuple(j / (m + 1) for j in range(1, m + 1)))
            for m in M_GRID for T in T_GRID
            if T >= T_FROM[m] or (T == T_CHECK and m in M_CHECK)]


def cell_key(m, T):
    """The key of a cell in the reference: m<m>_T<T>_const (the const arm)."""
    return f"m{m}_T{T}_const"


def cell_seed(base, m, T, lam):
    """The seed a cell draws from, in the block whose first seed is `base`.

    It does not depend on (m, T, lam): every cell of a block consumes the same
    stream. It is a function, so that the guard and the producer read the same
    rule; a guard that rebuilt the rule for itself could not fail when the
    producer changed.
    """
    return base


def block_base(b):
    return SEED_BASE + b * STRIDE


def select_specs(cell_keys=None):
    """The (m, T, lambdas) to compute, in the order of the table."""
    allspecs = cells()
    valid = {cell_key(m, T) for (m, T, _lam) in allspecs}
    if cell_keys:
        bad = [k for k in cell_keys if k not in valid]
        if bad:
            raise SystemExit(f"unknown cell(s) {bad}; a key reads "
                             f"m<m>_T<T>_const, for example m0_T300_const")
    keep = set(cell_keys) if cell_keys else None
    out = [s for s in allspecs
           if (keep is None or cell_key(s[0], s[1]) in keep)]
    if not out:
        raise SystemExit("no cell selected")
    return out


# ============================================================== the producer
def _chunk(b_seed, r0, r1, specs):
    """Statistics for replications [r0, r1) at every cell, one shared stream.

    Returns {(m, T, lam): (n, 5) array}. The innovations are drawn once
    per replication and reused by every cell; the prefix eps[:T] is what a cell
    of size T receives. Rows with a degenerate draw are NaN.
    """
    from mlb import kernel_numba as kern

    n = r1 - r0
    out = {k: np.full((n, len(STATS)), np.nan) for k in specs}
    Zs = {}
    for (m, T, lam) in specs:
        if (m, T, lam) not in Zs:
            bp = kern.break_pos_from_lambdas(T, list(lam))
            Zs[(m, T, lam)] = kern.build_z_nb(T, bp)
    code = kern._METHOD_CODE["const"]
    # Every cell of the block draws the same stream. That is asserted here, once,
    # against the producer's own rule, and it is what licenses drawing the
    # innovations once per replication instead of once per cell. If the rule
    # stops being cell-independent this raises rather than silently producing a
    # different experiment.
    seeds = {cell_seed(b_seed, m, T, lam) for (m, T, lam) in specs}
    if len(seeds) != 1:
        raise AssertionError(
            f"{len(seeds)} distinct seeds inside one block; cell_seed must not "
            f"depend on (m, T, lambda)")
    seed = seeds.pop()
    for j in range(n):
        rng = np.random.default_rng((seed + r0 + j) % SEED_MOD)
        eps_full = rng.standard_normal(T_MAX)
        for (m, T, lam) in specs:
            # zero break magnitudes (the statistics are invariant to them,
            # supplement S2.1): the null path is the driftless walk of eps[:T]
            y = kern.gen_dgp_nb(T, 0.0, eps_full[:T])
            mza, msb, mzt, pt, mpt, ok = kern.mstats_nb(
                y, Zs[(m, T, lam)], CBAR, code, KMAX)
            if ok > 0.5:
                out[(m, T, lam)][j] = (mza, msb, mzt, pt, mpt)
    return out


def guard(blocks, r_block):
    """The stream guard: complete sharing inside a block, no sharing between.

    Every cell of a block must draw the same seed sequence, two blocks must
    share no seed at all, and no seed may reach the modulus of the generator
    seed (which would fold two streams onto one).
    """
    ok, msgs = True, []
    spec = cells()
    # 1. inside a block every cell must resolve to the identical stream. The
    #    seed is taken from cell_seed, the function the producer calls.
    for b in (0, blocks - 1):
        base = block_base(b)
        seeds = {(m, T, lam): cell_seed(base, m, T, lam) for (m, T, lam) in spec}
        distinct = sorted(set(seeds.values()))
        if len(distinct) != 1:
            ok = False
            msgs.append(f"[x] block {b}: cells do NOT share one stream; "
                        f"{len(distinct)} distinct seed bases, spanning "
                        f"{distinct[-1] - distinct[0]:,}")
    # 2. blocks must not overlap: [base_b, base_b + r_block) disjoint
    bases = sorted(cell_seed(block_base(b), 0, T_GRID[0], ())
                   for b in range(blocks))
    gap = min(y - x for x, y in zip(bases, bases[1:])) if blocks > 1 else 1 << 62
    if gap < r_block:
        ok = False
        msgs.append(f"[x] blocks overlap: gap {gap:,} < R_block {r_block:,}")
    # 3. no seed reaches the modulus
    top = bases[-1] + r_block
    if top >= SEED_MOD:
        ok = False
        msgs.append(f"[x] the largest seed {top:,} reaches the modulus {SEED_MOD:,}")
    for msg in msgs:
        print(msg)
    if ok:
        slack = f"{gap - r_block:,}" if blocks > 1 else "unbounded"
        print(f"  [guard] {blocks} blocks x {len(spec)} "
              f"cell-runs: sharing complete within each block, blocks "
              f"disjoint with slack {slack} replications")
    return ok


def compute_block(b, r_block, specs, jobs=1, chunk=CHUNK):
    """The quantiles of block b for the given cells.

    Returns {cell key: {"n_ok": int, "q": {stat: {"<q>%": float}}}}. With
    jobs = 1 the chunks run in this process; with more, in worker processes.
    The result does not depend on jobs or chunk.
    """
    base = block_base(b)
    edges = list(range(0, r_block, chunk)) + [r_block]
    pairs = list(zip(edges[:-1], edges[1:]))
    if jobs <= 1:
        parts = [_chunk(base, x, y, specs) for x, y in pairs]
    else:
        from joblib import Parallel, delayed
        parts = Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
            delayed(_chunk)(base, x, y, specs) for x, y in pairs)
    res = {}
    for k in specs:
        arr = np.concatenate([p[k] for p in parts], axis=0)
        good = ~np.isnan(arr[:, 0])
        m, T, _lam = k
        key = cell_key(m, T)
        res[key] = {"n_ok": int(good.sum()), "q": {}}
        for si, s in enumerate(STATS):
            col = arr[good, si]
            res[key]["q"][s] = {f"{q}%": float(np.percentile(col, q)) for q in QS}
    return res


# ======================================================== summary and table
def summarize(per_block):
    """Per cell and label: the mean over the blocks, the standard error of that
    mean between blocks, and the block quantiles. per_block is {b: {key: ...}}
    in the layout of compute_block; the order of the keys is sorted."""
    bs = sorted(per_block)
    summary = {}
    for key in sorted(per_block[bs[0]]):
        vals = {}
        for s in STATS:
            for q in QS:
                lab = f"{s}_{q}%"
                xs = [per_block[b][key]["q"][s][f"{q}%"] for b in bs]
                mu = float(np.mean(xs))
                se = float(np.std(xs, ddof=1) / len(xs) ** 0.5)
                vals[lab] = {"mean": mu, "se_between_blocks": se, "blocks": xs}
        summary[key] = vals
    return summary


def blocks_from_summary(summary):
    """The inverse of summarize on the 'blocks' arrays: {b: {key: layout}}."""
    k = len(next(iter(next(iter(summary.values())).values()))["blocks"])
    return {b: {key: {"n_ok": None,
                      "q": {s: {f"{q}%": v[f"{s}_{q}%"]["blocks"][b] for q in QS}
                            for s in STATS}}
                for key, v in summary.items()}
            for b in range(k)}


def document(summary, blocks, r_block):
    return {"design": DESIGN, "K": blocks, "R_block": r_block,
            "R_total": blocks * r_block, "stride": STRIDE,
            "seed_base": SEED_BASE, "T_max": T_MAX, "truncation": TRUNCATION,
            "same_T_different_m": SAME_T, "cbar": CBAR,
            "stats": STATS, "quantiles": QS, "cells": summary}


def document_bytes(doc):
    """The bytes of a summary document: indent 1, LF, no final newline."""
    return json.dumps(doc, indent=1).encode("utf-8")


def table_bytes(summary):
    """The extraction table: four comment lines, the header, one row per
    (statistic, m, T) with the mean rounded to 6 places."""
    mT = sorted({(int(k.split("_")[0][1:]), int(k.split("_")[1][1:]))
                 for k in summary})
    need = len(cells())
    if len(mT) != need:
        raise SystemExit(f"the table needs the {need} cells of the reference; the "
                         f"summary has {len(mT)}")
    rows = []
    for st in TABLE_STATS:
        for m, T in mT:
            c = summary[cell_key(m, T)]
            rows.append([st, m, T]
                        + [round(float(c[f"{st}_{q}"]["mean"]), 6)
                           for _, q in TABLE_COLS])
    buf = io.StringIO()
    buf.write("\n".join(TABLE_HEAD) + "\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["stat", "m", "T"] + [c for c, _ in TABLE_COLS])
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


# ================================================================ the checks
def load_reference():
    if not REFERENCE_JSON.exists():
        raise SystemExit(f"the reference {REFERENCE_JSON} is not in this tree")
    return json.loads(REFERENCE_JSON.read_bytes().decode("utf-8"))


def compare_blocks(per_block, reference, r_block):
    """Block quantile by block quantile against the reference's 'blocks' arrays.

    Returns the counts and the list of differences; equality is ==."""
    cells_r = reference["cells"]
    compared = equal = n_ok_full = n_cells = beyond = 0
    diffs = []
    for b, res in sorted(per_block.items()):
        if b >= reference["K"]:
            beyond += 1  # the reference has no block b
            continue
        for key, v in res.items():
            n_cells += 1
            if v.get("n_ok") == r_block:
                n_ok_full += 1
            for s in STATS:
                for q in QS:
                    got = v["q"][s][f"{q}%"]
                    want = cells_r[key][f"{s}_{q}%"]["blocks"][b]
                    compared += 1
                    if got == want:
                        equal += 1
                    else:
                        diffs.append((b, key, f"{s}_{q}%", got, want))
    return dict(compared=compared, equal=equal, cell_blocks=n_cells,
                n_ok_equals_r_block=n_ok_full, blocks_beyond_reference=beyond,
                differences=diffs)


def compare_summary(summary, reference):
    """Mean, standard error and blocks of every label against the reference's."""
    cells_r = reference["cells"]
    tot = mean_eq = se_eq = blocks_eq = 0
    for key, vals in summary.items():
        for lab, v in vals.items():
            r = cells_r[key][lab]
            tot += 1
            mean_eq += v["mean"] == r["mean"]
            se_eq += v["se_between_blocks"] == r["se_between_blocks"]
            blocks_eq += v["blocks"] == r["blocks"]
    return dict(labels=tot, mean_equal=mean_eq, se_equal=se_eq,
                blocks_equal=blocks_eq)


# =================================================================== files
def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)


def _meta(r_block):
    return {"r_block": r_block, "seed_base": SEED_BASE, "stride": STRIDE,
            "t_max": T_MAX, "cbar": CBAR, "stats": STATS,
            "quantiles": QS}


def check_cache(out, r_block, create):
    """A folder holds the blocks of ONE set of parameters; refuse any other."""
    f = out / "cache_meta.json"
    want = _meta(r_block)
    if f.exists():
        got = json.loads(f.read_bytes().decode("utf-8"))
        if got != want:
            raise SystemExit(
                f"{out} holds blocks of other parameters ({got}); use --name "
                f"for another folder")
    elif any(out.glob("block_*.json")):
        raise SystemExit(f"{out} holds block files without cache_meta.json")
    elif create:
        _write(f, (json.dumps(want, indent=1) + "\n").encode("utf-8"))
    else:
        raise SystemExit(f"{out} has no cache_meta.json: no blocks to read")


def _kernel_sha_lf():
    raw = open(os.path.join(_ROOT, "mlb", "kernel_numba.py"), "rb").read()
    return _sha(raw.replace(b"\r\n", b"\n"))


def _versions():
    import numba
    nb = numba.__version__
    return dict(python=platform.python_version(), numpy=np.__version__, numba=nb)


def _manifest(out, name, **info):
    man = dict(info, **_versions(), kernel_numba_sha256_lf=_kernel_sha_lf())
    _write(out / name, (json.dumps(man, indent=1) + "\n").encode("utf-8"))


def parse_blocks(text, blocks):
    """'19', '0,19', '0-3', '0-3,19' -> sorted block numbers inside range(blocks)."""
    got = set()
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        try:
            lo_i = int(lo)
            hi_i = int(hi) if hi else lo_i
        except ValueError:
            raise SystemExit(f"--only-blocks: cannot read {part!r}")
        got.update(range(lo_i, hi_i + 1))
    if not got or min(got) < 0 or max(got) >= blocks:
        raise SystemExit(f"--only-blocks must lie in 0..{blocks - 1}; got {text!r}")
    return sorted(got)


def _positive(text):
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return v


def _blocks_arg(text):
    v = _positive(text)
    if v > MAX_BLOCKS:
        raise argparse.ArgumentTypeError(f"at most {MAX_BLOCKS}")
    return v


def _name_arg(text):
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", text):
        raise argparse.ArgumentTypeError(
            "a plain folder name: letters, digits, '_', '-' and '.'")
    return text


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", type=_blocks_arg, default=K_BLOCKS)
    ap.add_argument("--r-block", type=_positive, default=R_BLOCK)
    ap.add_argument("--jobs", type=_positive,
                    default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--chunk", type=_positive, default=CHUNK)
    ap.add_argument("--only-blocks", default=None)
    ap.add_argument("--cells", default=None)
    ap.add_argument("--name", type=_name_arg, default=OBJ)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--from-reference", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    if a.from_reference and not (a.aggregate or a.table):
        ap.error("--from-reference goes with --aggregate and/or --table")
    return a


# ================================================================ the modes
def _simulate(a, out, specs, only):
    kn.warm_up_numba(kmax=KMAX)
    check_cache(out, a.r_block, create=True)
    t0 = time.time()
    per_block = {}
    wanted = {cell_key(s[0], s[1]) for s in specs}
    for b in only:
        f = out / f"block_{b:03d}.json"
        have = json.loads(f.read_bytes().decode("utf-8")) if f.exists() else {}
        todo = [s for s in specs if cell_key(s[0], s[1]) not in have]
        if not todo:
            print(f"  block {b}: from cache", flush=True)
        else:
            have.update(compute_block(b, a.r_block, todo, a.jobs, a.chunk))
            tmp = out / f"block_{b:03d}.json.tmp"
            _write(tmp, json.dumps(dict(sorted(have.items())), indent=1)
                   .encode("utf-8"))
            os.replace(tmp, f)
            print(f"  block {b}: {len(todo)} cell-runs [{time.time() - t0:.0f}s]",
                  flush=True)
        per_block[b] = {k: have[k] for k in sorted(have) if k in wanted}
    complete = only == list(range(a.blocks))
    data, files = None, {}
    if complete:
        doc = document(summarize(per_block), a.blocks, a.r_block)
        data = document_bytes(doc)
        _write(out / f"{OBJ}.json", data)
        sha = files[f"{OBJ}.json"] = _sha(data)
        print(f"wrote {OBJ}.json ({len(doc['cells'])} cells, sha256 {sha[:12]})")
    else:
        print(f"{len(only)} of {a.blocks} blocks: no summary written; --aggregate "
              f"builds it once all {a.blocks} block files are in the folder")
    rc = 0
    if a.check:
        rc = _check_run(per_block, a.r_block, data if complete else None)
    _manifest(out, f"{OBJ}.manifest.json", mode="simulate", complete=complete,
              blocks=a.blocks, blocks_run=only, r_block=a.r_block,
              cells=[cell_key(s[0], s[1]) for s in specs],
              output_sha256=files)
    return rc


def _check_run(per_block, r_block, data):
    reference = load_reference()
    if r_block != reference["R_block"]:
        print(f"--check: R_block {r_block:,} is not the reference's "
              f"{reference['R_block']:,}; the blocks are not comparable")
        return 1
    c = compare_blocks(per_block, reference, r_block)
    print(f"--check: {c['equal']}/{c['compared']} block quantiles equal the "
          f"reference's; n_ok = R_block in {c['n_ok_equals_r_block']}/"
          f"{c['cell_blocks']} cell-blocks")
    if c["blocks_beyond_reference"]:
        print(f"  {c['blocks_beyond_reference']} block(s) have no counterpart in "
              f"the reference and were not compared")
    for d in c["differences"][:10]:
        print("  differs:", d)
    ok = c["equal"] == c["compared"]
    if data is not None and set(per_block[0]) == set(reference["cells"]):
        same = data == REFERENCE_JSON.read_bytes()
        print(f"--check: the summary document is "
              f"{'byte-identical to' if same else 'DIFFERENT from'} the reference")
        ok = ok and same
    return 0 if ok else 1


def _post(a, out, specs_given):
    """--aggregate and/or --table: no simulation."""
    rc, files = 0, {}
    reference = load_reference() if (a.from_reference or a.check) else None
    summary = None
    if a.aggregate:
        if a.from_reference:
            per_block = blocks_from_summary(reference["cells"])
            blocks, r_block = reference["K"], reference["R_block"]
        else:
            blocks, r_block = a.blocks, a.r_block
            check_cache(out, r_block, create=False)
            per_block = {}
            for b in range(blocks):
                f = out / f"block_{b:03d}.json"
                if not f.exists():
                    raise SystemExit(f"--aggregate: {f.name} is missing; "
                                     f"{blocks} blocks are needed")
                per_block[b] = json.loads(f.read_bytes().decode("utf-8"))
            keys = set.intersection(*(set(v) for v in per_block.values()))
            if specs_given:
                keys &= {cell_key(s[0], s[1]) for s in specs_given}
            if not keys:
                raise SystemExit("--aggregate: no cell is in every block file")
            per_block = {b: {k: v[k] for k in sorted(keys)}
                         for b, v in per_block.items()}
        summary = summarize(per_block)
        doc = document(summary, blocks, r_block)
        data = document_bytes(doc)
        _write(out / f"{OBJ}.json", data)
        files[f"{OBJ}.json"] = _sha(data)
        print(f"--aggregate: {len(summary)} cells, {blocks} blocks, "
              f"wrote {OBJ}.json (sha256 {files[f'{OBJ}.json'][:12]})")
        if a.check:
            c = compare_summary(summary, reference)
            same = data == REFERENCE_JSON.read_bytes()
            print(f"--check: {c['mean_equal']}/{c['labels']} means, "
                  f"{c['se_equal']}/{c['labels']} standard errors, "
                  f"{c['blocks_equal']}/{c['labels']} block lists equal the "
                  f"reference's; document bytes "
                  f"{'identical' if same else 'differ'}")
            full = len(summary) == len(reference["cells"])
            ok = (c["mean_equal"] == c["se_equal"] == c["blocks_equal"]
                  == c["labels"]) and (same or not full)
            rc = rc or (0 if ok else 1)
    if a.table:
        if summary is None:
            if a.from_reference:
                summary = reference["cells"]
            else:
                f = out / f"{OBJ}.json"
                if not f.exists():
                    raise SystemExit(f"--table: {f} does not exist; run or "
                                     f"--aggregate first")
                summary = json.loads(f.read_bytes().decode("utf-8"))["cells"]
        data = table_bytes(summary)
        _write(out / TABLE_FILE, data)
        files[TABLE_FILE] = _sha(data)
        print(f"--table: wrote {TABLE_FILE} ({len(data)} bytes, "
              f"sha256 {files[TABLE_FILE]})")
        if a.check:
            ref = paths.resolve("tabcv_full")
            if ref.exists():
                same = data == ref.read_bytes()
                print(f"--check: the table is "
                      f"{'byte-identical to' if same else 'DIFFERENT from'} {ref.name}")
                rc = rc or (0 if same else 1)
            else:
                print(f"--check: {ref} is not in this tree; nothing to compare")
    mode = "_".join(m for m, on in (("aggregate", a.aggregate), ("table", a.table))
                    if on)
    _manifest(out, f"{OBJ}.{mode}.manifest.json", mode=mode,
              source="reference" if a.from_reference else "output folder",
              output_sha256=files)
    return rc


def main(argv=None):
    a = parse_args(argv)
    post = a.aggregate or a.table
    only = (parse_blocks(a.only_blocks, a.blocks) if a.only_blocks
            else list(range(a.blocks)))
    given = bool(a.cells)
    listed = [c for c in a.cells.replace(" ", ",").split(",") if c] if a.cells else None
    specs = select_specs(listed) if (given or not post) else None
    print(f"blocks: K = {a.blocks} x R_block = {a.r_block:,} "
          f"(R_total = {a.blocks * a.r_block:,})")
    print(f"T_max = {T_MAX}, truncation = prefix eps[:T]; stride = {STRIDE:,}; "
          f"seed base = {SEED_BASE}")
    print(f"{len(cells())} cell-runs per block, cbar = {CBAR}")
    if not guard(a.blocks, a.r_block):
        raise SystemExit("stream guard failed")
    if a.dry_run:
        print("--dry-run: guard passed, no draws taken.")
        return 0
    out = paths.out_dir(SECTION) / a.name
    out.mkdir(parents=True, exist_ok=True)
    if post:
        return _post(a, out, specs if given else None)
    print(f"computing blocks {only} for {len(specs)} cell(s) with {a.jobs} job(s)",
          flush=True)
    return _simulate(a, out, specs, only)


if __name__ == "__main__":
    sys.exit(main())
