"""tab:cv regeneration driver (brief s.0.7): fixed cbar, uniform R, all five
M statistics, quantiles {1, 2.5, 5, 10}% in one pass.

Kernels are IMPORTED from src.mlb.kernel_numba, never reimplemented (the boot_ppp_cbar
equivalence-contract discipline). The null-distribution seed recipe is the
published one, read at src/mlb/kernel_numba.py:669-670:

    seed0 = config_seed(P, T, m, lambdas) + 2_000_000
    replication r draws np.random.default_rng((seed0 + r) % (2**63 - 1))

so the first 10_000 draws of an R=200_000 run ARE the published run's draws
(brief s.0.7 prefix diagnostics), and chunked accumulation is bit-identical
to unchunked by construction (chunk [a,b) calls _stats_under with seed0+a).

Lambda modes (both supported, author rules which one is printed):
  grid  : the published lambda grid via mlb_core.enumerate_configs
          (m_min_T respected; the eight never-simulated (m,T) are absent)
  equal : lambda_j = j/(m+1) as the tab:cv caption states, full 6x9 (m,T)
          grid INCLUDING the eight cells (simulated legitimately here,
          Phase-1 F18 exclusion applies to the registry only)

Per-cell outputs under --outdir: <key>.json (quantiles at R-prefixes
10k/50k/R, SEs, timestamps, seed0) and <key>_raw.npz (float32 statistic
vectors, compressed - the package's own raw-vector convention,
src/mlb/kernel_numba.py:701). Cells already complete are skipped on restart (resume).
A run_meta.json is written at the run root. DEGENERATE cells (non-finite
share > 1%% or empty lambda simplex) are reported, not coerced.

PowerShell examples:
    python tools\\regen_tabcv.py --outdir <run-root>\\smoke --R 200 --jobs 8 --lambda-mode grid
    python tools\\regen_tabcv.py --outdir <run-root> --R 200000 --jobs 8 --pilot T60_m2  (single cell)
"""

import os

for _v in (
    "NUMBA_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "mlb_numba_cache"
    ),
)

import argparse
import datetime
import json
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.mlb import kernel_numba as mlb_core  # noqa: E402

QS = ["1%", "2.5%", "5%", "10%"]
STATS = ["mza", "msb", "mzt", "pt", "mpt"]
PREFIXES = [10_000, 50_000]  # nested-prefix diagnostics (brief s.0.7)


def enumerate_cells(mode, P):
    """Yield (T, m, lambdas) per lambda mode."""
    if mode == "grid":
        return mlb_core.enumerate_configs(P)
    cells = []
    for m in P["m_grid"]:
        for T in P["T_grid"]:
            lam = tuple(j / (m + 1) for j in range(1, m + 1))
            cells.append((T, m, lam))
    return cells


def cell_key(T, m, lambdas, method):
    lam = "_".join(f"{x:g}" for x in lambdas)
    return f"T{T}_m{m}_{method}" + (f"_lam{lam}" if lam else "")


def density_at(a, q_val, h=None):
    """Gaussian-KDE density estimate at the quantile (for SE(q) reporting)."""
    a = a[np.isfinite(a)]
    n = len(a)
    if n < 100:
        return float("nan")
    if h is None:
        h = 1.06 * np.std(a) * n ** (-0.2)
    return float(
        np.mean(np.exp(-0.5 * ((a - q_val) / h) ** 2)) / (h * np.sqrt(2 * np.pi))
    )


def run_cell(P, T, m, lambdas, method, cbar, R, chunk, outdir,
             seed_stride=0, cell_index=0, block_base=0):
    key = cell_key(T, m, lambdas, method)
    jpath = os.path.join(outdir, key + ".json")
    if os.path.exists(jpath):
        try:
            if json.load(open(jpath)).get("complete"):
                return key, "skipped (checkpoint)"
        except json.JSONDecodeError:
            pass
    t0 = time.time()
    mc = mlb_core._METHOD_CODE[method]
    bp = mlb_core.break_pos_from_lambdas(T, list(lambdas))
    cfg_seed = mlb_core.config_seed(P, T, m, list(lambdas))
    seed0 = cfg_seed + 2_000_000  # published recipe, src/mlb/kernel_numba.py:670
    # [G_STRIDE] The config hash separates adjacent cells by as little as
    # 26,419 seeds, so a cell that draws R replications from seed0 shares
    # innovations with its neighbour once R exceeds that gap -- already true of
    # nine adjacent pairs at the published R = 200,000. Each cell's own quantile
    # is unaffected (a quantile of R valid draws is valid however its draws were
    # obtained); what the overlap costs is INDEPENDENCE ACROSS CELLS, which is
    # what any statement comparing cells rests on. seed_stride = 0 reproduces
    # the published recipe exactly and is the default.
    seed0 += seed_stride * cell_index
    seed0 += block_base  # K=20 block design: block k uses block_base = k*10**9
    # (disjoint across blocks); block-mean quantiles + se_between_blocks recover
    # the g48 3.3M full-table production run from feasible per-block R.
    out = {k: np.empty(R) for k in STATS}
    done = 0
    while done < R:
        b = min(chunk, R - done)
        block = mlb_core._stats_under(P, T, bp, 0.0, cbar, mc, b, seed0 + done, 12)
        for k in STATS:
            out[k][done : done + b] = block[k]
        done += b
    res = {
        "T": T,
        "m": m,
        "lambdas": list(lambdas),
        "method": method,
        "cbar": cbar,
        "R": R,
        "seed0": int(seed0),
        "cfg_seed": int(cfg_seed),
        "seed_stride": int(seed_stride),
        "cell_index": int(cell_index),
        "chunk_B": chunk,
        "started_iso": datetime.datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
        "quantiles": {},
        "se": {},
        "prefix_quantiles": {},
        "nonfinite_share": {},
    }
    degenerate = False
    for k in STATS:
        a = out[k]
        fin = np.isfinite(a)
        share_bad = 1.0 - fin.mean()
        res["nonfinite_share"][k] = float(share_bad)
        af = a[fin]
        if share_bad > 0.01 or len(af) < 100:
            degenerate = True
            res["quantiles"][k] = {q: None for q in QS}
            continue
        res["quantiles"][k] = {q: float(np.percentile(af, float(q[:-1]))) for q in QS}
        res["se"][k] = {}
        for q in QS:
            p = float(q[:-1]) / 100.0
            f = density_at(af, res["quantiles"][k][q])
            res["se"][k][q] = (
                float(np.sqrt(p * (1 - p) / len(af)) / f) if f and f > 0 else None
            )
        res["prefix_quantiles"][k] = {}
        for n in [p for p in PREFIXES if p < R]:
            an = a[:n]
            an = an[np.isfinite(an)]
            res["prefix_quantiles"][k][str(n)] = {
                q: float(np.percentile(an, float(q[:-1]))) for q in QS
            }
    res["degenerate"] = degenerate
    res["wall_s"] = round(time.time() - t0, 2)
    res["finished_iso"] = (
        datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    )
    res["complete"] = True
    np.savez_compressed(
        os.path.join(outdir, key + "_raw.npz"),
        **{k: out[k].astype(np.float32) for k in STATS},
    )
    with open(jpath, "w") as f:
        json.dump(res, f, indent=1)
    return key, ("DEGENERATE" if degenerate else "ok") + f" ({res['wall_s']}s)"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--R", type=int, default=200_000)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--cbar", type=float, default=-7.0)
    ap.add_argument("--lambda-mode", choices=["grid", "equal"], default="grid")
    ap.add_argument(
        "--methods", nargs="+", default=["const", "maic"], choices=["const", "maic"]
    )
    ap.add_argument("--chunk", type=int, default=10_000)
    ap.add_argument(
        "--pilot", default=None, help="run only cells whose key starts with this prefix"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="run the [G_STRIDE] guard and the cell enumeration, then exit "
        "without taking a single draw. The guard is the thing that must be "
        "testable without spending the run it protects.",
    )
    ap.add_argument(
        "--block-base", type=int, default=0,
        help="uniform seed offset added to every cell (K=20 block design: block "
        "k uses block_base = k*10**9, disjoint across blocks). The full-table "
        "producer aggregates block-mean quantiles + se_between_blocks over K "
        "blocks to recover the g48 3.3M production run at feasible per-block R.")
    ap.add_argument(
        "--seed-stride",
        type=int,
        default=0,
        help="add stride*cell_index to each cell's seed0, so the cells' "
        "innovation streams are provably disjoint. 0 (default) reproduces the "
        "published recipe. The guard below refuses to run when R exceeds the "
        "smallest gap between cell seeds and no stride was given.",
    )
    ap.add_argument(
        "--seed-base",
        type=int,
        default=None,
        help="override P['seed_base'] (default: published "
        "20240601 from src.mlb.kernel_numba.DEFAULTS)",
    )
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    P = dict(mlb_core.DEFAULTS)
    if args.seed_base is not None:
        P["seed_base"] = args.seed_base

    cells = enumerate_cells(args.lambda_mode, P)
    work = [(T, m, lam, method) for (T, m, lam) in cells for method in args.methods]
    if args.pilot:
        work = [
            w for w in work if cell_key(w[0], w[1], w[2], w[3]).startswith(args.pilot)
        ]
    print(
        f"[regen_tabcv] cells={len(work)} R={args.R} cbar={args.cbar} "
        f"lambda_mode={args.lambda_mode} methods={args.methods} "
        f"chunk={args.chunk}"
    )
    print(
        f"[regen_tabcv] n_jobs={args.jobs} "
        f"NUMBA_NUM_THREADS={os.environ['NUMBA_NUM_THREADS']} "
        f"OMP={os.environ['OMP_NUM_THREADS']} "
        f"MKL={os.environ['MKL_NUM_THREADS']} "
        f"OPENBLAS={os.environ['OPENBLAS_NUM_THREADS']}"
    )

    try:
        commit = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        if dirty:
            commit = f"DIRTY:{commit}"
    except OSError:
        commit = "UNKNOWN"
    meta = {
        "git_commit": commit,
        "seed_fn": "default_rng((config_seed(P,T,m,lam)+2e6 + r) % "
        "(2**63-1)); published recipe src/mlb/kernel_numba.py:669-670",
        "n_jobs": args.jobs,
        "env_threads": "NUMBA=1 OMP=1 MKL=1 OPENBLAS=1",
        "R": args.R,
        "cbar": args.cbar,
        "lambda_mode": args.lambda_mode,
        "methods": args.methods,
        "chunk_B": args.chunk,
        "grid": f"{len(work)} cells",
        "kernel": "canonical (src.mlb.kernel_numba)",
        "seed_base": P["seed_base"],
        "started_iso": datetime.datetime.now()
        .astimezone()
        .isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.outdir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)

    # [G_STRIDE] guard -- checked before a single draw is taken.
    # Grouped BY METHOD: const and maic at the same (T,m,lambda) share a config
    # seed on purpose -- that is the common-random-numbers design that makes the
    # two estimators comparable, and it must not be read as an overlap. What
    # must be disjoint is one cell against ANOTHER CELL within the same method.
    _by_method = {}
    for i, (T, m, lam, mt) in enumerate(work):
        _by_method.setdefault(mt, []).append(
            mlb_core.config_seed(P, T, m, list(lam)) + 2_000_000
            + args.seed_stride * i)
    _gap = 1 << 62
    for _mt, _bs in _by_method.items():
        _bs = sorted(_bs)
        if len(_bs) > 1:
            _gap = min(_gap, min((b - a) for a, b in zip(_bs, _bs[1:])))
    _bases = [x for v in _by_method.values() for x in v]
    if _gap < args.R:
        msg = (f"[G_STRIDE] cell seed streams overlap: smallest gap {_gap:,} "
               f"< R = {args.R:,}. Nine adjacent pairs already overlap at the "
               f"published R = 200,000. Pass --seed-stride (e.g. 10**9) to make "
               f"them disjoint, or accept correlated cells deliberately.")
        if args.seed_stride == 0:
            raise SystemExit(msg)
        raise SystemExit(msg + " Stride given but still insufficient.")
    print(f"  [G_STRIDE] {len(_bases)} cell streams disjoint, stride="
          f"{args.seed_stride:,}, minimum slack {_gap - args.R:,} replications",
          flush=True)

    if args.dry_run:
        print(f"[regen_tabcv] --dry-run: {len(work)} cells enumerated, guard "
              f"passed, no draws taken.")
        return
    mlb_core.warm_up_numba(kmax=12)
    t0 = time.time()
    if args.jobs > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=args.jobs, backend="loky", batch_size=1)(
            delayed(run_cell)(
                P, T, m, lam, method, args.cbar, args.R, args.chunk,
                args.outdir, args.seed_stride, i, args.block_base
            )
            for i, (T, m, lam, method) in enumerate(work)
        )
    else:
        results = [
            run_cell(P, T, m, lam, method, args.cbar, args.R, args.chunk,
                     args.outdir, args.seed_stride, i, args.block_base)
            for i, (T, m, lam, method) in enumerate(work)
        ]
    n_deg = sum("DEGENERATE" in s for _, s in results)
    n_skip = sum("skipped" in s for _, s in results)
    print(
        f"[regen_tabcv] done in {time.time() - t0:.1f}s  "
        f"cells={len(results)} skipped={n_skip} degenerate={n_deg}"
    )
    for k, s in results:
        if "DEGENERATE" in s:
            print(f"  DEGENERATE: {k}")


if __name__ == "__main__":
    main()
