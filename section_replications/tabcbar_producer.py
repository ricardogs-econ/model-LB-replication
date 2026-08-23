#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tabcbar_producer.py -- the single producer of tab:cbar (Model LB calibration
surface: c-bar(m,T), all admissible break-fraction configurations).

The finite-sample calibration surface (inputs/cbar_surface.csv) is a frozen,
externally-produced input (produced_outside; regenerable_here=false) built over
~142 wall-hours; this package never regenerates it. tab:cbar is rendered FROM
that frozen surface by the const-arm convention: the printed cell c-bar(m,T) is
the mean of cbar_otimo over the enumerated lambda configurations (1/7/15/20/5/6
for m=0..5), rounded to one decimal. m=0 is the single-configuration case under
the SAME code -- one methodology for every row.

  NUMBER            PRODUCER (this file)              ARTEFACT
  tab:cbar          render from the frozen surface    output/golden/printed_tables/tab_cbar_mT.tex
  calibration       run_grid + aggregate (numba)      cbar_surface.csv  [surface generation:
  surface                                              transparency only -- produced_outside]

The numerical core is imported from src/mlb/kernel_numba.py (calibrate_config_sigma,
enumerate_configs, make_lambdas, config_seed, DEFAULTS) and never reimplemented.
The tangency is the interpolated power=0.50 crossing of the ERS/CKP point-optimal
statistic; R_cv=R_pow=200,000 for m>=1, a refined pass at 40,000 for m=0;
seed_base=20240601.

USAGE
    python replicate.py cv           # render tab:cbar from the frozen surface (--quick)
    python section_replications/tabcbar_producer.py --outdir output/cbar_run --method const
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.mlb import kernel_numba as kn  # the ONE production kernel; never reimplemented here

QS = [("1%", "cv1"), ("2.5%", "cv2_5"), ("5%", "cv5"), ("10%", "cv10")]
STATS = ["mza", "msb", "mzt", "pt", "mpt"]


# ------------------------------------------------------------------------------
# Per-cell compute, atomic write, digest-keyed resume
# ------------------------------------------------------------------------------
def cell_key(T, m, lambdas, method):
    lam = "_".join(f"{x:g}" for x in lambdas)
    return f"T{T}_m{m}_{method}" + (f"_lam{lam}" if lam else "")


def cell_digest(P, T, m, lambdas, method):
    """Fingerprint of every parameter that can change a cell's result.
    A resume that finds a completed cell whose digest disagrees with the
    CURRENT parameters recomputes it -- see module docstring."""
    payload = dict(
        T=T, m=m, lambdas=list(lambdas), method=method,
        R_cv=P["R_cv"], R_pow=P["R_pow"],
        refine_R_cv=P["refine_R_cv"], refine_R_pow=P["refine_R_pow"],
        refine_cheap_configs=P["refine_cheap_configs"],
        refine_max_lambdas=P["refine_max_lambdas"],
        refine_grid_step=P["refine_grid_step"],
        refine_grid_halfwidth=P["refine_grid_halfwidth"],
        seed_base=P["seed_base"], cbar_grid=list(P["cbar_grid"]),
        kmax=P["kmax"], target_power=P["target_power"],
    )
    blob = json.dumps(payload, sort_keys=True, default=float).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=float)
    os.replace(tmp, path)  # atomic on POSIX and Windows (same volume)


def run_cell(P, T, m, lambdas, outdir, method):
    key = cell_key(T, m, lambdas, method)
    jpath = os.path.join(outdir, key + ".json")
    digest = cell_digest(P, T, m, lambdas, method)
    if os.path.exists(jpath):
        try:
            existing = json.load(open(jpath))
            if existing.get("complete") and existing.get("digest") == digest:
                return key, "skipped", None
        except json.JSONDecodeError:
            pass  # truncated/partial -- recompute

    t0 = time.time()
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    res = kn.calibrate_config_sigma(T, m, list(lambdas), P, method)
    wall = time.time() - t0

    raw = res.pop("raw", None)
    if raw:
        np.savez_compressed(
            os.path.join(outdir, key + "_raw.npz"),
            **{k: np.asarray(v, dtype=np.float32) for k, v in raw.items()})
    out = {k: v for k, v in res.items()
           if isinstance(v, (int, float, str, list, dict, type(None)))}
    refined = bool(res.get("refined"))
    out.update(
        T=T, m=m, lambdas=list(lambdas), method=method, digest=digest,
        R_cv=P["R_cv"], R_pow=P["R_pow"],
        r_cv_final=P["refine_R_cv"] if refined else P["R_cv"],
        r_pow_final=P["refine_R_pow"] if refined else P["R_pow"],
        started_iso=started,
        finished_iso=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        wall_s=round(wall, 2), complete=True,
    )
    _atomic_write_json(jpath, out)
    return key, f"ok ({wall:.1f}s)", out.get("fallback_used")


def heartbeat(outdir, done, total, key, status, elapsed):
    stamp = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    line = f"{stamp} [{done}/{total}] {key} {status} elapsed={elapsed:.0f}s"
    print(line, flush=True)
    with open(os.path.join(outdir, "progress.log"), "a") as f:
        f.write(line + "\n")


def guard_tangency(flags):
    """A whole run of edge-fallback c-bar* is a configuration failure, not a
    result (Rule from tools/regen_surface_maic.py, carried over verbatim)."""
    vals = [f for f in flags if f is not None]
    if len(vals) >= 9 and all(bool(f) for f in vals):
        raise RuntimeError(
            f"CONFIGURATION FAILURE: c-bar* was an edge fallback in all "
            f"{len(vals)} completed cells -- the tangency is not being "
            f"bracketed by the c-bar grid. Aborting before more cells are wasted.")


def _git_commit():
    try:
        commit = subprocess.run(["git", "-C", _ROOT, "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        if subprocess.run(["git", "-C", _ROOT, "status", "--porcelain"],
                          capture_output=True, text=True).stdout.strip():
            commit = f"DIRTY:{commit}"
        return commit
    except OSError:
        return "UNKNOWN"


def _run_manifest_path(outdir):
    return os.path.join(outdir, "RUN_MANIFEST.json")


def check_or_write_run_manifest(outdir, P, method, force):
    """Coherence guard: refuse to resume into an --outdir whose recorded
    parameters disagree with this invocation's, unless --force. Prevents a
    stale --outdir silently mixing cells computed under different R/seed_base
    into one 'resumed' run."""
    path = _run_manifest_path(outdir)
    fingerprint = dict(
        R_cv=P["R_cv"], R_pow=P["R_pow"], seed_base=P["seed_base"],
        refine_R_cv=P["refine_R_cv"], refine_R_pow=P["refine_R_pow"],
        method=method,
    )
    if os.path.exists(path):
        prior = json.load(open(path))
        if prior.get("fingerprint") != fingerprint and not force:
            raise RuntimeError(
                f"RUN_MANIFEST coherence guard: {path} was written with "
                f"{prior.get('fingerprint')}, this invocation asks for "
                f"{fingerprint}. Refusing to resume a mismatched run. Use a "
                f"fresh --outdir, or --force to override (recomputes nothing "
                f"automatically -- stale cells with a matching digest are "
                f"still skipped per-cell; only the coherence check is bypassed).")
        prior.setdefault("resumes", []).append(dict(
            iso=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            git_commit=_git_commit()))
        json.dump(prior, open(path, "w"), indent=1, default=str)
    else:
        json.dump(dict(
            fingerprint=fingerprint, created_iso=datetime.datetime.now()
            .astimezone().isoformat(timespec="seconds"),
            git_commit=_git_commit(), resumes=[]),
            open(path, "w"), indent=1, default=str)


# ------------------------------------------------------------------------------
# Grid driver
# ------------------------------------------------------------------------------
def run_grid(P, outdir, method, m_list=None, limit=None, jobs=8):
    cfgs = kn.enumerate_configs(P)
    if m_list:
        cfgs = [c for c in cfgs if c[1] in set(m_list)]
    if limit:
        cfgs = cfgs[:limit]
    print(f"[tabcbar_producer] configs={len(cfgs)} R_cv=R_pow={P['R_cv']} "
          f"method={method} seed_base={P['seed_base']}")

    kn.warm_up_numba(kmax=P["kmax"])
    t0 = time.time()
    powers, done = [], 0
    if jobs > 1:
        from joblib import Parallel, delayed
        for key, status, pw in Parallel(n_jobs=jobs, backend="loky",
                                        batch_size=1, return_as="generator")(
                delayed(run_cell)(P, T, m, lam, outdir, method)
                for (T, m, lam) in cfgs):
            done += 1
            powers.append(pw)
            guard_tangency(powers)
            heartbeat(outdir, done, len(cfgs), key, status, time.time() - t0)
    else:
        for (T, m, lam) in cfgs:
            key, status, pw = run_cell(P, T, m, lam, outdir, method)
            done += 1
            powers.append(pw)
            guard_tangency(powers)
            heartbeat(outdir, done, len(cfgs), key, status, time.time() - t0)
    print(f"[tabcbar_producer] grid done: {done} cells in {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------
# Aggregation: per-cell JSON -> one CSV (57+1 columns, matching SCHEMA.md)
# ------------------------------------------------------------------------------
def load_cells(outdir, method):
    rows = []
    for fn in sorted(os.listdir(outdir)):
        if not fn.endswith(".json") or fn.startswith("RUN_MANIFEST"):
            continue
        with open(os.path.join(outdir, fn), "r", encoding="utf-8") as fh:
            d = json.load(fh)
        if d.get("method") != method or not d.get("complete"):
            continue
        lam = d["lambdas"]
        row = dict(
            config_key=kn.config_key(d["T"], d["m"], lam),
            T=d["T"], m=d["m"], sigma2_method=method,
            cbar_otimo=d["cbar_star"], poder_no_otimo=d["poder_star"],
            se_poder=d["se_poder_star"], se_cbar=d.get("se_cbar_star", np.nan),
            fallback_used=int(d.get("fallback_used", False)),
            refined=int(d.get("refined", False)),
            inv_dev=d["inv_dev"], cfg_seed=d["cfg_seed"],
        )
        for i in range(5):
            row[f"lambda{i + 1}"] = lam[i] if i < len(lam) else np.nan
        for stat in STATS:
            for q, tag in QS:
                row[f"{stat}_{tag}"] = d["cvs"].get(stat, {}).get(q, np.nan)
                row[f"{stat}_{tag}_se"] = d["cvs_se"].get(stat, {}).get(q, np.nan)
        rows.append(row)
    return rows


def aggregate(outdir, methods):
    import pandas as pd
    rows = []
    for method in methods:
        cells = load_cells(outdir, method)
        if not cells:
            raise RuntimeError(f"{outdir}: zero complete '{method}' cells -- "
                               f"refusing to write an empty surface")
        rows.extend(cells)
    df = pd.DataFrame(rows).sort_values(["sigma2_method", "T", "m"]).reset_index(drop=True)
    cols = (["config_key", "T", "m", "sigma2_method", "cbar_otimo",
              "poder_no_otimo", "se_poder", "se_cbar", "fallback_used",
              "refined", "inv_dev", "cfg_seed"]
            + [f"lambda{i + 1}" for i in range(5)]
            + [f"{s}_{t}{suf}" for s in STATS for _, t in QS for suf in ("", "_se")])
    assert len(cols) == 57, len(cols)
    df = df[cols]
    csv_path = os.path.join(outdir, "cbar_surface.csv")
    df.to_csv(csv_path, index=False)
    manifest = dict(
        job="tabcbar_producer.aggregate", out=csv_path,
        built_at_commit=_git_commit(),
        built_iso=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        n_rows=len(df), n_rows_by_arm=df.groupby("sigma2_method").size().to_dict(),
        columns=list(df.columns),
    )
    json.dump(manifest, open(csv_path + ".manifest.json", "w"), indent=1, default=str)
    print(f"[tabcbar_producer] aggregated: {csv_path} | {len(df)} rows")
    return csv_path, df


# ------------------------------------------------------------------------------
# tab:cbar rendering (const arm only -- closes the registered "MISSING STEP"
# emits the surface AND the printed .tex in one run)
# ------------------------------------------------------------------------------
def render_tab_cbar(df, outdir):
    d = df[df.sigma2_method == "const"]
    if d.empty:
        raise RuntimeError("const arm absent -- refusing to write an empty tab:cbar")
    piv = d.groupby(["m", "T"])["cbar_otimo"].mean().unstack().round(1)
    path = os.path.join(outdir, "tab_cbar_mT.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(piv.to_latex(
            na_rep="---", float_format="%.1f",
            caption="Optimal $\\bar c(m,T)$ for Model~LB ($\\sigma^2$: const).",
            label="tab:cbar"))
    print(f"[tabcbar_producer] written: {path}")
    return path


# ------------------------------------------------------------------------------
# The parameter set -- ONE source of truth (the producer and the Parte-2
# validator both build P from here, so the validation runs the SAME convention
# the producer would, not a hand-copied approximation).
# ------------------------------------------------------------------------------
def build_P(outdir=None, R=200_000, quick=False, seed_base=None):
    """The production convention behind the frozen cbar_surface.csv, as P.

    Defaults ARE production: R_cv=R_pow=200000, single-seed m=0
    (m0_seed_averaging=False), seed_base 20240601 (from DEFAULTS). --R and
    quick are experiment overrides layered on top, never the baseline."""
    P = dict(kn.DEFAULTS)
    rr = 200 if quick else R
    P.update(R_cv=rr, R_pow=rr, save_raw_vectors=True)
    # m=0 is the single-seed calibrate_config_sigma path, NOT the K=20 wrapper.
    # This file never calls calibrate_config (the only reader of the flag), so
    # the wrapper is already unreachable; setting False makes that explicit and
    # cannot be undone by a future refactor that routes through calibrate_config.
    P["m0_seed_averaging"] = False
    if quick:
        # DEFAULTS' refine_R_cv/refine_R_pow=40,000 apply to m=0 regardless of
        # R_cv/R_pow above (calibrate_config_sigma substitutes them for the
        # refined row) -- without scaling these down too, a "--quick" smoke
        # test still pays the full 40,000-replication refine pass for m=0 and
        # is not actually quick (measured: ~55-70s per m=0 config at R=200
        # coarse alone, entirely from the un-scaled refine pass).
        P.update(refine_R_cv=200, refine_R_pow=200)
    if outdir is not None:
        P["checkpoint_dir"] = outdir
    if seed_base is not None:
        P["seed_base"] = seed_base
    return P


# ------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="output/cbar_surface",
                    help="run root (default output/cbar_surface). NEVER the "
                         "repo-root frozen cbar_surface.csv -- this writes its "
                         "own surface here.")
    ap.add_argument("--R", type=int, default=200_000,
                    help="replications per grid node (default 200000 = "
                         "production). An experiment override, not production knowledge.")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: tiny R (200) incl. the m=0 refine pass, "
                         "first 2 configs per arm")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seed-base", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N configs per method (smoke)")
    ap.add_argument("--method", choices=["const", "maic", "both"], default="both",
                    help="default 'both' reproduces the full frozen surface "
                         "(854 rows); 'const' alone suffices for the printed "
                         "tab:cbar; 'maic' is tab:cv / robustness cross-check")
    ap.add_argument("--m-list", type=int, nargs="+", default=None)
    ap.add_argument("--force", action="store_true",
                    help="bypass the RUN_MANIFEST coherence guard")
    ap.add_argument("--skip-tex", action="store_true",
                    help="write the CSV but not tab_cbar_mT.tex")
    a = ap.parse_args(argv)

    os.makedirs(a.outdir, exist_ok=True)
    P = build_P(outdir=a.outdir, R=a.R, quick=a.quick, seed_base=a.seed_base)
    limit = 2 if a.quick else a.limit

    methods = ["const", "maic"] if a.method == "both" else [a.method]

    check_or_write_run_manifest(a.outdir, P, a.method, a.force)

    for method in methods:
        run_grid(P, a.outdir, method, m_list=a.m_list, limit=limit, jobs=a.jobs)

    csv_path, df = aggregate(a.outdir, methods)
    if not a.skip_tex and "const" in methods:
        render_tab_cbar(df, a.outdir)

    print(f"[tabcbar_producer] DONE -> {a.outdir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
