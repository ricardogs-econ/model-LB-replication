"""Exact summaries of the envelope surface. No Monte Carlo.

Object: the surface behind tab:cbar (reference files under
reference/calibration/surface_summaries/):

  the displacement (surface_summaries.csv), T <= 60 and m <= 3: the largest
      |cell mean + 7| (with the cell that attains it) and the largest
      |cbar_half + 7| over configurations (with the configuration);
  the cells (surface_cells.csv): per (m, T) cell of the surface, the number of
      configurations and the mean, range and standard deviation (ddof 1) of
      cbar_half;
  trimming (surface_trimming*.csv): exact cbar_half with eps in {0.10, 0.15, 0.20}
      in (1,60), (2,60), (3,60), (2,100) (surface_trimming_configs.csv), the cell
      means and their displacement against eps = 0.15 (surface_trimming.csv), and
      the largest displacement of each cell (surface_trimming_range.csv).

The displacement and the cells read the exact surface (the reference under
reference/calibration/cbar_surface_exact/); the trimming summaries compute
exact tangencies with mlb/envelope_exact.py.

THE GRID RULE OF THE TRIMMING SUMMARIES. The surface's generator is
kernel_numba.make_lambdas (m, trim, min_spacing, n_grid), called by
enumerate_configs with DEFAULTS trim = 0.15, min_spacing = 0.15, n_grid = 7. The
minimum spacing is its own parameter: it does NOT follow the trim. Two conventions
are computed and reported side by side, the two lattices the supplement describes
(S4.1); none is chosen here.

  generator    make_lambdas(m, eps, DEFAULTS['min_spacing'], DEFAULTS['n_grid'])
               -- the generator's rule with eps swapped and nothing else.
  theory       make_lambdas(m, eps, eps, DEFAULTS['n_grid']) -- minimum spacing
               equal to eps, as in the trimmed simplex Lambda_eps of the
               paper (eq:trimmed-simplex).

    python mlb/surface_summaries.py [--jobs 4]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
# the loky workers import this file as `mlb.surface_summaries`
_pp = os.environ.get("PYTHONPATH", "")
if _REPO not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _REPO + (os.pathsep + _pp if _pp else "")

import numpy as np  # noqa: E402

from mlb import paths  # noqa: E402
from mlb import envelope_exact as ee  # noqa: E402
from mlb import kernel_numba as kn  # noqa: E402  (the grid rule only)

SECTION = "critical_values"
OBJ = "surface_summaries"
SURFACE_DIR = paths.reference_dir(SECTION) / "cbar_surface_exact"
SURFACE_SHA256 = "d020889e7a0e5a7756e4bee458b2e51e9b985bb204cdf75c0f18d17d3541d6dc"
N_CONFIGS, N_CELLS = 404, 43
TRIMMING_CELLS = ((1, 60), (2, 60), (3, 60), (2, 100))
EPS = (0.10, 0.15, 0.20)
EPS_BASE = 0.15
SURF_M0_T = (60, 100)
SURF_TOL = 1e-8
CONVENTIONS = ("generator", "theory")
# the outputs compared byte for byte with reference/calibration/surface_summaries/
REFERENCE_OUTPUTS = ("surface_cells.csv", "surface_summaries.csv",
                     "surface_trimming.csv", "surface_trimming_configs.csv",
                     "surface_trimming_range.csv")


# ================================================================== grids
def lambdas_for(conv: str, m: int, eps: float) -> list:
    if conv == "generator":
        return [tuple(l) for l in kn.make_lambdas(m, eps, kn.DEFAULTS["min_spacing"],
                                                  kn.DEFAULTS["n_grid"])]
    if conv == "theory":
        return [tuple(l) for l in kn.make_lambdas(m, eps, eps, kn.DEFAULTS["n_grid"])]
    raise KeyError(conv)


def key(T: int, lambdas) -> str:
    return kn.config_key(T, len(lambdas), list(lambdas))


# ================================================================ compute
def _one(T: int, lambdas: tuple):
    t0 = time.time()
    cb, b = ee.half_power_point(T, tuple(lambdas))
    return dict(config_key=key(T, lambdas), T=T, m=len(lambdas),
                lambdas=" ".join(repr(float(x)) for x in lambdas),
                cbar_half_exact=cb, b_T=b), time.time() - t0


def compute(configs: list, jobs: int) -> dict:
    from joblib import Parallel, delayed
    out = {}
    for row, _w in Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
            delayed(_one)(T, lam) for (T, lam) in configs):
        out[row["config_key"]] = row
    assert len(out) == len(configs), "config_key collapse in the exact batch"
    return out


# ================================================================ surface
def load_surface():
    p = SURFACE_DIR / "cbar_surface_exact.csv"
    sha = hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    if sha != SURFACE_SHA256:
        raise SystemExit("[surface_summaries] the surface is not the shipped reference: "
                         f"sha256 {sha}")
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    keys = [r["config_key"] for r in rows]
    assert len(set(keys)) == len(rows), "surface: config_key collapse"
    grid = {kn.config_key(T, m, list(l)) for (T, m, l) in kn.enumerate_configs(kn.DEFAULTS)}
    if set(keys) != grid:
        raise SystemExit("[surface_summaries] the surface's configurations are not the "
                         "public generator's enumerate_configs(DEFAULTS)")
    for r in rows:
        r["T"], r["m"] = int(r["T"]), int(r["m"])
        r["c"] = float(r["cbar_half_exact"])
    assert len(rows) == N_CONFIGS
    return rows, sha


def cells_of(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["m"], r["T"]), []).append(r)
    return cells


# ===================================================== the displacement, cells
def summaries(rows):
    out = []

    def put(quantity, scope, value, where=""):
        out.append(dict(quantity=quantity, scope=scope, value=value, where=where))

    cells = cells_of(rows)
    assert len(cells) == N_CELLS
    mean = {k: float(np.mean([r["c"] for r in v])) for k, v in cells.items()}

    # the displacement
    sub_c = {k: v for k, v in mean.items() if k[1] <= 60 and k[0] <= 3}
    kc = max(sub_c, key=lambda k: abs(sub_c[k] + 7.0))
    put("max |cell mean + 7|", "T<=60, m<=3", abs(sub_c[kc] + 7.0),
        f"(m={kc[0]},T={kc[1]}) mean={sub_c[kc]!r}")
    sub_r = [r for r in rows if r["T"] <= 60 and r["m"] <= 3]
    rr = max(sub_r, key=lambda r: abs(r["c"] + 7.0))
    put("max |cbar_half + 7|", "T<=60, m<=3", abs(rr["c"] + 7.0),
        f"{rr['config_key']} cbar={rr['c']!r}")

    cell_rows = []
    for (m, T), v in sorted(cells.items()):
        c = [r["c"] for r in v]
        cell_rows.append(dict(m=m, T=T, n_configs=len(c), mean=float(np.mean(c)),
                              range=max(c) - min(c),
                              sd_ddof1=(float(np.std(c, ddof=1)) if len(c) > 1
                                        else "")))
    ks = [(r["m"], r["T"]) for r in cell_rows]
    assert len(set(ks)) == len(cell_rows)
    ks = [(r["quantity"], r["scope"]) for r in out]
    assert len(set(ks)) == len(out)
    return out, cell_rows


# =============================================================== trimming
def trimming_plan():
    plan = {}
    for conv in CONVENTIONS:
        for m, T in TRIMMING_CELLS:
            for eps in EPS:
                plan[(conv, m, T, eps)] = lambdas_for(conv, m, eps)
    return plan


def trimming(plan, exact):
    rows = []
    for (conv, m, T, eps), lams in plan.items():
        c = [exact[key(T, l)]["cbar_half_exact"] for l in lams]
        rows.append(dict(convention=conv, m=m, T=T, eps=eps,
                         mean=float(np.mean(c))))
    base = {(r["convention"], r["m"], r["T"]): r for r in rows if r["eps"] == EPS_BASE}
    for r in rows:
        b = base[(r["convention"], r["m"], r["T"])]
        r["disp_mean_vs_015"] = r["mean"] - b["mean"]
    ks = [(r["convention"], r["m"], r["T"], r["eps"]) for r in rows]
    assert len(set(ks)) == len(rows)
    rng = []
    for conv in CONVENTIONS:
        for m, T in TRIMMING_CELLS:
            v = [r for r in rows if (r["convention"], r["m"], r["T"]) == (conv, m, T)]
            rng.append(dict(convention=conv, m=m, T=T,
                            max_abs_disp_mean=max(abs(x["disp_mean_vs_015"]) for x in v)))
    ks = [(r["convention"], r["m"], r["T"]) for r in rng]
    assert len(set(ks)) == len(rng)
    return rows, rng


# ======================================================== agreement checks
def surface_check(surface, exact):
    """surface_agreement: every configuration computed here that the surface also
    holds must reproduce it to 1e-8; m = 0 at T = 60 and 100 are among them."""
    s = {r["config_key"]: r["c"] for r in surface}
    rows = []
    for k, e in sorted(exact.items()):
        if k in s:
            d = abs(e["cbar_half_exact"] - s[k])
            rows.append(dict(config_key=k, T=e["T"], m=e["m"], surface=s[k],
                             recomputed=e["cbar_half_exact"], abs_diff=d,
                             pass_=d <= SURF_TOL))
    m0 = {r["T"] for r in rows if r["m"] == 0}
    ok = all(r["pass_"] for r in rows) and m0 == set(SURF_M0_T)
    return ok, rows


def generator_eps015_is_surface(surface, plan):
    """The generator convention at eps = 0.15 must be exactly the surface's
    configurations in each trimming cell, so that its eps = 0.15 mean IS the
    tab:cbar cell mean."""
    cells = cells_of(surface)
    res = {}
    for m, T in TRIMMING_CELLS:
        mine = {key(T, l) for l in plan[("generator", m, T, EPS_BASE)]}
        res[f"({m},{T})"] = mine == {r["config_key"] for r in cells[(m, T)]}
    return all(res.values()), res


# ================================================================== output
def _wcsv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def _sha(p):
    return hashlib.sha256(open(p, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def command_line(argv):
    """The command of the run, package-relative (no path of the machine), without
    --jobs (the number of worker processes does not change the outputs)."""
    cmd, it = ["python", f"mlb/{OBJ}.py"], iter(argv)
    for x in it:
        if x == "--jobs":
            next(it, None)
        elif not x.startswith("--jobs="):
            cmd.append(x)
    return cmd


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args(argv)
    if sys.flags.optimize:
        raise SystemExit("refusing to run under -O: the checks are asserts")
    out = paths.out_dir(SECTION) / OBJ
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    surface, surface_sha = load_surface()
    plan = trimming_plan()
    same, same_res = generator_eps015_is_surface(surface, plan)

    todo = {}
    for (conv, m, T, eps), lams in plan.items():
        for l in lams:
            todo[key(T, l)] = (T, l)
    for T in SURF_M0_T:
        todo[key(T, ())] = (T, ())
    s_keys = {r["config_key"] for r in surface}
    # the configurations shared with the surface
    first = [v for k, v in todo.items() if k in s_keys]
    rest = [v for k, v in todo.items() if k not in s_keys]
    sources = ("mlb/surface_summaries.py",
               "mlb/envelope_exact.py", "mlb/kernel_numba.py",
               "mlb/paths.py")
    generator = ("make_lambdas(m, eps, DEFAULTS['min_spacing']="
                 f"{kn.DEFAULTS['min_spacing']!r}, DEFAULTS['n_grid']="
                 f"{kn.DEFAULTS['n_grid']!r})")
    # no time, worker count or machine field: the manifests of two runs differ at most
    # in the command and the library versions
    man = {"job": OBJ,
           "command": command_line(sys.argv[1:] if argv is None else argv),
           "surface": {"path": SURFACE_DIR.relative_to(paths.REPO_ROOT).as_posix(),
                       "sha256": surface_sha},
           "conventions": {
               "generator": generator,
               "theory": "make_lambdas(m, eps, eps, DEFAULTS['n_grid'])",
               "min_spacing_follows_eps_in_generator": False},
           "envelope": {"conventions": ee.conventions(),
                        "call": ("half_power_point(T, lambdas) with the module "
                                 "defaults (xtol 1e-6, bracket (-25, -1))")},
           "n_exact": {"shared_with_surface": len(first), "not_in_surface": len(rest)},
           "python": sys.version.split()[0], "numpy": np.__version__,
           "scipy": __import__("scipy").__version__,
           "joblib": __import__("joblib").__version__,
           "sources": {p: _sha(os.path.join(_REPO, p)) for p in sources}}

    print(f"[surface_summaries] surface_agreement: {len(first)} configurations shared "
          f"with the surface, jobs={a.jobs}")
    exact = compute(first, a.jobs)
    ok, agree_rows = surface_check(surface, exact)
    _wcsv(out / "surface_agreement.csv",
          [{("pass" if k == "pass_" else k): v for k, v in r.items()}
           for r in agree_rows])
    man["agreement_checks"] = dict(
        surface_agreement=dict(passed=ok, n=len(agree_rows), tol=SURF_TOL,
                max_abs_diff=max(r["abs_diff"] for r in agree_rows),
                m0_T=sorted(r["T"] for r in agree_rows if r["m"] == 0)),
        generator_eps015_is_surface=dict(passed=same, cells=same_res))
    print(f"[surface_summaries] surface_agreement {'PASS' if ok else 'FAIL'}: "
          f"n={len(agree_rows)}, max |d| = "
          f"{man['agreement_checks']['surface_agreement']['max_abs_diff']:.3e}; "
          f"generator eps=0.15 is the "
          f"surface: {same_res}")
    if not (ok and same):
        json.dump(man,
                  open(out / "surface_summaries.manifest.json", "w", encoding="utf-8",
                       newline="\n"), indent=1, default=str)
        print("[surface_summaries] a check FAILED; nothing else runs")
        return 3

    print(f"[surface_summaries] trimming: {len(rest)} exact configurations not in the "
          "surface")
    exact.update(compute(rest, a.jobs))
    assert set(exact) == set(todo)

    summ, cell_rows = summaries(surface)
    trim_rows, trim_rng = trimming(plan, exact)
    cfg_rows = []
    for (conv, m, T, eps), lams in plan.items():
        for l in lams:
            e = exact[key(T, l)]
            cfg_rows.append(dict(convention=conv, m=m, T=T, eps=eps, config_key=e["config_key"],
                                 lambdas=e["lambdas"], cbar_half_exact=e["cbar_half_exact"],
                                 b_T=e["b_T"], in_surface=e["config_key"] in s_keys))
    ks = [(r["convention"], r["eps"], r["config_key"]) for r in cfg_rows]
    assert len(set(ks)) == len(cfg_rows)

    files = {"surface_summaries.csv": summ, "surface_cells.csv": cell_rows,
             "surface_trimming.csv": trim_rows, "surface_trimming_range.csv": trim_rng,
             "surface_trimming_configs.csv": cfg_rows}
    for name, rows in files.items():
        _wcsv(out / name, rows)

    rdir = paths.reference_dir(SECTION) / OBJ
    regen = {"references_present": rdir.is_dir()}
    if rdir.is_dir():
        for name in REFERENCE_OUTPUTS:
            regen[name] = (_sha(out / name) == _sha(rdir / name))
        print("[surface_summaries] regenerated outputs identical to the reference files: "
              f"{regen}")
    man["outputs"] = {f.name: _sha(f) for f in sorted(out.iterdir())
                      if f.suffix in (".csv", ".tex")}
    json.dump(man,
              open(out / "surface_summaries.manifest.json", "w", encoding="utf-8",
                   newline="\n"),
              indent=1, default=str)
    if rdir.is_dir() and not all(v for k, v in regen.items() if k != "references_present"):
        print("[surface_summaries] regeneration differs from the reference files")
        return 4
    print(f"[surface_summaries] DONE in {(time.time() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
