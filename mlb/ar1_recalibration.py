"""Table S2 (stab:recal): recalibration under AR(1) errors, exact; no Monte Carlo.

Reference file: reference/serial_dependence/ar1_recalibration/
ar1_recalibration_envelope.csv (the run compares it byte for byte; stab_recal.tex
is rendered into output/).

Object: the half-power point of the envelope L_T = S(abar) - S(1) when the
data are generated with AR(1) innovations, in the null AND on the diagonal:

    v_t = rho v_{t-1} + e_t, v_0 = 0;   u_t = (1 + c/T) u_{t-1} + v_t, u_0 = 0,

the convention of `make_ar1_eps` in ar1_size_power.py. L_T is still a
Gaussian quadratic form with a known matrix, so the tangency is exact
(mlb/envelope_exact.py, half_power_point_ar1; rho = 0 calls the i.i.d.
code unchanged).

Grid: T in {60, 100}, m in {0, 1, 2}, lambda_j = j/(m+1) (tab:cv's
convention), rho in {0, 0.5} (the printed grid of Table S2). Output
Delta = cbar_AR - cbar_iid.

Agreement check surface_agreement (first; a failure stops the run): at rho = 0 the
code reproduces cbar_surface_exact.csv to 1e-8 at every configuration the two share
(listed in recal_agreement.csv), which include m = 0 at T = 60 and 100.

Second-route check of the AR(1) spectrum (a failure stops the run): the non-zero
spectrum of K = L' A L, L = L_c P_rho, is recomputed by a second route, the
eigenvalues of A (L L') -- the same spectrum through the covariance of u -- and
must agree to 1e-9 relative at the located tangency.

    python mlb/ar1_recalibration.py [--jobs 4]
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
_pp = os.environ.get("PYTHONPATH", "")
if _REPO not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _REPO + (os.pathsep + _pp if _pp else "")

import numpy as np  # noqa: E402
from scipy import linalg  # noqa: E402

from mlb import paths  # noqa: E402
from mlb import envelope_exact as ee  # noqa: E402
from mlb import kernel_numba as kn  # noqa: E402  (config_key only)

SECTION = "robustness"
REFERENCE_DIR = "calibration_robustness"
OBJ = "ar1_recalibration"
SURFACE = paths.reference_dir("critical_values") / "cbar_surface_exact" / "cbar_surface_exact.csv"
SURFACE_SHA256 = "d020889e7a0e5a7756e4bee458b2e51e9b985bb204cdf75c0f18d17d3541d6dc"
TS = (60, 100)
MS = (0, 1, 2)
RHOS = (0.0, 0.5)
RHO_PRINTED = 0.5
SURF_TOL = 1e-8
EIG_ROUTE_RTOL = 1e-9
BRACKET = (-25.0, -1.0)
# the outputs compared byte for byte with reference/serial_dependence/ar1_recalibration/
REFERENCE_OUTPUTS = ("ar1_recalibration_envelope.csv",)


def lambdas(m):
    return tuple(j / (m + 1) for j in range(1, m + 1))


def _one(T, m, rho):
    lam = lambdas(m)
    cb, b = ee.half_power_point_ar1(T, lam, rho, bracket=BRACKET)
    Z = ee.design_Z(T, lam)
    pw, b2 = ee.power_at_ar1(cb, T, Z, rho)
    # the second route to the spectrum, at the located tangency
    A = ee._Q(1.0 + cb / T, T, Z) - ee._Q(1.0, T, Z)
    route = {}
    for tag, c in (("null", 0.0), ("diag", cb)):
        L = ee._Lc(c, T) @ ee._Pr(rho, T)
        w1 = np.sort(ee.eig_K_ar1(c, cb, T, Z, rho, A))
        w2 = linalg.eigvals(A @ (L @ L.T)).real
        w2 = w2[np.argsort(-np.abs(w2))]
        top, rest = np.sort(w2[:w1.size]), w2[w1.size:]
        scale = np.max(np.abs(w1))
        route[tag] = float(max(np.max(np.abs(w1 - top)),
                               np.max(np.abs(rest)) if rest.size else 0.0) / scale)
    return dict(T=T, m=m, rho=rho, lambdas=" ".join(repr(x) for x in lam),
                break_positions=" ".join(str(p) for p in ee.break_positions(T, lam)),
                cbar_half=cb, b_T=b, power_at_cbar_half=pw,
                bracket=f"({BRACKET[0]}, {BRACKET[1]})",
                eig_route_null_rel=route["null"],
                eig_route_diag_rel=route["diag"])


def compute(tasks, jobs):
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
        delayed(_one)(T, m, rho) for (T, m, rho) in tasks)
    ks = [(r["T"], r["m"], r["rho"]) for r in rows]
    assert len(set(ks)) == len(rows)
    return {(r["T"], r["m"], r["rho"]): r for r in rows}


def load_surface():
    sha = hashlib.sha256(
        open(SURFACE, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    if sha != SURFACE_SHA256:
        raise SystemExit("[ar1_recalibration] the surface is not the shipped reference: "
                         f"sha256 {sha}")
    rows = list(csv.DictReader(open(SURFACE, encoding="utf-8")))
    assert len({r["config_key"] for r in rows}) == len(rows)
    return {r["config_key"]: float(r["cbar_half_exact"]) for r in rows}, sha


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


def _tex(res, path):
    """The stab:recal layout at rho = 0.5; three decimals (the numbers are
    exact; the printed digits are the paper's)."""
    L = [r"% Table S2 (stab:recal): exact envelope tangency, rho = 0.5",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"$T$ & $m$ & $\cb_{\mathrm{iid}}$ & $\cb_{\mathrm{AR}}$ & $\Delta$ \\",
         r"\midrule"]
    for m in MS:
        for T in TS:
            a, b = res[(T, m, 0.0)]["cbar_half"], res[(T, m, RHO_PRINTED)]["cbar_half"]
            L.append(f"{T:<3} & {m} & ${a:.3f}$ & ${b:.3f}$ & ${b - a:+.3f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L))


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
    surf, surf_sha = load_surface()
    sources = ("mlb/ar1_recalibration.py",
               "mlb/envelope_exact.py", "mlb/kernel_numba.py",
               "mlb/paths.py")
    # no time, worker count or machine field: the run records of two runs differ
    # only in the command and the library versions
    man = {"reference_dir": REFERENCE_DIR, "job": OBJ,
           "command": command_line(sys.argv[1:] if argv is None else argv),
           "grid": {"T": TS, "m": MS, "rho": RHOS, "lambdas": "j/(m+1)"},
           "dgp": ee.AR1_CONVENTION,
           "envelope": {
               "conventions": ee.conventions(),
               "call": ("half_power_point_ar1(T, lambdas, rho) with the module "
                        f"defaults (xtol 1e-6); bracket {BRACKET}")},
           "surface": {"path": SURFACE.relative_to(paths.REPO_ROOT).as_posix(),
                       "sha256": surf_sha},
           "python": sys.version.split()[0], "numpy": np.__version__,
           "scipy": __import__("scipy").__version__,
           "joblib": __import__("joblib").__version__,
           "sources": {p: _sha(os.path.join(_REPO, p)) for p in sources}}

    # ---- surface_agreement at rho = 0
    print("[ar1_recalibration] surface_agreement: rho = 0 on the "
          f"{len(TS) * len(MS)} cells, "
          f"jobs={a.jobs}", flush=True)
    res = compute([(T, m, 0.0) for m in MS for T in TS], a.jobs)
    agree_rows = []
    for (T, m, _), r in sorted(res.items()):
        k = kn.config_key(T, m, list(lambdas(m)))
        if k in surf:
            d = abs(r["cbar_half"] - surf[k])
            agree_rows.append(dict(config_key=k, T=T, m=m, surface=surf[k],
                                   recomputed=r["cbar_half"], abs_diff=d, **{"pass": d <= SURF_TOL}))
        else:
            agree_rows.append(dict(config_key=k, T=T, m=m, surface="", recomputed=r["cbar_half"],
                                   abs_diff="", **{"pass": "not in the surface"}))
    shared = [r for r in agree_rows if r["surface"] != ""]
    ok = (all(r["pass"] is True for r in shared)
          and {r["T"] for r in shared if r["m"] == 0} == set(TS))
    _wcsv(out / "recal_agreement.csv", agree_rows)
    man["agreement_checks"] = dict(surface_agreement=dict(
        passed=ok, n_shared=len(shared), tol=SURF_TOL,
        max_abs_diff=max(r["abs_diff"] for r in shared),
        shared=[r["config_key"] for r in shared]))
    print(f"[ar1_recalibration] surface_agreement {'PASS' if ok else 'FAIL'}: "
          f"{len(shared)} shared, max |d| = "
          f"{man['agreement_checks']['surface_agreement']['max_abs_diff']:.3e}",
          flush=True)
    if not ok:
        json.dump(man,
                  open(out / "ar1_recalibration.manifest.json", "w", encoding="utf-8",
                       newline="\n"), indent=1, default=str)
        print("[ar1_recalibration] surface_agreement FAILED; nothing else runs")
        return 3

    # ---- AR(1)
    print(f"[ar1_recalibration] rho in {RHOS[1:]}", flush=True)
    res.update(compute([(T, m, rho) for rho in RHOS[1:] for m in MS for T in TS], a.jobs))
    worst_route = max(max(r["eig_route_null_rel"], r["eig_route_diag_rel"])
                      for r in res.values() if r["rho"] != 0.0)
    worst_pw = max(abs(r["power_at_cbar_half"] - 0.5) for r in res.values())
    man["checks"] = dict(eig_route_max_rel=worst_route, eig_route_tol=EIG_ROUTE_RTOL,
                         eig_route_pass=worst_route <= EIG_ROUTE_RTOL,
                         max_abs_power_minus_half=worst_pw,
                         brackets_used=sorted({r["bracket"] for r in res.values()}))
    if worst_route > EIG_ROUTE_RTOL:
        json.dump(man,
                  open(out / "ar1_recalibration.manifest.json", "w", encoding="utf-8",
                       newline="\n"), indent=1, default=str)
        print(f"[ar1_recalibration] the two routes to the spectrum disagree ({worst_route:.2e}); stopped")
        return 3

    rows = []
    for rho in RHOS:
        for m in MS:
            for T in TS:
                r = dict(res[(T, m, rho)])
                iid = res[(T, m, 0.0)]["cbar_half"]
                r.update(cbar_iid=iid, delta=r["cbar_half"] - iid)
                rows.append(r)
    ks = [(r["T"], r["m"], r["rho"]) for r in rows]
    assert len(set(ks)) == len(rows) == len(TS) * len(MS) * len(RHOS)
    _wcsv(out / "ar1_recalibration_envelope.csv", rows)

    _tex(res, out / "stab_recal.tex")

    ref_dir = paths.reference_dir(REFERENCE_DIR) / OBJ
    regen = {"references_present": ref_dir.is_dir()}
    if ref_dir.is_dir():
        for name in REFERENCE_OUTPUTS:
            regen[name] = ((ref_dir / name).is_file()
                           and _sha(out / name) == _sha(ref_dir / name))
        print("[ar1_recalibration] regenerated outputs identical to the reference files: "
              f"{regen}")
    man["outputs"] = {f.name: _sha(f) for f in sorted(out.iterdir())
                      if f.suffix in (".csv", ".tex")}
    json.dump(man,
              open(out / "ar1_recalibration.manifest.json", "w", encoding="utf-8",
                   newline="\n"),
              indent=1, default=str)
    if ref_dir.is_dir() and not all(v for k, v in regen.items() if k != "references_present"):
        print("[ar1_recalibration] regeneration differs from the reference files")
        return 4
    print(f"[ar1_recalibration] DONE in {(time.time() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
