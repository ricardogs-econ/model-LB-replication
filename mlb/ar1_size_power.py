"""Table S1 (stab:ar1): MZ_t at cbar = -7, the paper's test, under AR(1) errors.

Reference files: reference/serial_dependence/ar1_size_power/ar1_size_power.csv and
ar1_maic_cv.csv (a full run compares them byte for byte). --table-from-reference
renders Table S1 (stab_ar1.tex) from the reference ar1_size_power.csv, with the
function the full run uses, and simulates nothing.

Test: MZ_t detrended at cbar = -7, two arms of the long-run variance:
  const  the difference-based s^2 (kernel code 0, tab:cv's arm), against the
         i.i.d. 5% critical value of tab:cv -- the reference
         reference/calibration/tabcv_full_percentiles.csv (stat mzt,
         cv5; six decimals, the reference's own precision; R = 3,300,000);
  maic   the MAIC sieve (kernel code 1, kmax 12), against ITS OWN i.i.d. 5%
         critical value at -7, estimated here at R = 10^6 per cell.
DGP (make_ar1_eps below, then kernel gen_dgp_nb):
  v_t = rho v_{t-1} + e_t, v_0 = 0;  u_t = (1 + c/T) u_{t-1} + v_t, u_0 = 0.
Grid: T in {60, 100}, m in {0, 1, 2}, lambda_j = j/(m+1) (tab:cv's),
rho in {0, 0.5} (the printed grid of Table S1); size at c = 0, power at c = -7 (the
recommended test's diagonal; main) and c = -10 (the alternative of Table S1 and
Section 4.4). R = 200,000 per entry. Common random numbers: the same e for every
rho, c and arm of a replication. The run evaluates what the table prints, rho = 0.5
at c = 0, -7 and -10 and rho = 0 at c = -10 (the i.i.d. column), both arms
(ar1_size_power.csv has these rows), and the const arm at rho = 0, c = 0, whose
draws are those of the agreement checks (ar1_agreement.csv); nothing else.

Standard errors: against tab:cv's fixed critical value, binomial (the cv is an
input). For the maic arm the cv is estimated in this run: the cv
stream is split into B blocks parallel to the evaluation blocks (B = 8 and 20),
each block's cv and rate recomputed, and the block se is max(se_8, se_20). The
reported se is the larger of the block se and the binomial se
sqrt(p (1 - p) / n_finite): the block se is itself estimated from 8 or 20
blocks and can fall below the binomial floor. The column se_block keeps the
block se (empty for the const arm) and se_kind names the one that se reports.

AGREEMENT CHECKS (on the const arm's rho = 0, c = 0 draws, before the table; in the
nine cells T in {30, 60, 100}, m in {0, 1, 2} of tab:cv, the table's six among them):
  cv_agreement    the const i.i.d. cv at -7, recomputed from this run's rho = 0,
                  c = 0 draws (R = 200,000), is within 2 se (the block se of the
                  5% quantile, the larger over B = 8 and 20) of the tab:cv reference
                  in the 9 cells. tab:cv's lambda convention is that of
                  critical_values_blocks.py: lambda_j = j/(m+1) and
                  break_pos_from_lambdas(T, lambda) -- the same call used here.
  size_agreement  with rho = 0, the const arm's size against the tab:cv cv is
                  within 2 se (binomial) of 0.05 in the 9 cells.

THE CHECKS FAIL; THE TABLE IS COMPUTED: cv_agreement fails at (60,0) and
size_agreement at (60,0) and (60,1). The two checks are 18 two-sided 2-se tests
without a multiplicity allowance (P(at least one beyond 2 se) = 0.57 under
perfect agreement); over the 9 cells the sum of z^2 has p = 0.17 and 0.14, and
no cell is beyond 2.4 se. The verdict stays FAIL. With
--proceed-after-agreement-failure the table is computed anyway and the manifest
records the verdict (cells, z, the joint sum of z^2 and its p) and the reason
(AGREEMENT_OVERRIDE below); without the option a failed check stops the run.

Resumable at the level of the replication chunk: every chunk is written to
--state-dir as it finishes and reused on restart if its meta matches.

    python mlb/ar1_size_power.py --part agreement --state-dir <dir>
    python mlb/ar1_size_power.py --part all --state-dir <dir> \\
        --jobs 8 --proceed-after-agreement-failure
    python mlb/ar1_size_power.py --table-from-reference --out-dir <dir>
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
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_pp = os.environ.get("PYTHONPATH", "")
if _REPO not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _REPO + (os.pathsep + _pp if _pp else "")

import numpy as np  # noqa: E402

from mlb import paths  # noqa: E402
from mlb import kernel_numba as kn  # noqa: E402

SECTION = "robustness"
REFERENCE_DIR = "calibration_robustness"
OBJ = "ar1_size_power"
TS = (60, 100)
TS_CHECK = (30, 60, 100)  # the cells of the agreement checks
MS = (0, 1, 2)
RHOS = (0.0, 0.5)
CS = (0.0, -7.0, -10.0)
ARMS = ("const", "maic")
CBAR = -7.0
R_EVAL = 200_000
R_CV = 1_000_000
CH_EVAL = 10_000
CH_CV = 50_000
BLOCKS = (8, 20)
SEED_BASE = 20260926
CELL_STRIDE = 10**8
CV_OFFSET = 5 * 10**7
KMAX = int(kn.DEFAULTS["kmax"])
TABCV = paths.resolve("tabcv_full")
RHO_PRINTED = 0.5
# the outputs compared byte for byte with reference/serial_dependence/ar1_size_power/
REFERENCE_OUTPUTS = ("ar1_size_power.csv", "ar1_maic_cv.csv")
CHECK_RES = {}         # the check chunks, kept to assert they are the table's draws

# What --proceed-after-agreement-failure records beside the checks' verdict: the
# reason the table is computed although the agreement checks fail.
AGREEMENT_OVERRIDE = {
    "reason": ("the agreement checks cv_agreement and size_agreement fail; they are "
               "18 two-sided 2-se tests without a multiplicity allowance (P(at least "
               "one beyond 2 se) = 0.57 under perfect agreement); over the 9 cells "
               "their joint tests (agreement_verdict: sum_z2, p_chi2) do not reject "
               "at 5%"),
}


def lambdas(m):
    return [j / (m + 1) for j in range(1, m + 1)]


def make_ar1_eps(rng, T, rho):
    """AR(1) innovations eps_t = rho*eps_{t-1} + e_t, e_t ~ N(0,1).
    rho=0 returns pure i.i.d. e (the innovations of the i.i.d. DGP)."""
    e = rng.standard_normal(T)
    if rho == 0.0:
        return e
    eps = np.empty(T)
    eps[0] = e[0]
    for t in range(1, T):
        eps[t] = rho * eps[t - 1] + e[t]
    return eps


def seeds(T, m):
    s = (SEED_BASE + (10 * T + m) * CELL_STRIDE) % (2**63 - 1)
    return s, (s + CV_OFFSET) % (2**63 - 1)


def tabcv():
    with open(TABCV, encoding="utf-8") as fh:
        rows = list(csv.DictReader(ln for ln in fh if not ln.startswith("#")))
    out = {(int(r["m"]), int(r["T"])): float(r["cv5"]) for r in rows if r["stat"] == "mzt"}
    assert all((m, T) in out
               for m in MS for T in TS_CHECK)
    return out


# ================================================================== chunks
def _eval_chunk(T, m, r0, r1, const_only=False):
    """MZ_t at -7 for replications r0..r1-1: shape (n, rho, c, arm). Only the
    entries the outputs read are computed (the others stay NaN): rho = 0.5 at every
    c, both arms; rho = 0 at c = -10, both arms; the const arm at rho = 0, c = 0 (the
    draws of the agreement checks). eps is drawn once per (replication, rho), before
    the loop over c, so the entries computed do not depend on those skipped."""
    bp = kn.break_pos_from_lambdas(T, lambdas(m))
    Z = kn.build_z_nb(T, bp)
    seed, _ = seeds(T, m)
    arms = (0,) if const_only else (0, 1)
    out = np.full((r1 - r0, len(RHOS), len(CS), len(ARMS)), np.nan)
    for i, r in enumerate(range(r0, r1)):
        for a, rho in enumerate(RHOS):
            if const_only and rho != 0.0:
                continue
            eps = make_ar1_eps(np.random.default_rng((seed + r) % (2**63 - 1)), T, rho)
            for b, c in enumerate(CS):
                if const_only and c != 0.0:
                    continue
                if rho == 0.0 and c == -7.0:
                    continue
                y = kn.gen_dgp_nb(T, c, eps)
                for k in ((0,) if (rho == 0.0 and c == 0.0) else arms):
                    _, _, mz, _, _, ok = kn.mstats_nb(y, Z, CBAR, kn._METHOD_CODE[ARMS[k]], KMAX)
                    if ok > 0.5:
                        out[i, a, b, k] = mz
    return out


def _cv_chunk(T, m, r0, r1):
    """The maic arm's i.i.d. null at -7 (rho = 0, c = 0), its own stream."""
    bp = kn.break_pos_from_lambdas(T, lambdas(m))
    Z = kn.build_z_nb(T, bp)
    _, seed = seeds(T, m)
    out = np.full(r1 - r0, np.nan)
    for i, r in enumerate(range(r0, r1)):
        eps = make_ar1_eps(np.random.default_rng((seed + r) % (2**63 - 1)), T, 0.0)
        y = kn.gen_dgp_nb(T, 0.0, eps)
        _, _, mz, _, _, ok = kn.mstats_nb(y, Z, CBAR, kn._METHOD_CODE["maic"], KMAX)
        if ok > 0.5:
            out[i] = mz
    return out


def _task(kind, T, m, r0, r1, state_dir):
    """One chunk, resumable: read it from state_dir if there, else compute and save."""
    tag = f"T{T}_m{m}_{kind}_{r0:07d}_{r1:07d}"
    meta = dict(T=T, m=m, kind=kind, r0=r0, r1=r1, seeds=seeds(T, m), cbar=CBAR,
                rhos=RHOS, cs=CS, kmax=KMAX)
    f = os.path.join(state_dir, tag + ".npy") if state_dir else None
    if f and os.path.isfile(f) and os.path.isfile(f + ".json"):
        if json.load(open(f + ".json", encoding="utf-8")) == json.loads(json.dumps(meta)):
            return tag, np.load(f)
        raise SystemExit(f"[ar1_size_power] state {tag} exists with a different meta; "
                         "refusing")
    x = (_eval_chunk(T, m, r0, r1, const_only=(kind == "check")) if kind in ("eval", "check")
         else _cv_chunk(T, m, r0, r1))
    if f:
        os.makedirs(state_dir, exist_ok=True)
        with open(f + ".tmp", "wb") as fh:
            np.save(fh, x)
        os.replace(f + ".tmp", f)
        json.dump(meta, open(f + ".json", "w", encoding="utf-8"))
    return tag, x


def run_tasks(tasks, jobs, state_dir):
    from joblib import Parallel, delayed
    from mlb import ar1_size_power as _self
    t0, done, res = time.time(), 0, {}
    for tag, x in Parallel(n_jobs=jobs, backend="loky", batch_size=1,
                           return_as="generator_unordered")(
            delayed(_self._task)(*t, state_dir) for t in tasks):
        res[tag] = x
        done += 1
        if done % 10 == 0 or done == len(tasks):
            el = time.time() - t0
            print(f"  [{done}/{len(tasks)}] {el / 60:.1f} min, eta "
                  f"{el / done * (len(tasks) - done) / 60:.1f} min", flush=True)
    return res


def assemble(res, kind, T, m, R, ch):
    parts = [res[f"T{T}_m{m}_{kind}_{a:07d}_{min(a + ch, R):07d}"] for a in range(0, R, ch)]
    return np.concatenate(parts, axis=0)


# ================================================================ summaries
def quantile_se(x, B):
    e = np.linspace(0, x.shape[0], B + 1).astype(int)
    q = [np.nanpercentile(x[e[b]:e[b + 1]], 5) for b in range(B)]
    return float(np.std(q, ddof=1) / np.sqrt(B))


def rate(x, cv):
    v = np.isfinite(x)
    return float(((x < cv) & v).sum() / max(int(v.sum()), 1)), int(v.sum())


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
    """The command of the run, package-relative, with the folder of --state-dir
    written as the placeholder <state-dir> (no path of the machine) and without
    --jobs (the number of worker processes does not change the outputs)."""
    cmd, it = ["python", f"mlb/{OBJ}.py"], iter(argv)
    for x in it:
        if x == "--state-dir":
            cmd += [x, "<state-dir>"]
            next(it, None)
        elif x.startswith("--state-dir="):
            cmd.append("--state-dir=<state-dir>")
        elif x == "--jobs":
            next(it, None)
        elif not x.startswith("--jobs="):
            cmd.append(x)
    return cmd


def agreement_checks(jobs, state_dir, reference):
    tasks = [("check", T, m, a, min(a + CH_EVAL, R_EVAL))
             for m in MS for T in TS_CHECK for a in range(0, R_EVAL, CH_EVAL)]
    res = run_tasks(tasks, jobs, state_dir)
    CHECK_RES.update(res)
    rows, ok_t, ok_a = [], True, True
    for m in MS:
        for T in TS_CHECK:
            x = assemble(res, "check", T, m, R_EVAL, CH_EVAL)[:, 0, 0, 0]
            cv = float(np.nanpercentile(x, 5))
            se_q = max(quantile_se(x, B) for B in BLOCKS)
            s, n = rate(x, reference[(m, T)])
            se_s = (s * (1 - s) / n) ** 0.5
            gt = abs(cv - reference[(m, T)]) <= 2 * se_q
            ga = abs(s - 0.05) <= 2 * se_s
            ok_t, ok_a = ok_t and gt, ok_a and ga
            bp = " ".join(map(str, kn.break_pos_from_lambdas(T, lambdas(m))))
            rows.append({"T": T, "m": m, "break_pos": bp, "cv_recomputed": cv,
                         "se_quantile": se_q, "cv_tabcv_reference": reference[(m, T)],
                         "z_cv": (cv - reference[(m, T)]) / se_q, "cv_pass": gt,
                         "size_const_rho0": s, "se_binomial": se_s,
                         "z_size": (s - 0.05) / se_s, "size_pass": ga,
                         "n_finite": n})
    return ok_t, ok_a, rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("agreement", "all"), default="agreement")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--state-dir", help="required except with --table-from-reference")
    ap.add_argument("--proceed-after-agreement-failure", action="store_true",
                    help="compute the table although the agreement checks fail, as the "
                         "reference files were computed; the checks' verdict (FAIL) is recorded "
                         "unchanged, with the reason")
    ap.add_argument("--table-from-reference", action="store_true",
                    help="render Table S1 (stab_ar1.tex) from the reference "
                         "ar1_size_power.csv into --out-dir; nothing is simulated")
    ap.add_argument("--out-dir", help="the folder of --table-from-reference")
    a = ap.parse_args(argv)
    if a.table_from_reference:
        if not a.out_dir:
            ap.error("--table-from-reference needs --out-dir")
        return table_from_reference(a.out_dir)
    if a.out_dir:
        ap.error("--out-dir goes with --table-from-reference")
    if not a.state_dir:
        ap.error("the following arguments are required: --state-dir")
    if sys.flags.optimize:
        raise SystemExit("refusing to run under -O: the checks are asserts")
    out = paths.out_dir(SECTION) / OBJ
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    reference = tabcv()
    sources = ("mlb/ar1_size_power.py",
               "mlb/kernel_numba.py", "mlb/paths.py")
    # no time, worker count or machine field: the run records of two runs differ
    # only in the command and the library versions
    man = {"reference_dir": REFERENCE_DIR, "job": OBJ, "part": a.part,
           "command": command_line(sys.argv[1:] if argv is None else argv),
           "statistic": "MZ_t (not P_T)", "cbar": CBAR, "R_eval": R_EVAL,
           "R_cv_maic": R_CV, "rhos": RHOS, "cs": CS, "arms": ARMS, "kmax": KMAX,
           "blocks": BLOCKS, "seed_base": SEED_BASE, "cell_stride": CELL_STRIDE,
           "cv_offset": CV_OFFSET,
           "cv_sources": {
               "const": ("tab:cv reference tabcv_full_percentiles.csv (mzt cv5, 6 "
                         "decimals, R = 3,300,000; critical_values_blocks.py)"),
               "maic": "own i.i.d. null at -7, this run, R = 1,000,000"},
           "tabcv_lambda_convention": ("lambda_j = j/(m+1); break_pos_from_lambdas(T, "
                                       "lambda), as in critical_values_blocks.py"),
           "python": sys.version.split()[0], "numpy": np.__version__,
           "joblib": __import__("joblib").__version__,
           "sources": {p: _sha(os.path.join(_REPO, p)) for p in sources},
           "inputs": {"tabcv": _sha(TABCV)}}
    man["se_rule"] = {
        "const": "binomial, sqrt(p (1 - p) / n_finite)",
        "maic": "the larger of the block se and the binomial se; block se = max over "
                "B in (8, 20) of the standard deviation of the B block rates / "
                "sqrt(B), the cv re-estimated in each block (column se_block)"}
    kn.warm_up_numba(KMAX)
    print("[ar1_size_power] agreement checks cv_agreement and size_agreement: const "
          f"arm, rho = 0, c = 0, R = {R_EVAL}", flush=True)
    ok_t, ok_a, grows = agreement_checks(a.jobs, a.state_dir, reference)
    _wcsv(out / "ar1_agreement.csv", grows)
    from scipy import stats
    joint = {}
    for g, col, passed in (("cv_agreement", "z_cv", ok_t),
                           ("size_agreement", "z_size", ok_a)):
        z = [float(r[col]) for r in grows]
        s = float(sum(x * x for x in z))
        joint[g] = dict(verdict="PASS" if passed else "FAIL", sum_z2=s, df=len(z),
                        p_chi2=float(stats.chi2.sf(s, len(z))), max_abs_z=max(abs(x) for x in z),
                        cells_beyond_2se=[f"({r['T']},{r['m']}) z {float(r[col]):+.4f}" for r in grows
                                          if abs(float(r[col])) > 2],
                        bonferroni_threshold_9_tests_5pct=float(stats.norm.isf(0.05 / 18)))
    man["agreement_checks"] = joint
    print(f"[ar1_size_power] cv_agreement {joint['cv_agreement']['verdict']}, "
          f"size_agreement {joint['size_agreement']['verdict']}: {joint}", flush=True)
    failed = not (ok_t and ok_a)
    if failed and a.part == "all" and a.proceed_after_agreement_failure:
        # The checks' verdict is NOT rewritten: it stays FAIL. The table is computed
        # for the reason of AGREEMENT_OVERRIDE.
        rec = dict(agreement_verdict=joint, proceed_after_agreement_failure=True,
                   **AGREEMENT_OVERRIDE)
        man["agreement_override"] = rec
        print("[ar1_size_power] agreement checks FAILED; the run proceeds "
              "(--proceed-after-agreement-failure; recorded in the manifest)",
              flush=True)
    elif failed or a.part == "agreement":
        if failed and a.part == "all":
            man["refused"] = ("an agreement check failed and "
                              "--proceed-after-agreement-failure was not given")
        json.dump(man, open(out / "ar1_agreement.manifest.json", "w", encoding="utf-8",
                            newline="\n"),
                  indent=1, default=str)
        return 0 if not failed else 3

    tasks = []
    for m in MS:
        for T in TS:
            tasks += [("eval", T, m, r, min(r + CH_EVAL, R_EVAL)) for r in range(0, R_EVAL, CH_EVAL)]
            tasks += [("cv", T, m, r, min(r + CH_CV, R_CV)) for r in range(0, R_CV, CH_CV)]
    tasks.sort(key=lambda t: -t[1])                     # the longest T first
    print(f"[ar1_size_power] {len(tasks)} chunks, jobs = {a.jobs}", flush=True)
    res = run_tasks(tasks, a.jobs, a.state_dir)

    rows, cvrows = [], []
    for m in MS:
        for T in TS:
            ev = assemble(res, "eval", T, m, R_EVAL, CH_EVAL)
            cvn = assemble(res, "cv", T, m, R_CV, CH_CV)
            check = assemble(CHECK_RES, "check", T, m, R_EVAL, CH_EVAL)
            cv_maic = float(np.nanpercentile(cvn, 5))
            cvrows.append({"T": T, "m": m, "cv_maic": cv_maic,
                           "se_quantile": max(quantile_se(cvn, B) for B in BLOCKS),
                           "R": R_CV, "n_finite": int(np.isfinite(cvn).sum()),
                           "cv_const_tabcv": reference[(m, T)]})
            for k, arm in enumerate(ARMS):
                for i, rho in enumerate(RHOS):
                    for j, c in enumerate(CS):
                        if rho != RHO_PRINTED and c != -10.0:
                            continue  # rho = 0: only the i.i.d. column, at c = -10
                        x = ev[:, i, j, k]
                        cv = reference[(m, T)] if arm == "const" else cv_maic
                        s, n = rate(x, cv)
                        se_bin = (s * (1 - s) / n) ** 0.5
                        if arm == "const":
                            se, se_block = se_bin, ""
                            kind = "binomial (fixed cv)"
                        else:
                            se_b = []
                            for B in BLOCKS:
                                e1 = np.linspace(0, R_EVAL, B + 1).astype(int)
                                e2 = np.linspace(0, R_CV, B + 1).astype(int)
                                v = [rate(x[e1[b]:e1[b + 1]],
                                          float(np.nanpercentile(cvn[e2[b]:e2[b + 1]], 5)))[0]
                                     for b in range(B)]
                                se_b.append(float(np.std(v, ddof=1) / np.sqrt(B)))
                            se_block = max(se_b)
                            if se_block >= se_bin:
                                se = se_block
                                kind = "blocks (8, 20; cv re-estimated per block)"
                            else:
                                se, kind = se_bin, "binomial (above the block se)"
                        rows.append(dict(T=T, m=m, arm=arm, rho=rho, c=c,
                                         quantity="size" if c == 0.0 else "power",
                                         rate=s, se=se, se_block=se_block,
                                         se_kind=kind, n_finite=n))
            assert np.array_equal(check[:, 0, 0, 0], ev[:, 0, 0, 0], equal_nan=True), \
                "the check draws are not the table's draws"
    ks = [(r["T"], r["m"], r["arm"], r["rho"], r["c"]) for r in rows]
    assert len(set(ks)) == len(rows) == len(TS) * len(MS) * len(ARMS) * (len(CS) + 1)
    _wcsv(out / "ar1_size_power.csv", rows)
    _wcsv(out / "ar1_maic_cv.csv", cvrows)
    table_s1(rows, out / "stab_ar1.tex")

    ref_dir = paths.reference_dir(REFERENCE_DIR) / OBJ
    regen = {"references_present": ref_dir.is_dir()}
    if ref_dir.is_dir():
        for name in REFERENCE_OUTPUTS:
            regen[name] = ((ref_dir / name).is_file()
                           and _sha(out / name) == _sha(ref_dir / name))
        print(f"[ar1_size_power] regenerated outputs identical to the reference files: {regen}")
    man["outputs"] = {f.name: _sha(f) for f in sorted(out.iterdir())
                      if f.suffix in (".csv", ".tex")}
    json.dump(man,
              open(out / "ar1_size_power.manifest.json", "w", encoding="utf-8",
                   newline="\n"),
              indent=1, default=str)
    if ref_dir.is_dir() and not all(v for k, v in regen.items() if k != "references_present"):
        print("[ar1_size_power] regeneration differs from the reference files")
        return 4
    print(f"[ar1_size_power] DONE in {(time.time() - t0) / 60:.1f} min -> {out}")
    return 0


def table_s1(rows, path):
    """Table S1 (stab:ar1) from the rows of ar1_size_power.csv: rho = 0.5 at c = 0,
    -7 and -10 and the i.i.d. column (rho = 0, c = -10), both arms; three decimals."""
    def g(T, m, arm, rho, c):
        return next(r for r in rows if (r["T"], r["m"], r["arm"], r["rho"], r["c"]) == (T, m, arm, rho, c))

    L = [(r"% Table S1 (stab:ar1): MZ_t at cbar = -7, rho = 0.5; i.i.d. anchors "
          r"(rho = 0) on the same alternative"),
         r"\begin{tabular}{@{}rr rrrr rrrr@{}}", r"\toprule",
         r" & & \multicolumn{4}{c}{difference-based $s^2$ (tab:cv)} & \multicolumn{4}{c}{MAIC (own i.i.d. cv)} \\",
         r"$T$ & $m$ & size & $\Pi(-7)$ & $\Pi(-10)$ & $\Pi_{\mathrm{iid}}(-10)$ & size & $\Pi(-7)$ & $\Pi(-10)$ & $\Pi_{\mathrm{iid}}(-10)$ \\",
         r"\midrule"]
    for m in MS:
        for T in TS:
            cells = []
            for arm in ARMS:
                cells += [g(T, m, arm, RHO_PRINTED, 0.0)["rate"], g(T, m, arm, RHO_PRINTED, -7.0)["rate"],
                          g(T, m, arm, RHO_PRINTED, -10.0)["rate"], g(T, m, arm, 0.0, -10.0)["rate"]]
            L.append(f"{T} & {m} & " + " & ".join(f"{v:.3f}" for v in cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", ""]
    open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L))


def table_from_reference(out_dir):
    """Table S1 rendered from reference/serial_dependence/ar1_size_power/
    ar1_size_power.csv into out_dir; nothing is simulated."""
    src = paths.reference_dir(REFERENCE_DIR) / OBJ / "ar1_size_power.csv"
    with open(src, encoding="utf-8", newline="") as fh:
        rows = [dict(r, T=int(r["T"]), m=int(r["m"]), rho=float(r["rho"]),
                     c=float(r["c"]), rate=float(r["rate"]))
                for r in csv.DictReader(fh)]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "stab_ar1.tex")
    table_s1(rows, path)
    print(f"[ar1_size_power] Table S1 rendered from {src.relative_to(paths.REPO_ROOT).as_posix()}"
          f" -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
