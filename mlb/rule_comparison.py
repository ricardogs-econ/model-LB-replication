"""The rule comparison of Section 4.2: following the surface against -7, whole grid.

Objects: the comparison of Section 4.2 (sec:rule) on the configurations of Table 1
(tab:cbar), with the critical-value convention of Table 2 (tab:cv) (reference files under
reference/calibration/rule_comparison/).

Statistic: feasible MZ_t, GLS-detrended at the node cbar, difference-based s^2
(the kernel's code 0 -- tab:cv's arm: s^2 = sum (Delta ytilde)^2 / (T - 1)),
each node against the 5% critical value of its OWN finite-sample null on the
same draws. DGP i.i.d. Gaussian, u_0 = 0, break magnitudes zero (invariance).
The engine is exact-linear and matrix-valued (Dmat, Lmat and mzt below; the
detrender with the break POSITIONS given); the agreement check engine_agreement
compares it with the kernel's mstats_nb(code 0) draw for draw.

Configurations: the 43 (m, T) cells of Table 1 x {lambda_j = j/(m+1); lambda_ext =
argmax_lambda |cbar_half(m, T, lambda) + 7| on the exact surface}; one for
m = 0. Nodes: -7, cbar_half(config) exact (from the surface; if the equispaced
configuration is not on the surface grid its cbar_half is computed here with
envelope_exact.py, column cbar_half_source), and cbarbar_half(m, T), the cell mean that
tab:cbar prints. Alternatives c in {-5, -10, -15, -20} and their simple mean
W4. R = 200,000, CRN across nodes; the null and each alternative draw on
their own streams (seed + p * 10^9).

Standard errors: blocks B = 8 and 20, critical value AND power recomputed per
block, for the levels and the differences; se = max(se_8, se_20).

Classification: Delta_X(c) = Pi(-7; c) - Pi(X; c), X in {cbar_half(config),
cbarbar_half(m,T)}. In W4 each configuration is "-7 better" (z > 2),
"indistinguishable" (|z| <= 2) or "X better" (z < -2) (column classification of
rule_configs.csv).
Fixed-index grid: at (2,60), equispaced, the grid G = {-2.0, -2.5, ..., -12.0} u
{-13.5, -21.7} u {cbar_half}; at c = -10, the argmax, the distance of -7 to the
max, and the ratios (max - Pi(-13.5))/(max - Pi(-7)) and
(max - Pi(-21.7))/(max - Pi(-7)) (rule_fixed_index.csv, read by fixed_index.py).

AGREEMENT CHECKS (first; a failure stops the run): engine_agreement (the engine is
kernel code 0, draw for draw, on 200 draws); power_agreement (Pi(-7; c = -10) at
(2,60), lambda = (1/3, 2/3), within 2 se of the reference overdetrend_power.json).

Resumable: each configuration's summary is written to --state-dir as it
finishes and reused on restart if its meta matches.

    python mlb/rule_comparison.py --part agreement
    python mlb/rule_comparison.py --part all --state-dir <dir> \\
        --jobs 4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import zlib

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

from mlb import paths  # noqa: E402
from mlb import envelope_exact as ee  # noqa: E402
from mlb import kernel_numba as kn  # noqa: E402

SECTION = "critical_values"
OBJ = "rule_comparison"
R = 200_000
CHUNK = 25_000
ALPHA = 0.05
ALTS = (-5.0, -10.0, -15.0, -20.0)
BLOCKS = (8, 20)
SEED = 20260925
STRIDE = 10**9
SURFACE = paths.reference_dir("critical_values") / "cbar_surface_exact" / "cbar_surface_exact.csv"
SURFACE_SHA256 = "d020889e7a0e5a7756e4bee458b2e51e9b985bb204cdf75c0f18d17d3541d6dc"
OD_POWER = paths.resolve("overdetrend_power")
FIXED_INDEX_CELLS = ((2, 60),)
GRID_G = tuple(np.round(np.arange(-2.0, -12.0 - 1e-9, -0.5), 2)) + (-13.5, -21.7)
N_SAME = 200
# the outputs compared with their reference file in reference/ after --part all
REFERENCE_OUTPUTS = ("rule_configs.csv", "rule_fixed_index.csv", "rule_levels.csv")


# ============================================================ configurations
def load_surface():
    sha = hashlib.sha256(
        open(SURFACE, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    assert sha == SURFACE_SHA256, f"surface sha256 {sha}"
    rows = list(csv.DictReader(open(SURFACE, encoding="utf-8")))
    assert len({r["config_key"] for r in rows}) == len(rows) == 404
    for r in rows:
        r["T"], r["m"], r["c"] = int(r["T"]), int(r["m"]), float(r["cbar_half_exact"])
        r["lam"] = tuple(float(r[f"lambda{i}"]) for i in range(1, r["m"] + 1))
    return rows


def configurations(rows):
    cells = {}
    for r in rows:
        cells.setdefault((r["m"], r["T"]), []).append(r)
    assert len(cells) == 43
    cfgs = []
    for (m, T), v in sorted(cells.items()):
        mean = float(np.mean([r["c"] for r in v]))
        lam_eq = tuple(j / (m + 1) for j in range(1, m + 1))
        k_eq = kn.config_key(T, m, list(lam_eq))
        hit = [r for r in v if r["config_key"] == k_eq]
        if hit:
            c_eq, src_eq = hit[0]["c"], "surface"
        else:
            c_eq, src_eq = float(ee.half_power_point(T, lam_eq)[0]), "computed here (envelope_exact)"
        cfgs.append(dict(m=m, T=T, kind="equispaced", lam=lam_eq, key=k_eq, cbar_half=c_eq,
                         cbar_half_source=src_eq, cbarbar=mean))
        if m >= 1:
            ext = max(v, key=lambda r: (abs(r["c"] + 7.0), r["config_key"]))
            if ext["config_key"] != k_eq:
                cfgs.append(dict(m=m, T=T, kind="lambda_ext", lam=ext["lam"], key=ext["config_key"],
                                 cbar_half=ext["c"], cbar_half_source="surface", cbarbar=mean))
    for c in cfgs:
        c["bp"] = tuple(ee.break_positions(c["T"], c["lam"]))
        c["id"] = f"{c['key']}__{c['kind']}"
        nodes = [-7.0, c["cbar_half"], c["cbarbar"]]
        if (c["m"], c["T"]) in FIXED_INDEX_CELLS and c["kind"] == "equispaced":
            nodes += [x for x in GRID_G if x not in nodes]
        c["nodes"] = nodes
        c["seed"] = (SEED + 1000003 * c["T"]
                     + zlib.crc32(c["id"].encode("utf-8")) % 10**6) % (2**63 - 1)
    ids = [c["id"] for c in cfgs]
    assert len(set(ids)) == len(cfgs)
    return cfgs


# ================================================================== engine
def Dmat(T: int, a: float) -> np.ndarray:
    """Quasi-differencing operator (1 - a L), first row left in levels."""
    D = np.eye(T)
    idx = np.arange(1, T)
    D[idx, idx - 1] = -a
    return D


def Lmat(T: int, c: float) -> np.ndarray:
    """Lower-triangular partial-sum operator: u = L(c) eps with u_0 = 0,
    u_t = (1 + c/T) u_{t-1} + eps_t.  L[t, s] = a^(t-s) for s <= t."""
    a = 1.0 + c / T
    L = np.zeros((T, T))
    pw = a ** np.arange(T)
    for t in range(T):
        L[t, : t + 1] = pw[: t + 1][::-1]
    return L


def mzt(Ytil: np.ndarray, T: int) -> np.ndarray:
    """Demeaned MZ_t for each column of the detrended-level array (T x N).

    MZ_t = MZ_alpha * MSB, with
        MZ_alpha = (T^{-1} yt_T^2 - s^2) / (2 T^{-2} sum_{t=1}^{T} yt_{t-1}^2)
        MSB      = ( T^{-2} sum yt_{t-1}^2 / s^2 )^{1/2}
        s^2      = (T-1)^{-1} sum_{t=2}^{T} (yt_t - yt_{t-1})^2   [k = 0]
    """
    ss = np.sum(Ytil * Ytil, axis=0)
    yT2 = Ytil[-1] ** 2
    Sy2 = (ss - yT2) / T ** 2                    # T^{-2} sum_{t=1}^{T} yt_{t-1}^2
    dy = np.diff(Ytil, axis=0)
    s2 = np.sum(dy * dy, axis=0) / (T - 1)
    MZa = (yT2 / T - s2) / (2.0 * Sy2)
    MSB = np.sqrt(Sy2 / s2)
    return MZa * MSB


def detrender_bp(T, bp, cbar):
    """The exact-linear GLS detrender M (ytilde = M y) at abar = 1 + cbar/T, with the
    break positions given (DU_j = 1{t > tau_j})."""
    a = 1.0 + cbar / T
    cols = [np.ones(T)]
    for tau in bp:
        du = np.zeros(T)
        du[tau:] = 1.0
        cols.append(du)
    Z = np.column_stack(cols)
    D = Dmat(T, a)
    Zt = D @ Z
    M = np.eye(T) - Z @ np.linalg.solve(Zt.T @ Zt, Zt.T @ D)
    assert float(np.max(np.abs(M @ Z))) < 1e-8
    return M


def simulate(T, bp, nodes, seed, nrep=R, chunk=CHUNK):
    """null and each alternative on its own stream seed + p * 10^9; within a
    pass, chunks of E (T x n) as run_cell draws them; CRN across nodes."""
    Ms = [detrender_bp(T, bp, cb) for cb in nodes]
    out = {}
    for p, c in enumerate((0.0,) + ALTS):
        rng = np.random.default_rng((seed + p * STRIDE) % (2**63 - 1))
        L = Lmat(T, c)
        X = np.empty((nrep, len(nodes)))
        done = 0
        while done < nrep:
            n = min(chunk, nrep - done)
            U = L @ rng.standard_normal((T, n))
            for j, M in enumerate(Ms):
                X[done:done + n, j] = mzt(M @ U, T)
            done += n
        out[c] = X
    return out


def _levels(null, alts_arr):
    cv = np.quantile(null, ALPHA, axis=0)
    pw = np.array([(a < cv[None, :]).mean(axis=0) for a in alts_arr])     # (n_alt, k)
    return cv, pw


def summarize(sim, nodes):
    """Full-sample levels and differences vs node 0 (-7), and the block se."""
    null = sim[0.0]
    alts_arr = [sim[c] for c in ALTS]
    cv, pw = _levels(null, alts_arr)
    w4 = pw.mean(axis=0)
    se = {}
    for B in BLOCKS:
        e = np.linspace(0, null.shape[0], B + 1).astype(int)
        PB = np.empty((B, len(ALTS), len(nodes)))
        for b in range(B):
            _, PB[b] = _levels(null[e[b]:e[b + 1]], [a[e[b]:e[b + 1]] for a in alts_arr])
        W4B = PB.mean(axis=1)
        D = PB[:, :, [0]] - PB                 # Delta per alternative
        DW = W4B[:, [0]] - W4B
        f = np.sqrt(B)
        se[B] = dict(pw=PB.std(axis=0, ddof=1) / f, w4=W4B.std(axis=0, ddof=1) / f,
                     d=D.std(axis=0, ddof=1) / f, dw=DW.std(axis=0, ddof=1) / f)
    mx = {k: np.maximum(se[8][k], se[20][k]) for k in se[8]}
    return dict(nodes=list(nodes), cv=cv.tolist(), power=pw.tolist(), w4=w4.tolist(),
                delta=(pw[:, [0]] - pw).tolist(), delta_w4=(w4[0] - w4).tolist(),
                se_power=mx["pw"].tolist(),
                se_delta=mx["d"].tolist(), se_delta_w4=mx["dw"].tolist())


def run_config(c):
    s = summarize(simulate(c["T"], c["bp"], c["nodes"], c["seed"]), c["nodes"])
    return c["id"], s


def meta_of(c):
    return dict(id=c["id"], T=c["T"], bp=list(c["bp"]), nodes=c["nodes"], seed=c["seed"],
                R=R, chunk=CHUNK, alts=list(ALTS), blocks=list(BLOCKS))


# ======================================================== agreement checks
def agreement_checks(cfgs):
    res, ok = {}, True

    def rec(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        res[name] = dict(passed=bool(passed), **detail)
        print(f"[rule_comparison] {name}: {'PASS' if passed else 'FAIL'} {detail}",
              flush=True)

    # engine_agreement: the matrix engine is the kernel's MZ_t (code 0), draw for draw
    worst = 0.0
    for T, bp, cb in ((60, (20, 40), -7.0), (30, (15,), -9.0), (300, (), -7.2)):
        M = detrender_bp(T, bp, cb)
        Z = kn.build_z_nb(T, np.asarray(bp, dtype=np.int64))
        rng = np.random.default_rng(12345 + T)
        for _ in range(N_SAME // 3 + 1):
            e = rng.standard_normal(T)
            u = Lmat(T, -5.0) @ e
            mine = float(mzt((M @ u)[:, None], T)[0])
            _, _, mz, _, _, okk = kn.mstats_nb(u, Z, float(cb), 0, 12)
            if okk > 0.5:
                worst = max(worst, abs(mine - mz) / max(1.0, abs(mz)))
    rec("engine_agreement", worst < 1e-7,
        dict(max_rel_diff_vs_kernel_code0=worst, draws=N_SAME))

    # power_agreement: Pi(-7; c = -10) at (2,60), lambda = (1/3, 2/3), vs the reference
    gold = json.load(open(OD_POWER, encoding="utf-8"))
    assert int(gold["T"]) == 60 and int(gold["m"]) == 2
    g = gold["by_c"]["-10.0"]["cells"]["-7.0"]
    c260 = next(c for c in cfgs if (c["m"], c["T"], c["kind"]) == (2, 60, "equispaced"))
    assert c260["bp"] == tuple(int(x) for x in kn.break_pos_from_lambdas(60, [1 / 3, 2 / 3]))
    s = summarize(simulate(60, c260["bp"], [-7.0], c260["seed"]), [-7.0])
    i10 = ALTS.index(-10.0)
    p, se = s["power"][i10][0], s["se_power"][i10][0]
    rec("power_agreement", abs(p - g["power"]) <= 2 * se,
        {"power": p, "se": se, "reference": g["power"], "reference_se": g.get("se_total"),
         "z": (p - g["power"]) / se, "bp": list(c260["bp"])})
    return ok, res


# ========================================================== classification
def classify(z):
    return "-7 better" if z > 2 else ("X better" if z < -2 else "indistinguishable")


def read_all(cfgs, S):
    rows = []
    for c in cfgs:
        s = S[c["id"]]
        nodes = s["nodes"]
        for X, idx in (("cbar_half(config)", 1), ("cbarbar_half(m,T)", 2)):
            d, se = s["delta_w4"][idx], s["se_delta_w4"][idx]
            z = d / se if se > 0 else (0.0 if d == 0 else float("inf"))
            rows.append(dict(m=c["m"], T=c["T"], kind=c["kind"], config_key=c["key"],
                             break_pos=" ".join(map(str, c["bp"])), X=X, X_value=nodes[idx],
                             cbar_half_source=c["cbar_half_source"],
                             power_w4_minus7=s["w4"][0], power_w4_X=s["w4"][idx],
                             delta_w4=d, se_delta_w4=se, z=z,
                             classification=classify(z),
                             **{f"delta_c{int(a)}": s["delta"][i][idx] for i, a in enumerate(ALTS)},
                             **{f"se_delta_c{int(a)}": s["se_delta"][i][idx] for i, a in enumerate(ALTS)},
                             **{f"power_minus7_c{int(a)}": s["power"][i][0] for i, a in enumerate(ALTS)},
                             cv_minus7=s["cv"][0], cv_X=s["cv"][idx]))
    ks = [(r["m"], r["T"], r["kind"], r["X"]) for r in rows]
    assert len(set(ks)) == len(rows)
    return rows


def fixed_index_rows(cfgs, S):
    out = []
    for (m, T) in FIXED_INDEX_CELLS:
        c = next(x for x in cfgs if (x["m"], x["T"], x["kind"]) == (m, T, "equispaced"))
        s = S[c["id"]]
        nodes = s["nodes"]
        G = [j for j, x in enumerate(nodes) if x in GRID_G or j in (0, 1)]
        for i, a in enumerate(ALTS):
            if a != -10.0:
                continue
            pw = s["power"][i]
            jmax = max(G, key=lambda j: pw[j])
            row = dict(m=m, T=T, c=a, argmax_cbar=nodes[jmax], power_max=pw[jmax],
                       power_minus7=pw[0], dist_minus7_to_max=pw[jmax] - pw[0])
            j135, j217 = nodes.index(-13.5), nodes.index(-21.7)
            den = pw[jmax] - pw[0]
            row.update(ratio_135=(pw[jmax] - pw[j135]) / den if den > 0 else "",
                       ratio_217=(pw[jmax] - pw[j217]) / den if den > 0 else "")
            out.append(row)
    return out


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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=("agreement", "all"), default="agreement")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--state-dir", default=None)
    a = ap.parse_args(argv)
    if sys.flags.optimize:
        raise SystemExit("refusing to run under -O: the checks are asserts")
    out = paths.out_dir(SECTION) / OBJ
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfgs = configurations(load_surface())
    configs = [{"id": c["id"], "m": c["m"], "T": c["T"], "kind": c["kind"],
                "bp": c["bp"], "lam": c["lam"], "cbar_half": c["cbar_half"],
                "cbar_half_source": c["cbar_half_source"], "cbarbar": c["cbarbar"],
                "seed": c["seed"], "n_nodes": len(c["nodes"])} for c in cfgs]
    sources = ("mlb/rule_comparison.py",
               "mlb/envelope_exact.py", "mlb/kernel_numba.py",
               "mlb/paths.py")
    # no time, worker count or machine field: the run records of two runs are equal
    # apart from the command and the library versions
    man = {"job": OBJ, "part": a.part,
           "command": command_line(sys.argv[1:] if argv is None else argv),
           "R": R, "chunk": CHUNK, "alts": ALTS, "blocks": BLOCKS, "seed": SEED,
           "stride": STRIDE, "n_configs": len(cfgs),
           "fixed_index_cells": FIXED_INDEX_CELLS,
           "fixed_index_grid": GRID_G, "configs": configs, "envelope": ee.conventions(),
           "python": sys.version.split()[0], "numpy": np.__version__,
           "sources": {p: _sha(os.path.join(_REPO, p)) for p in sources},
           "inputs": {"surface": _sha(SURFACE), "overdetrend_power": _sha(OD_POWER)}}
    print(f"[rule_comparison] {len(cfgs)} configurations", flush=True)
    ok, gres = agreement_checks(cfgs)
    man["agreement_checks"] = gres
    if not ok or a.part == "agreement":
        json.dump(man, open(out / "rule_agreement.manifest.json", "w", encoding="utf-8",
                            newline="\n"),
                  indent=1, default=str)
        print(f"[rule_comparison] agreement checks {'PASS' if ok else 'FAILED'} "
              f"({time.time() - t0:.0f}s)")
        return 0 if ok else 3

    S, todo = {}, []
    for c in cfgs:
        f = os.path.join(a.state_dir, c["id"] + ".json") if a.state_dir else None
        if f and os.path.isfile(f):
            d = json.load(open(f, encoding="utf-8"))
            if d["meta"] != json.loads(json.dumps(meta_of(c))):
                raise SystemExit(f"[rule_comparison] state {c['id']} has a different "
                                 "meta; refusing")
            S[c["id"]] = d["summary"]
        else:
            todo.append(c)
    print(f"[rule_comparison] {len(S)} configurations from state, {len(todo)} to run, jobs = {a.jobs}",
          flush=True)
    todo.sort(key=lambda c: -(c["T"] ** 2) * len(c["nodes"]))     # longest first
    from joblib import Parallel, delayed
    from mlb import rule_comparison as _self
    n = len(S)
    for cid, s in Parallel(n_jobs=a.jobs, backend="loky", batch_size=1,
                           return_as="generator_unordered")(
            delayed(_self.run_config)(c) for c in todo):
        S[cid] = s
        n += 1
        if a.state_dir:
            c = next(x for x in cfgs if x["id"] == cid)
            os.makedirs(a.state_dir, exist_ok=True)
            tmp = os.path.join(a.state_dir, cid + ".json.tmp")
            json.dump(dict(meta=meta_of(c), summary=s), open(tmp, "w", encoding="utf-8"))
            os.replace(tmp, os.path.join(a.state_dir, cid + ".json"))
        print(f"  [{n}/{len(cfgs)}] {cid}  ({(time.time() - t0) / 60:.1f} min)",
              flush=True)
    assert set(S) == {c["id"] for c in cfgs}

    rows = read_all(cfgs, S)
    _wcsv(out / "rule_configs.csv", rows)
    _wcsv(out / "rule_fixed_index.csv", fixed_index_rows(cfgs, S))
    lv = []
    for c in cfgs:
        s = S[c["id"]]
        for j, x in enumerate(s["nodes"]):
            for i, alt in enumerate(ALTS):
                lv.append(dict(config=c["id"], m=c["m"], T=c["T"], kind=c["kind"],
                               node_index=j, node_role=(["-7", "cbar_half(config)",
                                                         "cbarbar_half(m,T)"][j] if j < 3
                                                        else "fixed-index grid"),
                               node=x, c=alt, cv=s["cv"][j], power=s["power"][i][j],
                               se_power=s["se_power"][i][j]))
    ks = [(r["config"], r["node_index"], r["c"]) for r in lv]
    assert len(set(ks)) == len(lv)
    _wcsv(out / "rule_levels.csv", lv)
    gdir = paths.reference_dir(SECTION) / OBJ
    regen = {"references_present": gdir.is_dir()}
    if gdir.is_dir():
        for name in REFERENCE_OUTPUTS:
            regen[name] = (gdir / name).is_file() and _sha(out / name) == _sha(gdir / name)
        print("[rule_comparison] regenerated outputs identical to the reference files: "
              f"{regen}")
    man["outputs"] = {f.name: _sha(f) for f in sorted(out.iterdir())
                      if f.suffix in (".csv", ".tex")}
    json.dump(man,
              open(out / "rule_comparison.manifest.json", "w", encoding="utf-8",
                   newline="\n"),
              indent=1, default=str)
    if gdir.is_dir() and not all(v for k, v in regen.items() if k != "references_present"):
        print("[rule_comparison] regeneration differs from the reference files")
        return 4
    print(f"[rule_comparison] DONE in {(time.time() - t0) / 60:.1f} min -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
