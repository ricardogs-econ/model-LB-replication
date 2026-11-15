"""The scope of the variance in the PPP application: the unit-root tests under a
variance that changes at the regime dates, for the principal design E and the
sensitivity SJ.

The application's registered null has a constant innovation
variance. This producer measures what a variance that changes at the dates of the
regimes does to the registered columns, without changing them.

MODEL (LB-V). y_t = mu_0 + sum_j delta_j DU_jt + u_t; under H0, Delta u_t =
sum_{i=1}^{p} phi_i Delta u_{t-i} + sigma_t eps_t, eps_t i.i.d. (0, 1), u_0 = 0, and
sigma_t = sigma_k in variance regime k.

RULE V (the variance dates), applied to the level dates of the design before any sigma
is estimated:
  (a) the candidates are the level dates of the design;
  (b) a variance regime has at least H_VAR = ceil(0.15 T) = 8 observations: a date whose
      regime at either end of the sample is shorter is dropped (the regime joins its
      neighbor);
  (c) two candidates less than H_VAR years apart: the one that is the adoption of a
      float (the end of a peg, a band or a floor: clause E2(i) of the events file, the
      float_mark of sj_dates.csv) stays; if both are, or neither is, the older stays.
sigma_k^2 is the mean of e_t^2 over the regime, e_t the residuals of the sieve ADF
regression of the principal run (the one that gives p_hat and phi_hat), t = p + 1, ...,
T - 1. The annual profile is rel_t = sigma_k(t) / (the root of the mean of sigma^2). A
cell whose sieve is not stationary is not simulated, as in the principal run, and
neither is a cell with a regime of fewer than MIN_RESIDUALS residuals.

WHAT IS COMPUTED, per cell (E: the nine currencies; SJ: the nine of inputs/sj_dates.csv;
CAD and NZD have the same cell in both, simulated once):
  * item vnull|<currency>|<cfg>, R = 200,000 replications, two arms on the same draws
    (common random numbers): "homo", the registered null (u_0 = 0, BURN = 100, the
    sieve at phi_hat, eps standard normal), and "profile", the same eps scaled by rel_t
    (the burn-in by rel of the first regime). From them: the size of (I) (-1.98) under
    both arms; the size of (II) under the profile at the registered critical value of
    the principal run, and at the critical value of the homoscedastic arm of the same
    draws (the registered null simulated again); the critical
    value (II-V) and the p-value of the data under the profile; the p-value under the
    homoscedastic arm.
  * item vwild|<currency>|<cfg>, B = 19,999: the wild sieve bootstrap of the data,
    Delta u*_t = sum_i phi_i Delta u*_{t-i} + e_t eta_t, eta Rademacher, Delta u*_t = 0
    for t <= p, u*_0 = 0 (IV-W): the critical value and the p-value.
  * Holm over the cells with a verdict of each universe (N8, N9) for the registered p
    (II), for (II-V) and for (IV-W).
Not computed: the half-lives under weights, coverage, any decision rule.

    python mlb/ppp_variance_scope.py --out-dir DIR --work WORKDIR \\
        --jobs 10

--out-dir holds the outputs of the principal run (ppp_null.csv, ppp_ar_diagnostic.csv,
readings.csv, ...) and of SJ (sj_null.csv, sj_statistics.csv, sj_readings.csv,
sj_cbar_half_configs.csv) that the registered values are read from. The outputs are
written only after the checks and the ledger check have passed:
ppp_variance_scope.csv, ppp_variance_scope.manifest.json,
ppp_variance_scope_provenance.csv, stab_app_var.tex.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
import time

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_v, "1")

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
# the loky workers import this file as mlb.ppp_variance_scope
_pp = os.environ.get("PYTHONPATH", "")
if _REPO not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _REPO + (os.pathsep + _pp if _pp else "")

import numpy as np  # noqa: E402

from mlb import app_seeds as A  # noqa: E402
from mlb import ppp_application as P  # noqa: E402
from mlb import ppp_sj as PS  # noqa: E402
from mlb import regime_events as RE  # noqa: E402
from mlb import sj_events as SJE  # noqa: E402

T = P.T
START = P.START_YEAR
BURN = P.BURN
H_VAR = math.ceil(0.15 * T)  # 8
MIN_RESIDUALS = 5
R_VNULL = 200_000
B_VWILD = 19_999
# a family level of about 0.1% over the checks that compare two simulations
Z_CHECK = 4.2
TOL = 1e-10
SJ_TAG = "SJ"
DESIGNS = ("E", SJ_TAG)
# the adoptions of a float that the events and the dates of SJ mark, by design
EXPECTED_FLOATS = {
    "E": {
        ("AUD", 1984),
        ("NZD", 1985),
        ("GBP", 1993),
        ("NOK", 1993),
        ("SEK", 1993),
        ("CHF", 2015),
    },
    SJ_TAG: {
        ("AUD", 1984),
        ("NZD", 1985),
        ("GBP", 1992),
        ("NOK", 1993),
        ("SEK", 1992),
        ("CHF", 2015),
    },
}


# ================================================================== rule V
def variance_dates(taus, floats, start=START, n_obs=T, h=H_VAR):
    """The variance dates of a design by rule V: taus, the level dates; floats, the set
    of dates among them that are adoptions of a float."""
    kept = []
    for t in sorted(taus):
        if t - start < h:  # a short initial regime
            continue
        if not kept or t - kept[-1] >= h:
            kept.append(t)
            continue
        if t in floats and kept[-1] not in floats:
            # the float replaces the earlier date; the date before that one is at least
            # h years before the earlier date, hence before this one, and the first date
            # is at least h years after the start of the sample: nothing else changes
            kept[-1] = t
    while kept and start + n_obs - kept[-1] < h:  # a short final regime
        kept.pop()
    return tuple(kept)


def float_marks(design, currency):
    """The dates of the design that are adoptions of a float, from the events file (E)
    or from the float_mark column of inputs/sj_dates.csv (SJ)."""
    if design == SJ_TAG:
        return {int(r["tau"]) for r in SJE.load()[currency] if r["float_mark"] == "Y"}
    taus, _, _, keep = SJE.configuration(RE.load(), [], currency)
    if tuple(taus) != tuple(RE.configuration(RE.load(), currency, "E")[0]):
        raise SystemExit(
            f"{currency}: the E dates of sj_events and of regime_events differ"
        )
    return {t for t, members in keep if any(f for _, _, f in members)}


def regime_index(years, vdates):
    """The variance regime (0, 1, ...) of each year."""
    return np.array([sum(y >= t for t in vdates) for y in years], dtype=int)


def profile(fit, vdates):
    """The variance profile of a cell: sigma_k^2 by regime from the residuals of the
    sieve regression, the residual counts, the max/min ratio and the annual relative
    scale rel_t = sigma_k(t) / sqrt(mean sigma^2) for the T years of the sample."""
    k = regime_index(fit["e_years"], vdates)
    n_k = [int(np.sum(k == j)) for j in range(len(vdates) + 1)]
    if min(n_k) < MIN_RESIDUALS:
        return dict(
            n_k=n_k,
            status="profile not estimable: a regime with fewer than "
            f"{MIN_RESIDUALS} residuals",
        )
    s2 = np.array([np.mean(fit["e"][k == j] ** 2) for j in range(len(vdates) + 1)])
    years = np.arange(START, START + T)
    w = s2[regime_index(years, vdates)]
    return dict(
        n_k=n_k,
        status="ok",
        s2=s2,
        ratio=float(s2.max() / s2.min()),
        rel=np.sqrt(w / np.mean(w)),
    )


# ================================================================== simulation
def _vchunk(bp, phi, rel, seed0, r0, r1):
    """Replications r0 .. r1-1 of a vnull item: eps = standard_normal(T + BURN) from
    default_rng(seed0 + r); arm "homo" feeds it to the generator of the registered null
    (u_0 = 0, c = 0), arm "profile" feeds eps scaled by rel (the burn-in by rel of the
    first regime). Returns (r1 - r0, 2) MZ_t, NaN where the kernel flags ok = 0."""
    K, gen = P.kernels()
    Z = K["build_z"](T, np.asarray(bp, dtype=np.int64))
    mc = K["method_code"]["maic"]
    phi = np.asarray(phi, dtype=np.float64)
    scale = np.ones(T + BURN)
    scale[:BURN] = rel[0]
    scale[BURN:] = rel
    out = np.full((r1 - r0, 2), np.nan)
    for i, r in enumerate(range(r0, r1)):
        eps = np.random.default_rng(seed0 + r).standard_normal(T + BURN)
        for j, e in enumerate((eps, eps * scale)):
            y = gen(T, BURN, 0.0, phi, e)
            _, _, mz, _, _, ok = K["mstats"](y, Z, P.CBAR, mc, P.KMAX)
            if ok > 0.5:
                out[i, j] = mz
    return out


def _wchunk(bp, phi, e, seed0, b0, b1):
    """Replications b0 .. b1-1 of a vwild item: Delta u*_t = sum_i phi_i Delta u*_{t-i}
    + e_t eta_t for t = p + 1, ..., T - 1, zero before, u* the cumulative sum, eta
    Rademacher from default_rng(seed0 + b). Returns (b1 - b0, 1) MZ_t."""
    K, _ = P.kernels()
    Z = K["build_z"](T, np.asarray(bp, dtype=np.int64))
    mc = K["method_code"]["maic"]
    phi = np.asarray(phi, dtype=np.float64)
    e = np.asarray(e, dtype=np.float64)
    p, n = len(phi), len(e)
    out = np.full((b1 - b0, 1), np.nan)
    for i, b in enumerate(range(b0, b1)):
        eta = 2.0 * np.random.default_rng(seed0 + b).integers(0, 2, size=n) - 1.0
        d = np.zeros(T)
        for t in range(p + 1, T):
            acc = e[t - p - 1] * eta[t - p - 1]
            for j in range(p):
                acc += phi[j] * d[t - 1 - j]
            d[t] = acc
        _, _, mz, _, _, ok = K["mstats"](np.cumsum(d), Z, P.CBAR, mc, P.KMAX)
        if ok > 0.5:
            out[i, 0] = mz
    return out


def _parallel(fn, args, n, jobs):
    """fn(*args, i0, i1) over n replications, jobs workers, 4 chunks per worker."""
    if jobs <= 1:
        return fn(*args, 0, n)
    from joblib import Parallel, delayed

    from mlb import ppp_variance_scope as _self

    f = getattr(_self, fn.__name__)
    edges = np.linspace(0, n, jobs * 4 + 1).astype(int)
    parts = Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
        delayed(f)(*args, int(a), int(b))
        for a, b in zip(edges[:-1], edges[1:])
        if b > a
    )
    return np.vstack(parts)


def sim_vnull(bp, phi, rel, item):
    s0 = A.seed0(item)
    spec = repr((tuple(bp), tuple(float(x) for x in phi), tuple(float(x) for x in rel)))
    return (
        lambda R, jobs: _parallel(
            _vchunk, (tuple(bp), tuple(phi), tuple(rel), s0), R, jobs
        ),
        spec,
    )


def sim_vwild(bp, phi, e, item):
    s0 = A.seed0(item)
    spec = repr((tuple(bp), tuple(float(x) for x in phi), tuple(float(x) for x in e)))
    return (
        lambda B, jobs: _parallel(
            _wchunk, (tuple(bp), tuple(phi), tuple(e), s0), B, jobs
        ),
        spec,
    )


# ================================================================== the cells
class VData(PS.SJData):
    """The principal run's and SJ's cells; the variance side is derived on demand."""

    def design_cells(self):
        """[(design, currency, break positions, level dates)] in the order E then SJ."""
        out = []
        for design in DESIGNS:
            for c in A.CANDIDATES:
                if design == "E":
                    bp, taus = self.conf[("E", c)], self.taus[("E", c)]
                else:
                    bp, taus = self.sj_bp[c], self.sj_tau[c]
                out.append((design, c, tuple(bp), tuple(taus)))
        return out


def cfg_item(family, c, bp):
    return f"{family}|{c}|{A.label(bp)}"


def stats_of(arr, stat, cv_registered, cv_own=None):
    """The statistics of a vnull or a vwild column against the data's statistic: its 5%
    critical value and p-value; its size at -1.98, at the registered critical value and
    at `cv_own` (the critical value of the homoscedastic arm of the same draws; the
    column's own 5% value if None, which makes its size 0.05)."""
    fin = arr[np.isfinite(arr)]
    cv = P.cv5(arr)
    cv_own = cv if cv_own is None else cv_own
    s8, s20, se = P.block_se(P.cv_blocks(arr))
    p, sep, n = P.pvalue(arr, stat)
    out = dict(cv=cv, cv_se=se, p=p, p_se=sep, n_nan=len(arr) - len(fin))
    for tag, thr in (("I", P.CV_ASY), ("IIreg", cv_registered), ("IIown", cv_own)):
        if thr is None or not np.isfinite(thr):
            continue
        size = float(np.mean(fin <= thr))
        out[f"size_{tag}"] = size
        out[f"size_{tag}_se"] = float(np.sqrt(size * (1 - size) / len(fin)))
    return out


def z_between(x, sx, y, sy):
    """(x - y) / sqrt(sx^2 + sy^2); 0 if they are equal with no standard error."""
    den = math.hypot(sx, sy)
    if den == 0.0:
        return 0.0 if x == y else math.inf
    return (x - y) / den


def z_check(checks):
    """The homoscedastic arm is the registered null: every z within Z_CHECK. Returns the
    largest |z|."""
    zs = [g for k in ("z_cv_homo", "z_p_homo") for g in checks[k]]
    bad = [g for g in zs if abs(g["z"]) > Z_CHECK]
    if bad:
        raise SystemExit(f"the homoscedastic arm is not the registered null: {bad}")
    return max((abs(g["z"]) for g in zs), default=0.0)


def check_float_marks(marks):
    """The adoptions of a float found for each design are EXPECTED_FLOATS'."""
    for d in DESIGNS:
        if marks[d] != EXPECTED_FLOATS[d]:
            raise SystemExit(
                f"{d}: the float marks are {sorted(marks[d])}, not the expected ones"
            )


# =========================================================== the registered values
def parse_phi(s):
    return [float(x) for x in s.split("|")] if s else []


def read_registered(out_dir):
    """{(design, currency): registered row} from the outputs of the principal run (E)
    and of SJ: the status, the order, phi_hat, MZ_t at -7, the critical value (II) with
    its se and the p-value (II) with its se."""
    null_e = {
        (r["currency"], r["cfg"]): r
        for r in P.read_csv(os.path.join(out_dir, "ppp_null.csv"))
    }
    diag = {
        r["currency"]: r
        for r in P.read_csv(os.path.join(out_dir, "ppp_ar_diagnostic.csv"))
        if r["set"] == "E"
    }
    null_sj = {
        r["currency"]: r for r in P.read_csv(os.path.join(out_dir, "sj_null.csv"))
    }
    stat_sj = {
        r["currency"]: r for r in P.read_csv(os.path.join(out_dir, "sj_statistics.csv"))
    }
    out = {}
    for c in A.CANDIDATES:
        cfg = A.label(P.configurations()[("E", c)])
        out[("E", c)] = dict(null_e[(c, cfg)], phi=diag[c]["phi"])
        out[(SJ_TAG, c)] = dict(null_sj[c], phi=stat_sj[c]["phi"])
    return out


def read_registered_holm(out_dir):
    """{(design, universe, currency): p_II_holm}, from readings.csv and
    sj_readings.csv."""
    out = {}
    for design, name in (("E", "readings.csv"), (SJ_TAG, "sj_readings.csv")):
        for r in P.read_csv(os.path.join(out_dir, name)):
            if r["reading"] == "p_II_holm":
                out[(design, r["universe"], r["arm"])] = float(r["value"])
    return out


# ================================================================== the run
def check_registered(design, c, x, reg):
    """The cell the data side rebuilds is the registered one."""
    bad = []
    if x["p"] != int(reg["p_hat"]):
        bad.append(f"p_hat {x['p']} != {reg['p_hat']}")
    if abs(x["mzt_m7"] - float(reg["mzt_I"])) > TOL:
        bad.append(f"MZ_t {x['mzt_m7']} != {reg['mzt_I']}")
    phi, want = list(x["fit"]["phi"]), parse_phi(reg["phi"])
    if len(phi) != len(want) or any(abs(a - b) > TOL for a, b in zip(phi, want)):
        bad.append("phi_hat differs")
    if bad:
        raise SystemExit(f"{design} {c}: not the registered cell ({'; '.join(bad)})")


def check_sign(D, M):
    """q -> -q: the variance dates, the profile and MZ_t are unchanged (e_hat flips
    sign, e_hat^2 does not)."""
    worst = 0.0
    for design, c, bp, taus in D.design_cells():
        x, y = D.cells[(c, bp)], M.cells[(c, bp)]
        vd = variance_dates(taus, float_marks(design, c))
        px, py = profile(x["fit"], vd), profile(y["fit"], vd)
        if px["status"] == "ok":
            worst = max(worst, float(np.max(np.abs(px["rel"] - py["rel"]))))
        worst = max(worst, abs(x["mzt_m7"] - y["mzt_m7"]))
    if worst > 1e-12:
        raise SystemExit(f"variance sign check: max difference {worst}")
    return worst


def cell_row(design, c, bp, taus, x, r, vd, pf):
    """The columns of a cell that do not need a simulation."""
    row = dict(
        design=design,
        currency=c,
        cfg=A.label(bp),
        level_dates="|".join(map(str, taus)),
        variance_dates="|".join(map(str, vd)),
        n_regimes=len(vd) + 1,
        n_residuals="|".join(map(str, pf["n_k"])),
        p_hat=x["p"],
        mzt_I=x["mzt_m7"],
        registered_status=r["status"],
    )
    if r["status"] != "ok":
        row["status"] = r["status"]
        return row
    row.update(
        cv_II_registered=float(r["cv_II"]),
        p_II_registered=float(r["p_II"]),
        p_II_registered_se=float(r["p_II_se"]),
    )
    if pf["status"] != "ok":
        row["status"] = pf["status"]
    else:
        row.update(
            status="ok",
            sigma2="|".join(repr(float(v)) for v in pf["s2"]),
            variance_ratio=pf["ratio"],
        )
    return row


def simulated_columns(x, r, s, R, B):
    """The columns that come from the two items of a cell."""
    h, v, w = s["homo"], s["het"], s["wild"]
    cvr = float(r["cv_II"])
    return dict(
        R=R,
        B=B,
        cv_II_homo_own=h["cv"],
        cv_II_homo_own_se=h["cv_se"],
        p_II_homo_own=h["p"],
        size_I_homo=h["size_I"],
        size_I_homo_se=h["size_I_se"],
        size_II_homo_registered=h["size_IIreg"],
        size_I_het=v["size_I"],
        size_I_het_se=v["size_I_se"],
        size_II_het_registered=v["size_IIreg"],
        size_II_het_registered_se=v["size_IIreg_se"],
        size_II_het_own=v["size_IIown"],
        size_II_het_own_se=v["size_IIown_se"],
        cv_IIV=v["cv"],
        cv_IIV_se=v["cv_se"],
        p_IIV=v["p"],
        p_IIV_se=v["p_se"],
        cv_IVW=w["cv"],
        cv_IVW_se=w["cv_se"],
        p_IVW=w["p"],
        p_IVW_se=w["p_se"],
        n_nan_homo=h["n_nan"],
        n_nan_het=v["n_nan"],
        n_nan_wild=w["n_nan"],
        reject_II_registered=int(x["mzt_m7"] < cvr),
        reject_IIV=int(x["mzt_m7"] < v["cv"]),
        reject_IVW=int(x["mzt_m7"] < w["cv"]),
    )


def simulate_cell(store, c, bp, x, pf, cvr, R, B):
    """The two items of a cell and their statistics."""
    phi = tuple(x["fit"]["phi"])
    it_n, it_w = cfg_item("vnull", c, bp), cfg_item("vwild", c, bp)
    vn = store.get(it_n, ["homo", "profile"], R, *sim_vnull(bp, phi, pf["rel"], it_n))
    vw = store.get(it_w, ["wild"], B, *sim_vwild(bp, phi, x["fit"]["e"], it_w))
    stat = x["mzt_m7"]
    cv_homo = P.cv5(vn[:, 0])
    return dict(
        homo=stats_of(vn[:, 0], stat, cvr),
        het=stats_of(vn[:, 1], stat, cvr, cv_own=cv_homo),
        wild=stats_of(vw[:, 0], stat, cvr),
    )


def run(D, store, out_dir, scale, B):
    """The checks, the simulation and the rows: (rows, checks)."""
    reg = read_registered(out_dir)
    R = R_VNULL // scale
    rows, done = [], {}
    marks = {d: set() for d in DESIGNS}
    checks = dict(z_cv_homo=[], z_p_homo=[])
    for design, c, bp, taus in D.design_cells():
        x, r = D.cells[(c, bp)], reg[(design, c)]
        check_registered(design, c, x, r)
        floats = float_marks(design, c)
        marks[design].update((c, t) for t in floats)
        vd = variance_dates(taus, floats)
        pf = profile(x["fit"], vd)
        row = cell_row(design, c, bp, taus, x, r, vd, pf)
        if row["status"] == "ok":
            if (c, bp) not in done:
                cvr = float(r["cv_II"])
                done[(c, bp)] = simulate_cell(store, c, bp, x, pf, cvr, R, B)
                h = done[(c, bp)]["homo"]
                for key, got, got_se, want, want_se in (
                    ("z_cv_homo", h["cv"], h["cv_se"], cvr, float(r["cv_II_se"])),
                    (
                        "z_p_homo",
                        h["p"],
                        h["p_se"],
                        float(r["p_II"]),
                        float(r["p_II_se"]),
                    ),
                ):
                    checks[key].append(
                        dict(
                            design=design,
                            currency=c,
                            z=z_between(got, got_se, want, want_se),
                        )
                    )
            row.update(simulated_columns(x, r, done[(c, bp)], R, B))
        rows.append(row)
    check_float_marks(marks)
    by = {(r["design"], r["currency"]): r for r in rows}
    for c in A.CANDIDATES:
        if (
            D.shared(c)
            and by[("E", c)]["variance_dates"] != by[(SJ_TAG, c)]["variance_dates"]
        ):
            raise SystemExit(f"{c}: the shared cell has two sets of variance dates")
    checks["max_abs_z"] = z_check(checks)
    holm = holm_columns(D, rows, read_registered_holm(out_dir))
    for row in rows:
        row.update(holm.get((row["design"], row["currency"]), {}))
    return rows, checks


def holm_columns(D, rows, registered_holm):
    """Holm of the registered p (II) over the cells that have a verdict in the
    principal run or in SJ (it must be theirs), and of (II-V) and (IV-W) over the cells
    that were simulated."""
    out = {}
    by = {(r["design"], r["currency"]): r for r in rows}
    for design in DESIGNS:
        for uname, univ in (("N8", D.N8), ("N9", D.N9)):
            for col, tag in (
                ("p_II_registered", "II"),
                ("p_IIV", "IIV"),
                ("p_IVW", "IVW"),
            ):
                key = "registered_status" if tag == "II" else "status"
                ok = [c for c in univ if by[(design, c)][key] == "ok"]
                adj = P.holm({c: by[(design, c)][col] for c in ok})
                for c in ok:
                    out.setdefault((design, c), {})[f"holm_{tag}_{uname}"] = adj[c]
                    if (
                        tag == "II"
                        and abs(adj[c] - registered_holm[(design, uname, c)]) > 1e-15
                    ):
                        raise SystemExit(
                            f"{design} {uname} {c}: Holm of p (II) is {adj[c]}, not "
                            f"the registered {registered_holm[(design, uname, c)]}"
                        )
    return out


# ================================================================== the outputs
SCOPE_COLS = [
    "design", "currency", "cfg", "status", "level_dates", "variance_dates", "n_regimes",
    "n_residuals", "sigma2", "variance_ratio", "p_hat", "mzt_I", "registered_status",
    "cv_II_registered", "p_II_registered", "p_II_registered_se", "reject_II_registered",
    "R", "B", "cv_II_homo_own", "cv_II_homo_own_se", "p_II_homo_own", "size_I_homo",
    "size_I_homo_se", "size_II_homo_registered", "size_I_het", "size_I_het_se",
    "size_II_het_registered", "size_II_het_registered_se", "size_II_het_own",
    "size_II_het_own_se", "cv_IIV", "cv_IIV_se", "p_IIV", "p_IIV_se", "reject_IIV",
    "cv_IVW", "cv_IVW_se", "p_IVW", "p_IVW_se", "reject_IVW", "n_nan_homo", "n_nan_het",
    "n_nan_wild", "holm_II_N8", "holm_IIV_N8", "holm_IVW_N8", "holm_II_N9",
    "holm_IIV_N9", "holm_IVW_N9",
]  # fmt: skip
TEX_FILE = "stab_app_var.tex"
PROV_COLS = [
    "object", "table_file", "body_row", "token", "printed", "value_full", "reference",
    "reference_key", "reference_column", "rule", "class", "decimals_printed", "se",
    "d_sustained", "resolved_at_printed",
]  # fmt: skip
STATUS_TEXT = {
    "not simulated": "not simulated",
    "profile not estimable": "not estimable",
}


def write_text(path, text):
    """Writes path.tmp; P.finalize() renames it once the checks have passed."""
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    P._PENDING.append(path)


def build_tex(rows):
    """The table of the variance scope and the provenance of each printed token. Numbers
    are rounded once, half up, from the full-precision value of ppp_variance_scope.csv;
    the places printed follow the principal run's convention (the smallest place at
    which every standard error of the class is sustained)."""
    from mlb import ppp_tables as TB

    cols = [
        ("variance_ratio", "ratio", None),
        ("size_I_het", "size", "size_I_het_se"),
        ("size_II_het_registered", "size", "size_II_het_registered_se"),
        ("p_II_registered", "p", "p_II_registered_se"),
        ("p_IIV", "p", "p_IIV_se"),
        ("p_IVW", "p", "p_IVW_se"),
    ]
    ok = [r for r in rows if r["status"] == "ok"]
    places = {"ratio": 2, "int": 0}
    for cls in ("size", "p"):
        ds = [
            TB.d_sustained(repr(float(r[se])))
            for r in ok
            for col, k, se in cols
            if k == cls and se
        ]
        places[cls] = min(d for d in ds if d is not None)
    lines = [
        r"\begin{tabular}{llcrrrrrr}",
        r"\hline",
        r"Design & Currency & Variance dates & Ratio & Size (I) & Size (II) "
        r"& $p$ (II) & $p$ (II-V) & $p$ (IV-W) \\",
        r"\hline",
    ]
    prov = []
    for i, r in enumerate(rows):
        key = f"design={r['design']}|currency={r['currency']}"

        def token(printed, value, column, cls, se=None):
            d = places[cls]
            prov.append(
                dict(
                    object="stab_app_var",
                    table_file=TEX_FILE,
                    body_row=i,
                    token=len(prov),
                    printed=printed,
                    value_full=value,
                    reference="ppp_variance_scope.csv",
                    reference_key=key,
                    reference_column=column,
                    rule="round half up",
                    **{"class": cls},
                    decimals_printed=d,
                    se=se if se is not None else "",
                    d_sustained=TB.d_sustained(se) if se else "",
                    resolved_at_printed=TB.resolved(value, d, se)
                    if cls != "int"
                    else "",
                )
            )
            return printed

        dates = [t for t in r["variance_dates"].split("|") if t]
        if r["status"] == "ok":
            date_txt = (
                "; ".join(token(d, d, "variance_dates", "int") for d in dates) or "none"
            )
            cells = [
                token(
                    TB.rnd(repr(float(r[col])), places[cls]),
                    repr(float(r[col])),
                    col,
                    cls,
                    repr(float(r[se])) if se else None,
                )
                for col, cls, se in cols
            ]
        else:
            tag = r["status"].split(" (")[0].split(":")[0]
            date_txt = "; ".join(token(d, d, "variance_dates", "int") for d in dates)
            cells = [TB.tex_escape(STATUS_TEXT.get(tag, tag))] + [""] * (len(cols) - 1)
        lines.append(
            " & ".join([r["design"], r["currency"], date_txt] + cells) + r" \\"
        )
    lines += [r"\hline", r"\end{tabular}", ""]
    return "\n".join(lines), prov


def write_outputs(out_dir, rows, checks, store, scale, B, D, fp):
    P.write_csv(os.path.join(out_dir, "ppp_variance_scope.csv"), SCOPE_COLS, rows)
    tex, prov = build_tex(rows)
    write_text(os.path.join(out_dir, TEX_FILE), tex)
    P.write_csv(
        os.path.join(out_dir, "ppp_variance_scope_provenance.csv"), PROV_COLS, prov
    )
    inputs = [
        ("panel", P.PANEL),
        ("sweep_gates", P.SWEEP),
        ("regime_events", RE.EVENTS),
        ("sj_dates", SJE.SJ_DATES),
    ]
    registered = (
        "ppp_null.csv",
        "ppp_ar_diagnostic.csv",
        "readings.csv",
        "cbar_half_configs.csv",
        "sj_null.csv",
        "sj_statistics.csv",
        "sj_readings.csv",
        "sj_cbar_half_configs.csv",
    )
    inputs += [(name, os.path.join(out_dir, name)) for name in registered]
    items = sorted({it for it, _, _ in store.ledger})
    man = dict(
        synthetic=D.synthetic,
        inputs_sha256={
            k: P.sha_lf(f)
            for k, f in inputs
            if not (D.synthetic and k.startswith(("panel", "sweep")))
        },
        code_sha256={
            os.path.basename(f): P.sha_lf(f)
            for f in (os.path.abspath(__file__), PS.__file__, P.__file__, A.__file__)
        },
        fingerprint=fp,
        R=R_VNULL // scale,
        B=B,
        H_VAR=H_VAR,
        MIN_RESIDUALS=MIN_RESIDUALS,
        Z_CHECK=Z_CHECK,
        seeds={it: dict(seed0=A.seed0(it), n=A.item_R(it)) for it in items},
        checks=dict(
            max_abs_z_homoscedastic_arm=checks["max_abs_z"],
            n_comparisons=sum(len(checks[k]) for k in ("z_cv_homo", "z_p_homo")),
            sign=checks["sign"],
        ),
    )
    P.write_json(os.path.join(out_dir, "ppp_variance_scope.manifest.json"), man)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work", required=True, help="the item store and the ledgers")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    scale, B = 1, B_VWILD
    assert R_VNULL % scale == 0 and (R_VNULL // scale) % 20 == 0
    t0 = time.time()
    D = VData(a.out_dir, False)
    fp = hashlib.sha256(
        (PS.sj_fingerprint(a.out_dir, D) + P.sha_lf(os.path.abspath(__file__))).encode()
    ).hexdigest()
    store = P.Store(a.work, fp, a.jobs, "variance")
    rows, checks = run(D, store, a.out_dir, scale, B)
    checks["sign"] = check_sign(D, VData(a.out_dir, False, sign=-1.0))
    write_outputs(a.out_dir, rows, checks, store, scale, B, D, fp)
    n = P.check_ledgers(a.work, store)
    P.finalize()
    print(
        f"variance scope: {len(rows)} cells in {time.time() - t0:.1f}s; "
        f"max |z| of the homoscedastic arm {checks['max_abs_z']:.2f}; "
        f"{n} consumers in the ledgers",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
