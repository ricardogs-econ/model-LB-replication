"""The sensitivity design SJ of the PPP application: the principal design E plus the
systemic banking crises of the issuers (inputs/sj_dates.csv, sj_events.py).

SJ is read beside E, never in place of it: the outputs of
the principal run are not touched, the objects of SJ are written to sj_*.csv, and what
SJ shares with E is read from the principal run's outputs instead of being computed
again, so that no seed item is consumed twice. Two things are shared: a currency whose
SJ design is its E design (CAD and NZD) and a half-life cell whose key (scheme, model,
currency, configuration, order) already belongs to the principal run.

Stages (each reads the principal run's outputs from --out-dir):

    design    the exact cbar_half of every SJ configuration (sj_cbar_half_configs.csv)
    data      the data side of the nine SJ cells (sj_statistics.csv), with the
              q -> -q check
    null      columns (II) and (III) and the size of (I) for the SJ cells that are not
              E's (sj_null.csv); the draws are the items of the registry's SJ block
    hl        the grid-t half-lives, LB and MP, recursive and wild, at each cell's own
              order p_hat (sj_hl_cells.csv, sj_hl_nodes.csv, sj_hl_checks.json,
              sj_hl_run.json)
    sets      E beside SJ, two rows per currency (sj_sets.csv)
    readings  the readings by universe (sj_readings.csv): the number of rejections,
              the currencies rejecting with (I) and (II), Holm and Bonferroni

    python mlb/ppp_sj.py --stage design --out-dir DIR
    python mlb/ppp_sj.py --stage null --out-dir DIR --work WORKDIR \\
        --jobs 10

A stage writes its outputs only after its checks and the ledger check have passed.

RULES: those of ppp_application.py and ppp_half_lives.py, whose functions this module
calls, with one difference in the half-life stage.

  * A currency whose sieve is non-stationary is not simulated: its row says so, it has
    no (II) verdict and it leaves the count of rejections, as in E.
  * In E every stop of the grid-t aborts the run. Here the lower bound K_MIN of the
    grid does not: a cell whose grid reaches it is recorded as failed ("failed: grid
    reached K_MIN", the node, the OLS alpha and its standard error), its result fields
    are blank and the run goes on. Every other stop (a non-finite t*, an empty
    confidence set, a median crossing below the grid, alpha_OLS differing from the
    sieve's) still aborts. A failed cell is stored under the fingerprint, so a resumed
    run does not repeat it.
  * A half-life cell without result keeps its status and blank fields in sj_sets.csv;
    every count of sj_readings.csv states the number of currencies it rests on.
  * C4 and C6 are descriptive on the SJ dates: kappa2 against the asymptotic
    thresholds and the Wald statistic with its chi-square p-value; no critical value is
    simulated for the SJ designs and no verdict of the universe rests on them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
# the loky workers import this file as mlb.ppp_sj
_pp = os.environ.get("PYTHONPATH", "")
if _REPO not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _REPO + (os.pathsep + _pp if _pp else "")

import numpy as np  # noqa: E402

from mlb import app_seeds as A  # noqa: E402
from mlb import ppp_application as P  # noqa: E402
from mlb import ppp_half_lives as HL  # noqa: E402
from mlb import sj_events as SJE  # noqa: E402

STAGES = ("design", "data", "null", "hl", "sets", "readings")
SJ_CBAR = "sj_cbar_half_configs.csv"
BOUND_STATUS = "failed: grid reached K_MIN"
BOUND_PATTERN = re.compile(r"the grid reached the bound at node (-?\d+)")
SIGN_TOL = 1e-12
SJ_TAG = "SJ"


# ================================================================== the designs
def designs():
    """({currency: break positions}, {currency: taus}) of SJ, from inputs/sj_dates.csv.

    The cells that are not E's, in currency order, must be the registry's SJ_CELLS."""
    by = SJE.load()
    if sorted(by) != sorted(A.CANDIDATES):
        raise SystemExit(f"sj_dates.csv: currencies {sorted(by)}")
    bp = {c: tuple(int(r["break_pos"]) for r in by[c]) for c in A.CANDIDATES}
    tau = {c: tuple(int(r["tau"]) for r in by[c]) for c in A.CANDIDATES}
    for c in A.CANDIDATES:
        if tuple(t - P.START_YEAR for t in tau[c]) != bp[c]:
            raise SystemExit(f"sj_dates.csv: {c}: tau and break_pos disagree")
    sj_cells_not_in_e = tuple(
        (c, bp[c]) for c in A.CANDIDATES if (c, bp[c]) not in A.CELLS)
    if sj_cells_not_in_e != A.SJ_CELLS:
        raise SystemExit("sj_dates.csv: the designs are not the registry's SJ_CELLS")
    return bp, tau


def read_sj_cbar(out_dir):
    """{break positions: cbar_half} of the SJ configurations."""
    return {
        P.parse_bp(r["break_pos"]): float(r["cbar_half"])
        for r in P.read_csv(os.path.join(out_dir, SJ_CBAR))
    }


class SJData(P.Data):
    """The principal run's data side plus the SJ cells (the ones that are not E's are
    fitted here; CAD and NZD keep E's cell)."""

    def __init__(self, out_dir, synthetic=False, sign=1.0):
        super().__init__(out_dir, synthetic, sign)
        self.sj_bp, self.sj_tau = designs()
        cbar = read_sj_cbar(out_dir)
        for c in A.CANDIDATES:
            bp = self.sj_bp[c]
            if (c, bp) in self.cells:
                if abs(cbar[bp] - self.cells[(c, bp)]["cbar_half"]) > P.CBAR_XTOL:
                    raise SystemExit(f"{c} {P.cfg_label(bp)}: cbar_half is not E's")
            else:
                self.cells[(c, bp)] = P.cell_fit(self.q[c], bp, cbar[bp])

    def sj_cell(self, c):
        return self.cells[(c, self.sj_bp[c])]

    def shared(self, c):
        """True if the SJ design of c is one of the principal run's cells."""
        return (c, self.sj_bp[c]) in A.CELLS

    def sj_cells_not_in_e(self):
        """[(currency, break positions)] of the SJ cells that are not E's."""
        return [(c, self.sj_bp[c]) for c in A.CANDIDATES if not self.shared(c)]


def tau_label(D, c):
    return "|".join(map(str, D.sj_tau[c]))


def sj_fingerprint(out_dir, D):
    """The code and inputs every stored SJ item depends on: the principal run's
    fingerprint plus this module, the dates and the SJ configurations' cbar_half."""
    h = hashlib.sha256(P.fingerprint(out_dir, D).encode())
    for f in (os.path.abspath(__file__), SJE.SJ_DATES, os.path.join(out_dir, SJ_CBAR)):
        h.update(P.sha_lf(f).encode())
    return h.hexdigest()


# ================================================================== stage: design
def stage_design(out_dir, jobs):
    from mlb import envelope_exact as ee

    bp, _ = designs()
    cfgs = sorted(set(bp.values()), key=lambda b: (len(b), b))
    if jobs > 1:
        from joblib import Parallel, delayed

        res = Parallel(n_jobs=min(jobs, len(cfgs)), backend="loky")(
            delayed(P._cbar_one)(b) for b in cfgs
        )
    else:
        res = [P._cbar_one(b) for b in cfgs]
    principal = P.read_cbar(out_dir)
    for r in res:
        want = principal.get(r["bp"])
        if want is not None and abs(r["cbar"] - want) > P.CBAR_XTOL:
            raise SystemExit(
                f"cbar_half {r['bp']}: {r['cbar']} differs from E's {want}"
            )
    rows = [
        dict(
            cfg=P.cfg_label(r["bp"]),
            m=len(r["bp"]),
            break_pos="|".join(map(str, r["bp"])),
            lam="|".join(repr(x) for x in r["lam"]),
            cbar_half=r["cbar"],
            b_T=r["bT"],
            displacement=abs(r["cbar"] - P.CBAR),
            bracket_lo=r["lo"],
            bracket_hi=r["hi"],
            xtol=P.CBAR_XTOL,
        )
        for r in res
    ]
    path = os.path.join(out_dir, SJ_CBAR)
    cols = [
        "cfg",
        "m",
        "break_pos",
        "lam",
        "cbar_half",
        "b_T",
        "displacement",
        "bracket_lo",
        "bracket_hi",
        "xtol",
    ]
    P.write_csv(path, cols, rows)
    man = dict(
        T=P.T,
        conventions=ee.conventions(),
        envelope_exact_sha256=P.sha_lf(ee.__file__),
        sj_dates_sha256=P.sha_lf(SJE.SJ_DATES),
        shared_with_E=sorted(P.cfg_label(r["bp"]) for r in res if r["bp"] in principal),
        csv_sha256=P.sha_lf(path + ".tmp"),
    )
    P.write_json(os.path.join(out_dir, "sj_cbar_half_configs.manifest.json"), man)


# ================================================================== stage: data
def stage_data(D, out_dir):
    rows = []
    for c in A.CANDIDATES:
        bp = D.sj_bp[c]
        x = D.sj_cell(c)
        f = x["fit"]
        row = dict(
            currency=c,
            tau=tau_label(D, c),
            break_pos="|".join(map(str, bp)),
            m=len(bp),
            cfg=P.cfg_label(bp),
            same_cell_as_E=int(D.shared(c)),
            p_hat_lagdiff=x["p"],
            phi="|".join(repr(float(v)) for v in f["phi"]),
            sum_phi=f["sum_phi"],
            flag09=f["flag09"],
            max_root=f["max_root"],
            stationary=f["stationary"],
            mzt_I=x["mzt_m7"],
            reject_I=x["mzt_m7"] < P.CV_ASY,
            cbar_half=x["cbar_half"],
            mzt_III=x["mzt_half"],
        )
        row.update({f"bic_{p}": x["bics"][p] for p in range(len(x["bics"]))})
        rows.append(row)
        if not np.isfinite(x["mzt_m7"]) or not np.isfinite(x["mzt_half"]):
            raise SystemExit(f"{c} SJ: degenerate MZ_t on the data")
    cols = (
        ["currency", "tau", "break_pos", "m", "cfg", "same_cell_as_E", "p_hat_lagdiff"]
        + [f"bic_{p}" for p in range(len(D.sj_cell("AUD")["bics"]))]
        + ["phi", "sum_phi", "flag09", "max_root", "stationary", "mzt_I", "reject_I"]
        + ["cbar_half", "mzt_III"]
    )
    P.write_csv(os.path.join(out_dir, "sj_statistics.csv"), cols, rows)


def check_sign_data(D, out_dir):
    """q -> -q with the same code: MZ_t, p_hat and phi_hat of the SJ cells equal to
    1e-12 (every value finite on both sides); e_hat flips sign."""
    M = SJData(out_dir, D.synthetic, sign=-1.0)
    worst = 0.0

    def diff(a, b):
        a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
        if a.shape != b.shape or not (
            np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        ):
            raise SystemExit("sj sign check: a non-finite value or a shape mismatch")
        return float(np.max(np.abs(a - b), initial=0.0))

    for c in A.CANDIDATES:
        x, y = D.sj_cell(c), M.sj_cell(c)
        if x["p"] != y["p"]:
            raise SystemExit(f"sj sign check: p_hat differs at {c}")
        worst = max(
            worst,
            diff(x["mzt_m7"], y["mzt_m7"]),
            diff(x["mzt_half"], y["mzt_half"]),
            diff(x["fit"]["phi"], y["fit"]["phi"]),
            diff(x["fit"]["e"], -y["fit"]["e"]),
        )
    if worst > SIGN_TOL:
        raise SystemExit(f"sj sign check: max difference {worst}")
    return worst


# ================================================================== stage: null
SJ_NULL_COLS = ["currency", "tau", "same_cell_as_E"] + P.NULL_COLS[1:]


def stage_null(D, store, out_dir, scale):
    """Columns (II), (III) and the size of (I): simulated for the SJ cells that are
    not E's, copied from the principal run's rows for the others."""
    principal = {
        (r["currency"], r["cfg"]): r
        for r in P.read_csv(os.path.join(out_dir, "ppp_null.csv"))
    }
    rows, arrays = [], {}
    for c in A.CANDIDATES:
        bp = D.sj_bp[c]
        if D.shared(c):
            row = dict(principal[(c, P.cfg_label(bp))], same_cell_as_E=1)
        else:
            t0 = time.time()
            arrays[(c, bp)] = P.null_arrays(D, store, c, bp, scale)
            row = dict(P.null_row(D, c, bp, arrays[(c, bp)]), same_cell_as_E=0)
            print(f"  null {c} {P.cfg_label(bp)}: {time.time() - t0:.0f}s", flush=True)
        rows.append(dict(row, tau=tau_label(D, c)))
    # sign check on the null draws: the first two SJ cells with p_hat >= 1 that are
    # simulated, their first SIGN_DRAWS draws recomputed with phi_hat from -q
    M = SJData(out_dir, D.synthetic, sign=-1.0)
    check = []
    n = min(P.SIGN_DRAWS, P.R_NULL // scale)
    for c, bp in D.sj_cells_not_in_e():
        nul = arrays[(c, bp)]
        if D.cells[(c, bp)]["p"] < 1 or nul is None or len(check) == 2:
            continue
        arms = (("u0zero", 0.0, tuple(M.cells[(c, bp)]["fit"]["phi"]), P.CBAR),)
        item = f"null|{c}|{P.cfg_label(bp)}"
        neg = P.simulate(bp, arms, A.seed0(item), n, store.jobs)[:, 0]
        pos = nul["sieve"][:n]
        if not np.array_equal(np.isnan(neg), np.isnan(pos)):
            raise SystemExit(f"sj null sign check {c}: NaN pattern differs")
        d = float(np.nanmax(np.abs(neg - pos)))
        if d > 1e-9:
            raise SystemExit(f"sj null sign check {c}: max difference {d}")
        check.append(dict(currency=c, cfg=P.cfg_label(bp), draws=n, max_diff=d))
    P.write_csv(os.path.join(out_dir, "sj_null.csv"), SJ_NULL_COLS, rows)
    P.write_json(
        os.path.join(out_dir, "sj_null_checks.json"),
        dict(
            sign_check=check or "not applicable (no simulated SJ cell with p_hat >= 1)",
            note="the first two simulated SJ cells with p_hat >= 1",
        ),
    )
    return rows


# ================================================================== stage: hl
SJ_HL_COLS = HL.CELL_COLS + ["status", "same_cell_as_E", "k_stop"]


def run_cell_sj(q, scheme, model, c, bp, p, B, store=None, fp=""):
    """HL.run_cell, except that a grid that reaches the lower bound K_MIN is recorded
    as a failed cell instead of stopping the run. Returns (summary, nodes); a failed
    cell has status BOUND_STATUS, no nodes and no result fields. Any other stop of the
    grid raises as in the principal run."""
    it = HL.item_of(scheme, model, c, bp, p)
    fail = (
        os.path.join(store, it.replace("|", "__") + ".failed.json") if store else None
    )
    if fail and os.path.exists(fail):
        with open(fail, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec["fingerprint"] == fp and rec["B"] == B:
            return dict(rec["summ"], item=it), []
    try:
        return HL.run_cell(q, scheme, model, c, bp, p, B, store, fp)
    except SystemExit as e:
        m = BOUND_PATTERN.search(str(e))
        if not m:
            raise
        u = HL.residual(q, bp if model == "LB" else ())
        alpha, se = HL.adf_const_nb(np.ascontiguousarray(u), p)
        summ = dict(
            status=BOUND_STATUS, k_stop=int(m.group(1)), alpha_ols=alpha, se_ols=se
        )
        if fail:
            with open(fail + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
                json.dump(dict(fingerprint=fp, B=B, summ=summ), fh)
            os.replace(fail + ".tmp", fail)
        return dict(summ, item=it), []


def sj_cells(D):
    """[(key, shared)] of the SJ half-life cells, key = (scheme, model, currency,
    break positions or (), order); LB at the SJ design and MP, both at the SJ cell's
    p_hat. A cell is shared when the principal run has the same key."""
    principal = {k[:5] for k in HL.cell_list(D)}
    out = []
    for scheme in A.SCHEMES:
        for c in A.CANDIDATES:
            p = D.sj_cell(c)["p"]
            for model, bp in (("LB", D.sj_bp[c]), ("MP", ())):
                key = (scheme, model, c, bp, p)
                out.append((key, key in principal))
    return out


def check_sign_cell(D, B, failed=()):
    """q -> -q on the recursive LB cell of an SJ cell that is not E's, whose sieve is
    stationary and whose grid did not fail (`failed`: currencies; one with p_hat >= 1 if
    there is one): the same nodes (t_obs and quantiles to 1e-9, the same accepted
    flags) and interval."""
    ok = [
        (c, bp)
        for c, bp in D.sj_cells_not_in_e()
        if D.cells[(c, bp)]["fit"]["stationary"] and c not in failed
    ]
    if not ok:
        return "not applicable (no stationary SJ cell outside E that completed)"
    c, bp = next((x for x in ok if D.cells[x]["p"] >= 1), ok[0])
    p = D.cells[(c, bp)]["p"]
    item = HL.item_of("rec", "LB", c, bp, p)
    (s1, n1), (s2, n2) = (
        HL.grid_t(sgn * D.q[c], bp, p, item, "rec", B) for sgn in (1.0, -1.0)
    )
    if [x["k"] for x in n1] != [x["k"] for x in n2]:
        raise SystemExit(f"sj hl sign check {c}: the grids differ")
    d = 0.0
    for x, y in zip(n1, n2):
        if x["accepted"] != y["accepted"]:
            raise SystemExit(f"sj hl sign check {c}: node {x['k']} flips")
        d = max(d, *(abs(x[k] - y[k]) for k in ("t_obs", "q_lo", "q_med", "q_hi")))
    if d > HL.SIGN_TOL or s1["k_lo"] != s2["k_lo"] or s1["k_hi"] != s2["k_hi"]:
        raise SystemExit(f"sj hl sign check {c}: max difference {d}")
    return dict(currency=c, cell="rec LB SJ", max_diff=d)


def stage_hl(D, store, out_dir, B, jobs, fp):
    run_record = os.path.join(out_dir, "hl_run.json")
    if os.path.exists(run_record):
        with open(run_record, encoding="utf-8") as fh:
            if json.load(fh)["B"] != B:
                raise SystemExit("the principal half-lives were made at another B")
    principal = {
        (r["scheme"], r["model"], r["currency"], r["cfg"], int(r["p"])): r
        for r in P.read_csv(os.path.join(out_dir, "hl_cells.csv"))
    }
    cells = sj_cells(D)
    # the cells to compute, the LB cells of CHF first (the slowest, so that they overlap
    # the others)
    todo = [k for k, shared in cells if not shared]
    todo.sort(
        key=lambda k: (not (k[2] == "CHF" and k[1] == "LB"), A.CANDIDATES.index(k[2]))
    )
    args = [
        (D.q[c], sch, mod, c, bp, p, B, store.dir_hl, fp) for sch, mod, c, bp, p in todo
    ]
    if jobs > 1:
        from joblib import Parallel, delayed

        from mlb import ppp_sj as _self

        res = Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
            delayed(_self.run_cell_sj)(*a) for a in args
        )
    else:
        res = [run_cell_sj(*a) for a in args]
    done = dict(zip(todo, res))
    rows, nrows = [], []
    for key, shared in cells:
        sch, mod, c, bp, p = key
        cfg = P.cfg_label(bp) if mod == "LB" else ""
        base = dict(scheme=sch, model=mod, currency=c, cfg=cfg, p=p)
        specs = f"{mod}@{SJ_TAG}"
        if shared:
            row = dict(
                principal[(sch, mod, c, cfg, p)],
                specs=specs,
                status="ok",
                same_cell_as_E=1,
            )
            rows.append(row)
            continue
        summ, nodes = done[key]
        summ = dict(summ)
        summ.setdefault("status", "ok")
        rows.append(dict(base, **summ, specs=specs, same_cell_as_E=0))
        for nd in nodes:
            nrows.append(dict(base, **nd))
    check = check_sign_cell(
        D,
        B,
        {
            r["currency"]
            for r in rows
            if r["scheme"] == "rec" and r["model"] == "LB" and r["status"] != "ok"
        },
    )
    failed = [
        dict(
            item=r["item"], currency=r["currency"], scheme=r["scheme"], model=r["model"]
        )
        for r in rows
        if r["status"] != "ok"
    ]
    store.ledger.extend(
        (r["item"], store.stage, "grid") for r in rows if not r["same_cell_as_E"]
    )
    P.write_csv(os.path.join(out_dir, "sj_hl_cells.csv"), SJ_HL_COLS, rows)
    P.write_csv(os.path.join(out_dir, "sj_hl_nodes.csv"), HL.NODE_COLS, nrows)
    P.write_json(
        os.path.join(out_dir, "sj_hl_checks.json"),
        dict(sign_check=check, failed=failed),
    )
    P.write_json(
        os.path.join(out_dir, "sj_hl_run.json"),
        dict(B=B, n_cells=len(todo)),
    )
    return rows


# ================================================================== stage: sets
def sets_rows(D, e_rows, null, hl):
    """E and SJ rows, two per currency. e_rows: {(set, currency): row} of ppp_sets.csv;
    null: {(currency, cfg): row} of sj_null.csv; hl: {(scheme, model, currency, cfg,
    p): row} of sj_hl_cells.csv."""
    rows = []
    for c in A.CANDIDATES:
        e = dict(e_rows[("E", c)])
        bp = D.sj_bp[c]
        x = D.sj_cell(c)
        cfg = P.cfg_label(bp)
        n = null[(c, cfg)]
        row = dict(
            set=SJ_TAG,
            currency=c,
            tau=tau_label(D, c),
            break_pos="|".join(map(str, bp)),
            status=n["status"],
            cfg=cfg,
            p_hat=x["p"],
            phi="|".join(repr(float(v)) for v in x["fit"]["phi"]),
            same_cell_as_E=int(D.shared(c)),
        )
        for k in P.CELL_FIELDS[3:]:
            row[k] = n.get(k, "")
        cell = {
            "LB": hl.get(("rec", "LB", c, cfg, str(x["p"]))),
            "MP": hl.get(("rec", "MP", c, "", str(x["p"]))),
        }
        good = {t: r is not None and r["status"] == "ok" for t, r in cell.items()}
        for tag, r in cell.items():
            for k in P.HL_FIELDS:
                row[f"{tag}_{k}"] = r[k] if good[tag] else ""
            row[f"{tag}_status"] = r["status"] if r is not None else "missing"
        if good["LB"] and good["MP"]:
            row["collapse"] = int(
                int(cell["MP"]["k_hi"]) >= 200 and int(cell["LB"]["k_hi"]) <= 199
            )
        if good["LB"]:
            row["LB_meets_3_5"] = int(
                float(cell["LB"]["hl_lo"]) <= 5 and float(cell["LB"]["hl_hi"]) >= 3
            )
        for k in ("reject_II", "collapse", "LB_meets_3_5"):
            a, b = str(row.get(k, "")), str(e.get(k, ""))
            row[f"{k}_differs_from_E"] = int(a != b) if a and b else ""
        rows.append(e)
        rows.append(row)
    return rows


def stage_sets(D, out_dir):
    e_all = P.read_csv(os.path.join(out_dir, "ppp_sets.csv"))
    e_rows = {(r["set"], r["currency"]): r for r in e_all}
    null = {
        (r["currency"], r["cfg"]): r
        for r in P.read_csv(os.path.join(out_dir, "sj_null.csv"))
    }
    hl = {
        (r["scheme"], r["model"], r["currency"], r["cfg"], r["p"]): r
        for r in P.read_csv(os.path.join(out_dir, "sj_hl_cells.csv"))
    }
    cols = list(e_all[0].keys()) + ["LB_status", "MP_status", "same_cell_as_E"]
    P.write_csv(
        os.path.join(out_dir, "sj_sets.csv"), cols, sets_rows(D, e_rows, null, hl)
    )


# ================================================================== stage: readings
READING_COLS = [
    "universe",
    "reading",
    "arm",
    "value",
    "N",
    "currencies",
]


def reading_rows(null, n8, n9):
    """The readings of SJ by universe. null: {currency: row of sj_null.csv}. Currencies
    without a (II) verdict leave the count of rejections and Holm; each reading states
    N."""
    out = []

    def add(universe, reading, value, arm="", N="", currencies=""):
        out.append(
            dict(
                universe=universe,
                reading=reading,
                arm=arm,
                value=value,
                N=N,
                currencies=currencies,
            )
        )

    for uname, univ in (("N8", n8), ("N9", n9)):
        ok = [c for c in univ if null[c]["status"] == "ok"]
        rej = [c for c in ok if null[c]["reject_II"] == "1"]
        add(uname, "K_obs", len(rej), N=len(ok), currencies="|".join(ok))
        add(uname, "rejecting_II", "|".join(rej), N=len(ok))
        add(
            uname,
            "rejecting_I",
            "|".join(c for c in univ if null[c]["reject_I"] == "1"),
            N=len(univ),
        )
        pv = {c: float(null[c]["p_II"]) for c in ok}
        hb = P.holm(pv)
        for c in ok:
            add(uname, "p_II_bonferroni", min(1.0, len(ok) * pv[c]), arm=c, N=len(ok))
            add(uname, "p_II_holm", hb[c], arm=c, N=len(ok))
    return out


def stage_readings(D, out_dir):
    null = {r["currency"]: r for r in P.read_csv(os.path.join(out_dir, "sj_null.csv"))}
    P.write_csv(
        os.path.join(out_dir, "sj_readings.csv"),
        READING_COLS,
        reading_rows(null, D.N8, D.N9),
    )


# ================================================================== main
class HLStore(P.Store):
    """The Store of the stage, with the folder of its half-life cells."""

    def __init__(self, work, fingerprint, jobs, stage):
        super().__init__(work, fingerprint, jobs, stage)
        self.dir_hl = os.path.join(work, "hl_sj") if work else None
        if self.dir_hl:
            os.makedirs(self.dir_hl, exist_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", required=True, choices=list(STAGES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work", default=None, help="the item store and the ledgers")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    scale, B = 1, HL.B_MAIN
    assert P.R_NULL % scale == 0 and (P.R_NULL // scale) % 20 == 0
    t0 = time.time()
    if a.stage == "design":
        stage_design(a.out_dir, a.jobs)
        P.finalize()
        return 0
    if a.stage in ("null", "hl") and not a.work:
        raise SystemExit(f"--stage {a.stage} needs --work (the item store)")
    D = SJData(a.out_dir, False)
    fp = sj_fingerprint(a.out_dir, D)
    if a.stage == "hl":
        fp = hashlib.sha256(
            (fp + P.sha_lf(HL.__file__) + f"B={B}").encode()
        ).hexdigest()
    store = HLStore(a.work, fp, a.jobs, f"sj_{a.stage}")
    if a.stage == "data":
        stage_data(D, a.out_dir)
        print(f"sign invariance: max difference {check_sign_data(D, a.out_dir)}")
    elif a.stage == "null":
        stage_null(D, store, a.out_dir, scale)
    elif a.stage == "hl":
        stage_hl(D, store, a.out_dir, B, a.jobs, fp)
    elif a.stage == "sets":
        stage_sets(D, a.out_dir)
    elif a.stage == "readings":
        stage_readings(D, a.out_dir)
    n = P.check_ledgers(a.work, store)
    P.finalize()
    print(
        f"stage sj {a.stage}: {time.time() - t0:.1f}s; {n} consumers in the ledgers",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
