"""The half-lives of the PPP application -- median-unbiased persistence and grid-t
confidence intervals (Hansen 1999), LB against MP.

MODEL. The AR(p_hat + 1) in levels with a constant, fitted as the ADF regression
    Delta y_t = mu + (alpha - 1) y_{t-1} + sum_{j=1}^{p} c_j Delta y_{t-j} + e_t,
t = p + 1, ..., T - 1 (0-based), on y = u_hat = M_Z q (LB: Z = the level-break design of
the set; MP: Z = a constant). alpha is the sum of the level coefficients (equal to
ppp_sieve.ar_levels's alpha to 1e-12: a check). p_hat is the LB residual's sieve order
of the set; MP uses the same p_hat (MP_own: bic_order(q - mean q), a sensitivity).

GRID-t (Hansen 1999). Nodes alpha_k = k / 200. At each node:
  * the restricted fit: Delta y_t - (alpha_k - 1) y_{t-1} on (1, Delta y_{t-1..t-p}),
    giving c_r and the residual e_r (n = T - p - 1 values); the intercept is not part
    of the bootstrap DGP;
  * B = 19,999 replicates: y*_t = y_t for t <= p (the observed initial values), then
    y*_t = sum_j a_j y*_{t-j} + e*_t with a the level coefficients of (alpha_k, c_r);
    recursive scheme: e*_t drawn with replacement from e_r; wild scheme: e*_t = e_r,t
    eta_t, eta Rademacher. Then M_Z y*, the ADF regression with a constant, and
    t* = (alpha* - alpha_k) / se*;
  * the node is accepted iff q_lo <= t_obs(alpha_k) <= q_hi, t_obs = (alpha_hat -
    alpha_k) / se_hat, with q_lo, q_hi the order statistics 500 and 19,500 of the 19,999
    sorted t* (in general (B + 1) 0.025 and (B + 1) 0.975); the median is the order
    statistic 10,000;
  * draws of node k: np.random.default_rng(seed0(item) + k - K_MIN), item
    hl|<scheme>|<LB|MP>|<currency>|[<cfg>|]p<p> (app_seeds): a node's draws do not
    depend on which other nodes were evaluated. Recursive: integers(0, n, (B, n));
    wild: 2 integers(0, 2, (B, n)) - 1;
  * a node with a non-finite t* stops the run.

GRID. k = 100..210 first. The upper end is a cap: alpha = 1.05 (k = 210). Above it the
replicates are dominated by the explosive propagation of the observed initial values and
the distribution of t* collapses, so the grid never grows upward. The lower side grows
by blocks of 20 nodes while its lowest 20-node block holds an accepted node, or while
its lowest node lies below the band or below the median (t_obs < q_lo or t_obs < q_med);
K_MIN = -200 is a hard bound (reaching it stops the run). The CI is the hull
[alpha_{k_lo}, alpha_{k_hi}] of the accepted nodes, with the gaps inside it recorded;
the lower limit is interior. At the cap:
  * if t_obs lies above the band there (alpha = 1.05 rejected in favor of larger
    values), the upper side is open whatever is accepted below: "alpha_hi > 1.05"
    (upper_open = 1, k_hi = 211 codes it, hl_hi = inf); the nodes between the highest
    accepted one and the cap count as gaps (disconnected = 1 if there are any). If no
    node up to the cap is accepted, the whole interval reads "alpha > 1.05"
    (ci_above_cap = 1, k_lo = k_hi = 211, both half-life limits inf);
  * otherwise, if the cap itself is accepted, the upper limit reads "alpha_hi >= 1.05"
    (upper_at_cap = 1); its half-life is inf, as for any alpha_hi >= 1;
  * any other empty set stops the run.

MEDIAN-UNBIASED POINT: Hansen's grid-bootstrap median, from the same draws: the alpha at
which t_obs(alpha) = q_med(alpha). With f(k) = t_obs(alpha_k) - q_med(k), linearly
interpolated between nodes, and f >= 0 at the lowest node: if f >= 0 at the cap, the
point reads "alpha_MU >= 1.05" (mu_at_cap = 1; HL_MU = inf), whatever the crossings
below. Otherwise each sign change gives a crossing point by linear interpolation; near
the unit root the median function's slope can come close to t_obs's, and Monte Carlo
noise can then produce more than one crossing (their number is then odd): alpha_MU is
the median of the crossing points, itself a crossing. The number of crossings is
recorded.

HALF-LIVES. HL(alpha) = ln 0.5 / ln alpha for 0 < alpha < 1, 0 for alpha <= 0, inf for
alpha >= 1; the CI's limits are HL(alpha_lo) and HL(alpha_hi). The IRF half-life at the
MU point uses the restricted fit's coefficients at alpha_MU (ppp_sieve.irf_half_life:
first passage below 1/2, linear interpolation). Collapse (strict), per scheme and
currency: k_hi(MP) >= 200 and k_hi(LB) <= 199.

CELLS. E, both schemes: LB and MP for the nine candidates. E, recursive only: MP_own;
the decomposition LB with one break at a time for AUD, CHF, GBP, NOK and SEK, at p_hat
of E; GBP's general-to-specific order (LB and MP). The S-sets, recursive: LB for their
own designs and MP at their p_hat. A cell is the key (scheme, model, currency, cfg, p):
two specifications with the same key are the same computation.

CHECKS. alpha_OLS = ar_levels's; q -> -q on the LB recursive E cells of the first two
candidates gives the same nodes (t_obs and quantiles to 1e-9, the same accepted flags);
the seed ledger. With --work, every finished cell is stored (keyed by the fingerprint of
the code and inputs) and read back on a relaunch. The outputs are written only after the
checks pass.

    python mlb/ppp_half_lives.py --out-dir DIR --work DIR --jobs 10
"""
from __future__ import annotations

import argparse
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
from numba import njit  # noqa: E402

from mlb import app_seeds as A  # noqa: E402
from mlb import ppp_sieve as S  # noqa: E402

T = 52
B_MAIN = 19_999
LEVEL = 0.95
K_MIN = -200  # node slots r = k - K_MIN in 0..500 (app_seeds)
K_CAP = 210  # alpha = 1.05: the grid never grows above it
K_INIT = (100, K_CAP)
BLOCK = 20
STEP = 200  # alpha_k = k / STEP
SIGN_TOL = 1e-9


def order_stats(B):
    """0-based indices of the (B + 1) 0.025, (B + 1) 0.975 and (B + 1) / 2 order
    statistics; (B + 1) 0.025 must be an integer."""
    lo = (B + 1) * (1 - LEVEL) / 2
    assert abs(lo - round(lo)) < 1e-9 and (B + 1) % 2 == 0, B
    lo = int(round(lo))
    return lo - 1, B + 1 - lo - 1, (B + 1) // 2 - 1


# ================================================================== numba kernels
@njit(cache=True)
def _ls(X, y):
    """Least squares by Householder QR: (beta, ssr, [(X'X)^-1]_{11})."""
    n, k = X.shape
    R = X.copy()
    b = y.copy()
    for j in range(k):
        nrm = 0.0
        for i in range(j, n):
            nrm += R[i, j] * R[i, j]
        nrm = np.sqrt(nrm)
        alpha = -nrm if R[j, j] > 0 else nrm
        v0 = R[j, j] - alpha
        vv = v0 * v0
        for i in range(j + 1, n):
            vv += R[i, j] * R[i, j]
        if vv > 0.0:
            for c in range(j + 1, k):
                s = v0 * R[j, c]
                for i in range(j + 1, n):
                    s += R[i, j] * R[i, c]
                f = 2.0 * s / vv
                R[j, c] -= f * v0
                for i in range(j + 1, n):
                    R[i, c] -= f * R[i, j]
            s = v0 * b[j]
            for i in range(j + 1, n):
                s += R[i, j] * b[i]
            f = 2.0 * s / vv
            b[j] -= f * v0
            for i in range(j + 1, n):
                b[i] -= f * R[i, j]
        R[j, j] = alpha
    beta = np.zeros(k)
    for j in range(k - 1, -1, -1):
        s = b[j]
        for c in range(j + 1, k):
            s -= R[j, c] * beta[c]
        beta[j] = s / R[j, j]
    ssr = 0.0
    for i in range(k, n):
        ssr += b[i] * b[i]
    # [(R'R)^-1]_{11} = ||w||^2 with R' w = e_1 (index 1)
    w = np.zeros(k)
    for j in range(k):
        s = 1.0 if j == 1 else 0.0
        for c in range(j):
            s -= R[c, j] * w[c]
        w[j] = s / R[j, j]
    v11 = 0.0
    for j in range(k):
        v11 += w[j] * w[j]
    return beta, ssr, v11


@njit(cache=True)
def adf_const_nb(y, p):
    """(alpha_hat, se) of Delta y_t on (1, y_{t-1}, Delta y_{t-1..t-p}),
    t = p + 1..T - 1; s^2 = SSR / (n - p - 2)."""
    nt = y.shape[0]
    n = nt - p - 1
    k = p + 2
    X = np.empty((n, k))
    Y = np.empty(n)
    for i in range(n):
        t = p + 1 + i
        Y[i] = y[t] - y[t - 1]
        X[i, 0] = 1.0
        X[i, 1] = y[t - 1]
        for j in range(1, p + 1):
            X[i, 1 + j] = y[t - j] - y[t - j - 1]
    beta, ssr, v11 = _ls(X, Y)
    s2 = ssr / (n - k)
    return 1.0 + beta[1], np.sqrt(s2 * v11)


@njit(cache=True)
def tstar_nb(u0, p, alpha0, a, e_r, idx, signs, scheme, Q):
    """The B replicates of t* at one node (see the module docstring)."""
    nt = Q.shape[0]
    q = Q.shape[1]
    B = idx.shape[0] if scheme == 0 else signs.shape[0]
    out = np.empty(B)
    y = np.empty(nt)
    z = np.empty(nt)
    coef = np.empty(q)
    for b in range(B):
        for t in range(p + 1):
            y[t] = u0[t]
        for t in range(p + 1, nt):
            s = 0.0
            for j in range(p + 1):
                s += a[j] * y[t - 1 - j]
            i = t - p - 1
            if scheme == 0:
                s += e_r[idx[b, i]]
            else:
                s += e_r[i] * signs[b, i]
            y[t] = s
        for m in range(q):
            c = 0.0
            for t in range(nt):
                c += Q[t, m] * y[t]
            coef[m] = c
        for t in range(nt):
            s = y[t]
            for m in range(q):
                s -= Q[t, m] * coef[m]
            z[t] = s
        ah, se = adf_const_nb(z, p)
        out[b] = (ah - alpha0) / se
    return out


# ================================================================== the model
def residual(q, bp):
    """u_hat = M_Z q: the LB design for bp, the constant for bp = ()."""
    q = np.asarray(q, dtype=float)
    if bp:
        return S.lb_residual(q, bp)
    return q - q.mean()


def basis(bp):
    Z = S.lb_design(T, bp) if bp else np.ones((T, 1))
    return np.ascontiguousarray(np.linalg.qr(Z)[0])


def restricted(u, p, alpha0):
    """The restricted ADF fit at alpha0 with an intercept: (c_r, e_r)."""
    du = np.diff(u)
    rows = range(p + 1, len(u))
    y = np.array([du[t - 1] - (alpha0 - 1.0) * u[t - 1] for t in rows])
    X = np.array([[1.0] + [du[t - 1 - j] for j in range(1, p + 1)] for t in rows])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return b[1:].copy(), y - X @ b


def half_life(alpha):
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return float("inf")
    return float(np.log(0.5) / np.log(alpha))


def node(u, p, Q, item, k, scheme, B, stat, r=None):
    """One grid node; its draws come from rng(item, k - K_MIN) (r overrides the
    replication index, for the synthetic tests only)."""
    a0 = k / STEP
    c_r, e_r = restricted(u, p, a0)
    a = S.adf_to_levels(a0, c_r)
    g = A.rng(item, k - K_MIN if r is None else r)
    n = T - p - 1
    if scheme == "rec":
        idx = g.integers(0, n, size=(B, n))
        signs = np.zeros((1, 1))
        sc = 0
    else:
        idx = np.zeros((1, 1), dtype=np.int64)
        signs = (2 * g.integers(0, 2, size=(B, n)) - 1).astype(np.float64)
        sc = 1
    ts = tstar_nb(np.ascontiguousarray(u[:p + 1]), p, a0, a, e_r, idx, signs, sc, Q)
    if not np.all(np.isfinite(ts)):
        raise SystemExit(f"{item} node {k}: a non-finite t*")
    ts.sort()
    ilo, ihi, imed = order_stats(B)
    t_obs = (stat[0] - a0) / stat[1]
    return dict(k=k, alpha0=a0, t_obs=t_obs, q_lo=float(ts[ilo]), q_med=float(ts[imed]),
                q_hi=float(ts[ihi]), accepted=bool(ts[ilo] <= t_obs <= ts[ihi]))


def grid_t(q, bp, p, item, scheme, B=B_MAIN, k_init=K_INIT):
    """The grid-t of one cell: (summary, nodes)."""
    assert k_init[1] <= K_CAP
    u = residual(q, bp)
    Q = basis(bp)
    stat = adf_const_nb(np.ascontiguousarray(u), p)
    nodes = {}

    def ev(ks):
        for k in ks:
            if k < K_MIN:
                raise SystemExit(f"{item}: the grid reached the bound at node {k}")
            nodes[k] = node(u, p, Q, item, k, scheme, B, stat)

    ev(range(k_init[0], k_init[1] + 1))
    while True:
        lo = min(nodes)
        nl = nodes[lo]
        if (any(nodes[k]["accepted"] for k in range(lo, lo + BLOCK))
                or nl["t_obs"] < nl["q_lo"] or nl["t_obs"] < nl["q_med"]):
            ev(range(lo - BLOCK, lo))
        else:
            break
    ks = sorted(nodes)
    acc = [k for k in ks if nodes[k]["accepted"]]
    top = nodes[ks[-1]]
    upper_open = ks[-1] == K_CAP and top["t_obs"] > top["q_hi"]
    above = upper_open and not acc
    if not acc and not above:
        raise SystemExit(f"{item}: empty confidence set")
    if above:
        k_lo = k_hi = K_CAP + 1
        gaps = []
    else:
        k_lo, k_hi = acc[0], (K_CAP + 1 if upper_open else acc[-1])
        gaps = [k for k in range(k_lo, min(k_hi, K_CAP) + 1)
                if not nodes[k]["accepted"]]
    disconnected = upper_open and bool(acc) and acc[-1] < K_CAP
    f = [nodes[k]["t_obs"] - nodes[k]["q_med"] for k in ks]
    if f[0] < 0:
        raise SystemExit(f"{item}: the median crossing lies below the grid")
    changes = [i for i in range(len(ks) - 1) if (f[i] >= 0) != (f[i + 1] >= 0)]
    xs = [(ks[i] + f[i] / (f[i] - f[i + 1])) / STEP for i in changes]
    mu_at_cap = f[-1] >= 0
    if mu_at_cap:
        a_mu = ks[-1] / STEP
        irf_mu = float("inf")
    else:
        assert len(xs) % 2 == 1
        a_mu = xs[len(xs) // 2]
        c_mu, _ = restricted(u, p, a_mu)
        irf_mu = S.irf_half_life(S.adf_to_levels(a_mu, c_mu))
    alpha_ols = stat[0]
    diff = abs(alpha_ols - S.ar_levels(u, p)[2])
    if not diff <= 1e-12:
        raise SystemExit(f"{item}: alpha_OLS differs from ar_levels by {diff}")
    summ = dict(alpha_ols=alpha_ols, se_ols=stat[1], alpha_mu=a_mu,
                mu_at_cap=mu_at_cap, n_crossings=len(xs),
                hl_mu=float("inf") if mu_at_cap else half_life(a_mu),
                irf_hl_mu=irf_mu, k_lo=k_lo, k_hi=k_hi, ci_above_cap=above,
                upper_open=upper_open, disconnected=disconnected,
                alpha_lo=min(k_lo, K_CAP) / STEP, alpha_hi=min(k_hi, K_CAP) / STEP,
                upper_at_cap=k_hi >= ks[-1],
                hl_lo=float("inf") if above else half_life(k_lo / STEP),
                hl_hi=half_life(min(k_hi, K_CAP) / STEP),
                bounded=k_hi <= STEP - 1, n_gaps=len(gaps),
                gaps="|".join(map(str, gaps)), grid_lo=ks[0], grid_hi=ks[-1],
                n_nodes=len(ks), ar_levels_diff=diff)
    return summ, [nodes[k] for k in ks]


# ================================================================== the cells
def cell_list(D):
    """[(scheme, model, currency, bp, p, [(spec, set)])], deduplicated by the key."""
    cells = {}

    def add(scheme, model, c, bp, p, spec, s):
        key = (scheme, model, c, tuple(bp) if model == "LB" else (), int(p))
        cells.setdefault(key, []).append((spec, s))

    for c in A.CANDIDATES:
        e = D.cell("E", c)
        for scheme in A.SCHEMES:
            add(scheme, "LB", c, e["bp"], e["p"], "LB", "E")
            add(scheme, "MP", c, (), e["p"], "MP", "E")
        add("rec", "MP", c, (), D.pmp[c], "MP_own", "E")
        if len(e["bp"]) == 2:
            add("rec", "LB", c, e["bp"][:1], e["p"], "LB_b1", "E")
            add("rec", "LB", c, e["bp"][1:], e["p"], "LB_b2", "E")
    g = D.pgs["GBP"]
    add("rec", "LB", "GBP", D.cell("E", "GBP")["bp"], g, "LB_gs", "E")
    add("rec", "MP", "GBP", (), g, "MP_gs", "E")
    for c in A.CANDIDATES:
        for s in A.SETS[1:]:
            x = D.cell(s, c)
            if x is None:
                continue
            add("rec", "LB", c, x["bp"], x["p"], "LB", s)
            add("rec", "MP", c, (), x["p"], "MP", s)
    return [(k[0], k[1], k[2], k[3], k[4], v) for k, v in cells.items()]


def item_of(scheme, model, c, bp, p):
    if model == "LB":
        return f"hl|{scheme}|LB|{c}|{A.label(bp)}|p{p}"
    return f"hl|{scheme}|MP|{c}|p{p}"


def run_cell(q, scheme, model, c, bp, p, B, store=None, fp=""):
    """One cell, read back from the store if a matching record exists (its lowest
    node is then recomputed from its seed and must equal the stored one)."""
    it = item_of(scheme, model, c, bp, p)
    path = os.path.join(store, it.replace("|", "__") + ".json") if store else None
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            rec = json.load(fh)
        if rec["fingerprint"] == fp and rec["B"] == B:
            b = bp if model == "LB" else ()
            u = residual(q, b)
            stat = adf_const_nb(np.ascontiguousarray(u), p)
            n0 = rec["nodes"][0]
            if node(u, p, basis(b), it, n0["k"], scheme, B, stat) != n0:
                raise SystemExit(f"{it}: the stored cell does not reproduce")
            return dict(rec["summ"], item=it), rec["nodes"]
    summ, nodes = grid_t(q, bp if model == "LB" else (), p, it, scheme, B)
    if path:
        with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(dict(fingerprint=fp, B=B, summ=summ, nodes=nodes), fh)
        os.replace(path + ".tmp", path)
    return dict(summ, item=it), nodes


def check_sign(D, B):
    """q -> -q on the LB recursive E cells of the first two candidates."""
    out = []

    def one(c, sgn):
        e = D.cell("E", c)
        return grid_t(sgn * D.q[c], e["bp"], e["p"], item_of("rec", "LB", c, e["bp"],
                                                                 e["p"]), "rec", B)

    for c in A.CANDIDATES[:2]:
        (s1, n1), (s2, n2) = one(c, 1.0), one(c, -1.0)
        if [x["k"] for x in n1] != [x["k"] for x in n2]:
            raise SystemExit(f"hl sign check {c}: the grids differ")
        d = 0.0
        for x, y in zip(n1, n2):
            if x["accepted"] != y["accepted"]:
                raise SystemExit(f"hl sign check {c}: node {x['k']} flips")
            d = max(d, *(abs(x[k] - y[k]) for k in ("t_obs", "q_lo", "q_med", "q_hi")))
        if d > SIGN_TOL or s1["k_lo"] != s2["k_lo"] or s1["k_hi"] != s2["k_hi"]:
            raise SystemExit(f"hl sign check {c}: max difference {d}")
        out.append(dict(currency=c, cell="rec LB E", max_diff=d,
                        alpha_mu_diff=abs(s1["alpha_mu"] - s2["alpha_mu"])))
    return out


CELL_COLS = ["scheme", "model", "currency", "cfg", "p", "item", "alpha_ols", "se_ols",
             "alpha_mu", "mu_at_cap", "n_crossings", "hl_mu", "irf_hl_mu", "k_lo",
             "k_hi", "ci_above_cap", "upper_open", "disconnected", "alpha_lo",
             "alpha_hi", "upper_at_cap", "hl_lo", "hl_hi",
             "bounded", "n_gaps", "gaps", "grid_lo", "grid_hi", "n_nodes",
             "ar_levels_diff", "specs"]
NODE_COLS = ["scheme", "model", "currency", "cfg", "p", "k", "alpha0", "t_obs", "q_lo",
             "q_med", "q_hi", "accepted"]


def main(argv=None):
    from mlb import ppp_application as P

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--work", default=None,
                    help="the cell store and the consumers' ledgers")
    a = ap.parse_args(argv)
    B = B_MAIN
    D = P.Data(a.out_dir, False)
    fp = hashlib.sha256((P.fingerprint(a.out_dir, D) + P.sha_lf(__file__)
                         + f"B={B}").encode()).hexdigest()
    store = None
    if a.work:
        store = os.path.join(a.work, "hl")
        os.makedirs(store, exist_ok=True)
    cells = cell_list(D)
    t0 = time.time()
    jobs = [(D.q[c], sch, mod, c, bp, p, B, store, fp)
            for sch, mod, c, bp, p, _ in cells]
    if a.jobs > 1:
        from joblib import Parallel, delayed

        from mlb import ppp_half_lives as _self
        res = Parallel(n_jobs=a.jobs, backend="loky", batch_size=1)(
            delayed(_self.run_cell)(*j) for j in jobs)
    else:
        res = [run_cell(*j) for j in jobs]
    rows, nrows = [], []
    for (sch, mod, c, bp, p, specs), (summ, nodes) in zip(cells, res):
        base = dict(scheme=sch, model=mod, currency=c,
                    cfg=A.label(bp) if mod == "LB" else "", p=p)
        summ = dict(summ)
        rows.append(dict(base, **summ, specs="|".join(f"{sp}@{s}" for sp, s in specs)))
        for nd in nodes:
            nrows.append(dict(base, **nd))
    checks = check_sign(D, B)
    led = [(r["item"], "hl", "grid") for r in rows]
    if a.work:
        import glob

        with open(os.path.join(a.work, "ledger_hl.json"), "w", encoding="utf-8",
                  newline="\n") as fh:
            json.dump(led, fh)
        led = []
        for f in sorted(glob.glob(os.path.join(a.work, "ledger_*.json"))):
            led += [tuple(x) for x in json.load(open(f, encoding="utf-8"))]
    A.assert_disjoint({it: A.item_R(it) for it, _, _ in led})
    A.assert_consumers(led)
    P.write_csv(os.path.join(a.out_dir, "hl_cells.csv"), CELL_COLS, rows)
    P.write_csv(os.path.join(a.out_dir, "hl_nodes.csv"), NODE_COLS, nrows)
    P.write_json(os.path.join(a.out_dir, "hl_checks.json"), dict(sign_check=checks))
    P.finalize()
    with open(os.path.join(a.out_dir, "hl_run.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(dict(B=B, n_cells=len(rows)), fh, indent=1)
    print(f"hl: {len(rows)} cells in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
