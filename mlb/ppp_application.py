"""The PPP application: unit-root tests, the band's power, the law of the number of
rejections and the sensitivity sets.

The producer of tab:ppp's columns (I), (II) and (III), the
true size of (I), the Berger-Boos p-values, the power at the Rogoff band's half-lives
and its sensitivities, the joint law of the number of rejections K, Pesaran's CD
and the mean residual correlation, and the per-set rows of the supplement. The
half-lives are mlb/ppp_half_lives.py.

Stages (each reads the outputs of the stages before it from --out-dir):

    design    the exact cbar_half (envelope_exact, Imhof) of every configuration of E
              and S1-S5; reads only the events file (no series, no universe)
    data      the data side: LB residual, sieve order and fit, MZ_t at -7 and at
              cbar_half, the band c_{h,i}, rho_{h,i}, the guards and the IRF
              half-lives, p_MP, GBP's general-to-specific order, R_hat and CD
    null      columns (II) and (III), the size of (I), the i.i.d. null (E)
    bb        Berger-Boos (E)
    power     the band: the main arm, the i.i.d. and stationary arms, the irf and
              unscaled arms, the power of (III) (E)
    joint     the law of K per universe, h (or H0) and kappa
    sets      the per-set rows (reads the half-lives' output too)
    readings  K_obs, the Holm/Bonferroni-adjusted p (II), P(K >= 1) under H0, and the
              half-life medians, counts and collapses, from the outputs above

    python mlb/ppp_application.py --stage design --out-dir DIR
    python mlb/ppp_application.py --stage null --out-dir DIR \\
        --work WORKDIR --jobs 10

A stage writes its outputs only after its checks and the ledger check have passed.

RULES:
  * Work items are keyed by (currency, configuration): rows of two sets that share a
    currency's design are the same cell and byte-identical.
  * Every per-currency object runs for the nine candidates; aggregates for N8 (main)
    and N9 (sensitivity), read from inputs/ppp_sweep/sweep_gates.csv. The panel is
    inputs/ppp_panel.csv.
  * Sieve: ppp_sieve (p = 0..10 lagged differences with a constant; BIC on the common
    sample t = 11..51; smallest p on a tie; p = 0 legal; re-selected per
    configuration). V_hat with s^2 = SSR / (n - k).
  * Draws: replication r of item w is np.random.default_rng(seed0(w) + r),
    eps = standard_normal(T + BURN) (ppp_ar_null.stats_under's recipe); u_0 = 0
    with the AR nuisance started at zero BURN = 100 periods before the sample
    (make_gen_arp; an approximation to its stationary law when a root is close to
    one). R = 200,000 per
    null and per power arm, 50,000 per Berger-Boos point, 100,000 per joint item. The
    first 5 replications of every item used are recomputed from their seeds and must
    equal the stored ones. NaN draws (ok = 0): excluded from critical values and
    p-values, counted as non-rejections in powers and in K; their number is reported.
    cv = the 5% nanpercentile (linear interpolation); p = mean(null <= stat) without a
    +1; the verdict is stat < cv (strict).
  * Standard errors of a critical value, a power and a probability of K: blocks B = 8
    and 20 contiguous in r, with the critical value re-estimated in each block from
    the matching block of the null; se = max(se_8, se_20). p-values and sizes:
    binomial.
  * Guards: the flag sum phi_hat >= 0.9 is printed with the exact bounds of rho_{h,i}
    (< 0 once sum phi_hat > alpha_h, <= -1 once sum phi_hat >= (1 + alpha_h) / 2). A
    (currency, h) with rho_{h,i} <= -1 drops the arms at c_{h,i} (the main arm, the
    stationary arm and the power of (III)) from the power, the band mean and the joint
    K, with a row that says so; the i.i.d., irf and unscaled arms, which do not use
    c_{h,i}, stay. A currency whose phi_hat is non-stationary is not simulated: its
    statistic alone is reported, and it has no (II) verdict, so it leaves K_obs and the
    joint. Rows with rho_{h,i} < 0, or with a power below 0.05 by more than 2 se, carry
    a flag.
  * Berger-Boos: p_hat = 0 not applicable; p_hat = 1, 9 equally spaced points on
    phi_hat +- z_.995 se; p_hat = 2, 16 points at angles 2 pi k / 16 on the
    chi2_2(0.99) ellipse through chol(V_hat), plus the center; p_hat >= 3, 2 p_hat
    points at +- sqrt(chi2_p(0.99)) along the principal axes of V_hat, plus the
    center. The root bound is max(0.995, max|root| of phi_hat): a point beyond it is
    moved along the segment from phi_hat to the first point where the bound is reached
    (a scan in steps of 0.001, then bisection to 1e-12; the factor is recorded). A
    point whose factor is 0 is the center itself: it is reported "= center" and not
    simulated again. p_BB = max_k p(phi_k) + 0.01 over the distinct points; the argmax
    point is re-simulated on the item bbchk.
  * The band: c_{h,i} = T (alpha_h - 1) / (1 - sum phi_hat), alpha_h = 0.5^(1/h).
    Arms on the alt item (common random numbers): main (sieve, c_{h,i}, u_0 = 0); iid
    (phi = 0, c_h, the i.i.d. null's cv); stationary (sieve, c_{h,i}, the pre-sample
    state drawn from the exact stationary law of the AR(p_hat + 1), with p_hat + 1
    normals drawn after eps). The irf arm: rho from ppp_sieve.rho_irf, on its own
    item; the unscaled arm: c_h = 52 (0.5^(1/h) - 1) (the i.i.d. arm's c, not divided
    by 1 - sum phi_hat) with phi_hat, on its own item; for p_hat = 0 both are
    "= main", not simulated. The power of (III): c_{h,i}, the statistic at
    cbar_half against cv (III), on the item alt3.
  * The count: R_hat = corrcoef of the unrestricted ADF residuals e_hat at p_hat of
    the universe's currencies with a stationary phi_hat (those simulated under H0),
    aligned by year on their common sample (the years at which every one of them has a
    residual), estimated once per universe and restricted to the currencies kept at
    each h; positive definiteness asserted.
    R(kappa) = (1 - kappa) I + kappa R_hat, kappa = 0, 0.5, 1. Replication r:
    standard_normal((T + BURN, N)) @ chol(R(kappa)).T; the stationary arm draws
    sum(p_i + 1) further normals after them, from the joint stationary law (one
    discrete Lyapunov equation on the stacked states, innovation covariance
    R(kappa) kron e1 e1'). K_obs is the number of (II) rejections on the data among the
    currencies in the joint. The two arms are common random numbers.
  * CD and rho-bar on Delta u_hat (E) and on e_hat of every currency of the universe
    over their common sample.
  * Readings: the median of N half-lives is the middle value, or for even N the mean of
    the two middle values (inf if either is inf); an LB interval meets [3, 5] iff
    hl_lo <= 5 and hl_hi >= 3; unbounded iff hl_hi = inf; collapse (strict) iff
    k_hi(MP) >= 200 and k_hi(LB) <= 199; Holm and Bonferroni adjust p (II) over the
    universe's currencies with a verdict.
  * Sign check of the null draws: the first two E currencies with p_hat >= 1 (those
    that exist; "not applicable" if none), their first 2,000 draws (or R if fewer)
    recomputed with phi_hat from -q, to 1e-9.
  * --work (the item store) is required by the stages null, bb, power and joint.
  * Monte Carlo consistency checks (a failure stops the run): the Berger-Boos center
    within 4 combined se of p (II); the joint marginals against the band's powers (and
    5% under H0) by z-tests at a Bonferroni 0.1% family level; at kappa = 0 the law of
    K against the Poisson binomial of its marginals (chi-square, p > 1e-4).
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
# the loky workers import this file as mlb.ppp_application
_pp = os.environ.get("PYTHONPATH", "")
if _REPO not in _pp.split(os.pathsep):
    os.environ["PYTHONPATH"] = _REPO + (os.pathsep + _pp if _pp else "")

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402
from scipy import linalg, stats  # noqa: E402

from mlb import app_seeds as A  # noqa: E402
from mlb import ppp_ar_null as AN  # noqa: E402
from mlb import ppp_sieve as S  # noqa: E402
from mlb import regime_events as RE  # noqa: E402

T = 52
START_YEAR = 1973
KMAX = 12  # MAIC
BURN = 100
CBAR = -7.0
CV_ASY = -1.98
H = (3, 4, 5)
KAPPA = {"0": 0.0, "0.5": 0.5, "1": 1.0}
BLOCKS = (8, 20)
R_NULL, R_BB, R_ALT, R_JOINT = 200_000, 50_000, 200_000, 100_000
HEAD_CHECK = 5  # replications of every item recomputed from their seeds
BB_LEVEL = 0.99
BB_ROOT_BOUND = 0.995
CBAR_BRACKETS = ((-25.0, -1.0), (-40.0, -1.0))
CBAR_XTOL = 1e-9
N8 = ("AUD", "CAD", "DKK", "GBP", "JPY", "NOK", "NZD", "SEK")
N9 = A.CANDIDATES
PANEL = os.path.join(_REPO, "inputs", "ppp_panel.csv")
SWEEP = os.path.join(_REPO, "inputs", "ppp_sweep", "sweep_gates.csv")
NAN = float("nan")


# ================================================================== small helpers
def fmt(x):
    """Full precision: repr of a float, 'nan'/'inf' literal, ints and strings as is."""
    if isinstance(x, (bool, np.bool_)):
        return str(int(x))
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return repr(float(x))
    return str(x)


_PENDING = []


def write_csv(path, header, rows):
    """Writes path.tmp; finalize() renames it once the stage's checks have passed."""
    with open(path + ".tmp", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([fmt(r.get(h, "")) for h in header])
    _PENDING.append(path)


def write_json(path, obj):
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1)
    _PENDING.append(path)


def finalize():
    for p in _PENDING:
        os.replace(p + ".tmp", p)
    _PENDING.clear()


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def sha_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def cfg_label(bp):
    return A.label(bp)


def parse_bp(s):
    return tuple(int(x) for x in s.split("|")) if s else ()


def block_edges(R, B):
    return np.linspace(0, R, B + 1).astype(int)


def cv5(x):
    return float(np.nanpercentile(x, 5))


def pvalue(null, stat):
    fin = np.isfinite(null)
    n = int(fin.sum())
    p = float(np.mean(null[fin] <= stat))
    return p, float(np.sqrt(p * (1 - p) / n)), n


def power(alt, cv):
    """Share of the R draws with stat < cv; a NaN draw is a non-rejection."""
    return float(np.sum(alt < cv) / len(alt))


def block_se(per_block):
    """{B: per-block values} -> (se_8, se_20, max)."""
    s = {B: float(np.std(v, ddof=1) / np.sqrt(B)) for B, v in per_block.items()}
    return s[8], s[20], max(s[8], s[20])


def cv_blocks(null):
    R = len(null)
    return {B: np.array([cv5(null[a:b]) for a, b in
                         zip(block_edges(R, B)[:-1], block_edges(R, B)[1:])])
            for B in BLOCKS}


def power_blocks(alt, cvb):
    R = len(alt)
    return {B: np.array([power(alt[a:b], cvb[B][i]) for i, (a, b) in
                         enumerate(zip(block_edges(R, B)[:-1], block_edges(R, B)[1:]))])
            for B in BLOCKS}


def same(a, b):
    """Bit-for-bit equality with NaN in the same places."""
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and \
        np.array_equal(np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0))


# ================================================================== kernels
_KG = {}


def kernels():
    """The kernels and the AR(p) generator, bound once per process."""
    if not _KG:
        K = AN.load_kernels()
        _KG["K"] = K
        _KG["gen"] = AN.make_gen_arp(K["njit"])
    return _KG["K"], _KG["gen"]


@njit(cache=True)
def gen_levels_nb(nt, burn, a, s0, eps):
    """u_t = sum_j a_j u_{t-j} + eps[burn + t - 1], t = 1..nt, from the pre-sample
    state s0 = (u_0, u_{-1}, ..., u_{-p}); returns (u_1, ..., u_nt). The innovations
    are the ones make_gen_arp feeds its u_1..u_nt."""
    k = a.shape[0]
    buf = np.empty(k + nt)
    for j in range(k):
        buf[k - 1 - j] = s0[j]
    for t in range(nt):
        acc = eps[burn + t]
        for j in range(k):
            acc += a[j] * buf[k - 1 + t - j]
        buf[k + t] = acc
    return buf[k:].copy()


def companion(a):
    a = np.asarray(a, dtype=float)
    k = len(a)
    C = np.zeros((k, k))
    C[0, :] = a
    if k > 1:
        C[1:, :-1] = np.eye(k - 1)
    return C


def stationary_cov(a):
    """The covariance of the state (u_t, ..., u_{t-p}) of a stationary AR with level
    coefficients a and unit innovation variance: Gamma = C Gamma C' + e1 e1'."""
    C = companion(a)
    Q = np.zeros_like(C)
    Q[0, 0] = 1.0
    G = linalg.solve_discrete_lyapunov(C, Q)
    return (G + G.T) / 2


def stationary_cov_joint(a_list, Rk):
    """The joint covariance of the stacked states of N ARs whose innovations have
    covariance Rk: Gamma = C Gamma C' + Q, C = blockdiag(C_i), Q = Rk kron e1 e1'."""
    sizes = [len(a) for a in a_list]
    C = linalg.block_diag(*[companion(a) for a in a_list])
    Q = np.zeros_like(C)
    first = np.cumsum([0] + sizes[:-1])
    for i, fi in enumerate(first):
        for j, fj in enumerate(first):
            Q[fi, fj] = Rk[i, j]
    G = linalg.solve_discrete_lyapunov(C, Q)
    return (G + G.T) / 2, first


def observed(q, bp, cbar):
    """MZ_t of the series at the configuration; NaN if the kernel flags ok = 0."""
    K, _ = kernels()
    Z = K["build_z"](T, np.asarray(bp, dtype=np.int64))
    _, _, mz, _, _, ok = K["mstats"](np.asarray(q, dtype=np.float64), Z, float(cbar),
                                     K["method_code"]["maic"], KMAX)
    return float(mz) if ok > 0.5 else NAN


# ================================================================== simulation
def _chunk(bp, arms, seed0, r0, r1):
    """Replications r0 .. r1-1 of one item; every arm on the same eps.

    arms: tuple of (kind, c, phi, cbar); kind "u0zero" is make_gen_arp (u_0 = 0),
    "stationary" the AR(p + 1) of (1 - rho L) Phi(L), rho = 1 + c/T, from the exact
    stationary state, whose p + 1 normals are drawn after eps. Returns (r1 - r0, arms)
    MZ_t, NaN where ok = 0.
    """
    K, gen = kernels()
    Z = K["build_z"](T, np.asarray(bp, dtype=np.int64))
    mc = K["method_code"]["maic"]
    prep = []
    k0 = 0
    for kind, c, phi, cbar in arms:
        phi = np.asarray(phi, dtype=np.float64)
        if kind == "stationary":
            a = S.level_coeffs(1.0 + c / T, phi)
            L = np.linalg.cholesky(stationary_cov(a))
            k0 = max(k0, len(a))
            prep.append((kind, float(c), phi, float(cbar), a, L))
        else:
            assert kind == "u0zero", kind
            prep.append((kind, float(c), phi, float(cbar), None, None))
    out = np.full((r1 - r0, len(arms)), np.nan)
    for i, r in enumerate(range(r0, r1)):
        g = np.random.default_rng(seed0 + r)
        eps = g.standard_normal(T + BURN)
        z0 = g.standard_normal(k0) if k0 else None
        for j, (kind, c, phi, cbar, a, L) in enumerate(prep):
            if kind == "stationary":
                y = gen_levels_nb(T, BURN, a, L @ z0[:len(a)], eps)
            else:
                y = gen(T, BURN, c, phi, eps)
            _, _, mz, _, _, ok = K["mstats"](y, Z, cbar, mc, KMAX)
            if ok > 0.5:
                out[i, j] = mz
    return out


def _joint_chunk(bps, phis, cs, cbar, L, kind, Lstat, sizes, seed0, r0, r1):
    """Replications r0 .. r1-1 of a joint item: eps = standard_normal((T + BURN, N))
    @ L.T; currency i uses column i. kind "stationary": sum(sizes) extra normals, the
    stacked pre-sample state Lstat @ z. Returns (r1 - r0, N) MZ_t."""
    K, gen = kernels()
    mc = K["method_code"]["maic"]
    N = len(bps)
    Zs = [K["build_z"](T, np.asarray(b, dtype=np.int64)) for b in bps]
    phis = [np.asarray(p, dtype=np.float64) for p in phis]
    avec = [S.level_coeffs(1.0 + c / T, p) for c, p in zip(cs, phis)]
    first = np.cumsum([0] + list(sizes[:-1])) if sizes else []
    out = np.full((r1 - r0, N), np.nan)
    for i, r in enumerate(range(r0, r1)):
        g = np.random.default_rng(seed0 + r)
        E = g.standard_normal((T + BURN, N)) @ L.T
        if kind == "stationary":
            s = Lstat @ g.standard_normal(int(sum(sizes)))
        for j in range(N):
            eps = np.ascontiguousarray(E[:, j])
            if kind == "stationary":
                y = gen_levels_nb(T, BURN, avec[j], s[first[j]:first[j] + sizes[j]],
                                  eps)
            else:
                y = gen(T, BURN, float(cs[j]), phis[j], eps)
            _, _, mz, _, _, ok = K["mstats"](y, Zs[j], float(cbar), mc, KMAX)
            if ok > 0.5:
                out[i, j] = mz
    return out


def _parallel(fn, args, R, jobs):
    """fn(*args, r0, r1) over R replications, jobs workers, 4 chunks per worker."""
    if jobs <= 1:
        return fn(*args, 0, R)
    from joblib import Parallel, delayed

    from mlb import ppp_application as _self
    f = getattr(_self, fn.__name__)
    edges = np.linspace(0, R, jobs * 4 + 1).astype(int)
    parts = Parallel(n_jobs=jobs, backend="loky", batch_size=1)(
        delayed(f)(*args, int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])
        if b > a)
    return np.vstack(parts)


def simulate(bp, arms, seed0, R, jobs):
    return _parallel(_chunk, (tuple(bp), tuple(arms), int(seed0)), R, jobs)


def simulate_joint(bps, phis, cs, cbar, L, kind, Lstat, sizes, seed0, R, jobs):
    args = (tuple(tuple(b) for b in bps), tuple(tuple(p) for p in phis), tuple(cs),
            float(cbar), L, kind, Lstat, tuple(sizes), int(seed0))
    return _parallel(_joint_chunk, args, R, jobs)


# ================================================================== item store
class Store:
    """One npz per item under WORK/items, keyed by the fingerprint of the code and the
    inputs and by the item's own specification; a stored item that does not match is
    recomputed, never reused. Every item used has its first HEAD_CHECK replications
    recomputed from their seeds and compared bit for bit."""

    def __init__(self, work, fingerprint, jobs, stage):
        self.dir = os.path.join(work, "items") if work else None
        self.fp = fingerprint
        self.jobs = jobs
        self.stage = stage
        self.ledger = []
        if self.dir:
            os.makedirs(self.dir, exist_ok=True)

    def path(self, item):
        return os.path.join(self.dir, item.replace("|", "__") + ".npz")

    def get(self, item, arms, R, sim, spec, consume=True):
        """sim(R, jobs) -> the item's (R, ...) array; spec: the repr of every
        parameter of its arms. consume=False reads an earlier stage's item without
        recording it in this stage's ledger."""
        if consume:
            for arm in arms:
                self.ledger.append((item, self.stage, arm))
        arr = None
        if self.dir and os.path.exists(self.path(item)):
            with np.load(self.path(item)) as z:
                if (str(z["fingerprint"]) == self.fp and int(z["R"]) == R
                        and list(z["arms"]) == list(arms) and str(z["spec"]) == spec):
                    arr = z["mzt"]
        if arr is None:
            if not consume:
                raise SystemExit(f"{item}: not in the store; run its stage first")
            arr = sim(R, self.jobs)
            if self.dir:
                tmp = self.path(item) + ".tmp.npz"
                np.savez(tmp, mzt=arr, fingerprint=self.fp, R=R, arms=np.array(arms),
                         spec=spec, seed0=A.seed0(item))
                os.replace(tmp, self.path(item))
        if not same(arr[:HEAD_CHECK], sim(HEAD_CHECK, 1)):
            raise SystemExit(f"{item}: the stored draws do not reproduce from their "
                             "seeds")
        return arr


# ================================================================== the data
def configurations():
    """{(set, currency): bp} over the candidates and E, S1-S5 (bp () = no design)."""
    rows = RE.load()
    return {(s, c): tuple(RE.configuration(rows, c, s)[1])
            for c in A.CANDIDATES for s in A.SETS}


def taus():
    rows = RE.load()
    return {(s, c): tuple(RE.configuration(rows, c, s)[0])
            for c in A.CANDIDATES for s in A.SETS}


def design_configs():
    """The distinct non-empty configurations, sorted by (m, positions)."""
    return sorted({b for b in configurations().values() if b},
                  key=lambda b: (len(b), b))


def load_panel(path=PANEL):
    """The panel's q."""
    q = {}
    for r in read_csv(path):
        y = int(r["year"])
        if y >= START_YEAR:
            q.setdefault(r["currency"], []).append((y, float(r["q"])))
    out = {}
    for c in A.CANDIDATES:
        obs = sorted(q[c])
        assert [y for y, _ in obs] == list(range(START_YEAR, START_YEAR + T)), c
        out[c] = np.array([v for _, v in obs])
    return out


SYNTH_DGP = {"AUD": (0.90, ()), "CAD": (0.85, (0.6,)), "CHF": (0.95, (0.5, 0.3)),
             "DKK": (0.80, (0.3, -0.4, 0.5)), "GBP": (0.92, (0.7,)),
             "JPY": (0.97, (0.4, 0.3)), "NOK": (0.88, (0.2, 0.1, 0.6)),
             "NZD": (0.75, (0.5,)), "SEK": (0.93, (0.6, -0.3))}


def synthetic_panel():
    """Nine synthetic series on the E designs: u_t = rho u_{t-1} + w_t, Phi(L) w_t =
    e_t (SYNTH_DGP), plus a common factor; seed item synth|panel, replication j for
    currency j and 100 for the factor. A check of the code on known series; no step of
    the section scripts uses it."""
    conf = configurations()
    f = A.rng("synth|panel", 100).standard_normal(T + 50) * 0.03
    out = {}
    for j, c in enumerate(A.CANDIDATES):
        rho, phi = SYNTH_DGP[c]
        g = A.rng("synth|panel", j)
        e = g.standard_normal(T + 50) * 0.05 + f
        w = np.zeros(T + 50)
        u = np.zeros(T + 50)
        for t in range(1, T + 50):
            w[t] = e[t] + sum(ph * w[t - 1 - i] for i, ph in enumerate(phi)
                              if t - 1 - i >= 0)
            u[t] = rho * u[t - 1] + w[t]
        Z = S.lb_design(T, conf[("E", c)])
        beta = g.normal(0.0, 0.15, Z.shape[1])
        out[c] = Z @ beta + u[50:]
    return out


def universes(synthetic):
    """(N8, N9) from sweep_gates.csv: C4 and C6 at 5% for the main set; the 1% set
    (equal to the set without C6) for the sensitivity; asserted against N8 and N9."""
    if synthetic:
        return N8, N9
    rows = [r for r in read_csv(SWEEP) if r.get("kappa2", "") != ""]
    main = tuple(sorted(r["currency"] for r in rows
                        if r["C4_5"] == "1" and r["C6_5"] == "1"))
    one = tuple(sorted(r["currency"] for r in rows
                       if r["C4_1"] == "1" and r["C6_1"] == "1"))
    noc6 = tuple(sorted(r["currency"] for r in rows if r["C4_5"] == "1"))
    assert main == N8 and one == N9 and noc6 == N9, (main, one, noc6)
    return main, one


def band_entry(h, fit):
    if not fit["stationary"]:
        return dict(c=NAN, rho=NAN, guard="nonstationary_phi", irf_hl=NAN)
    c = S.c_band(h, fit["phi"], T)
    rho = 1.0 + c / T
    if rho <= -1.0:
        return dict(c=c, rho=rho, guard="explosive_rho", irf_hl=NAN)
    return dict(c=c, rho=rho, guard="ok",
                irf_hl=S.irf_half_life(S.level_coeffs(rho, fit["phi"])))


def cell_fit(q, bp, cbar_half):
    """Everything the data side needs for one (currency, configuration) cell."""
    assert len(bp) >= 1 and all(1 <= b <= T - 1 for b in bp), bp
    u = S.lb_residual(q, bp)
    p, bics = S.bic_order(u)
    fit = S.sieve_fit(u, p, START_YEAR)
    return dict(bp=tuple(bp), u=u, p=p, bics=bics, fit=fit,
                mzt_m7=observed(q, bp, CBAR), mzt_half=observed(q, bp, cbar_half),
                cbar_half=cbar_half, band={h: band_entry(h, fit) for h in H})


def read_cbar(out_dir):
    return {parse_bp(r["break_pos"]): float(r["cbar_half"])
            for r in read_csv(os.path.join(out_dir, "cbar_half_configs.csv"))}


class Data:
    """The data side, recomputed identically by every stage (< 1 s)."""

    def __init__(self, out_dir, synthetic=False, sign=1.0):
        self.synthetic = synthetic
        q = synthetic_panel() if synthetic else load_panel()
        self.q = {c: sign * np.asarray(v, dtype=float) for c, v in q.items()}
        self.conf = configurations()
        self.taus = taus()
        self.cbar = read_cbar(out_dir)
        self.cells = {}
        for c, bp in A.CELLS:
            self.cells[(c, bp)] = cell_fit(self.q[c], bp, self.cbar[bp])
        got = {(c, b) for (s, c), b in self.conf.items() if b}
        assert got == set(A.CELLS), "the events' designs differ from the registry"
        self.N8, self.N9 = universes(synthetic)
        self.pmp = {c: S.bic_order(self.q[c] - self.q[c].mean())[0]
                    for c in A.CANDIDATES}
        self.pgs = {"GBP": S.gs_order(self.cell("E", "GBP")["u"])}

    def cell(self, s, c):
        bp = self.conf[(s, c)]
        return self.cells[(c, bp)] if bp else None

    def is_E(self, c, bp):
        return self.conf[("E", c)] == tuple(bp)

    def simulated(self, univ):
        """The universe's currencies with a stationary phi_hat (E)."""
        return [c for c in univ if self.cell("E", c)["fit"]["stationary"]]

    def residual_matrix(self, cur):
        """The e_hat of the currencies cur, aligned on the years at which all have
        one."""
        fits = [self.cell("E", c)["fit"] for c in cur]
        years = sorted(set.intersection(*[set(f["e_years"].tolist()) for f in fits]))
        M = np.column_stack([f["e"][np.searchsorted(f["e_years"], years)]
                             for f in fits])
        return np.array(years), M

    def rhat(self, univ):
        """R_hat over the universe's simulated currencies (positive definite)."""
        cur = self.simulated(univ)
        years, M = self.residual_matrix(cur)
        R = np.corrcoef(M.T)
        np.linalg.cholesky(R)
        return cur, years, R


def fingerprint(out_dir, D):
    """The code and inputs every stored item depends on."""
    from mlb import kernel_numba
    files = [os.path.abspath(__file__), S.__file__, A.__file__, AN.__file__,
             kernel_numba.__file__, RE.__file__, RE.EVENTS,
             os.path.join(out_dir, "cbar_half_configs.csv")]
    if not D.synthetic:
        files += [PANEL, SWEEP]
    h = hashlib.sha256()
    for f in files:
        h.update(sha_lf(f).encode())
    for c in A.CANDIDATES:
        h.update(np.ascontiguousarray(D.q[c]).tobytes())
    h.update(f"synthetic={D.synthetic}".encode())
    return h.hexdigest()


# ================================================================== stage: design
def _cbar_one(bp):
    from mlb import envelope_exact as ee
    lam = tuple((b + 0.5) / T for b in bp)
    assert ee.break_positions(T, lam) == list(bp), (bp, lam)
    last = None
    for lo, hi in CBAR_BRACKETS:
        try:
            cb, bT = ee.half_power_point(T, lam, bracket=(lo, hi), xtol=CBAR_XTOL)
            return dict(bp=bp, lam=lam, cbar=cb, bT=bT, lo=lo, hi=hi)
        except ValueError as e:  # no sign change in the bracket
            last = e
    raise SystemExit(f"cbar_half {bp}: no root in {CBAR_BRACKETS} ({last})")


def stage_design(out_dir, jobs):
    from mlb import envelope_exact as ee
    cfgs = design_configs()
    if jobs > 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=min(jobs, len(cfgs)), backend="loky")(
            delayed(_cbar_one)(b) for b in cfgs)
    else:
        res = [_cbar_one(b) for b in cfgs]
    rows = []
    for r in res:
        rows.append(dict(cfg=cfg_label(r["bp"]), m=len(r["bp"]),
                         break_pos="|".join(map(str, r["bp"])),
                         lam="|".join(repr(x) for x in r["lam"]), cbar_half=r["cbar"],
                         b_T=r["bT"], displacement=abs(r["cbar"] - CBAR),
                         bracket_lo=r["lo"], bracket_hi=r["hi"], xtol=CBAR_XTOL))
    path = os.path.join(out_dir, "cbar_half_configs.csv")
    write_csv(path, ["cfg", "m", "break_pos", "lam", "cbar_half", "b_T",
                     "displacement", "bracket_lo", "bracket_hi", "xtol"], rows)
    man = dict(T=T, conventions=ee.conventions(),
               envelope_exact_sha256=sha_lf(ee.__file__),
               regime_events_sha256=sha_lf(RE.EVENTS),
               csv_sha256=sha_lf(path + ".tmp"))
    write_json(os.path.join(out_dir, "cbar_half_configs.manifest.json"), man)


# ================================================================== stage: data
def stage_data(D, out_dir):
    diag, stat = [], []
    for c in A.CANDIDATES:
        for s in A.SETS:
            bp = D.conf[(s, c)]
            base = dict(set=s, currency=c, tau="|".join(map(str, D.taus[(s, c)])),
                        break_pos="|".join(map(str, bp)), m=len(bp),
                        cfg=cfg_label(bp) if bp else "none")
            if not bp:
                diag.append(dict(base, status="no design"))
                stat.append(dict(base, status="no design"))
                continue
            x = D.cells[(c, bp)]
            f = x["fit"]
            d = dict(base, status="ok", p_hat_lagdiff=x["p"],
                     phi="|".join(repr(float(v)) for v in f["phi"]),
                     sum_phi=f["sum_phi"], flag09=f["flag09"], max_root=f["max_root"],
                     stationary=f["stationary"],
                     p_hat_mp=D.pmp[c] if s == "E" else "",
                     p_gs=D.pgs.get(c, "") if s == "E" else "")
            d.update({f"bic_{p}": x["bics"][p] for p in range(S.K_MAX_BIC + 1)})
            diag.append(d)
            st = dict(base, status="ok", p_hat_lagdiff=x["p"], mzt_I=x["mzt_m7"],
                      reject_I=x["mzt_m7"] < CV_ASY, cbar_half=x["cbar_half"],
                      mzt_III=x["mzt_half"])
            for h in H:
                b = x["band"][h]
                st.update({f"c_h{h}": b["c"], f"rho_h{h}": b["rho"],
                           f"guard_h{h}": b["guard"], f"irf_hl_h{h}": b["irf_hl"],
                           f"rho_negative_h{h}": int(b["rho"] < 0)})
            stat.append(st)
            if not np.isfinite(x["mzt_m7"]) or not np.isfinite(x["mzt_half"]):
                raise SystemExit(f"{c} {s}: degenerate MZ_t on the data")
    write_csv(os.path.join(out_dir, "ppp_ar_diagnostic.csv"),
              ["set", "currency", "tau", "break_pos", "m", "cfg", "status",
               "p_hat_lagdiff"] + [f"bic_{p}" for p in range(S.K_MAX_BIC + 1)]
              + ["phi", "sum_phi", "flag09", "max_root", "stationary", "p_hat_mp",
                 "p_gs"], diag)
    write_csv(os.path.join(out_dir, "ppp_statistics.csv"),
              ["set", "currency", "tau", "break_pos", "m", "cfg", "status",
               "p_hat_lagdiff", "mzt_I", "reject_I", "cbar_half", "mzt_III"]
              + [f"{k}_h{h}" for h in H
                 for k in ("c", "rho", "guard", "irf_hl", "rho_negative")], stat)
    bounds = [dict(h=h, alpha_h=S.alpha_h(h), sum_phi_rho_negative=S.alpha_h(h),
                   sum_phi_rho_minus_one=(1 + S.alpha_h(h)) / 2) for h in H]
    write_csv(os.path.join(out_dir, "band_bounds.csv"),
              ["h", "alpha_h", "sum_phi_rho_negative", "sum_phi_rho_minus_one"], bounds)
    cdrows, samp = [], []
    for name, univ in (("N8", D.N8), ("N9", D.N9)):
        cur, years, _ = D.rhat(univ)
        samp.append(dict(universe=name, currencies="|".join(cur),
                         excluded_nonstationary="|".join(c for c in univ
                                                         if c not in cur),
                         first_year=int(years[0]), last_year=int(years[-1]),
                         n_years=len(years)))
        cdrows += cd_rows(D, name, univ)
    write_csv(os.path.join(out_dir, "rhat_sample.csv"),
              ["universe", "currencies", "excluded_nonstationary", "first_year",
               "last_year", "n_years"], samp)
    write_csv(os.path.join(out_dir, "pesaran_cd.csv"),
              ["universe", "series", "N", "first_year", "last_year", "CD", "pvalue",
               "n_pairs", "mean_rho", "mean_abs_rho"], cdrows)
    man = dict(panel_sha256=None if D.synthetic else sha_lf(PANEL),
               sweep_sha256=None if D.synthetic else sha_lf(SWEEP),
               regime_events_sha256=sha_lf(RE.EVENTS),
               cbar_half_sha256=sha_lf(os.path.join(out_dir, "cbar_half_configs.csv")),
               synthetic=D.synthetic, universes=dict(N8=list(D.N8), N9=list(D.N9)))
    write_json(os.path.join(out_dir, "data.manifest.json"), man)


def cd_rows(D, name, univ):
    """Pesaran's CD and rho-bar on Delta u_hat (E) and on e_hat (common sample)."""
    from mlb import pesaran_cd as PCD
    out = []
    U = {c: np.diff(D.cell("E", c)["u"]) for c in univ}
    nm, rho, Tij = PCD.pairwise_corr(U)
    out.append(dict(universe=name, series="delta_u_hat", N=len(nm),
                    first_year=START_YEAR + 1, last_year=START_YEAR + T - 1,
                    **PCD.cd_statistic(rho, Tij)))
    years, M = D.residual_matrix(list(univ))
    U = {c: M[:, i] for i, c in enumerate(univ)}
    nm, rho, Tij = PCD.pairwise_corr(U)
    out.append(dict(universe=name, series="e_hat", N=len(nm), first_year=int(years[0]),
                    last_year=int(years[-1]), **PCD.cd_statistic(rho, Tij)))
    return out


def check_sign(D, out_dir):
    """q -> -q with the same code: MZ_t, p_hat, phi_hat, R_hat and CD equal to 1e-12
    (every value finite on both sides); e_hat flips sign."""
    M = Data(out_dir, D.synthetic, sign=-1.0)
    worst = 0.0

    def cmp(a, b):
        a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
        finite = np.all(np.isfinite(a)) and np.all(np.isfinite(b))
        if not finite or a.shape != b.shape:
            raise SystemExit("sign check: a non-finite value or a shape mismatch")
        return float(np.max(np.abs(a - b), initial=0.0))

    for key, x in D.cells.items():
        y = M.cells[key]
        if x["p"] != y["p"]:
            raise SystemExit(f"sign check: p_hat differs at {key}")
        worst = max(worst, cmp(x["mzt_m7"], y["mzt_m7"]),
                    cmp(x["mzt_half"], y["mzt_half"]),
                    cmp(x["fit"]["phi"], y["fit"]["phi"]),
                    cmp(x["fit"]["e"], -y["fit"]["e"]))
    for name, univ in (("N8", D.N8), ("N9", D.N9)):
        worst = max(worst, cmp(D.rhat(univ)[2], M.rhat(univ)[2]))
        for a, b in zip(cd_rows(D, name, univ), cd_rows(M, name, univ)):
            worst = max(worst, cmp([a["CD"], a["mean_rho"]], [b["CD"], b["mean_rho"]]))
    if worst > 1e-12:
        raise SystemExit(f"sign check: max difference {worst}")
    return worst


# ================================================================== stage: null
def null_arms(D, c, bp):
    x = D.cells[(c, bp)]
    arms2 = [("u0zero", 0.0, tuple(x["fit"]["phi"]), CBAR)]
    names2 = ["sieve"]
    if D.is_E(c, bp):
        arms2.append(("u0zero", 0.0, (), CBAR))
        names2.append("iid")
    arms3 = [("u0zero", 0.0, tuple(x["fit"]["phi"]), x["cbar_half"])]
    return tuple(arms2), names2, tuple(arms3)


def sim_items(bp, arms, item):
    """The simulator of an item, with its specification."""
    s0 = A.seed0(item)
    return (lambda R, jobs: simulate(bp, arms, s0, R, jobs)), repr((tuple(bp), arms))


def null_arrays(D, store, c, bp, scale, consume=True):
    """{"sieve", "iid" (E), "III"}: the R null draws, or None if not simulated."""
    x = D.cells[(c, bp)]
    if not x["fit"]["stationary"]:
        return None
    R = R_NULL // scale
    arms2, names2, arms3 = null_arms(D, c, bp)
    it2, it3 = f"null|{c}|{cfg_label(bp)}", f"null3|{c}|{cfg_label(bp)}"
    a2 = store.get(it2, names2, R, *sim_items(bp, arms2, it2), consume)
    a3 = store.get(it3, ["sieve"], R, *sim_items(bp, arms3, it3), consume)
    out = {n: a2[:, i] for i, n in enumerate(names2)}
    out["III"] = a3[:, 0]
    return out


def null_row(D, c, bp, nul):
    x = D.cells[(c, bp)]
    row = dict(currency=c, cfg=cfg_label(bp), break_pos="|".join(map(str, bp)),
               m=len(bp), p_hat=x["p"], E=D.is_E(c, bp), mzt_I=x["mzt_m7"],
               reject_I=x["mzt_m7"] < CV_ASY, cbar_half=x["cbar_half"],
               mzt_III=x["mzt_half"])
    if nul is None:
        row["status"] = "not simulated (non-stationary phi_hat)"
        return row
    row["status"] = "ok"
    R = len(nul["sieve"])
    row["R"] = R
    for tag, arr, stat_ in (("II", nul["sieve"], x["mzt_m7"]),
                            ("III", nul["III"], x["mzt_half"])):
        cv = cv5(arr)
        s8, s20, se = block_se(cv_blocks(arr))
        p, sep, n = pvalue(arr, stat_)
        row.update({f"cv_{tag}": cv, f"cv_{tag}_se8": s8, f"cv_{tag}_se20": s20,
                    f"cv_{tag}_se": se, f"p_{tag}": p, f"p_{tag}_se": sep,
                    f"n_valid_{tag}": n, f"n_nan_{tag}": R - n,
                    f"reject_{tag}": stat_ < cv})
    fin = np.isfinite(nul["sieve"])
    size = float(np.mean(nul["sieve"][fin] <= CV_ASY))
    row.update(size_I=size, size_I_se=float(np.sqrt(size * (1 - size) / fin.sum())))
    if "iid" in nul:
        s8, s20, se = block_se(cv_blocks(nul["iid"]))
        row.update(cv_iid=cv5(nul["iid"]), cv_iid_se=se,
                   n_nan_iid=int(np.sum(~np.isfinite(nul["iid"]))))
    return row


NULL_COLS = ["currency", "cfg", "break_pos", "m", "p_hat", "E", "status", "R", "mzt_I",
             "reject_I", "size_I", "size_I_se", "cv_II", "cv_II_se8", "cv_II_se20",
             "cv_II_se", "p_II", "p_II_se", "n_valid_II", "n_nan_II", "reject_II",
             "cbar_half", "mzt_III", "cv_III", "cv_III_se8", "cv_III_se20", "cv_III_se",
             "p_III", "p_III_se", "n_valid_III", "n_nan_III", "reject_III", "cv_iid",
             "cv_iid_se", "n_nan_iid"]
SIGN_DRAWS = 2000


def stage_null(D, store, out_dir, scale):
    rows, arrays = [], {}
    for c, bp in A.CELLS:
        t0 = time.time()
        nul = null_arrays(D, store, c, bp, scale)
        arrays[(c, bp)] = nul
        rows.append(null_row(D, c, bp, nul))
        print(f"  null {c} {cfg_label(bp)}: {time.time() - t0:.0f}s", flush=True)
    # sign check on the null draws: the first two E currencies with p_hat >= 1, their
    # first SIGN_DRAWS draws recomputed with phi_hat from -q
    M = Data(out_dir, D.synthetic, sign=-1.0)
    check = []
    n = min(SIGN_DRAWS, R_NULL // scale)
    for c in A.CANDIDATES:
        bp = D.conf[("E", c)]
        if D.cells[(c, bp)]["p"] < 1 or arrays[(c, bp)] is None or len(check) == 2:
            continue
        arms = (("u0zero", 0.0, tuple(M.cells[(c, bp)]["fit"]["phi"]), CBAR),)
        neg = simulate(bp, arms, A.seed0(f"null|{c}|{cfg_label(bp)}"), n,
                       store.jobs)[:, 0]
        pos = arrays[(c, bp)]["sieve"][:n]
        if not np.array_equal(np.isnan(neg), np.isnan(pos)):
            raise SystemExit(f"null sign check {c}: NaN pattern differs")
        d = float(np.nanmax(np.abs(neg - pos)))
        if d > 1e-9:
            raise SystemExit(f"null sign check {c}: max difference {d}")
        check.append(dict(currency=c, cfg=cfg_label(bp), draws=n, max_diff=d))
    write_csv(os.path.join(out_dir, "ppp_null.csv"), NULL_COLS, rows)
    write_json(os.path.join(out_dir, "ppp_null_checks.json"),
               dict(sign_check=check
                    or "not applicable (no E currency with p_hat >= 1)",
                    note="the first two E currencies with p_hat >= 1"))
    return rows


# ================================================================== stage: bb
def _shrink(center, point, bound):
    """The first s in (0, 1] along center + s (point - center) at which max|root|
    reaches bound (a scan in steps of 0.001, then bisection to 1e-12); 1 if the point
    is within the bound and no earlier exit occurs."""
    def mr(s):
        return S.companion_max_root(center + s * (point - center))

    prev = 0.0
    for i in range(1, 1001):
        s = i / 1000
        if mr(s) > bound:
            lo, hi = prev, s
            while hi - lo > 1e-12:
                mid = (lo + hi) / 2
                if mr(mid) <= bound:
                    lo = mid
                else:
                    hi = mid
            return lo
        prev = s
    return 1.0


def bb_points(phi, V):
    """[(k, raw point, shrink factor, point)]; k = 0 is the center. Empty for p = 0."""
    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    if p == 0:
        return []
    raw = [phi.copy()]
    if p == 1:
        half = stats.norm.ppf(0.5 + BB_LEVEL / 2) * np.sqrt(V[0, 0])
        for t in (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0):
            raw.append(phi + t * half)
    elif p == 2:
        L = np.linalg.cholesky(V)
        r = np.sqrt(stats.chi2.ppf(BB_LEVEL, 2))
        for k in range(16):
            th = 2 * np.pi * k / 16
            raw.append(phi + r * (L @ np.array([np.cos(th), np.sin(th)])))
    else:
        lam, vec = np.linalg.eigh(V)
        order = np.argsort(-lam, kind="stable")
        r2 = stats.chi2.ppf(BB_LEVEL, p)
        for j in order:
            v = vec[:, j]
            v = v * (1.0 if v[np.argmax(np.abs(v))] > 0 else -1.0)
            d = np.sqrt(r2 * lam[j]) * v
            raw.append(phi + d)
            raw.append(phi - d)
    bound = max(BB_ROOT_BOUND, S.companion_max_root(phi))
    out = []
    for k, pt in enumerate(raw):
        s = 1.0 if k == 0 else _shrink(phi, pt, bound)
        out.append((k, pt, s, phi + s * (pt - phi)))
    return out


def stage_bb(D, store, out_dir, scale, nulls):
    rows, summ = [], []
    R = R_BB // scale
    for c in A.CANDIDATES:
        bp = D.conf[("E", c)]
        x = D.cells[(c, bp)]
        f = x["fit"]
        base = dict(currency=c, cfg=cfg_label(bp), p_hat=x["p"], mzt_obs=x["mzt_m7"])
        if x["p"] == 0:
            summ.append(dict(base, status="not applicable (p_hat = 0)"))
            continue
        if not f["stationary"]:
            summ.append(dict(base, status="not simulated (non-stationary phi_hat)"))
            continue
        pts = bb_points(f["phi"], f["V"])
        best = None
        n_center = 0
        for k, raw, s, pt in pts:
            if k > 0 and s == 0.0:
                n_center += 1
                rows.append(dict(base, k=k, phi="= center",
                                 phi_raw="|".join(repr(float(v)) for v in raw),
                                 shrink=s, max_root=S.companion_max_root(pt)))
                continue
            it = f"bb|{c}|k{k}"
            arms = (("u0zero", 0.0, tuple(pt), CBAR),)
            null = store.get(it, ["sieve"], R, *sim_items(bp, arms, it))[:, 0]
            p, se, n = pvalue(null, x["mzt_m7"])
            rows.append(dict(base, k=k, phi="|".join(repr(float(v)) for v in pt),
                             phi_raw="|".join(repr(float(v)) for v in raw), shrink=s,
                             max_root=S.companion_max_root(pt), p=p, p_se=se,
                             n_valid=n))
            if best is None or p > best[1]:
                best = (k, p, se, pt)
        it = f"bbchk|{c}"
        arms = (("u0zero", 0.0, tuple(best[3]), CBAR),)
        chk = store.get(it, ["sieve"], R, *sim_items(bp, arms, it))[:, 0]
        pc, sec, _ = pvalue(chk, x["mzt_m7"])
        center = [r for r in rows if r["currency"] == c and r["k"] == 0][0]
        z = (center["p"] - nulls[c]["p_II"]) / np.sqrt(
            center["p_se"] ** 2 + nulls[c]["p_II_se"] ** 2)
        summ.append(dict(base, status="ok", n_points=len(pts) - n_center,
                         n_equal_center=n_center,
                         root_bound=max(BB_ROOT_BOUND, f["max_root"]),
                         argmax_k=best[0], p_max=best[1], p_max_se=best[2],
                         p_BB=best[1] + 0.01, p_check=pc, p_check_se=sec,
                         p_BB_check=pc + 0.01, center_vs_II_z=z))
        if abs(z) > 4.0:
            raise SystemExit(f"Berger-Boos {c}: the center is {z} se from p (II)")
    write_csv(os.path.join(out_dir, "ppp_berger_boos.csv"),
              ["currency", "cfg", "p_hat", "mzt_obs", "k", "phi", "phi_raw", "shrink",
               "max_root", "p", "p_se", "n_valid"], rows)
    write_csv(os.path.join(out_dir, "ppp_berger_boos_summary.csv"),
              ["currency", "cfg", "p_hat", "mzt_obs", "status", "n_points",
               "n_equal_center", "root_bound", "argmax_k", "p_max", "p_max_se", "p_BB",
               "p_check", "p_check_se", "p_BB_check", "center_vs_II_z"], summ)
    return summ


# ================================================================== stage: power
def stage_power(D, store, out_dir, scale, nulls):
    rows, band = [], []
    R = R_ALT // scale
    for c in A.CANDIDATES:
        bp = D.conf[("E", c)]
        x = D.cells[(c, bp)]
        phi = tuple(x["fit"]["phi"])
        nul = nulls.get(c)
        if nul is None:
            band.append(dict(currency=c, arm="main", status="not simulated "
                                                          "(non-stationary phi_hat)"))
            continue
        cvb = {k: cv_blocks(nul[k]) for k in ("sieve", "iid", "III")}
        cv = {k: cv5(nul[k]) for k in ("sieve", "iid", "III")}
        pw, pwb = {}, {}
        for h in H:
            b = x["band"][h]
            ok = b["guard"] == "ok"
            ch = S.c_band(h, (), T)
            it = f"alt|{c}|h{h}"
            if ok:
                arms = (("u0zero", b["c"], phi, CBAR), ("u0zero", ch, (), CBAR),
                        ("stationary", b["c"], phi, CBAR))
                names = ["main", "iid", "stationary"]
            else:
                arms, names = (("u0zero", ch, (), CBAR),), ["iid"]
                for arm in ("main", "stationary", "III"):
                    rows.append(dict(currency=c, h=h, arm=arm, c=b["c"], rho=b["rho"],
                                     guard=b["guard"],
                                     status="dropped (rho_{h,i} <= -1)"))
            alt = store.get(it, names, R, *sim_items(bp, arms, it))
            spec = []
            if ok:
                spec += [("main", alt[:, 0], "sieve", b["c"], b["rho"]),
                         ("iid", alt[:, 1], "iid", ch, 1 + ch / T),
                         ("stationary", alt[:, 2], "sieve", b["c"], b["rho"])]
                if x["p"] == 0 and not same(alt[:, 0], alt[:, 1]):
                    raise SystemExit(f"{c} h{h}: main and iid differ with p_hat = 0")
            else:
                spec.append(("iid", alt[:, 0], "iid", ch, 1 + ch / T))
            if x["p"] > 0:
                r_a = S.rho_irf(h, phi)
                if r_a is None:
                    rows.append(dict(currency=c, h=h, arm="irf", guard=b["guard"],
                                     status="no root"))
                else:
                    ca = T * (r_a - 1.0)
                    it_a = f"alt_irf|{c}|h{h}"
                    arms_a = (("u0zero", ca, phi, CBAR),)
                    alt_a = store.get(it_a, ["irf"], R, *sim_items(bp, arms_a, it_a))
                    spec.append(("irf", alt_a[:, 0], "sieve", ca, r_a))
                it_b = f"alt_unscaled|{c}|h{h}"
                arms_b = (("u0zero", ch, phi, CBAR),)
                alt_b = store.get(it_b, ["unscaled"], R, *sim_items(bp, arms_b, it_b))
                spec.append(("unscaled", alt_b[:, 0], "sieve", ch, 1 + ch / T))
            if ok:
                it3 = f"alt3|{c}|h{h}"
                arms3 = (("u0zero", b["c"], phi, x["cbar_half"]),)
                alt3 = store.get(it3, ["III"], R, *sim_items(bp, arms3, it3))
                spec.append(("III", alt3[:, 0], "III", b["c"], b["rho"]))
            for arm, arr, nk, cc, rr in spec:
                pwh = power(arr, cv[nk])
                pb = power_blocks(arr, cvb[nk])
                s8, s20, se = block_se(pb)
                pw.setdefault(arm, {})[h] = pwh
                pwb.setdefault(arm, {})[h] = pb
                rows.append(dict(currency=c, h=h, arm=arm, c=cc, rho=rr,
                                 guard=b["guard"], status="ok", power=pwh, se8=s8,
                                 se20=s20, se=se, n_nan=int(np.sum(~np.isfinite(arr))),
                                 rho_negative=int(rr < 0),
                                 power_below_size=int(pwh < 0.05 - 2 * se),
                                 irf_hl=(b["irf_hl"] if arm in ("main", "stationary",
                                                                "III") else "")))
            if x["p"] == 0:
                for arm in ("irf", "unscaled"):
                    rows.append(dict(currency=c, h=h, arm=arm, status="= main"))
        if "main" not in pw:
            band.append(dict(currency=c, arm="main", status="all h dropped "
                                                          "(rho_{h,i} <= -1)"))
        flagged = {(r["arm"], r["h"]) for r in rows
                   if r["currency"] == c and r.get("status") == "ok"
                   and (r.get("rho_negative") or r.get("power_below_size"))}
        for arm in pw:
            hs = sorted(pw[arm])
            m = float(np.mean([pw[arm][h] for h in hs]))
            mb = {B: np.mean([pwb[arm][h][B] for h in hs], axis=0) for B in BLOCKS}
            s8, s20, se = block_se(mb)
            fh = [h for h in hs if (arm, h) in flagged]
            band.append(dict(currency=c, arm=arm, status="ok",
                             h_used="|".join(map(str, hs)), band_mean=m, se8=s8,
                             se20=s20, se=se, n_flagged=len(fh),
                             flagged_h="|".join(map(str, fh))))
    write_csv(os.path.join(out_dir, "ppp_power.csv"),
              ["currency", "h", "arm", "c", "rho", "guard", "status", "power", "se8",
               "se20", "se", "n_nan", "rho_negative", "power_below_size", "irf_hl"],
              rows)
    write_csv(os.path.join(out_dir, "ppp_power_band.csv"),
              ["currency", "arm", "status", "h_used", "band_mean", "se8", "se20", "se",
               "n_flagged", "flagged_h"], band)
    return rows


# ================================================================== stage: joint
def k_stats(mzt, cvs, K_obs):
    rej = mzt < cvs[None, :]
    Kc = rej.sum(axis=1)
    N = mzt.shape[1]
    pmf = np.bincount(Kc, minlength=N + 1) / len(Kc)
    return dict(pmf=pmf, EK=float(Kc.mean()), P0=float(pmf[0]),
                Ple=float(pmf[:K_obs + 1].sum()), Pge1=float(pmf[1:].sum()),
                Pge=float(pmf[K_obs:].sum()), marg=rej.mean(axis=0))


K_KEYS = ("EK", "P0", "Ple", "Pge1", "Pge")


def k_blocks(mz, cvb_all, use, K_obs, R):
    """{key: {B: per-block values}} with block b's critical values."""
    out = {k: {} for k in K_KEYS}
    for B in BLOCKS:
        e = block_edges(R, B)
        vals = {k: [] for k in K_KEYS}
        for b in range(B):
            cvbb = np.array([cvb_all[i][B][b] for i in use])
            st = k_stats(mz[e[b]:e[b + 1]], cvbb, K_obs)
            for k in K_KEYS:
                vals[k].append(st[k])
        for k in K_KEYS:
            out[k][B] = np.array(vals[k])
    return out


def stage_joint(D, store, out_dir, scale, nulls, power_rows):
    R = R_JOINT // scale
    out, marg_rows = [], []
    for uname, univ in (("N8", D.N8), ("N9", D.N9)):
        cur, years, Rh = D.rhat(univ)
        cvs_all = np.array([cv5(nulls[c]["sieve"]) for c in cur])
        cvb_all = [cv_blocks(nulls[c]["sieve"]) for c in cur]
        rej_obs = {c: D.cell("E", c)["mzt_m7"] < cvs_all[i] for i, c in enumerate(cur)}
        for hk in ("H0",) + tuple(H):
            if hk == "H0":
                use = list(range(len(cur)))
            else:
                use = [i for i, c in enumerate(cur)
                       if D.cell("E", c)["band"][hk]["guard"] == "ok"]
            cu = [cur[i] for i in use]
            K_obs = int(sum(rej_obs[c] for c in cu))
            for kl, kv in KAPPA.items():
                Rk = (1 - kv) * np.eye(len(use)) + kv * Rh[np.ix_(use, use)]
                L = np.linalg.cholesky(Rk)
                bps = [D.conf[("E", c)] for c in cu]
                phis = [tuple(D.cell("E", c)["fit"]["phi"]) for c in cu]
                cs = ([0.0] * len(cu) if hk == "H0"
                      else [D.cell("E", c)["band"][hk]["c"] for c in cu])
                it = f"joint|{uname}|{'H0' if hk == 'H0' else f'h{hk}'}|k{kl}"
                arms = ["u0zero"] if hk == "H0" else ["u0zero", "stationary"]
                sizes = [len(p) + 1 for p in phis]
                if hk != "H0":
                    avec = [S.level_coeffs(1 + cc / T, p) for cc, p in zip(cs, phis)]
                    G, _ = stationary_cov_joint(avec, Rk)
                    Lst = np.linalg.cholesky(G)
                else:
                    Lst = np.zeros((1, 1))
                s0 = A.seed0(it)

                def sim(Rn, jobs, bps=bps, phis=phis, cs=cs, L=L, Lst=Lst,
                        sizes=sizes, hk=hk, s0=s0):
                    a = simulate_joint(bps, phis, cs, CBAR, L, "u0zero", Lst, sizes,
                                       s0, Rn, jobs)
                    if hk == "H0":
                        return a[:, :, None]
                    b = simulate_joint(bps, phis, cs, CBAR, L, "stationary", Lst,
                                       sizes, s0, Rn, jobs)
                    return np.stack([a, b], axis=2)

                spec = repr((uname, hk, kl, cu, [tuple(b) for b in bps], phis, cs,
                             Rk.tolist()))
                arr = store.get(it, arms, R, sim, spec)
                cvs = cvs_all[use]
                for ai, arm in enumerate(arms):
                    mz = arr[:, :, ai]
                    st = k_stats(mz, cvs, K_obs)
                    kb = k_blocks(mz, cvb_all, use, K_obs, R)
                    ses = {k: block_se(kb[k]) for k in K_KEYS}
                    row = dict(universe=uname, h=hk, kappa=kl, arm=arm, N=len(cu),
                               currencies="|".join(cu), K_obs=K_obs, R=R,
                               EK=st["EK"], EK_se=ses["EK"][2], P0=st["P0"],
                               P0_se=ses["P0"][2], P0_se_binom=np.sqrt(
                                   st["P0"] * (1 - st["P0"]) / R),
                               P_le_Kobs=st["Ple"], P_le_Kobs_se=ses["Ple"][2],
                               P_le_Kobs_se_binom=np.sqrt(st["Ple"] * (1 - st["Ple"])
                                                          / R),
                               P_ge1=st["Pge1"], P_ge1_se=ses["Pge1"][2],
                               P_ge_Kobs=st["Pge"], P_ge_Kobs_se=ses["Pge"][2],
                               pmf="|".join(repr(float(v)) for v in st["pmf"]),
                               n_nan=int(np.sum(~np.isfinite(mz))))
                    out.append(row)
                    for i, c in enumerate(cu):
                        marg_rows.append(dict(universe=uname, h=hk, kappa=kl, arm=arm,
                                              currency=c, rate=float(st["marg"][i])))
    check_joint(out, marg_rows, power_rows, R)
    write_csv(os.path.join(out_dir, "ppp_count_law.csv"),
              ["universe", "h", "kappa", "arm", "N", "currencies", "K_obs", "R", "EK",
               "EK_se", "P0", "P0_se", "P0_se_binom", "P_le_Kobs", "P_le_Kobs_se",
               "P_le_Kobs_se_binom", "P_ge1", "P_ge1_se", "P_ge_Kobs", "P_ge_Kobs_se",
               "pmf", "n_nan"], out)
    return out


def check_joint(out, marg_rows, power_rows, R):
    """The marginals of the joint draws against the band's powers (h) and 5% (H0): a
    z-test at a Bonferroni 0.1% family level; at kappa = 0 the law of K against the
    Poisson binomial of its own marginals (chi-square, p > 1e-4)."""
    pw = {(r["currency"], r["h"]): (r["power"], r["se"]) for r in power_rows
          if r.get("arm") == "main" and r.get("status") == "ok"}
    tests = [m for m in marg_rows if m["arm"] == "u0zero"]
    zc = stats.norm.ppf(1 - 0.001 / (2 * len(tests)))
    bad = []
    for m in tests:
        a = m["rate"]
        if m["h"] == "H0":
            b, seb = 0.05, np.sqrt(0.05 * 0.95 / (2 * R))
        else:
            b, seb = pw[(m["currency"], m["h"])]
        z = (a - b) / np.sqrt(a * (1 - a) / R + seb ** 2)
        if abs(z) > zc:
            bad.append((m["universe"], m["h"], m["kappa"], m["currency"], z))
    if bad:
        raise SystemExit(f"joint check: marginals {bad}")
    for r in out:
        if r["kappa"] != "0":
            continue
        rates = [m["rate"] for m in marg_rows if (m["universe"], m["h"], m["kappa"],
                                                  m["arm"]) == (r["universe"], r["h"],
                                                                "0", r["arm"])]
        ind = independence_pmf(rates)
        obs = np.array([float(v) for v in r["pmf"].split("|")]) * r["R"]
        exp = ind * r["R"]
        keep = exp >= 5
        chi = float(np.sum((obs[keep] - exp[keep]) ** 2 / exp[keep]))
        dof = int(keep.sum()) - 1
        p = float(stats.chi2.sf(chi, dof)) if dof > 0 else 1.0
        if p <= 1e-4:
            raise SystemExit(f"joint check: kappa = 0 law {r['universe']} {r['h']} "
                             f"{r['arm']}: chi2 {chi}, dof {dof}")


def independence_pmf(powers):
    pmf = np.array([1.0])
    for p in powers:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


# ================================================================== stage: sets
CELL_FIELDS = ("cfg", "p_hat", "phi", "mzt_I", "reject_I", "cv_II", "p_II", "p_II_se",
               "reject_II", "size_I", "size_I_se", "cbar_half", "mzt_III", "cv_III",
               "p_III", "p_III_se", "reject_III")
HL_FIELDS = ("alpha_mu", "mu_at_cap", "n_crossings", "hl_mu", "alpha_lo", "alpha_hi",
             "upper_at_cap", "upper_open", "ci_above_cap", "hl_lo", "hl_hi", "k_hi")


def stage_sets(D, out_dir):
    """One row per (set, currency): the design, the sieve, (I), (II), (III), the size of
    (I), cbar_half, the half-lives (recursive scheme), the collapse, whether the LB
    interval meets [3, 5], and the changes against E (only where both rows have the
    object)."""
    null = {(r["currency"], r["cfg"]): r for r in read_csv(
        os.path.join(out_dir, "ppp_null.csv"))}
    hl = {}
    for r in read_csv(os.path.join(out_dir, "hl_cells.csv")):
        if r["scheme"] == "rec":
            hl[(r["model"], r["currency"], r["cfg"], r["p"])] = r
    rows, seen = [], {}
    for c in A.CANDIDATES:
        e_row = None
        for s in A.SETS:
            bp = D.conf[(s, c)]
            base = dict(set=s, currency=c, tau="|".join(map(str, D.taus[(s, c)])),
                        break_pos="|".join(map(str, bp)))
            if not bp:
                rows.append(dict(base, status=f"no event under {s}"))
                continue
            x = D.cells[(c, bp)]
            n = null[(c, cfg_label(bp))]
            row = dict(base, status=n["status"], cfg=cfg_label(bp), p_hat=x["p"],
                       phi="|".join(repr(float(v)) for v in x["fit"]["phi"]))
            for k in CELL_FIELDS[3:]:
                row[k] = n.get(k, "")
            lb = hl.get(("LB", c, cfg_label(bp), str(x["p"])))
            mp = hl.get(("MP", c, "", str(x["p"])))
            for tag, r in (("LB", lb), ("MP", mp)):
                for k in HL_FIELDS:
                    row[f"{tag}_{k}"] = r[k] if r else ""
            if lb and mp:
                row["collapse"] = int(int(mp["k_hi"]) >= 200 and int(lb["k_hi"]) <= 199)
            if lb:
                row["LB_meets_3_5"] = int(float(lb["hl_lo"]) <= 5
                                          and float(lb["hl_hi"]) >= 3)
            key = (c, bp)
            vals = tuple(fmt(row.get(k, "")) for k in CELL_FIELDS + tuple(
                f"{t}_{k}" for t in ("LB", "MP") for k in HL_FIELDS))
            if key in seen and seen[key] != vals:
                raise SystemExit(f"sets: the rows of {key} differ between sets")
            seen[key] = vals
            if s == "E":
                e_row = row
            elif e_row is not None:
                for k in ("reject_II", "collapse", "LB_meets_3_5"):
                    a, b = str(row.get(k, "")), str(e_row.get(k, ""))
                    row[f"{k}_differs_from_E"] = int(a != b) if a and b else ""
            rows.append(row)
    cols = ["set", "currency", "tau", "break_pos", "status", "cfg", "p_hat", "phi",
            "mzt_I", "reject_I", "cv_II", "p_II", "p_II_se", "reject_II", "size_I",
            "size_I_se", "cbar_half", "mzt_III", "cv_III", "p_III", "p_III_se",
            "reject_III"] + [f"{t}_{k}" for t in ("LB", "MP") for k in HL_FIELDS] + [
                "collapse", "LB_meets_3_5", "reject_II_differs_from_E",
                "collapse_differs_from_E", "LB_meets_3_5_differs_from_E"]
    write_csv(os.path.join(out_dir, "ppp_sets.csv"), cols, rows)
    return rows


# ================================================================== stage: readings
def median_inf(v):
    """The median with the even-N rule; inf propagates when a middle value is inf."""
    v = sorted(float(x) for x in v)
    n = len(v)
    if n % 2:
        return v[n // 2]
    a, b = v[n // 2 - 1], v[n // 2]
    return float("inf") if (np.isinf(a) or np.isinf(b)) else (a + b) / 2


def holm(pvals):
    """{name: p} -> {name: Holm step-down adjusted p}."""
    items = sorted(pvals.items(), key=lambda kv: (kv[1], kv[0]))
    n = len(items)
    out, run = {}, 0.0
    for j, (k, p) in enumerate(items):
        run = max(run, min(1.0, (n - j) * p))
        out[k] = run
    return out


def hl_readings(cells, univ, scheme):
    """The half-life readings of one universe and scheme from hl_cells rows."""
    def pick(model):
        return {r["currency"]: r for r in cells
                if r["scheme"] == scheme and r["model"] == model
                and f"{model}@E" in r["specs"].split("|")}

    lb, mp = pick("LB"), pick("MP")
    f = float
    out = dict(median_hl_mu_LB=median_inf([lb[c]["hl_mu"] for c in univ]),
               median_hl_mu_MP=median_inf([mp[c]["hl_mu"] for c in univ]),
               LB_meet_3_5=sum(f(lb[c]["hl_lo"]) <= 5 and f(lb[c]["hl_hi"]) >= 3
                               for c in univ),
               LB_unbounded=sum(int(lb[c]["k_hi"]) >= 200 for c in univ),
               MP_unbounded=sum(int(mp[c]["k_hi"]) >= 200 for c in univ),
               collapse=sum(int(mp[c]["k_hi"]) >= 200 and int(lb[c]["k_hi"]) <= 199
                            for c in univ))
    return out


def stage_readings(D, out_dir):
    rows = []

    def add(universe, reading, value, se="", h="", kappa="", arm="", scheme="",
            K_obs="", N="", currencies=""):
        rows.append(dict(universe=universe, reading=reading, h=h, kappa=kappa, arm=arm,
                         scheme=scheme, value=value, se=se, K_obs=K_obs, N=N,
                         currencies=currencies))

    null = {r["currency"]: r for r in read_csv(os.path.join(out_dir, "ppp_null.csv"))
            if r["E"] == "1"}
    law = read_csv(os.path.join(out_dir, "ppp_count_law.csv"))
    cells = read_csv(os.path.join(out_dir, "hl_cells.csv"))
    for uname, univ in (("N8", D.N8), ("N9", D.N9)):
        ok = [c for c in univ if null[c]["status"] == "ok"]
        rej = [c for c in ok if null[c]["reject_II"] == "1"]
        add(uname, "K_obs", len(rej), N=len(ok), currencies="|".join(ok))
        pv = {c: float(null[c]["p_II"]) for c in ok}
        hb = holm(pv)
        for c in ok:
            add(uname, "p_II_bonferroni", min(1.0, len(ok) * pv[c]), arm=c)
            add(uname, "p_II_holm", hb[c], arm=c)
        for r in law:
            if r["universe"] != uname or r["h"] != "H0":
                continue
            kw = dict(h=r["h"], kappa=r["kappa"], arm=r["arm"], K_obs=int(r["K_obs"]),
                      N=int(r["N"]), currencies=r["currencies"])
            add(uname, "P_ge1_H0", float(r["P_ge1"]), float(r["P_ge1_se"]), **kw)
        for scheme in A.SCHEMES:
            for k, v in hl_readings(cells, univ, scheme).items():
                add(uname, k, v, scheme=scheme)
    write_csv(os.path.join(out_dir, "readings.csv"),
              ["universe", "reading", "h", "kappa", "arm", "scheme", "value", "se",
               "K_obs", "N", "currencies"], rows)
    return rows


# ================================================================== main
def load_nulls(D, store, scale):
    """The E cells' null arrays (II), iid and (III), read from the store."""
    return {c: null_arrays(D, store, c, D.conf[("E", c)], scale, consume=False)
            for c in A.CANDIDATES}


def check_ledgers(work, store):
    """Every item consumed by every stage run so far (the union of the ledgers):
    listed in the registry, disjoint, above tab:cv, and read only by whitelisted
    arms."""
    led = [tuple(x) for x in store.ledger]
    if work:
        import glob
        with open(os.path.join(work, f"ledger_{store.stage}.json"), "w",
                  encoding="utf-8", newline="\n") as fh:
            json.dump(store.ledger, fh)
        led = []
        for f in sorted(glob.glob(os.path.join(work, "ledger_*.json"))):
            led += [tuple(x) for x in json.load(open(f, encoding="utf-8"))]
    if led:
        A.assert_disjoint({it: A.item_R(it) for it, _, _ in led})
        A.assert_consumers(led)
    return len(led)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["design", "data", "null", "bb", "power", "joint", "sets",
                             "readings"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work", default=None, help="the item store (npz per item)")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    scale = 1
    for R in (R_NULL, R_BB, R_ALT, R_JOINT):
        assert R % scale == 0 and (R // scale) % 8 == 0 and (R // scale) % 20 == 0
    t0 = time.time()
    if a.stage == "design":
        stage_design(a.out_dir, a.jobs)
        finalize()
        return 0
    if a.stage in ("null", "bb", "power", "joint") and not a.work:
        raise SystemExit(f"--stage {a.stage} needs --work (the item store)")
    D = Data(a.out_dir, False)
    store = Store(a.work, fingerprint(a.out_dir, D), a.jobs, a.stage)
    if a.stage == "data":
        stage_data(D, a.out_dir)
        print(f"sign invariance: max difference {check_sign(D, a.out_dir)}")
    elif a.stage == "null":
        stage_null(D, store, a.out_dir, scale)
    elif a.stage == "bb":
        nr = {r["currency"]: dict(p_II=float(r["p_II"]), p_II_se=float(r["p_II_se"]))
              for r in read_csv(os.path.join(a.out_dir, "ppp_null.csv"))
              if r["E"] == "1" and r["status"] == "ok"}
        stage_bb(D, store, a.out_dir, scale, nr)
    elif a.stage == "power":
        stage_power(D, store, a.out_dir, scale, load_nulls(D, store, scale))
    elif a.stage == "joint":
        pr = [dict(currency=r["currency"], h=int(r["h"]), arm="main", status="ok",
                   power=float(r["power"]), se=float(r["se"]))
              for r in read_csv(os.path.join(a.out_dir, "ppp_power.csv"))
              if r["arm"] == "main" and r["status"] == "ok"]
        stage_joint(D, store, a.out_dir, scale, load_nulls(D, store, scale), pr)
    elif a.stage == "sets":
        stage_sets(D, a.out_dir)
    elif a.stage == "readings":
        stage_readings(D, a.out_dir)
    n = check_ledgers(a.work, store)
    finalize()
    print(f"stage {a.stage}: {time.time() - t0:.1f}s; {n} consumers in the ledgers",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
