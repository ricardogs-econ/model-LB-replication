"""Lemma 1 of the article (the Gram matrix of the quasi-differenced level regressors),
checked in exact rational arithmetic on 3,004 designs (proof in supplement S2).

A design is (T, tau_1 < ... < tau_m, cbar) with 1 <= tau_1 and tau_m <= T - 1. The
regressors are the constant and the level dummies DU_{j,t} = 1{t > tau_j},
t = 1, ..., T (tau_0 := 0 gives the constant), and quasi-differencing at
abar = 1 + cbar/T keeps the first row (supplement S1.1, N2): x^abar_1 = x_1 and
x^abar_t = x_t - abar x_{t-1} for t >= 2. Z^abar has the columns z_0, ..., z_m,
G_T = Z^abar' Z^abar and D = G_T - I. Nothing is rounded: with cbar = p/q in lowest
terms and s = T q, the columns s z_j are built from the dummies by the recursion
s x_t - (s + p) x_{t-1} in integers, G_T = (s Z^abar)'(s Z^abar) / s^2 is a
fractions.Fraction, and so is every quantity derived from it. Each design is checked
for:

  qd_step     eq. (qd-step): z_{j,t} = 1{t = tau_j + 1} - (cbar/T) 1{t > tau_j + 1};
  gram_exact  eq. (Gram-exact): (G_T)_jj = 1 + cbar^2 (T - tau_j - 1)/T^2 and
              (G_T)_jk = -cbar/T + cbar^2 (T - tau_max(j,k) - 1)/T^2 for j != k;
  expansion   eq. (Gram-qd): for lambda_j = (tau_j + f)/T with f in {0, 1/2,
              999/1000} (so that floor(lambda_j T) = tau_j) and lambda_0 = 0, every
              entry of G_T - I - B(cbar, lambda)/T lies in [-cbar^2/T^2, 0], and
              |B_jk| <= |cbar| + cbar^2;
  a_diagonal  proof step (a): D_jj in [0, cbar^2/T) for every j, and D_00 > 0;
  b_offdiag   step (b): D_jk in [|cbar|/T, (|cbar| + cbar^2)/T) for j != k;
  c_trace     step (c): 0 < tr D < (m + 1) cbar^2/T;
  d_upper     step (d): ||D||_F <= (m + 1) max_jk |D_jk| <= (m + 1)(|cbar| + cbar^2)/T
              and ||D|| <= (m + 1)(|cbar| + cbar^2)/T;
  e_lower     step (e): v'Dv >= m |cbar|/T with v = (m + 1)^(-1/2) (1, ..., 1)',
              e_0'D e_0 = cbar^2 (T - 1)/T^2, and
              ||D|| >= max{m |cbar|, cbar^2 (1 - 1/T)}/T.

||.|| is the spectral norm, and its bounds are decided exactly, without computing an
eigenvalue: for a symmetric D and a rational x > 0, ||D|| < x if and only if x I - D
and x I + D are both positive definite, and a symmetric matrix is positive definite
if and only if every pivot of Gaussian elimination without pivoting is positive. The
upper bound is verified as ||D|| < x (which implies the stated <=), and the lower
bound as "x I - D or x I + D is not positive definite", which is ||D|| >= x. No
floating point enters.

The designs: 4 fixed edge cases (T = 2 with one break at T - 1; T = 6 with a break at
every date; T = 60 with breaks at the first and the last admissible date; T = 2 with
the constant only) and 3,000 random draws with T uniform on {2, ..., 60}, m uniform on
{0, ..., min(5, T - 1)}, m distinct dates uniform on {1, ..., T - 1}, and cbar uniform
on {-3, -7, -20, -1/3, -13/2}. The draws use only the random() method of
random.Random(SEED), whose sequence Python keeps fixed for a given seed, and the
sha256 of the list of designs is part of the output.

Output (output/supplement_appendix/lemma1_bounds/, or --out-dir): lemma1_bounds.json
(the designs' description and sha256, the number of designs passing each check and
the failures) and lemma1_bounds.manifest.json (the parameters, the seed and the
output's sha256; nothing time- or machine-dependent). --check compares the JSON with
reference/theory/lemma1_bounds/ byte for byte (LF). The exit code is
non-zero on any failed check or difference. About 15 seconds.

    python mlb/lemma1_bounds.py [--out-dir D] [--check]
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from fractions import Fraction

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlb import paths  # noqa: E402

SECTION, OBJ = "supplement_appendix", "lemma1_bounds"
SEED = 20261011
N_RANDOM = 3000
T_MIN, T_MAX, M_MAX = 2, 60, 5
CBARS = (Fraction(-3), Fraction(-7), Fraction(-20), Fraction(-1, 3), Fraction(-13, 2))
FIXED = (
    (2, (1,), Fraction(-7)),
    (6, (1, 2, 3, 4, 5), Fraction(-20)),
    (60, (1, 59), Fraction(-3)),
    (2, (), Fraction(-13, 2)),
)
SHIFTS = (Fraction(0), Fraction(1, 2), Fraction(999, 1000))
CHECKS = (
    "qd_step",
    "gram_exact",
    "expansion",
    "a_diagonal",
    "b_offdiag",
    "c_trace",
    "d_upper",
    "e_lower",
)
DESCRIPTION = dict(
    qd_step="z_jt = 1{t = tau_j + 1} - (cbar/T) 1{t > tau_j + 1} (eq. qd-step)",
    gram_exact="(G_T)_jj = 1 + cbar^2 (T - tau_j - 1)/T^2, (G_T)_jk = -cbar/T + "
    "cbar^2 (T - tau_max(j,k) - 1)/T^2 (eq. Gram-exact)",
    expansion="G_T - I - B/T in [-cbar^2/T^2, 0] entrywise and |B_jk| <= |cbar| + "
    "cbar^2, lambda_j T - tau_j in {0, 1/2, 999/1000} (eq. Gram-qd)",
    a_diagonal="D_jj in [0, cbar^2/T), D_00 > 0 (proof step a)",
    b_offdiag="D_jk in [|cbar|/T, (|cbar| + cbar^2)/T), j != k (proof step b)",
    c_trace="0 < tr D < (m + 1) cbar^2/T (proof step c)",
    d_upper="||D||_F <= (m + 1) max|D_jk| <= (m + 1)(|cbar| + cbar^2)/T and "
    "||D|| <= (m + 1)(|cbar| + cbar^2)/T (proof step d)",
    e_lower="v'Dv >= m|cbar|/T, e_0'De_0 = cbar^2 (T - 1)/T^2 and ||D|| >= "
    "max{m|cbar|, cbar^2 (1 - 1/T)}/T (proof step e)",
)


# ------------------------------------------------------------------- the designs
def draw_designs(seed=SEED, n_random=N_RANDOM):
    """The 4 fixed designs, then n_random draws; each is (T, taus, cbar)."""
    u = random.Random(seed).random
    out = list(FIXED)
    for _ in range(n_random):
        T = T_MIN + int((T_MAX - T_MIN + 1) * u())
        m = int((min(M_MAX, T - 1) + 1) * u())
        dates = set()
        while len(dates) < m:
            dates.add(1 + int((T - 1) * u()))
        cbar = CBARS[int(len(CBARS) * u())]
        out.append((T, tuple(sorted(dates)), cbar))
    return out


def design_text(designs):
    """One line per design, 'T;tau_1 ... tau_m;cbar', the text that is hashed."""
    return "".join(
        f"{T};{' '.join(str(t) for t in taus)};{cbar}\n" for T, taus, cbar in designs
    )


# ------------------------------------------------------------ exact linear algebra
def scaled_columns(T, taus, cbar):
    """(s, [s z_0, ..., s z_m]): the columns of Z^abar times s = T q (cbar = p/q),
    integers, built from the dummies by the quasi-difference recursion (first row
    kept): s z_t = s x_t - (s + p) x_{t-1}, since s abar = s + p."""
    p, q = cbar.numerator, cbar.denominator
    s = T * q
    cols = []
    for tau in (0,) + tuple(taus):
        x = [int(t > tau) for t in range(1, T + 1)]
        cols.append([s * x[0]] + [s * x[t] - (s + p) * x[t - 1] for t in range(1, T)])
    return s, cols


def gram_exact(T, taus, cbar):
    """G_T = Z^abar' Z^abar, exactly (Fractions), from the integer columns."""
    s, cols = scaled_columns(T, taus, cbar)
    n = len(cols)
    g = [[0] * n for _ in range(n)]
    for j in range(n):
        for k in range(j, n):
            g[j][k] = g[k][j] = sum(a * b for a, b in zip(cols[j], cols[k]))
    return [[Fraction(g[j][k], s * s) for k in range(n)] for j in range(n)]


def positive_definite(A):
    """True iff the symmetric rational matrix A is positive definite (all pivots of
    Gaussian elimination without pivoting are positive)."""
    M = [list(row) for row in A]
    n = len(M)
    for k in range(n):
        p = M[k][k]
        if p <= 0:
            return False
        for i in range(k + 1, n):
            f = M[i][k] / p
            if f:
                for j in range(k + 1, n):
                    M[i][j] -= f * M[k][j]
    return True


def norm_below(D, x):
    """||D|| < x, exactly, for symmetric rational D and rational x > 0."""
    n = len(D)
    plus = [[(x if j == k else 0) - D[j][k] for k in range(n)] for j in range(n)]
    minus = [[(x if j == k else 0) + D[j][k] for k in range(n)] for j in range(n)]
    return positive_definite(plus) and positive_definite(minus)


# ------------------------------------------------------------------- the checks
def check_design(T, taus, cbar):
    """{check: bool} for one design."""
    m, n, c, k = len(taus), len(taus) + 1, cbar, abs(cbar)
    tt = (0,) + tuple(taus)
    s, cols = scaled_columns(T, taus, c)
    ok = dict.fromkeys(CHECKS, True)

    # s z_jt = s 1{t = tau_j + 1} - p 1{t > tau_j + 1}, as s (-cbar/T) = -p
    for j in range(n):
        want = [0] * tt[j] + [s] + [-c.numerator] * (T - tt[j] - 1)
        ok["qd_step"] &= cols[j] == want

    G = gram_exact(T, taus, c)
    D = [[G[j][q] - (1 if j == q else 0) for q in range(n)] for j in range(n)]
    for j in range(n):
        for q in range(n):
            if j == q:
                want = 1 + c * c * (T - tt[j] - 1) / Fraction(T * T)
            else:
                want = -c / T + c * c * (T - tt[max(j, q)] - 1) / Fraction(T * T)
            ok["gram_exact"] &= G[j][q] == want

    for f in SHIFTS:
        lam = [Fraction(0)] + [(tau + f) / T for tau in taus]
        ok["expansion"] &= all(math.floor(lam[j] * T) == tt[j] for j in range(n))
        for j in range(n):
            for q in range(n):
                if j == q:
                    B = c * c * (1 - lam[j])
                else:
                    B = -c + c * c * (1 - lam[max(j, q)])
                rem = D[j][q] - B / T
                ok["expansion"] &= -c * c / (T * T) <= rem <= 0
                ok["expansion"] &= abs(B) <= k + c * c

    for j in range(n):
        ok["a_diagonal"] &= 0 <= D[j][j] < c * c / T
    ok["a_diagonal"] &= D[0][0] > 0
    for j in range(n):
        for q in range(n):
            if j != q:
                ok["b_offdiag"] &= k / T <= D[j][q] < (k + c * c) / T
    tr = sum(D[j][j] for j in range(n))
    ok["c_trace"] = 0 < tr < (m + 1) * c * c / T

    upper = (m + 1) * (k + c * c) / T
    biggest = max(abs(D[j][q]) for j in range(n) for q in range(n))
    frob2 = sum(D[j][q] ** 2 for j in range(n) for q in range(n))
    ok["d_upper"] = (
        frob2 <= ((m + 1) * biggest) ** 2
        and (m + 1) * biggest <= upper
        and norm_below(D, upper)
    )

    lower = max(m * k, c * c * (1 - Fraction(1, T))) / T
    vdv = sum(D[j][q] for j in range(n) for q in range(n)) / (m + 1)
    ok["e_lower"] = (
        vdv >= m * k / T
        and D[0][0] == c * c * (T - 1) / Fraction(T * T)
        and not norm_below(D, lower)
    )
    return ok


def run(designs):
    """The summary of the checks over the designs (a dict that is the JSON)."""
    passed = dict.fromkeys(CHECKS, 0)
    failures = []
    for i, (T, taus, cbar) in enumerate(designs):
        ok = check_design(T, taus, cbar)
        for name in CHECKS:
            passed[name] += int(ok[name])
        if not all(ok.values()):
            failures.append(
                dict(
                    index=i,
                    T=T,
                    tau=list(taus),
                    cbar=str(cbar),
                    failed=[name for name in CHECKS if not ok[name]],
                )
            )

    count_m, count_c = {}, {}
    for T, taus, cbar in designs:
        count_m[str(len(taus))] = count_m.get(str(len(taus)), 0) + 1
        count_c[str(cbar)] = count_c.get(str(cbar), 0) + 1
    text = design_text(designs)
    return dict(
        statement="Lemma 1 (lem:design): eqs. qd-step, Gram-exact, Gram-qd and "
        "Gram-bounds, with proof steps (a)-(e) of supplement S2",
        arithmetic="exact rational (fractions.Fraction); spectral-norm bounds "
        "decided by positive definiteness of x I -/+ D",
        designs=dict(
            n=len(designs),
            n_fixed=len(FIXED),
            n_random=len(designs) - len(FIXED),
            seed=SEED,
            generator="random.Random(seed).random()",
            T_range=[T_MIN, T_MAX],
            m_max=M_MAX,
            cbar_values=[str(c) for c in CBARS],
            fixed=[
                dict(T=T, tau=list(taus), cbar=str(cbar)) for T, taus, cbar in FIXED
            ],
            count_by_m=dict(sorted(count_m.items())),
            count_by_cbar={str(c): count_c.get(str(c), 0) for c in CBARS},
            sha256_of_list=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            list_format="one line per design: T;tau_1 ... tau_m;cbar",
        ),
        checks=DESCRIPTION,
        expansion_shifts=[str(f) for f in SHIFTS],
        passed=passed,
        failures=failures,
        all_designs_pass=not failures,
    )


# ------------------------------------------------------------------ files and check
def _dump(obj):
    return (json.dumps(obj, indent=1) + "\n").encode("utf-8")


RTOL = 1e-9


def _same_json(x, y):
    """Structure, strings and integers equal; floats to a relative RTOL (a root or an
    input regenerated on another platform moves in its last digits)."""
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same_json(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(map(_same_json, x, y))
    if isinstance(x, float) or isinstance(y, float):
        ok = isinstance(x, (int, float)) and isinstance(y, (int, float))
        return ok and abs(x - y) <= RTOL * max(1.0, abs(x), abs(y))
    return type(x) is type(y) and x == y


def _lf(data):
    return data.replace(b"\r\n", b"\n")


def manifest(output_bytes):
    return dict(
        object=OBJ,
        command="python mlb/lemma1_bounds.py",
        command_note="--out-dir does not change the outputs",
        outputs_sha256={f"{OBJ}.json": hashlib.sha256(_lf(output_bytes)).hexdigest()},
        inputs_sha256_lf={},
        seeds=dict(designs=SEED),
        parameters=dict(
            n_fixed=len(FIXED),
            n_random=N_RANDOM,
            T_range=[T_MIN, T_MAX],
            m_max=M_MAX,
            cbar_values=[str(c) for c in CBARS],
            expansion_shifts=[str(f) for f in SHIFTS],
        ),
        check=dict(mode="LF bytes"),
    )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default=None, help="output folder")
    ap.add_argument(
        "--check", action="store_true", help="compare the output with the reference"
    )
    a = ap.parse_args(argv)
    out = a.out_dir or str(paths.out_dir(SECTION) / OBJ)
    os.makedirs(out, exist_ok=True)
    print(f"[{OBJ}] {len(FIXED) + N_RANDOM} designs in exact arithmetic, about 15 s",
          flush=True)
    t0 = time.time()
    res = run(draw_designs())
    data = _dump(res)
    with open(os.path.join(out, f"{OBJ}.json"), "wb") as fh:
        fh.write(data)
    with open(os.path.join(out, f"{OBJ}.manifest.json"), "wb") as fh:
        fh.write(_dump(manifest(data)))
    print(f"designs: {res['designs']['n']} ({res['designs']['n_fixed']} fixed, "
          f"{res['designs']['n_random']} random, seed {SEED})")
    for name in CHECKS:
        print(f"  {name:<11} {res['passed'][name]} of {res['designs']['n']}")
    print(f"[{OBJ}] all designs pass: {res['all_designs_pass']}; wrote {out} "
          f"in {time.time() - t0:.0f} s")
    status = 0 if res["all_designs_pass"] else 1
    if not a.check:
        return status
    ref = paths.reference_dir(SECTION) / OBJ / f"{OBJ}.json"
    if not ref.is_file():
        print(f"--check: MISSING {ref}")
        return 1
    raw = _lf(ref.read_bytes())
    if raw == _lf(data):
        word = "identical"
    elif _same_json(json.loads(raw), json.loads(data)):
        word = "equal within 1e-9"
    else:
        word = "DIFFERENT"
    print(f"--check: {OBJ}.json: {word}")
    ok = word in ("identical", "equal within 1e-9") and status == 0
    print(f"--check: {'PASS' if ok else 'FAIL'} against {ref.parent}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
