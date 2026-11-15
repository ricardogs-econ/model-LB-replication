"""The CKP response surface for c-bar: the -21.7 of Section 4.4 and Figure 2.

SOURCE. Carrion-i-Silvestre, J.L., Kim, D. and Perron, P. (2009), "GLS-based
unit root tests with multiple structural breaks under both the null and the
alternative hypotheses", Econometric Theory 25(6), 1754-1792; and the GAUSS
code distributed by the authors (procedure `c_bar_rs` of `msbur.src`, in
"GLS msbur GAUSS code 2009", the item published at
sites.google.com/view/carrion-i-silvestre/code-and-data). Only the 61
coefficients and the order of the regressors are reproduced here, with this
attribution; the GAUSS file itself is not part of this repository.

WHAT IT COMPUTES. c_bar_rs(lam5) is the authors' response surface of the
detrending parameter c-bar in the vector lam5 of five break fractions:

    xreg = 1 ~ lam_i ~ lam_i^2 ~ lam_i^3 ~ lam_i^4            (i = 1..5)
             ~ |lam_i - lam_j| ~ |.|^2 ~ |.|^3 ~ |.|^4       (i < j, 10 pairs)
    c_bar = xreg * param                                    (61 terms)

in the order of msbur.src (the pairs run (1,2),(1,3),(1,4),(1,5),(2,3),...,
(4,5) within each power). The surface always receives FIVE fractions: the
authors' caller pads the unused ones with zeros (msbur.src l.186,
`c_bar_rs((min_tb/nt)|zeros(5-nbrk,1))`), and cbar_ckp(break_pos, T) does the
same. Evaluated this way, the 61 coefficients reproduce all 45 simulated values
of CKP (ET 2009, Table 1, p.1761) to the one decimal printed (supplement S4.2,
ssub:trend; the functional form is eq. (9), p. 8, of the 2007 working paper of
Carrion-i-Silvestre, Kim and Perron, with lambda_i := 0 for i > m); --table1
prints the 45 evaluations.

At the cell of Figure 2, T = 60, m = 2, lambda = (1/3, 2/3):
    cbar_ckp([20, 40], 60) = -21.668863567901237  ->  -21.7 at one decimal.

Pure Python; no kernel, no RNG.
"""
from __future__ import annotations

# The 61 coefficients of c_bar_rs, in msbur.src order (param, l.685-690).
CBAR_PARAM = (
    -13.12832, -36.53045, 0.0, 20.2423, -4.596202, -10.31678, 115.2092, -29.18712,
    -68.36453, 5.873121, 0.0, -130.337, 74.64396, 85.48737, 0.0, 0.0, 51.98117,
    -53.03452, -36.27221, 0.0, 11.27727, -23.39517, -5.360149, 23.99683, 4.788676,
    -27.10002, -35.78388, 51.12371, -29.8518, -3.069174, -37.45898, 64.95842,
    5.825729, -88.78176, -11.54197, 83.48645, 125.2349, -173.1259, 80.95821,
    2.863782, 118.2829, -80.1287, 0.0, 128.872, 6.387147, -118.1043, -199.0615,
    247.6469, -98.05947, 0.0, -160.5713, 38.52177, 0.0, -65.21576, 0.0, 62.86494,
    117.9976, -127.5544, 46.2304, 0.0, 79.1693,
)
PAIRS = tuple((i, j) for i in range(5) for j in range(i + 1, 5))


def xreg(lam5):
    """The 61 regressor values at the five break fractions lam5."""
    lam = [float(x) for x in lam5]
    if len(lam) != 5:
        raise ValueError("c_bar_rs takes exactly five break fractions (pad with zeros)")
    x = [1.0]
    for p in (1, 2, 3, 4):
        x += [v ** p for v in lam]
    for p in (1, 2, 3, 4):
        x += [abs(lam[i] - lam[j]) ** p for i, j in PAIRS]
    return x


def c_bar_rs(lam5):
    """CKP (2009) response surface for c-bar at five break fractions."""
    x = xreg(lam5)
    assert len(x) == len(CBAR_PARAM) == 61
    s = 0.0
    for a, b in zip(x, CBAR_PARAM):
        s += a * b
    return s


def cbar_ckp(break_pos, T):
    """c_bar_rs at the break dates break_pos of a sample of T, zero-padded to 5.

    As msbur.src l.186: lam = break date / T for each break, in the order
    given (l.186 does not sort; the authors' search routines sort before they
    call c_bar_rs, and the surface is not symmetric in the order), then zeros.
    """
    bp = list(break_pos)
    if any(int(b) != b for b in bp):
        raise ValueError("break positions must be integers")
    if not 1 <= len(bp) <= 5:
        # msbur.src does not use the surface without breaks (it uses fixed
        # values there, l.162/l.165), and the surface covers at most five
        raise ValueError("the CKP surface takes one to five breaks")
    return c_bar_rs([int(b) / T for b in bp] + [0.0] * (5 - len(bp)))


def table1_grid():
    """c_bar_rs on the break-fraction lattice of CKP (2009), Table 1: 45 cells.

    m = 1: lambda in {0.1, ..., 0.9} (9 cells); m = 2: every pair
    lambda_1 < lambda_2 on the same 0.1 lattice (36 cells, lexicographic).
    Each cell is the zero-padded evaluation of msbur.src l.186, i.e.
    c_bar_rs([lambdas..., 0, ..., 0]). Pure: no I/O, no RNG. Returns a list of
    dicts (m, lambdas, cbar at full precision, cbar_1dp = round(cbar, 1)),
    the m = 1 cells first.
    """
    lattice = [k / 10 for k in range(1, 10)]
    cells = [(a,) for a in lattice] + [(a, b) for i, a in enumerate(lattice) for b in lattice[i + 1:]]
    out = []
    for lam in cells:
        v = c_bar_rs(list(lam) + [0.0] * (5 - len(lam)))
        out.append(dict(m=len(lam), lambdas=list(lam), cbar=v, cbar_1dp=round(v, 1)))
    return out


def format_table1(cells):
    """The 45 cells at one decimal: m = 1 as one row; m = 2 as the triangle
    with lambda_1 in the rows and lambda_2 in the columns."""
    lattice = [k / 10 for k in range(1, 10)]
    one = {c["lambdas"][0]: c for c in cells if c["m"] == 1}
    two = {tuple(c["lambdas"]): c for c in cells if c["m"] == 2}
    head = "  " + " " * 10 + "".join(f"{b:>7.1f}" for b in lattice)
    lines = ["c_bar_rs (CKP 2009 response surface), zero-padded as msbur.src l.186; one decimal",
             "", "m = 1", head,
             "  " + f"{'c-bar':<10}" + "".join(f"{one[a]['cbar_1dp']:>7.1f}" for a in lattice),
             "", "m = 2  (rows lambda_1, columns lambda_2)", head]
    for a in lattice[:-1]:
        lines.append("  " + f"{a:<10.1f}" + "".join(
            f"{two[(a, b)]['cbar_1dp']:>7.1f}" if b > a else " " * 7 for b in lattice))
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="The CKP (2009) response surface for c-bar.")
    ap.add_argument("--table1", action="store_true",
                    help="print the 45 cells of table1_grid() at one decimal")
    ap.add_argument("--table1-json", default=None,
                    help="also write the 45 cells, full precision and one decimal, to this JSON file")
    a = ap.parse_args()
    if a.table1 or a.table1_json:
        grid = table1_grid()
        print(format_table1(grid))
        if a.table1_json:
            with open(a.table1_json, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(dict(function="ckp_cbar_surface.table1_grid", n_cells=len(grid),
                               evaluation="c_bar_rs at the lambdas, zero-padded to five (msbur.src l.186)",
                               cells=grid), fh, indent=1)
                fh.write("\n")
    else:
        print(f"cbar_ckp([20, 40], 60) = {cbar_ckp([20, 40], 60)!r}")
