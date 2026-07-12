"""
size_power_cbar_comparison.py
=============================
Size and power comparison of three c-bar specifications for the Model 1
(constant + level dummies) point-optimal unit-root test. Produces the power
table and power-curve figure of the paper's robustness section.

Compares THREE c-bar specifications for the MZt test of MODEL 1 (constant + 2
level dummies DU), at T=60, m=2, alpha=0.05:

    1. "Calibrated Model 1"  -- the calibrated surface of this paper (const).
    2. "Linear break-count"  -- ad hoc linear scaling of the ERS constant by the
                                break count (-7,-13,-17,-20 for m=0,1,2,3).
    3. "Trend-break surface" -- the CKP response surface (Model 3, broken trend),
                                deliberately misapplied. Uses the trend-break
                                response surface c_bar_rs, INLINED below (no
                                external dependency).

All three use the SAME test (Model 1 detrending with the DU at the known dates
and the MZt statistic) -- only the VALUE of c-bar changes. Each c-bar gets its
own critical value (5% percentile of the null c=0), as a practitioner using
that c-bar would.

DGPs (table columns):
    (i)   SIZE, no breaks   : pure I(1) (c=0), theta=0; tested with 2 DU.
    (ii)  SIZE, with breaks : I(1) (c=0) with 2 level breaks (theta!=0); 2 DU.
          [by GLS-detrending invariance must match (i) -- a consistency check]
    (iii) POWER, broken mean: I(0) (c=C_ALT<0) around a broken mean (theta!=0).

The numerical engine is IMPORTED from mlb_core.py, so the validation uses
exactly the same primitives (build_z, GLS detrending, MZt) as the calibration.

OUTPUTS:
    - prints the power table (rejection rates +/- Monte Carlo error)
    - writes tab_power.tex (the table body)
    - generates fig_power.pdf / .png (power curves)

USAGE:
    python size_power_cbar_comparison.py            # production
    python size_power_cbar_comparison.py --speed    # fast test (small R)
"""
from __future__ import annotations
import os, sys, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore")

# Model 1 engine (same primitives as the calibration) ----------------------
# --- kernel adapter ------------------------------------------------------
# The calibration kernel is mlb_core, whose public primitives are numba
# functions. We build thin local wrappers around them so this script shares
# the SAME kernel used everywhere else in the paper:
#
#   build_z(nt, break_pos)                    -> build_z_nb
#   compute_M_statistics(y, break_pos, cbar)  -> mstats_nb(y, Z, cbar, code, kmax)
#   generate_dgp(T, lambdas, c, seed, beta)   -> gen_dgp_nb(nt, break_pos, c, beta, eps)
#
# Conventions: kmax=12; sigma2_method string -> int code via _METHOD_CODE;
# break positions from lambdas via the kernel helper; N(0,1) innovations drawn
# with a seeded default_rng so a given seed maps to a reproducible draw.
import numpy as _np
import mlb_core as _v6

# --- trend-break response surface c_bar_rs (INLINED from the CKP Model 3
# kernel to remove the cross-package dependency; used ONLY as the
# comparison curve in Table `power`/Figure `power`). --------------------
_CBAR_PARAM = np.array([
    -13.12832, -36.53045,  0,        20.2423,  -4.596202, -10.31678,
    115.2092,  -29.18712, -68.36453,  5.873121,  0,
    -130.337,   74.64396,  85.48737,  0,         0,
     51.98117, -53.03452, -36.27221,  0,        11.27727,
    -23.39517,  -5.360149, 23.99683,
      4.788676, -27.10002, -35.78388, 51.12371,
    -29.8518,   -3.069174, -37.45898, 64.95842,
      5.825729, -88.78176, -11.54197, 83.48645,
    125.2349,  -173.1259,  80.95821,   2.863782,
    118.2829,   -80.1287,   0,        128.872,
      6.387147,-118.1043, -199.0615,  247.6469,
    -98.05947,   0,       -160.5713,  38.52177,
      0,        -65.21576,  0,         62.86494,
    117.9976,  -127.5544,  46.2304,    0,        79.1693,
])


def _xreg_cbar(lam5):
    """
    Build the regressor vector for c_bar_rs (61 elements).
    lam5: array (5,) of break fractions, zeros for unused slots.
    Fiel ao Gauss linhas 809-815.
    """
    L = lam5
    xr = [1.0]
    for i in range(5):
        xr.append(L[i])          # λi
    for i in range(5):
        xr.append(L[i]**2)       # λi²
    for i in range(5):
        xr.append(L[i]**3)       # λi³
    for i in range(5):
        xr.append(L[i]**4)       # λi⁴
    pairs = [(i, j) for i in range(5) for j in range(i+1, 5)]
    for d in [1, 2, 3, 4]:
        for i, j in pairs:
            xr.append(abs(L[i] - L[j])**d)  # |λi-λj|^d
    return np.array(xr)


def c_bar_rs(break_pos, T):
    """
    c̄ via surface de resposta (Gauss: c_bar_rs).
    break_pos: list of (1-based) break positions.
    T: sample size.
    """
    lam5 = np.zeros(5)
    for k, bp in enumerate(break_pos[:5]):
        lam5[k] = bp / T
    xr = _xreg_cbar(lam5)
    return float(xr @ _CBAR_PARAM)



_KMAX_FIXED = 12
_CODE = _v6._METHOD_CODE            # {'const': 0, 'maic': 1}

def build_z(nt, break_pos):
    return _v6.build_z_nb(int(nt), _np.asarray(break_pos, dtype=_np.int64))

def compute_M_statistics(y, break_pos, cbar, sigma2_method='const',
                         kmax=_KMAX_FIXED):
    Z = _v6.build_z_nb(len(y), _np.asarray(break_pos, dtype=_np.int64))
    code = _CODE[sigma2_method] if isinstance(sigma2_method, str) \
        else int(sigma2_method)
    mza, msb, mzt, pt, mpt, ok = _v6.mstats_nb(
        _np.ascontiguousarray(y, dtype=_np.float64), Z,
        float(cbar), code, int(kmax))
    # the kernel returns a tuple; we index into it by statistic name used in
    # stat_sample (stat='mzt', 'pt', ...); expose both cases for safety.
    return {'mza': mza, 'msb': msb, 'mzt': mzt, 'pt': pt, 'mpt': mpt,
            'MZa': mza, 'MSB': msb, 'MZt': mzt, 'PT': pt, 'MPT': mpt,
            'ok': ok}

def generate_dgp(T, lambdas, c, seed, beta_scale=0.0):
    # lambdas (break fractions in (0,1)) -> integer break positions, the
    # same mapping the calibration uses.
    break_pos = _v6.break_pos_from_lambdas(int(T), tuple(lambdas))
    rng = _np.random.default_rng(int(seed))
    eps = rng.standard_normal(int(T))
    y = _v6.gen_dgp_nb(int(T), _np.asarray(break_pos, dtype=_np.int64),
                       float(c), float(beta_scale), eps)
    return y, break_pos
# --- end adapter ---------------------------------------------------------
# CKP trend-break surface (Model 3) ------------------------------------------


try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# =============================================================================
# CONFIGURATION
# =============================================================================
T        = 60
LAM      = (1.0/3.0, 2.0/3.0)                       # fractions of the 2 breaks
BREAK_POS = [int(np.floor(l * T)) for l in LAM]     # [20, 40]
ALPHA    = 0.05
SEED0    = 12345
BETA     = 5.0                                       # magnitude θ dos saltos (a
                                                     # statistic is invariant to theta;
                                                     # mantemos ≠0 por realismo)
C_ALT    = -10.0                                     # I(0) alternative of column (iii)
                                                     # (raiz AR ≈ 1−10/60 ≈ 0.83)
C_GRID   = list(np.round(np.arange(0.0, -30.01, -2.0), 1))   # alternatives (power curve)
CBAR_GRID = list(np.round(np.arange(-20.0, -2.9, 0.5), 2))   # grade p/ recalibrar c̄

DEF = dict(R_cv=10000, R_table=10000, R_curve=10000, R_calib_cv=3000, R_calib_pow=3000)
SPD = dict(R_cv=300,  R_table=300,  R_curve=150,  R_calib_cv=300,  R_calib_pow=300)


# =============================================================================
# STATISTIC SAMPLER (MZt by default; PT for the ERS tangency)
# =============================================================================
def stat_sample(cbar, c, n, seed, stat='mzt', beta=BETA, lambdas=LAM):
    """n draws of statistic `stat` under local parameter c, with the 2 breaks of
    level (magnitude beta), tested in Model 1 at the BREAK_POS com o c̄ dado."""
    out = np.empty(n)
    for i in range(n):
        y, _ = generate_dgp(T, lambdas, c, seed + i, beta_scale=beta)
        s = compute_M_statistics(y, BREAK_POS, cbar, 'const')
        out[i] = s[stat] if s else np.nan
    return out[np.isfinite(out)]


def cv5(cbar, n, seed, stat='mzt'):
    """5% critical value: 5th percentile of `stat` under the null (c=0), with breaks present.
    (All M/PT statistics reject for small valuenos → cauda inferior.)"""
    null = stat_sample(cbar, 0.0, n, seed, stat=stat, beta=BETA, lambdas=LAM)
    return float(np.percentile(null, 100.0 * ALPHA))


def rej_rate(cbar, c, cvval, n, seed, stat='mzt', beta=BETA, lambdas=LAM):
    draws = stat_sample(cbar, c, n, seed, stat=stat, beta=beta, lambdas=lambdas)
    p = float(np.mean(draws < cvval))
    se = float(np.sqrt(max(p * (1 - p), 0.0) / max(len(draws), 1)))
    return p, se


# =============================================================================
# RECOMPUTE THE CALIBRATED c-bar (ERS tangency) -- self-contained check
# =============================================================================
def calibrate_cbar(R_cv, R_pow):
    """Reproduce the ERS tangency criterion for (T=60, m=2, LAM): c-bar* is the value where the
    POINT-OPTIMAL (PT) test attains power 0.5 against the alternative c=c-bar -- exactly the
    criterion used in the surface calibration. This is an independent, lower-replication
    cross-check; it is expected to land near, but not bit-exactly reproduce, the paper's
    literal -8.40 (the lambda-exact tangency at (m,T)=(2,60), computed via the full production
    surface search in replicate_section3_4.py and hardcoded here as the default).
    Note: the tangency is defined on PT, not MZt; using MZt would give a different c-bar."""
    best_c, best_gap = None, 1e9
    for cb in CBAR_GRID:
        cvv = cv5(cb, R_cv, SEED0 + 101, stat='pt')
        p, _ = rej_rate(cb, cb, cvv, R_pow, SEED0 + 202, stat='pt')   # poder PT em c=c̄
        gap = abs(p - 0.5)
        if gap < best_gap:
            best_gap, best_c = gap, cb
    return best_c


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Size/power comparison of c-bar specifications")
    ap.add_argument("--speed", action="store_true", help="fast test (small R)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--recalib", action="store_true",
                    help="recompute c-bar via PT tangency (default: use -8.40, the "
                         "paper's lambda-exact tangency at (m,T)=(2,60))")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    R = SPD if args.speed else DEF
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 74)
    print(f"SIZE/POWER COMPARISON -- T={T}, m=2, breaks={BREAK_POS}, alpha={ALPHA}")
    print(f"R: {R}")
    print("=" * 74)

    # ---- calibrated c-bar (Model 1) ---------------------------------------
    if args.recalib:
        cbar_calib = calibrate_cbar(R['R_calib_cv'], R['R_calib_pow'])
        print(f"[calibrated c-bar] recomputed via PT tangency: {cbar_calib}  (paper literal: -8.40)")
    else:
        cbar_calib = -8.40
        print(f"[calibrated c-bar] lambda-exact tangency (const, T=60, m=2): {cbar_calib}"
              f"  [use --recalib for an independent lower-replication cross-check]")

    cbar_tb = round(c_bar_rs(BREAK_POS, T), 2)
    CBAR = {
        "Calibrated Model 1":  float(cbar_calib),
        "Linear break-count":  -17.0,
        "Trend-break surface": float(cbar_tb),
    }
    print("\n[c-bar specifications at (T=60, m=2)]")
    for k, v in CBAR.items():
        print(f"   {k:24s}: c-bar = {v:+.2f}")

    # ---- 5% critical values per c-bar -------------------------------------
    CV = {k: cv5(v, R['R_cv'], SEED0 + 11 + i) for i, (k, v) in enumerate(CBAR.items())}
    print("\n[MZt 5% critical values (null c=0)]")
    for k in CBAR:
        print(f"   {k:24s}: CV = {CV[k]:.3f}")

    # ---- table: (i) size no breaks, (ii) size with breaks, (iii) power
    print("\n[Table] rejection rates (Monte Carlo error in parentheses)")
    hdr = f"{'c̄ specification':24s} {'(i) I(1) no breaks':>20s} {'(ii) I(1)+breaks':>18s} {f'(iii) I(0) c={C_ALT:.0f}':>16s}"
    print(hdr); print("-" * len(hdr))
    table = {}
    for i, (name, cb) in enumerate(CBAR.items()):
        s0 = rej_rate(cb, 0.0,  CV[name], R['R_table'], SEED0 + 31 + i, beta=0.0,  lambdas=())   # (i) no breaks
        s1 = rej_rate(cb, 0.0,  CV[name], R['R_table'], SEED0 + 41 + i, beta=BETA, lambdas=LAM)  # (ii) with breaks
        pw = rej_rate(cb, C_ALT, CV[name], R['R_table'], SEED0 + 51 + i, beta=BETA, lambdas=LAM) # (iii) power
        table[name] = (s0, s1, pw)
        print(f"{name:24s} {s0[0]:.3f} ({s0[1]:.3f})   {s1[0]:.3f} ({s1[1]:.3f})   {pw[0]:.3f} ({pw[1]:.3f})")

    # ---- power curves ------------------------------------------------------
    print("\n[Figure] power curves over c ...")
    def one_point(name, cb, c, seed):
        p, _ = rej_rate(cb, c, CV[name], R['R_curve'], seed)
        return (name, c, p)
    tasks = [(name, cb, c, SEED0 + 1000 + int(abs(c)) * 101 + j)
             for j, (name, cb) in enumerate(CBAR.items()) for c in C_GRID]
    if _HAS_JOBLIB and not args.speed:
        res = Parallel(n_jobs=args.jobs)(delayed(one_point)(n, cb, c, sd) for (n, cb, c, sd) in tasks)
    else:
        res = [one_point(n, cb, c, sd) for (n, cb, c, sd) in tasks]
    curve = {name: {} for name in CBAR}
    for name, c, p in res:
        curve[name][c] = p

    # ---- LaTeX (table body) ------------------------------------------------
    def cell(t): return f"${t[0]:.3f}$ \\tiny$(\\pm{t[1]:.3f})$"
    rows = []
    label = {"Calibrated Model 1": "Calibrated Model~1 $\\bar c(m,T)$",
             "Linear break-count": "Linear break-count scaling",
             "Trend-break surface": "Trend-break surface"}
    for name in CBAR:
        s0, s1, pw = table[name]
        rows.append(f"{label[name]:34s} & {cell(s0)} & {cell(s1)} & {cell(pw)} \\\\")
    tex = ("% cbar values: " +
           ", ".join(f"{k}={v:+.2f}" for k, v in CBAR.items()) +
           f"; CV5 MZt per spec; T={T}, m=2, breaks={BREAK_POS}, alpha={ALPHA};\n"
           f"% columns: (i) I(1) no breaks [size], (ii) I(1)+level breaks [size],"
           f" (iii) I(0) broken mean at c={C_ALT:.0f} [power].\n" +
           "\n".join(rows) + "\n")
    with open(os.path.join(args.outdir, "tab_power.tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    print("\n[escrito] tab_power.tex")

    # ---- power curve figure -------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import rcParams
        rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 10, "legend.fontsize": 9})
        # Monochrome palette for Q1 print consistency with the Section-6 figures
        # (which are strictly greyscale). Curves are distinguished on three axes
        # orthogonal to hue -- marker shape, line style, and marker fill -- so the
        # figure stays legible in black-and-white and for colour-blind readers.
        # The paper's own proposal (Calibrated Model 1) carries maximal contrast
        # (solid black, filled marker); the two dominated shortcuts are grey with
        # open markers, visually subordinate.
        cols = {"Calibrated Model 1": "black", "Linear break-count": "0.35",
                "Trend-break surface": "0.55"}
        mks  = {"Calibrated Model 1": "o", "Linear break-count": "s",
                "Trend-break surface": "^"}
        lss  = {"Calibrated Model 1": "-", "Linear break-count": "--",
                "Trend-break surface": ":"}
        mfc  = {"Calibrated Model 1": "black", "Linear break-count": "white",
                "Trend-break surface": "white"}
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        xs = [abs(c) for c in C_GRID]                  # plot vs |c| (0 → 30)
        for name in CBAR:
            ys = [curve[name][c] for c in C_GRID]
            ax.plot(xs, ys, marker=mks[name], color=cols[name], ls=lss[name],
                    ms=4.5, lw=1.4, markerfacecolor=mfc[name],
                    markeredgecolor=cols[name],
                    label=f"{name} ($\\bar c={CBAR[name]:+.1f}$)")
        ax.axhline(ALPHA, ls=":", color="0.5", lw=1)
        ax.text(0.3, ALPHA + 0.012, r"nominal $\alpha=0.05$", fontsize=8, color="0.4")
        ax.axvline(abs(C_ALT), ls="--", color="0.7", lw=0.9)
        ax.text(abs(C_ALT) + 0.3, 0.95, f"col.~(iii)\n$c={C_ALT:.0f}$", fontsize=7.5,
                color="0.5", va="top")
        ax.set_xlabel(r"local alternative $|c|$  (root $=1-|c|/T$)")
        ax.set_ylabel(r"rejection rate (power)")
        ax.set_title(f"Power of $\\mathrm{{MZ}}_t$ under three $\\bar c$ choices, $T={T}$, $m=2$",
                     fontsize=10)
        ax.set_ylim(0, 1.02); ax.set_xlim(-0.5, 30.5)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fig_power.pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(args.outdir, "fig_power.png"), dpi=160, bbox_inches="tight")
        print("[escrito] fig_power.pdf / fig_power.png")
    except Exception as e:
        print(f"[warning] figure not generated: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
