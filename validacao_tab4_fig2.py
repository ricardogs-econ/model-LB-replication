"""
validacao_tab4_fig2.py
======================
Validação tamanho/poder para a Tabela 4 (tab:power) e a Figura 2 (fig:power) do
short_paper.tex.

Compara TRÊS especificações de c̄ para o teste MZt do MODELO 1 (constante + 2
level dummies DU), em T=60, m=2, α=0.05:

    1. "Calibrated Model 1"  — superfície calibrada deste artigo (Tabela 1, const)
    2. "Linear break-count"  — escala linear ad hoc da constante ERS pela contagem
                               de quebras (−7,−13,−17,−20 para m=0,1,2,3)
    3. "Trend-break surface" — superfície de resposta do CKP (Modelo 3, tendência
                               quebrada), reaproveitada indevidamente. Vem de
                               the trend-break response surface c_bar_rs, INLINED below (no external dependency).

Os três usam o MESMO teste (detrending Model 1 com os DU nas datas conhecidas e a
estatística MZt) — só muda o VALOR de c̄. Cada c̄ recebe seu próprio valor crítico
(percentil 5% da nula c=0), como faria um praticante usando aquele c̄.

DGPs (colunas da Tabela 4):
    (i)   TAMANHO, sem quebras : I(1) puro (c=0), θ=0; teste com 2 DU.
    (ii)  TAMANHO, com quebras : I(1) (c=0) com 2 level breaks (θ≠0); teste com 2 DU.
          [por invariância do detrending GLS deve coincidir com (i) — checagem]
    (iii) PODER, broken mean   : I(0) (c=C_ALT<0) em torno de média quebrada (θ≠0).

O motor numérico é IMPORTADO de cbar_ml1_final_production_v2.py, garantindo que a
validação use exatamente as mesmas primitivas (build_z, _glsd, MZt) da calibração.

SAÍDAS:
    - imprime a Tabela 4 (taxas de rejeição ± erro de Monte Carlo)
    - escreve tab_power.tex (corpo da Tabela 4)
    - gera fig_power.pdf / .png (curvas de poder, Figura 2)

USO:
    python validacao_tab4_fig2.py            # produção
    python validacao_tab4_fig2.py --speed    # teste rápido (R pequeno)
# (dependency-free: c_bar_rs is inlined above)
"""
from __future__ import annotations
import os, sys, argparse, warnings
import numpy as np

warnings.filterwarnings("ignore")

# motor Model 1 (mesmas primitivas da calibração v6) ------------------------
# --- v6 compatibility adapter -------------------------------------------
# The calibration kernel is now mlb_core, whose public
# primitives are numba functions with a different signature from the v2/v3
# API this script was written against. Rather than import obsolete symbols,
# we build thin wrappers that reproduce the v2/v3 call semantics EXACTLY on
# top of the v6 primitives, so tab:power and fig:power share the SAME v6
# kernel used everywhere else in the paper.
#
#   v2/v3  build_z(nt, break_pos)                       -> v6 build_z_nb
#   v2/v3  compute_M_statistics(y, break_pos, cbar,     -> v6 mstats_nb(y, Z,
#             sigma2_method='const', kmax=12)                 cbar, code, kmax)
#   v2/v3  gerar_dgp(T, lambdas, c, seed, beta_scale)   -> v6 gen_dgp_nb(nt,
#                                                          break_pos, c, bscale, eps)
#
# Conventions preserved: kmax=12 (v6 fixed); sigma2_method string -> v6 int
# code via _METHOD_CODE; break positions from lambdas via the v6 helper;
# N(0,1) innovations drawn with a seeded default_rng so a given (seed) maps
# to a reproducible draw exactly as gerar_dgp did.
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
    Constrói o vetor de regressores para c_bar_rs (61 elementos).
    lam5: array (5,) de break fractions, zeros para slots não usados.
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
    break_pos: lista de posições (1-based) das quebras.
    T: tamanho da amostra.
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
    # v2/v3 returned a dict indexed by the lowercase statistic key used in
    # stat_sample (stat='mzt', 'pt', ...); expose both cases for safety.
    return {'mza': mza, 'msb': msb, 'mzt': mzt, 'pt': pt, 'mpt': mpt,
            'MZa': mza, 'MSB': msb, 'MZt': mzt, 'PT': pt, 'MPT': mpt,
            'ok': ok}

def gerar_dgp(T, lambdas, c, seed, beta_scale=0.0):
    # lambdas (break fractions in (0,1)) -> integer break positions, the
    # same mapping the v6 calibration uses.
    break_pos = _v6.break_pos_from_lambdas(int(T), tuple(lambdas))
    rng = _np.random.default_rng(int(seed))
    eps = rng.standard_normal(int(T))
    y = _v6.gen_dgp_nb(int(T), _np.asarray(break_pos, dtype=_np.int64),
                       float(c), float(beta_scale), eps)
    return y, break_pos
# --- end adapter ---------------------------------------------------------
# superfície trend-break do CKP (Modelo 3) -----------------------------------


try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
T        = 60
LAM      = (1.0/3.0, 2.0/3.0)                       # frações das 2 quebras
BREAK_POS = [int(np.floor(l * T)) for l in LAM]     # [20, 40]
ALPHA    = 0.05
SEED0    = 12345
BETA     = 5.0                                       # magnitude θ dos saltos (a
                                                     # estatística é invariante a θ;
                                                     # mantemos ≠0 por realismo)
C_ALT    = -10.0                                     # alternativa I(0) da coluna (iii)
                                                     # (raiz AR ≈ 1−10/60 ≈ 0.83)
C_GRID   = list(np.round(np.arange(0.0, -30.01, -2.0), 1))   # alternativas (Fig 2)
CBAR_GRID = list(np.round(np.arange(-20.0, -2.9, 0.5), 2))   # grade p/ recalibrar c̄

DEF = dict(R_cv=6000, R_table=6000, R_curve=2500, R_calib_cv=3000, R_calib_pow=3000)
SPD = dict(R_cv=300,  R_table=300,  R_curve=150,  R_calib_cv=300,  R_calib_pow=300)


# =============================================================================
# AMOSTRADOR DE ESTATÍSTICA (MZt por padrão; PT para a tangência ERS)
# =============================================================================
def stat_sample(cbar, c, n, seed, stat='mzt', beta=BETA, lambdas=LAM):
    """n sorteios da estatística `stat` sob parâmetro local c, com as 2 quebras de
    nível (magnitude beta), testadas em Model 1 nas datas BREAK_POS com o c̄ dado."""
    out = np.empty(n)
    for i in range(n):
        y, _ = gerar_dgp(T, lambdas, c, seed + i, beta_scale=beta)
        s = compute_M_statistics(y, BREAK_POS, cbar, 'const')
        out[i] = s[stat] if s else np.nan
    return out[np.isfinite(out)]


def cv5(cbar, n, seed, stat='mzt'):
    """Valor crítico 5%: percentil 5 de `stat` sob a nula (c=0), com quebras presentes.
    (Todas as estatísticas M/PT rejeitam para valores pequenos → cauda inferior.)"""
    null = stat_sample(cbar, 0.0, n, seed, stat=stat, beta=BETA, lambdas=LAM)
    return float(np.percentile(null, 100.0 * ALPHA))


def rej_rate(cbar, c, cvval, n, seed, stat='mzt', beta=BETA, lambdas=LAM):
    draws = stat_sample(cbar, c, n, seed, stat=stat, beta=beta, lambdas=lambdas)
    p = float(np.mean(draws < cvval))
    se = float(np.sqrt(max(p * (1 - p), 0.0) / max(len(draws), 1)))
    return p, se


# =============================================================================
# RECÁLCULO DO c̄ CALIBRADO (tangência ERS) — checagem self-contained
# =============================================================================
def calibrate_cbar(R_cv, R_pow):
    """Reproduz o critério de tangência ERS para (T=60, m=2, LAM): c̄* é o valor onde o
    teste POINT-OPTIMAL (PT) atinge poder 0.5 contra a alternativa c=c̄ — exatamente o
    critério usado na calibração da superfície. Deve recuperar ≈ −8.7 (Tabela 1, const).
    Nota: a tangência é definida sobre PT, não MZt; usar MZt daria um c̄ diferente."""
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
    ap = argparse.ArgumentParser(description="Validação Tabela 4 / Figura 2")
    ap.add_argument("--speed", action="store_true", help="teste rápido (R pequeno)")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--recalib", action="store_true",
                    help="recalcular c̄ por tangência PT (default: usar -8.7 da Tabela 1)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    R = SPD if args.speed else DEF
    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 74)
    print(f"VALIDAÇÃO Tabela 4 / Figura 2 — T={T}, m=2, quebras={BREAK_POS}, α={ALPHA}")
    print(f"R: {R}")
    print("=" * 74)

    # ---- c̄ calibrado (Model 1) --------------------------------------------
    if args.recalib:
        cbar_calib = calibrate_cbar(R['R_calib_cv'], R['R_calib_pow'])
        print(f"[c̄ calibrado] recalculado por tangência PT: {cbar_calib}  (Tabela 1: -8.7)")
    else:
        cbar_calib = -8.7
        print(f"[c̄ calibrado] valor da Tabela 1 (const, T=60, m=2): {cbar_calib}"
              f"  [use --recalib para recalcular por tangência PT]")

    cbar_tb = round(c_bar_rs(BREAK_POS, T), 2)
    CBAR = {
        "Calibrated Model 1":  float(cbar_calib),
        "Linear break-count":  -17.0,
        "Trend-break surface": float(cbar_tb),
    }
    print("\n[Especificações de c̄ em (T=60, m=2)]")
    for k, v in CBAR.items():
        print(f"   {k:24s}: c̄ = {v:+.2f}")

    # ---- valores críticos 5% por c̄ ----------------------------------------
    CV = {k: cv5(v, R['R_cv'], SEED0 + 11 + i) for i, (k, v) in enumerate(CBAR.items())}
    print("\n[Valores críticos MZt 5% (nula c=0)]")
    for k in CBAR:
        print(f"   {k:24s}: CV = {CV[k]:.3f}")

    # ---- Tabela 4: (i) tamanho s/quebras, (ii) tamanho c/quebras, (iii) poder
    print("\n[Tabela 4] taxas de rejeição (erro de Monte Carlo entre parênteses)")
    hdr = f"{'c̄ specification':24s} {'(i) I(1) no breaks':>20s} {'(ii) I(1)+breaks':>18s} {f'(iii) I(0) c={C_ALT:.0f}':>16s}"
    print(hdr); print("-" * len(hdr))
    table = {}
    for i, (name, cb) in enumerate(CBAR.items()):
        s0 = rej_rate(cb, 0.0,  CV[name], R['R_table'], SEED0 + 31 + i, beta=0.0,  lambdas=())   # (i) sem quebras
        s1 = rej_rate(cb, 0.0,  CV[name], R['R_table'], SEED0 + 41 + i, beta=BETA, lambdas=LAM)  # (ii) com quebras
        pw = rej_rate(cb, C_ALT, CV[name], R['R_table'], SEED0 + 51 + i, beta=BETA, lambdas=LAM) # (iii) poder
        table[name] = (s0, s1, pw)
        print(f"{name:24s} {s0[0]:.3f} ({s0[1]:.3f})   {s1[0]:.3f} ({s1[1]:.3f})   {pw[0]:.3f} ({pw[1]:.3f})")

    # ---- curvas de poder (Figura 2) ----------------------------------------
    print("\n[Figura 2] curvas de poder sobre c ...")
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

    # ---- LaTeX (corpo da Tabela 4) -----------------------------------------
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

    # ---- Figura 2 -----------------------------------------------------------
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
        print(f"[aviso] figura não gerada: {e}")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
