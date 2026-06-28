"""
ar1_size_power_robustness_v2.py
==============================
Simulação REALISTA de robustez à correlação serial -- fecha o Ponto 2 dos
dois referee reports na parte que o experimento oráculo (oracle_serial_
robustness.py, já reportado na nota como Tabela 4) não cobre.

POR QUE ESTE SCRIPT, ALÉM DO ORÁCULO
-------------------------------------
O oráculo isola o efeito da correlação serial *per se*, usando a variância
de longo prazo VERDADEIRA. Ele mostra que a tangência LIMITE não é movida
pela correlação -- mas não diz nada sobre o que acontece quando s²_AR é
ESTIMADO, que é exatamente o que os dois referees pediram:

  - Referee 1 (revisão maior, ponto 4.2): "uma pequena simulação de
    robustez (por exemplo, para um AR(1) com coeficiente 0.5)... e
    comparar a taxa de rejeição usando a tabela calibrada sob i.i.d.
    versus uma recalibração sob dependência."
  - Referee 2 (revisão menor, major point 1): "incluir uma seção de Monte
    Carlo avaliando o comportamento do tamanho (size) e poder (power) sob
    processos de erro autorregressivos... usando o estimador de Perron e
    Ng (1998)" -- i.e., com s²_AR DE FATO estimado, não oráculo.

Este script reusa DIRETAMENTE compute_M_statistics da produção (nenhuma
fórmula é reimplementada), de modo que sar é estimado exatamente como no
artigo -- 'const' (difference-based) ou 'maic' (autoregressive spectral
density, Perron & Ng 1998, com MAIC de Ng & Perron 2001).

TRÊS EXPERIMENTOS
------------------
(0) ANCORAGEM (rho=0).  Deriva, com a MESMA máquina deste script (mesmo
    gerador, mesma estatística PT, mesmo critério de tangência), o cbar* e
    o valor crítico de 5% sob i.i.d.  Evita qualquer risco de divergência
    de transcrição com as Tabelas 1-2 do artigo e ata o experimento a uma
    única simulação internamente coerente.

(1) SIZE.  Usando o (cbar*, CV) ancorados em (0), gera dados sob H0 (c=0)
    com inovações AR(1) em rho>0, computa PT com sar ESTIMADO (não
    oráculo), e mede a taxa de rejeição empírica contra a nominal 5%.
    Isto mede a DISTORÇÃO DE TAMANHO de usar a tabela i.i.d. sob
    dependência genuína -- a pergunta central do Referee 2.

(2) POWER.  Idem, mas sob H1 (c=cbar*), medindo o poder empírico contra a
    âncora de 0.50. Mede a PERDA/GANHO DE PODER de usar a tabela i.i.d.
    sob dependência.

(3) RECALIBRAÇÃO (opcional, mais caro; ligar com --recalibrate).  Repete a
    busca de tangência completa, mas com sar ESTIMADO sob AR(1) em rho>0,
    encontrando o cbar*_AR(rho) "verdadeiro" sob dependência. A diferença
    cbar*_AR(rho) - cbar*_iid quantifica o que uma recalibração mudaria --
    exatamente a comparação que o Referee 1 pede.

USO
---
    python3 ar1_size_power_robustness.py              # experimentos 0+1+2
    python3 ar1_size_power_robustness.py --recalibrate  # também o (3)
    python3 ar1_size_power_robustness.py --quick       # checagem rápida

NÚCLEOS / PARALELIZAÇÃO
-------------------------
O script usa joblib (backend 'loky', baseado em processos) para paralelizar
entre CONFIGURAÇÕES independentes (cada combinação de m, T, rho, método é
uma tarefa) -- não dentro de uma configuração. Por padrão usa TODOS os
núcleos disponíveis (--n-jobs -1, o default). Para controlar:

    python3 ar1_size_power_robustness.py --recalibrate --n-jobs 4
    python3 ar1_size_power_robustness.py --recalibrate --n-jobs 1   # sequencial

Se joblib não estiver instalado (`pip install joblib`), o script roda em 1
núcleo automaticamente, com aviso. As primitivas de produção (_glsd) são
numba-jitadas mas SEM paralelismo interno (sem `parallel=True`); o ganho de
velocidade vem inteiramente de paralelizar entre configurações, não de
acelerar uma única avaliação.

Coloque este arquivo na mesma pasta de cbar_ml1_final_production_v3.py.
Saídas: ar1_size_power.csv, ar1_recalibration.csv (se --recalibrate),
tab_ar1_sizepower.tex, tab_ar1_recalibration.tex (se --recalibrate).
"""
from __future__ import annotations
import argparse
import csv
import importlib
import time

import numpy as np

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

# =============================================================================
# 0. IMPORTAR AS PRIMITIVAS DE PRODUÇÃO -- reuso direto, nenhuma fórmula nova
# =============================================================================
_PROD_CANDIDATES = [
    "cbar_ml1_final_production_v3",
    "cbar_ml1_final_production_v2",
    "cbar_ml1_final_production",
]
_prod = None
for _name in _PROD_CANDIDATES:
    try:
        _prod = importlib.import_module(_name)
        print(f"[ok] primitivas importadas de {_name}.py")
        break
    except ModuleNotFoundError:
        continue
if _prod is None:
    raise ImportError(
        "Não encontrei cbar_ml1_final_production_v3.py (nem v2/v1). "
        "Coloque este script na MESMA pasta do arquivo de produção."
    )

build_z = _prod.build_z
_glsd = _prod._glsd
compute_M_statistics = _prod.compute_M_statistics   # reuso DIRETO -- sem reimplementar

# =============================================================================
# 1. CONFIGURAÇÃO  (ajuste aqui)
# =============================================================================
M_LIST = [0, 1, 2]
T_LIST = [30, 60, 100]                 # cobre o T pequeno (preocupação dos referees) e moderado
RHO_LIST = [0.3, 0.5, 0.6]             # 0.5 é o valor que o Referee 1 pede explicitamente
SIGMA2_METHODS = ["const", "maic"]      # 'const' = headline da nota; 'maic' = Perron-Ng/MAIC
TRIM = 0.15
TARGET_POWER = 0.50
CBAR_GRID = np.round(np.arange(-16.0, -3.4, 0.5), 2)   # grade local; amplie se bater na borda
KMAX = 12
SEED_BASE = 20260624

# Réplicas para os experimentos (0)+(1)+(2): avaliação simples (sem busca em grade)
R_ANCHOR_CV = 10000    # réplicas para fixar a CV de 5% sob iid (experimento 0)
R_ANCHOR_POW = 5000    # réplicas para fixar cbar* sob iid (experimento 0) -- só usado se a busca em grade for necessária (método 'maic')
R_EVAL = 10000         # réplicas para size (1) e power (2) sob rho>0

# Réplicas para a recalibração (3) -- mais caro: grade completa em cbar
R_RECAL_CV = 10000
R_RECAL_POW = 5000
# Por padrão, recalibração só no ponto que o Referee 1 pediu (rho=0.5, estimador
# headline 'const'); amplie via --recal-rho / --recal-method se quiser mais.
RECAL_RHO_DEFAULT = [0.5]
RECAL_METHOD_DEFAULT = ["const"]

# Valores de cbar*(m,T) JÁ PUBLICADOS na Tabela 1 do artigo (estimador
# 'const', sob i.i.d., R_cv=10000/R_pow=5000 de produção). Usados para
# ANCORAR o experimento 'const' SEM repetir a busca em grade completa --
# só a CV de 5% é recomputada aqui (mais barato, e internamente coerente
# porque usa a mesma simulação que os passos (1)-(2) seguintes). Para o
# método 'maic' (não tabulado no artigo) a busca em grade É feita, mas
# estreitada em torno do valor 'const' correspondente (a robustez relatada
# no artigo mostra que os dois estimadores diferem por no máximo ~2.15).
TABLE1_CBAR_CONST = {
    (0, 30): -7.5, (0, 45): -7.0, (0, 50): -7.0, (0, 60): -7.0, (0, 80): -7.0,
    (0, 100): -7.0, (0, 150): -6.5, (0, 200): -7.0, (0, 300): -7.0,
    (1, 30): -8.8, (1, 45): -8.1, (1, 50): -8.1, (1, 60): -7.9, (1, 80): -7.6,
    (1, 100): -7.4, (1, 150): -7.4, (1, 200): -7.3, (1, 300): -7.2,
    (2, 30): -10.3, (2, 45): -9.4, (2, 50): -9.1, (2, 60): -8.7, (2, 80): -8.3,
    (2, 100): -8.0, (2, 150): -7.8, (2, 200): -7.6, (2, 300): -7.4,
}


def break_fractions(m, trim=TRIM):
    if m == 0:
        return []
    return [round(trim + (i + 1) * (1 - 2 * trim) / (m + 1), 4) for i in range(m)]


def config_seed(m, T, rho, method, seed_base=SEED_BASE):
    msig = 1 if method == "const" else 2
    return (seed_base + T * 1_000_000 + m * 100_000
            + int(round(rho * 1000)) * 17 + msig * 3)


# =============================================================================
# 2. DGP COM INOVAÇÕES AR(1)  (idêntico ao usado no experimento oráculo;
#    rho_ar=0 reproduz exatamente gerar_dgp da produção)
# =============================================================================
def gerar_dgp_ar1(T, lambdas, c, rho_ar, seed, beta_scale=0.0):
    rng = np.random.default_rng(int(abs(seed)) % (2**63 - 1))
    break_pos = [int(max(1, min(T - 1, np.floor(lam * T)))) for lam in lambdas]
    Z = build_z(T, break_pos)
    beta = np.full(Z.shape[1], float(beta_scale))
    e = rng.normal(0, 1, T)
    if rho_ar != 0.0:
        eps = np.empty(T)
        eps[0] = e[0]
        for t in range(1, T):
            eps[t] = rho_ar * eps[t - 1] + e[t]
    else:
        eps = e
    root = 1.0 + c / T
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = root * u[t - 1] + eps[t]
    return Z @ beta + u, break_pos


def pt_stat(y, break_pos, cbar, sigma2_method, kmax=KMAX):
    """PT real (sar estimado), via compute_M_statistics da produção -- sem oráculo."""
    out = compute_M_statistics(y, break_pos, cbar, sigma2_method=sigma2_method, kmax=kmax)
    return out["pt"] if out is not None else np.nan


# =============================================================================
# 3. EXPERIMENTO 0: ÂNCORA SOB I.I.D. (rho=0), com a MESMA máquina
# =============================================================================
def anchor_iid(m, T, method, seed, r_anchor_cv=None, r_anchor_pow=None, cbar_grid=None):
    """
    Para 'const': usa o cbar* JÁ PUBLICADO na Tabela 1 (TABLE1_CBAR_CONST) e
    recomputa apenas a CV de 5% nesse ponto (r_anchor_cv réplicas, sem busca
    em grade) -- mais barato e internamente coerente, já que a CV é derivada
    com a MESMA simulação usada nos passos (1)-(2) seguintes.

    Para 'maic' (não tabulado no artigo): faz a busca de tangência completa,
    mas em uma grade ESTREITA centrada no valor 'const' correspondente (a
    robustez já relatada no artigo mostra diferença máxima de ~2.15 entre os
    dois estimadores), reduzindo o custo computacional sem perder rigor.

    r_anchor_cv/r_anchor_pow/cbar_grid: passados EXPLICITAMENTE (não lidos de
    globais do módulo) para que o comportamento seja idêntico em execução
    sequencial ou paralela (joblib/loky não compartilha globais mutados após
    o import).
    """
    r_anchor_cv = R_ANCHOR_CV if r_anchor_cv is None else r_anchor_cv
    r_anchor_pow = R_ANCHOR_POW if r_anchor_pow is None else r_anchor_pow
    cbar_grid = CBAR_GRID if cbar_grid is None else cbar_grid
    lambdas = break_fractions(m)

    if method == "const" and (m, T) in TABLE1_CBAR_CONST:
        cbar_star = TABLE1_CBAR_CONST[(m, T)]
        h0 = np.empty(r_anchor_cv)
        for r in range(r_anchor_cv):
            y, bp = gerar_dgp_ar1(T, lambdas, 0.0, 0.0, seed + r)
            h0[r] = pt_stat(y, bp, cbar_star, method)
        h0 = h0[np.isfinite(h0)]
        cv5 = np.percentile(h0, 5) if len(h0) >= r_anchor_cv * 0.5 else np.nan
        rej = nval = 0
        for r in range(r_anchor_pow):
            y, bp = gerar_dgp_ar1(T, lambdas, cbar_star, 0.0, seed + 10**6 + r)
            v = pt_stat(y, bp, cbar_star, method)
            if np.isfinite(v):
                nval += 1
                rej += v < cv5
        power = rej / nval if nval else np.nan
        return cbar_star, cv5, power, 0   # widened=0 (âncora publicada, sem busca)

    # método 'maic' (ou (m,T) fora da tabela publicada): busca em grade,
    # estreitada em torno do valor 'const' quando disponível, com
    # AUTO-EXPANSÃO da janela se a tangência cair na borda (sinal de que a
    # janela inicial era estreita demais -- conhecido em T pequeno, onde o
    # MAIC infla s² e exige c̄ bem mais negativo; cf. Seção de Robustez).
    center = TABLE1_CBAR_CONST.get((m, T))

    def _search(grid):
        best = None
        for cbar in grid:
            h0 = np.empty(r_anchor_cv)
            for r in range(r_anchor_cv):
                y, bp = gerar_dgp_ar1(T, lambdas, 0.0, 0.0, seed + r)
                h0[r] = pt_stat(y, bp, cbar, method)
            h0 = h0[np.isfinite(h0)]
            if len(h0) < r_anchor_cv * 0.5:
                continue
            cv5 = np.percentile(h0, 5)
            rej = nval = 0
            for r in range(r_anchor_pow):
                y, bp = gerar_dgp_ar1(T, lambdas, cbar, 0.0, seed + 10**6 + r)
                v = pt_stat(y, bp, cbar, method)
                if np.isfinite(v):
                    nval += 1
                    rej += v < cv5
            power = rej / nval if nval else np.nan
            if best is None or abs(power - TARGET_POWER) < abs(best[2] - TARGET_POWER):
                best = (cbar, cv5, power)
        return best

    if center is not None:
        half_width = 4.0
        max_half_width = 16.0  # cobre a discrepância máxima já documentada (~12) com margem
        attempt = 0
        while True:
            grid = np.round(np.arange(center - half_width, center + half_width + 0.01, 0.5), 2)
            best = _search(grid)
            if best is None:
                break
            cbar_star, cv5, power = best
            hit_boundary = cbar_star in (float(grid[0]), float(grid[-1]))
            if not hit_boundary or half_width >= max_half_width:
                if hit_boundary:
                    print(f"  [!] AVISO: âncora ainda na BORDA após expandir a janela "
                          f"até ±{half_width} (m={m},T={T},{method}) -- "
                          f"amplie max_half_width ou center manualmente.")
                break
            attempt += 1
            print(f"  [auto-widen] cbar*={cbar_star} na borda (±{half_width}); "
                  f"expandindo para ±{half_width*2} e refazendo a busca "
                  f"(m={m},T={T},{method}, tentativa {attempt})")
            half_width *= 2.0
        grid_used = grid
    else:
        attempt = 0
        best = _search(cbar_grid)
        grid_used = cbar_grid

    if best is None:
        return np.nan, np.nan, np.nan, attempt
    cbar_star, cv5, power = best
    if cbar_star in (float(grid_used[0]), float(grid_used[-1])) and center is None:
        print(f"  [!] AVISO: âncora cbar*={cbar_star} na BORDA da grade "
              f"(m={m},T={T},{method}) -- amplie a janela em anchor_iid.")
    # widened = nº de auto-expansões; >0 marca âncora degenerada (MAIC em T curto)
    return cbar_star, cv5, power, attempt


# =============================================================================
# 4. EXPERIMENTOS 1+2: SIZE e POWER sob AR(1), usando (cbar*, CV) da âncora
# =============================================================================
def size_power_at_rho(m, T, method, rho, cbar_star, cv5, seed, r_eval=None):
    r_eval = R_EVAL if r_eval is None else r_eval
    lambdas = break_fractions(m)
    # SIZE: H0 (c=0) com inovações AR(1) em rho
    rej0 = nval0 = 0
    for r in range(r_eval):
        y, bp = gerar_dgp_ar1(T, lambdas, 0.0, rho, seed + r)
        v = pt_stat(y, bp, cbar_star, method)
        if np.isfinite(v):
            nval0 += 1
            rej0 += v < cv5
    size_emp = rej0 / nval0 if nval0 else np.nan
    # POWER: H1 (c=cbar_star) com inovações AR(1) em rho
    rej1 = nval1 = 0
    for r in range(r_eval):
        y, bp = gerar_dgp_ar1(T, lambdas, cbar_star, rho, seed + 10**6 + r)
        v = pt_stat(y, bp, cbar_star, method)
        if np.isfinite(v):
            nval1 += 1
            rej1 += v < cv5
    power_emp = rej1 / nval1 if nval1 else np.nan
    return size_emp, power_emp


# =============================================================================
# 5. EXPERIMENTO 3 (opcional): RECALIBRAÇÃO COMPLETA SOB AR(1), sar ESTIMADO
# =============================================================================
def recalibrate_ar1(m, T, method, rho, seed, r_recal_cv=None, r_recal_pow=None, cbar_grid=None):
    r_recal_cv = R_RECAL_CV if r_recal_cv is None else r_recal_cv
    r_recal_pow = R_RECAL_POW if r_recal_pow is None else r_recal_pow
    cbar_grid = CBAR_GRID if cbar_grid is None else cbar_grid
    lambdas = break_fractions(m)
    best = None
    for cbar in cbar_grid:
        h0 = np.empty(r_recal_cv)
        for r in range(r_recal_cv):
            y, bp = gerar_dgp_ar1(T, lambdas, 0.0, rho, seed + r)
            h0[r] = pt_stat(y, bp, cbar, method)
        h0 = h0[np.isfinite(h0)]
        if len(h0) < r_recal_cv * 0.5:
            continue
        cv5 = np.percentile(h0, 5)
        rej = nval = 0
        for r in range(r_recal_pow):
            y, bp = gerar_dgp_ar1(T, lambdas, cbar, rho, seed + 10**6 + r)
            v = pt_stat(y, bp, cbar, method)
            if np.isfinite(v):
                nval += 1
                rej += v < cv5
        power = rej / nval if nval else np.nan
        if best is None or abs(power - TARGET_POWER) < abs(best[1] - TARGET_POWER):
            best = (cbar, power)
    if best is None:
        return np.nan, np.nan
    cbar_star, power = best
    if cbar_star in (float(cbar_grid[0]), float(cbar_grid[-1])):
        print(f"  [!] AVISO: recalibração cbar*={cbar_star} na BORDA da grade "
              f"(m={m},T={T},rho={rho},{method}) -- amplie CBAR_GRID.")
    return cbar_star, power


# =============================================================================
# 6. LOOP PRINCIPAL
# =============================================================================
def _anchor_worker(m, T, method, r_anchor_cv, r_anchor_pow, cbar_grid):
    seed = config_seed(m, T, 0.0, method)
    t0 = time.time()
    cbar_star, cv5, power_iid, widened = anchor_iid(
        m, T, method, seed, r_anchor_cv=r_anchor_cv,
        r_anchor_pow=r_anchor_pow, cbar_grid=cbar_grid,
    )
    return m, T, method, cbar_star, cv5, power_iid, widened, time.time() - t0


def _sp_worker(m, T, method, rho, cbar_star, cv5, r_eval):
    seed = config_seed(m, T, rho, method)
    t0 = time.time()
    size_emp, power_emp = size_power_at_rho(
        m, T, method, rho, cbar_star, cv5, seed, r_eval=r_eval,
    )
    return m, T, method, rho, size_emp, power_emp, time.time() - t0


def _recal_worker(m, T, method, rho, r_recal_cv, r_recal_pow, cbar_grid):
    seed = config_seed(m, T, rho, method) + 999
    t0 = time.time()
    cbar_ar, power_ar = recalibrate_ar1(
        m, T, method, rho, seed, r_recal_cv=r_recal_cv,
        r_recal_pow=r_recal_pow, cbar_grid=cbar_grid,
    )
    return m, T, method, rho, cbar_ar, power_ar, time.time() - t0


def _run_parallel(jobs, n_jobs, desc):
    """jobs: lista de tuplas (worker_fn, args). Executa em paralelo via joblib
    (backend 'loky', baseado em processos) se disponível; senão, sequencial."""
    if _HAS_JOBLIB and n_jobs != 1:
        print(f"  [joblib] {len(jobs)} tarefas em '{desc}', n_jobs={n_jobs}")
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(fn)(*a) for fn, a in jobs
        )
    if not _HAS_JOBLIB:
        print(f"  [aviso] joblib não encontrado -- '{desc}' roda em 1 núcleo. "
              f"Instale com 'pip install joblib' para paralelizar.")
    return [fn(*a) for fn, a in jobs]


# =============================================================================
# 6. LOOP PRINCIPAL
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Robustez realista a AR(1): size/power + recalibração")
    ap.add_argument("--quick", action="store_true", help="checagem rápida (poucas réplicas, grade pequena)")
    ap.add_argument("--recalibrate", action="store_true", help="também roda o experimento (3), mais caro")
    ap.add_argument("--recal-rho", type=float, nargs="+", default=RECAL_RHO_DEFAULT)
    ap.add_argument("--recal-method", choices=["const", "maic"], nargs="+", default=RECAL_METHOD_DEFAULT)
    ap.add_argument("--n-jobs", type=int, default=-1,
                     help="núcleos para joblib (-1 = todos disponíveis; 1 = sequencial). Default: -1.")
    ap.add_argument("--t-list", type=int, nargs="+", default=None,
                     help="restringe os T rodados (ex.: --t-list 30) -- útil para reexecutar só "
                          "configs específicas sem refazer a grade completa.")
    ap.add_argument("--m-list", type=int, nargs="+", default=None,
                     help="restringe os m rodados (ex.: --m-list 0 1 2).")
    ap.add_argument("--methods", choices=["const", "maic"], nargs="+", default=None,
                     help="restringe os métodos de s²_AR rodados (ex.: --methods maic).")
    args = ap.parse_args()

    m_list, t_list, rho_list, methods = M_LIST, T_LIST, RHO_LIST, SIGMA2_METHODS
    cbar_grid = CBAR_GRID
    r_anchor_cv, r_anchor_pow = R_ANCHOR_CV, R_ANCHOR_POW
    r_eval = R_EVAL
    r_recal_cv, r_recal_pow = R_RECAL_CV, R_RECAL_POW
    if args.quick:
        m_list, t_list, rho_list, methods = [0, 1], [60], [0.5], ["const"]
        r_anchor_cv = r_anchor_pow = 500
        r_eval = 500
        r_recal_cv = r_recal_pow = 300
        cbar_grid = np.round(np.arange(-12.0, -4.9, 1.0), 2)
        print("[quick] grade/réplicas reduzidas -- só para checar que tudo roda.\n")

    # Filtros de reexecução pontual (aplicados DEPOIS do --quick, então combinam
    # com ele se necessário). Não afetam rho_list -- todos os rho são sempre
    # recomputados para as configs (m,T,método) selecionadas.
    if args.t_list is not None:
        t_list = args.t_list
    if args.m_list is not None:
        m_list = args.m_list
    if args.methods is not None:
        methods = args.methods

    import multiprocessing as _mp
    ncpu = _mp.cpu_count()
    eff_jobs = ncpu if args.n_jobs == -1 else args.n_jobs
    print(f"Núcleos detectados na máquina: {ncpu}  |  n_jobs solicitado: {args.n_jobs} "
          f"({'todos' if args.n_jobs == -1 else eff_jobs} serão usados)"
          f"{'' if _HAS_JOBLIB else '  [joblib AUSENTE -> execução sequencial, 1 núcleo]'}")

    t0 = time.time()
    anchors = {}     # (m,T,method) -> (cbar*, cv5, power_iid)
    sp_results = {}  # (m,T,method,rho) -> (size, power)

    print("\n=== Experimento 0: âncora sob i.i.d. ===")
    anchor_jobs = [
        (_anchor_worker, (m, T, method, r_anchor_cv, r_anchor_pow, cbar_grid))
        for m in m_list for T in t_list for method in methods
    ]
    for m, T, method, cbar_star, cv5, power_iid, widened, dt in _run_parallel(
        anchor_jobs, args.n_jobs, "âncora"
    ):
        anchors[(m, T, method)] = (cbar_star, cv5, power_iid, widened)
        flag = "  [marcada: âncora widened, ver footnote]" if widened else ""
        print(f"  m={m} T={T:>3} {method:>5}: cbar*={cbar_star:>6.1f} "
              f"CV5%={cv5:>7.3f} power={power_iid:.3f}  ({dt:.1f}s){flag}")

    print("\n=== Experimentos 1+2: size e power sob AR(1) (sar estimado) ===")
    sp_jobs = [
        (_sp_worker, (m, T, method, rho, anchors[(m, T, method)][0],
                      anchors[(m, T, method)][1], r_eval))
        for (m, T, method), (cbar_star, cv5, power_iid, _w) in anchors.items()
        if np.isfinite(cbar_star)
        for rho in rho_list
    ]
    for m, T, method, rho, size_emp, power_emp, dt in _run_parallel(
        sp_jobs, args.n_jobs, "size/power"
    ):
        sp_results[(m, T, method, rho)] = (size_emp, power_emp)
        power_iid = anchors[(m, T, method)][2]
        print(f"  m={m} T={T:>3} {method:>5} rho={rho:.1f}: "
              f"size={size_emp:.3f} (nom. 0.05)  power={power_emp:.3f} "
              f"(âncora {power_iid:.3f})  ({dt:.1f}s)")

    # ---- CSV: size/power ----
    with open("ar1_size_power.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "T", "method", "rho", "cbar_iid", "cv5_iid",
                     "power_iid_anchor", "size_emp", "power_emp", "anchor_widened"])
        for (m, T, method, rho), (size_emp, power_emp) in sp_results.items():
            cbar_star, cv5, power_iid, widened = anchors[(m, T, method)]
            w.writerow([m, T, method, rho, cbar_star, f"{cv5:.4f}",
                        f"{power_iid:.4f}", f"{size_emp:.4f}", f"{power_emp:.4f}",
                        int(widened > 0)])
    print("\nGravado: ar1_size_power.csv")

    # ---- Tabela LaTeX: foco em rho=0.5 (o pedido explícito do Referee 1) ----
    rho_focus = 0.5 if 0.5 in rho_list else rho_list[len(rho_list) // 2]
    with open("tab_ar1_sizepower.tex", "w") as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write(
            "\\caption{Empirical size and power of $P_T$ at $\\rho=" + f"{rho_focus:g}" +
            "$ AR(1) serial correlation, using the i.i.d.-calibrated $\\bar c$ and "
            "$5\\%$ critical value (no recalibration), with $s^2_{\\mathrm{AR}}$ "
            "\\emph{estimated} (not oracle) by the difference-based ('const') and "
            "autoregressive MAIC ('maic') methods. Size should be near the nominal "
            "$0.05$; power should be compared to the i.i.d.\\ anchor of $\\approx0.50$.}"
            "\n\\label{tab:ar1-sizepower}\n"
        )
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("& & \\multicolumn{2}{c}{const} & \\multicolumn{2}{c}{maic} & \\\\\n")
        f.write("$T$ & $m$ & size & power & size & power & power$_{\\mathrm{iid}}$ \\\\\n\\midrule\n")
        any_flagged = False
        for T in t_list:
            for m in m_list:
                row = [str(T), str(m)]
                p_iid = None
                for method in ["const", "maic"]:
                    anc = anchors.get((m, T, method))
                    flagged = anc is not None and len(anc) > 3 and anc[3] > 0
                    if flagged:
                        any_flagged = True
                    if anc is not None and (m, T, method, rho_focus) in sp_results:
                        s_e, p_e = sp_results[(m, T, method, rho_focus)]
                        mark = "^\\dagger" if flagged else ""
                        row += [f"${s_e:.3f}{mark}$", f"${p_e:.3f}{mark}$"]
                        p_iid = anc[2]
                    else:
                        row += ["--", "--"]
                row.append(f"{p_iid:.3f}" if p_iid is not None else "--")
                f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
        excl_note = (
            "$\\dagger$~The i.i.d.\\ MAIC anchor for this cell required widening "
            "the calibration grid beyond $\\pm4$ to locate the tangency, a known "
            "short-$T$ symptom of MAIC's small-sample instability; the entry is "
            "the genuine empirical outcome of that calibration and is reported "
            "for completeness, but the underlying $\\bar c_{\\mathrm{iid}}$ "
            "(see the accompanying CSV) is itself poorly identified, so the cell "
            "should be read with corresponding caution rather than excluded. " if any_flagged else ""
        )
        f.write(
            "\\par\\medskip\\footnotesize Entries from $R=" + f"{r_eval}" +
            "$ replications; Monte Carlo standard error of a proportion near "
            "$0.5$ is approx.\\ $\\sqrt{0.25/R}$. The i.i.d.\\ anchor (`power$_{"
            "\\mathrm{iid}}$') is the power at $\\rho=0$ used to calibrate "
            "$\\bar c$ and the critical value with the same estimator. " + excl_note +
            "\n\\end{table}\n"
        )
    print("Gravado: tab_ar1_sizepower.tex")

    # ---- Experimento 3 (opcional): recalibração ----
    if args.recalibrate:
        print("\n=== Experimento 3: recalibração completa sob AR(1) (mais caro) ===")
        recal_results = {}
        recal_jobs = [
            (_recal_worker, (m, T, method, rho, r_recal_cv, r_recal_pow, cbar_grid))
            for m in m_list for T in t_list
            for method in args.recal_method for rho in args.recal_rho
        ]
        for m, T, method, rho, cbar_ar, power_ar, dt in _run_parallel(
            recal_jobs, args.n_jobs, "recalibração"
        ):
            cbar_iid = anchors.get((m, T, method), (np.nan,))[0]
            recal_results[(m, T, method, rho)] = (cbar_iid, cbar_ar, power_ar)
            print(f"  m={m} T={T:>3} {method:>5} rho={rho:.1f}: "
                  f"cbar_iid={cbar_iid:.1f}  cbar_AR={cbar_ar:.1f}  "
                  f"(Δ={cbar_ar-cbar_iid:+.1f})  power={power_ar:.3f}  ({dt:.1f}s)")

        with open("ar1_recalibration.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["m", "T", "method", "rho", "cbar_iid", "cbar_AR", "delta", "power_at_cbar_AR"])
            for (m, T, method, rho), (cbar_iid, cbar_ar, power_ar) in recal_results.items():
                w.writerow([m, T, method, rho, cbar_iid, cbar_ar,
                            f"{cbar_ar-cbar_iid:+.1f}", f"{power_ar:.4f}"])
        print("Gravado: ar1_recalibration.csv")

        with open("tab_ar1_recalibration.tex", "w") as f:
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write(
                "\\caption{Recalibration under genuine AR(1) dependence: the "
                "optimal $\\bar c$ found by repeating the full tangency search "
                "with $s^2_{\\mathrm{AR}}$ estimated under $\\rho=0.5$ "
                "(`$\\bar c_{\\mathrm{AR}}$'), against the i.i.d.-calibrated "
                "value (`$\\bar c_{\\mathrm{iid}}$').}\n\\label{tab:ar1-recal}\n"
            )
            f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
            f.write("$T$ & $m$ & $\\bar c_{\\mathrm{iid}}$ & $\\bar c_{\\mathrm{AR}}$ & $\\Delta$ \\\\\n\\midrule\n")
            for (m, T, method, rho), (cbar_iid, cbar_ar, power_ar) in recal_results.items():
                f.write(f"{T} & {m} & {cbar_iid:.1f} & {cbar_ar:.1f} & "
                        f"{cbar_ar-cbar_iid:+.1f} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
        print("Gravado: tab_ar1_recalibration.tex")

    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
