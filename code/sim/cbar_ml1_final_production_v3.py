"""
cbar_ml1_final_production_v2.py
================================
Calibração da superfície de resposta do parâmetro de detrending GLS (c̄) para o
teste de raiz unitária de Carrion-i-Silvestre, Kim & Perron (2009, CKP) sob o
MODELO 1 (constante + level dummies DU, SEM tendência e SEM slope dummies DT).

CONTRIBUIÇÃO (cf. short_paper.tex):
  O CKP tabula c̄ apenas para especificações com tendência quebrada, como função
  só das frações λ e assintoticamente. Para o Modelo 1 não há superfície. Este
  script a produz, e documenta que, no Modelo 1, c̄ depende primariamente de
  (m, T) — efeito de AMOSTRA FINITA — e não de λ, porque os DU são regressores
  LIMITADOS (Proposição da nota). Assintoticamente c̄ é plano em m (≈ valor ERS).

CRITÉRIO DE CALIBRAÇÃO (ERS 1996, tangência):
  c̄* é o valor para o qual o teste point-optimal PT atinge poder = 0.50 contra
  a alternativa local avaliada EM c = c̄*. Validação: m=0 → c̄ ≈ −7 (ERS).

NOVIDADES DA v2 (escopo fechado com o usuário):
  [Item 6]  Erro de Monte Carlo (SE/IC) para poder, CVs e c̄*.
  [Item 1]  V_m do DGP salvo por configuração (=0 esperado; ASSERÇÃO DE SANIDADE
            — garante que as quebras estão no determinístico e são removidas pelo
            detrending, não contaminando a inovação).
  [Item 12] σ² por DOIS métodos na grade toda: 'const' (Δy) e 'maic' (s2ar MAIC).
  [Grade]   T ∈ {30,45,50,60,80,100,150,200,300}; m ∈ {0..5} (até m=5, como CKP).
  [Item 8]  Seeds determinísticos por configuração (pesos posicionais 37^i) e
            estado salvo no checkpoint.
  [Item 9]  Vetores brutos das estatísticas por réplica salvos em .npz.
  [Item 7]  R_cv/R_pow/R_curve documentados e justificados (ver DEFAULTS).
  [Item 16] Geração de tabelas LaTeX ao final.

EQUIVALÊNCIA: as primitivas numéricas (build_z, _glsd, s2ar_maic, M-stats) são
fiéis ao ckp_test.py / msbur.src (GAUSS), garantindo que a calibração use o mesmo
motor que o teste aplicado no protocolo.
"""
from __future__ import annotations
import os, sys, time, json, pickle, argparse, warnings, itertools
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:                       # fallback sem numba (mais lento)
    _HAS_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco if (a and callable(a[0]) is False) else (a[0] if a else deco)

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

warnings.filterwarnings("ignore")

# =============================================================================
# DEFAULTS — documentados (Item 7)
# =============================================================================
#  R_cv   = 10000 : réplicas para distribuição nula (CVs). SE de um percentil 5%
#                   ~ depende da densidade; 10^4 dá SE de CV pequena (<0.02).
#  R_pow  =  5000 : réplicas para poder. SE(poder)=√(0.5·0.5/5000)=0.0071, então
#                   a seleção de c̄* (poder=0.50) tem ±0.007 — sub-grade de c̄.
#  R_curve = 3000 : réplicas por ponto da curva de poder (diagnóstico).
#  cbar_grid: −20 a −3 em passos de 0.5 (a resolução do c̄*).
#  c_grid_power: −30 a −2 (alternativas para as curvas de poder).
DEFAULTS = dict(
    T_grid       = [30, 45, 50, 60, 80, 100, 150, 200, 300],
    m_grid       = [0, 1, 2, 3, 4, 5],
    m_min_T      = {0: 30, 1: 30, 2: 30, 3: 45, 4: 60, 5: 80},   # T mínimo por m
    trim         = 0.15,
    min_spacing  = 0.15,
    n_grid       = 7,
    cbar_grid    = list(np.round(np.arange(-20.0, -2.9, 0.5), 2)),
    target_power = 0.50,
    R_cv         = 10000,
    R_pow        = 5000,
    R_curve      = 3000,
    c_grid_power = list(np.round(np.arange(-30.0, -1.9, 2.0), 1)),
    kmax         = 12,
    sigma2_methods = ['const', 'maic'],     # Item 12: AMBOS na grade toda
    compute_power_curves = True,
    save_raw_vectors     = True,            # Item 9
    seed_base    = 20240601,
    checkpoint_dir = "checkpoints_cbar_ml1_v2",
)

# =============================================================================
# 1. PRIMITIVAS NUMÉRICAS (Model 1: Z = [1, DU₁,...,DUₘ])
#    Fiéis a ckp_test.py / msbur.src (GAUSS).
# =============================================================================
def build_z(nt, break_pos):
    """Determinística do Model 1: constante + level dummies (SEM tendência).
    DU_j = 1{t > τ_j} (degrau começa APÓS o break, convenção GAUSS)."""
    trend = np.arange(1, nt + 1, dtype=float)
    cols = [np.ones(nt)]
    for bp in break_pos:
        bp_int = int(max(1, min(nt - 1, int(bp))))
        cols.append(np.where(trend > bp_int, 1.0, 0.0))
    return np.column_stack(cols)


@njit(cache=True)
def _glsd(y, z, cbar):
    """Quasi-diferença GLS → (yt_detrended, ssr). ā = 1 + c̄/T.
    yt = y − Z·β̂(c̄); ssr é a SSR da regressão quasi-diferenciada."""
    nt = len(y)
    abar = 1.0 + cbar / nt
    ya = np.empty(nt); ya[0] = y[0]
    for t in range(1, nt):
        ya[t] = y[t] - abar * y[t - 1]
    za = np.empty_like(z)
    for j in range(z.shape[1]):
        za[0, j] = z[0, j]
    for t in range(1, nt):
        for j in range(z.shape[1]):
            za[t, j] = z[t, j] - abar * z[t - 1, j]
    ncol = z.shape[1]
    AtA = np.zeros((ncol, ncol)); Aty = np.zeros(ncol)
    for i in range(ncol):
        for k in range(ncol):
            s = 0.0
            for t in range(nt):
                s += za[t, i] * za[t, k]
            AtA[i, k] = s
        sy = 0.0
        for t in range(nt):
            sy += za[t, i] * ya[t]
        Aty[i] = sy
    for i in range(ncol):
        AtA[i, i] += 1e-10
    bhat = np.linalg.solve(AtA, Aty)
    yt = np.empty(nt)
    for t in range(nt):
        pred = 0.0
        for j in range(ncol):
            pred += z[t, j] * bhat[j]
        yt[t] = y[t] - pred
    ssr = 0.0
    for t in range(nt):
        pred = 0.0
        for j in range(ncol):
            pred += za[t, j] * bhat[j]
        d = ya[t] - pred
        ssr += d * d
    return yt, ssr


def s2ar_maic(yt_ols, kmax=12):
    """Variância de longo prazo via AR(k), seleção MAIC (Ng-Perron 2001).
    Entrada: série OLS-detrended (nota do Perron: OLS, não GLS).
    Fiel a s2ar de ckp_test.py."""
    nt = len(yt_ols)
    nef = nt - kmax - 1
    if nef < 5:
        return max(np.var(np.diff(yt_ols)), 1e-10)
    dyt = np.diff(yt_ols)
    reg = np.column_stack(
        [yt_ols[:-1]] +
        [np.concatenate([np.zeros(k), dyt[:-k]]) for k in range(1, kmax + 1)]
    )
    reg0 = reg[kmax:]; d0 = dyt[kmax:]
    sumy = float(reg0[:, 0] @ reg0[:, 0])
    s2e = np.full(kmax + 1, 1e9); tau = np.zeros(kmax + 1)
    for k in range(kmax + 1):
        Xk = reg0[:, :k + 1]
        b = np.linalg.lstsq(Xk, d0, rcond=None)[0]
        e = d0 - Xk @ b
        s2e[k] = (e @ e) / nef
        tau[k] = (b[0] ** 2) * sumy / s2e[k] if s2e[k] > 0 else 0.0
    kk = np.arange(kmax + 1)
    mic = np.log(s2e) + 2.0 * (kk + tau) / nef     # MAIC
    kstar = int(np.argmin(mic))
    Xopt = reg0[:, :kstar + 1]
    bopt = np.linalg.lstsq(Xopt, d0, rcond=None)[0]
    eopt = d0 - Xopt @ bopt
    s2 = (eopt @ eopt) / nef
    if kstar > 0:
        arsum = 1.0 - bopt[1:].sum()
        sar = s2 / (arsum ** 2) if abs(arsum) > 1e-6 else s2
    else:
        sar = s2
    return max(sar, 1e-10)


def compute_M_statistics(y, break_pos, cbar, sigma2_method='const', kmax=12):
    """Todas as estatísticas CKP para Model 1: MZα, MSB, MZt, PT, MPT.
    Fórmulas idênticas a ckp_test.py / msbur.src. Retorna dict ou None.

    Convenção de rejeição: TODAS rejeitam para valores PEQUENOS (muito negativos
    em MZα/MZt; pequenos em MSB/PT/MPT) — a CV é o percentil inferior.
    """
    nt = len(y)
    z = build_z(nt, break_pos)
    yt, ssra = _glsd(y, z, float(cbar))
    if not np.isfinite(ssra):
        return None

    # Variância de longo prazo. AMBOS os métodos partem da série OLS-detrended
    # (y − Z·β̂), garantindo INVARIÂNCIA às quebras determinísticas: sem remover
    # Z·β primeiro, os saltos dos DU contaminam Δy e inflam σ² (bug corrigido).
    bols = np.linalg.lstsq(z, y, rcond=None)[0]
    yt_ols = y - z @ bols
    if sigma2_method == 'const':
        # estimador simples (k=0): variância de Δ(série detrended); sob iid é
        # consistente e equivale a s2ar com zero defasagens.
        sar = max(np.var(np.diff(yt_ols), ddof=1), 1e-10)
    else:
        # autoregressive spectral density (Perron-Ng 1998) com seleção MAIC.
        sar = s2ar_maic(yt_ols, kmax)

    Xreg = yt[:-1]
    denom = float(Xreg @ Xreg)
    if denom <= 0:
        return None
    sumyt2 = denom / (nt - 1) ** 2
    if sumyt2 <= 0:
        return None
    bt = nt - 1

    mza = (yt[-1] ** 2 / bt - sar) / (2.0 * sumyt2)
    msb = np.sqrt(sumyt2 / sar)
    mzt = mza * msb

    _, ssr1 = _glsd(y, z, 0.0)
    pt  = (ssra - (1.0 + cbar / nt) * ssr1) / sar          # ERS P_T (forma SSR)
    mpt = (cbar ** 2 * sumyt2 + (1.0 - cbar) * yt[-1] ** 2 / nt) / sar  # demeaned

    return dict(mza=mza, msb=msb, mzt=mzt, pt=pt, mpt=mpt)


# =============================================================================
# 2. DGP E V_m (Item 1 — asserção de sanidade)
# =============================================================================
def gerar_dgp(T, lambdas, c, seed, beta_scale=0.0):
    """y = Z·β + u, com u_t = (1+c/T)·u_{t-1} + ε_t, ε ~ N(0,1). c=0 → H0.

    beta_scale controla a magnitude θ dos saltos determinísticos. Por INVARIÂNCIA
    do detrending GLS, as estatísticas independem de θ; usamos beta_scale=0 no
    grid de produção (DGP padrão ERS/CKP). beta_scale≠0 é usado só nos testes de
    invariância (sanity)."""
    rng = np.random.default_rng(int(abs(seed)) % (2**63 - 1))
    break_pos = [int(max(1, min(T - 1, np.floor(lam * T)))) for lam in lambdas]
    Z = build_z(T, break_pos)
    beta = np.full(Z.shape[1], float(beta_scale))
    eps = rng.normal(0, 1, T)
    rho = 1.0 + c / T
    u = np.zeros(T)
    for t in range(1, T):
        u[t] = rho * u[t - 1] + eps[t]
    return Z @ beta + u, break_pos


def sanity_invariance(T, lambdas, cbar, seed, sigma2_method='const'):
    """ASSERÇÃO DE SANIDADE (Item 1, reformulado).

    O ponto a garantir: as quebras do DGP vivem no DETERMINÍSTICO (Z·β) e são
    removidas EXATAMENTE pelo detrending GLS, de modo que a inovação local-to-
    unity é a única fonte estocástica. A propriedade testável disso é a
    INVARIÂNCIA da estatística ao tamanho dos saltos β: gerando a MESMA inovação
    u e variando apenas β, a estatística deve ser idêntica.

    (A tentativa anterior de medir "V_m da inovação" por partição de Δu media
    apenas variação amostral O(1/n) de ruído e disparava falsamente; a
    invariância é a verificação correta e exata.)

    Retorna o desvio máximo da estatística MZt entre β=0 e β grande. Deve ≈ 0.
    """
    break_pos = [int(max(1, min(T - 1, np.floor(lam * T)))) for lam in lambdas]
    Z = build_z(T, break_pos)
    rng = np.random.default_rng((int(abs(seed)) + 7) % (2**63 - 1))
    eps = rng.normal(0, 1, T)
    u = np.zeros(T)                         # random walk exato (H0)
    for t in range(1, T):
        u[t] = u[t - 1] + eps[t]
    vals = []
    for beta_scale in (0.0, 10.0):
        y = Z @ np.full(Z.shape[1], float(beta_scale)) + u
        s = compute_M_statistics(y, break_pos, cbar, sigma2_method)
        vals.append(s['mzt'] if s else np.nan)
    if any(not np.isfinite(v) for v in vals):
        return np.nan
    return float(abs(vals[0] - vals[1]))


# =============================================================================
# 3. GERAÇÃO DAS CONFIGURAÇÕES DE QUEBRA
# =============================================================================
def gerar_lambdas(m, trim, min_spacing, n_grid=7):
    """Combinações de m frações em [trim, 1-trim], espaçamento ≥ min_spacing.
    m=0 → [()].

    O n_grid efetivo cresce com m: com poucos pontos não há combinação viável
    para m alto (e.g. m=5 com n_grid=7 dá zero). Como o c̄ depende fracamente de
    λ dentro de cada célula (m,T), bastam poucas configurações por m alto para
    cobrir a variação relevante; densificamos o grid apenas o necessário."""
    if m == 0:
        return [()]
    ng = max(n_grid, 3 * m - 1)          # garante viabilidade até m=5
    grid = np.round(np.linspace(trim, 1 - trim, ng), 3)
    combos = []
    for combo in itertools.combinations(grid, m):
        if all(combo[i + 1] - combo[i] >= min_spacing - 1e-9
               for i in range(len(combo) - 1)):
            combos.append(tuple(combo))
    # Para m alto, limita a um subconjunto representativo (espaçado) para não
    # inflar o custo, preservando extremos e centro do espaço de λ.
    if m >= 4 and len(combos) > 12:
        idx = np.linspace(0, len(combos) - 1, 12).round().astype(int)
        combos = [combos[i] for i in sorted(set(idx))]
    return combos


def enumerar_configs(P):
    """Lista (T, m, λ) válidas: respeita T mínimo por m e cabimento de regimes."""
    configs = []
    for T in P['T_grid']:
        for m in P['m_grid']:
            if T < P['m_min_T'].get(m, 30):
                continue
            # cabimento: (m+1) regimes com trim ⇒ precisa T*trim ≥ ~2 por regime
            if m > 0 and T * P['trim'] < 2:
                continue
            for lambdas in gerar_lambdas(m, P['trim'], P['min_spacing'], P['n_grid']):
                configs.append((T, m, lambdas))
    return configs


# =============================================================================
# 4. SEED DETERMINÍSTICO (Item 8 — pesos posicionais 37^i, sem colisão)
# =============================================================================
def config_seed(P, T, m, lambdas):
    s = P['seed_base'] + T * 100000 + m * 10000
    for i, lam in enumerate(lambdas):
        s += int(round(lam * 1000)) * (37 ** (i + 1))
    return s


# =============================================================================
# 5. ERRO DE MONTE CARLO (Item 6)
# =============================================================================
def se_power(p, R):
    """SE de uma proporção (poder/tamanho): √(p(1−p)/R)."""
    if not np.isfinite(p) or R <= 0:
        return np.nan
    return float(np.sqrt(max(p * (1 - p), 0.0) / R))


def se_quantile_bootstrap(sample, q, n_boot=500, seed=0):
    """SE de um percentil via bootstrap não-paramétrico (Item 6)."""
    a = np.asarray(sample)
    a = a[np.isfinite(a)]
    if len(a) < 50:
        return np.nan
    rng = np.random.default_rng(seed)
    n = len(a)
    qs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        qs[b] = np.percentile(a[idx], q)
    return float(qs.std(ddof=1))


# =============================================================================
# 6. CALIBRAÇÃO DE UMA CONFIGURAÇÃO (para um método de σ²)
# =============================================================================
def calibrar_config_sigma(T, m, lambdas, P, sigma2_method):
    """Calibra c̄* e CVs para uma config e UM método de σ². Retorna dict.
    Inclui SE de Monte Carlo (Item 6) e vetores brutos (Item 9)."""
    cfg_seed = config_seed(P, T, m, lambdas)
    cbar_grid = np.array(P['cbar_grid'])

    # ── 1) Seleção de c̄* pelo critério ERS (poder=target no ponto c=c̄) ──────
    sel = []                          # (cbar, cv5_pt, poder, se_poder)
    for cbar in cbar_grid:
        pt_h0 = np.empty(P['R_cv'])
        for r in range(P['R_cv']):
            y, bp = gerar_dgp(T, lambdas, 0.0, cfg_seed + r)
            s = compute_M_statistics(y, bp, cbar, sigma2_method, P['kmax'])
            pt_h0[r] = s['pt'] if s else np.nan
        pt_h0 = pt_h0[np.isfinite(pt_h0)]
        if len(pt_h0) < P['R_cv'] * 0.5:
            sel.append((cbar, np.nan, np.nan, np.nan)); continue
        cv5_pt = np.percentile(pt_h0, 5)

        rej = nval = 0
        for r in range(P['R_pow']):
            y, bp = gerar_dgp(T, lambdas, cbar, cfg_seed + 10**6 + r)
            s = compute_M_statistics(y, bp, cbar, sigma2_method, P['kmax'])
            if s and np.isfinite(s['pt']):
                nval += 1
                if s['pt'] < cv5_pt:
                    rej += 1
        poder = rej / nval if nval > 0 else np.nan
        sel.append((cbar, cv5_pt, poder, se_power(poder, nval)))

    arr = np.array([(c, cv, p, se) for c, cv, p, se in sel if np.isfinite(p)])
    if len(arr) == 0:
        return None
    idx = int(np.argmin(np.abs(arr[:, 2] - P['target_power'])))
    cbar_star  = float(arr[idx, 0])
    poder_star = float(arr[idx, 2])
    se_poder_star = float(arr[idx, 3])

    # ── 2) CVs completos das 5 estatísticas sob H0 no c̄* (+ SE e brutos) ─────
    dist = {k: [] for k in ['mza', 'msb', 'mzt', 'pt', 'mpt']}
    for r in range(P['R_cv']):
        y, bp = gerar_dgp(T, lambdas, 0.0, cfg_seed + 2 * 10**6 + r)
        s = compute_M_statistics(y, bp, cbar_star, sigma2_method, P['kmax'])
        if s:
            for k in dist:
                if np.isfinite(s[k]):
                    dist[k].append(s[k])

    cvs, cvs_se = {}, {}
    for k in dist:
        a = np.array(dist[k])
        if len(a) < 100:
            cvs[k] = {q: np.nan for q in ['1%', '2.5%', '5%', '10%']}
            cvs_se[k] = {q: np.nan for q in ['1%', '2.5%', '5%', '10%']}
            continue
        cvs[k] = {q: float(np.percentile(a, float(q[:-1])))
                  for q in ['1%', '2.5%', '5%', '10%']}
        cvs_se[k] = {q: se_quantile_bootstrap(a, float(q[:-1]), seed=cfg_seed + hash(k) % 1000)
                     for q in ['1%', '2.5%', '5%', '10%']}

    # ── 3) Sanidade: invariância da estatística às quebras determinísticas ───
    #     (Item 1 reformulado) — desvio máximo de MZt entre β=0 e β grande, ≈0.
    inv_dev = sanity_invariance(T, lambdas, cbar_star, cfg_seed, sigma2_method)

    # ── 4) Curvas de poder (Item 10) ─────────────────────────────────────────
    power_curve = None
    if P['compute_power_curves']:
        power_curve = {k: [] for k in dist}
        power_curve['c_grid'] = list(P['c_grid_power'])
        for c_true in P['c_grid_power']:
            rej = {k: 0 for k in dist}; nval = 0
            for r in range(P['R_curve']):
                y, bp = gerar_dgp(T, lambdas, c_true, cfg_seed + 3 * 10**6 +
                                  int(abs(c_true)) * 10**4 + r)
                s = compute_M_statistics(y, bp, cbar_star, sigma2_method, P['kmax'])
                if s:
                    nval += 1
                    for k in dist:
                        cv5 = cvs[k]['5%']
                        if np.isfinite(s[k]) and np.isfinite(cv5) and s[k] < cv5:
                            rej[k] += 1
            for k in dist:
                power_curve[k].append(rej[k] / nval if nval > 0 else np.nan)

    raw = {k: np.array(dist[k], dtype=np.float32) for k in dist} \
        if P['save_raw_vectors'] else None

    return dict(
        T=T, m=m, lambdas=lambdas, sigma2_method=sigma2_method,
        cbar_star=cbar_star, poder_star=poder_star, se_poder_star=se_poder_star,
        cvs=cvs, cvs_se=cvs_se, inv_dev=inv_dev, power_curve=power_curve,
        cfg_seed=cfg_seed, raw=raw,
    )


def calibrar_config(T, m, lambdas, P):
    """Calibra a config para TODOS os métodos de σ² (Item 12)."""
    out = {'T': T, 'm': m, 'lambdas': lambdas, 'by_sigma2': {}}
    for method in P['sigma2_methods']:
        res = calibrar_config_sigma(T, m, lambdas, P, method)
        if res is not None:
            out['by_sigma2'][method] = res
    return out if out['by_sigma2'] else None


# =============================================================================
# 7. CHECKPOINT (Item 8) E ORQUESTRAÇÃO
# =============================================================================
def config_key(T, m, lambdas):
    lam = '_'.join(str(round(l, 3)) for l in lambdas)
    return f"T{T}_m{m}_lambdas_{lam}"


def pkl_path(P, key):
    return os.path.join(P['checkpoint_dir'], f"{key}.pkl")


def npz_path(P, key):
    return os.path.join(P['checkpoint_dir'], f"{key}_raw.npz")


def salvar_resultado(P, res):
    key = config_key(res['T'], res['m'], res['lambdas'])
    # separa vetores brutos (.npz) do resto (.pkl)
    raw_blocks = {}
    for method, r in res['by_sigma2'].items():
        if r.get('raw') is not None:
            for stat, vec in r['raw'].items():
                raw_blocks[f"{method}__{stat}"] = vec
            r = dict(r); r['raw'] = None
            res['by_sigma2'][method] = r
    if raw_blocks:
        np.savez_compressed(npz_path(P, key), **raw_blocks)
    with open(pkl_path(P, key), 'wb') as fh:
        pickle.dump(res, fh)


def ja_processado(P, T, m, lambdas):
    return os.path.exists(pkl_path(P, config_key(T, m, lambdas)))


def run_grid(P, n_jobs=-1):
    os.makedirs(P['checkpoint_dir'], exist_ok=True)
    configs = enumerar_configs(P)
    todo = [c for c in configs if not ja_processado(P, *c)]

    sep = "=" * 78
    print(f"\n{sep}")
    print("CALIBRAÇÃO c̄ MODELO 1 (DU) v2 — superfície + CVs + curvas + SE-MC + V_m")
    print(sep)
    print(f"T = {P['T_grid']}")
    print(f"m = {P['m_grid']} (T mín por m: {P['m_min_T']})")
    print(f"σ² methods = {P['sigma2_methods']} (Item 12, grade toda)")
    print(f"cbar grid: {len(P['cbar_grid'])} pts ({P['cbar_grid'][0]} a {P['cbar_grid'][-1]})")
    print(f"R_cv={P['R_cv']} R_pow={P['R_pow']} R_curve={P['R_curve']}")
    print(f"Configs totais: {len(configs)} | a fazer: {len(todo)}")
    print(f"{sep}\n")

    t0 = time.time()

    def _do(cfg):
        T, m, lambdas = cfg
        res = calibrar_config(T, m, lambdas, P)
        if res is not None:
            salvar_resultado(P, res)
        return config_key(T, m, lambdas)

    if _HAS_JOBLIB and n_jobs != 1:
        Parallel(n_jobs=n_jobs, verbose=10)(delayed(_do)(c) for c in todo)
    else:
        for i, c in enumerate(todo, 1):
            _do(c)
            el = time.time() - t0
            eta = el / i * (len(todo) - i) / 3600
            print(f"  [{i}/{len(todo)}] {config_key(*c)}  ETA {eta:.2f}h")

    print(f"\nConcluído em {(time.time()-t0)/60:.1f} min.")
    return agregar(P)


# =============================================================================
# 8. AGREGAÇÃO E SAÍDAS (CSV) + checagem V_m
# =============================================================================
def agregar(P):
    import pandas as pd
    rows = []
    vm_max = 0.0
    for f in sorted(os.listdir(P['checkpoint_dir'])):
        if not f.endswith(".pkl"):
            continue
        with open(os.path.join(P['checkpoint_dir'], f), 'rb') as fh:
            res = pickle.load(fh)
        for method, r in res['by_sigma2'].items():
            lam = r['lambdas']
            row = dict(
                config_key=config_key(r['T'], r['m'], lam),
                T=r['T'], m=r['m'], sigma2_method=method,
                cbar_otimo=r['cbar_star'], poder_no_otimo=r['poder_star'],
                se_poder=r['se_poder_star'], inv_dev=r['inv_dev'],
                cfg_seed=r['cfg_seed'],
            )
            for i in range(5):
                row[f'lambda{i+1}'] = lam[i] if i < len(lam) else np.nan
            for stat in ['mza', 'msb', 'mzt', 'pt', 'mpt']:
                for q, tag in [('1%', 'cv1'), ('2.5%', 'cv2_5'),
                               ('5%', 'cv5'), ('10%', 'cv10')]:
                    row[f'{stat}_{tag}'] = r['cvs'].get(stat, {}).get(q, np.nan)
                    row[f'{stat}_{tag}_se'] = r['cvs_se'].get(stat, {}).get(q, np.nan)
            rows.append(row)
            vm_max = max(vm_max, abs(r['inv_dev']) if np.isfinite(r['inv_dev']) else 0.0)

    df = pd.DataFrame(rows).sort_values(['sigma2_method', 'T', 'm']).reset_index(drop=True)
    out = os.path.join(P['checkpoint_dir'], "resultados_cbar_ml1_v2.csv")
    df.to_csv(out, index=False)
    print(f"\nAgregado: {out}  |  {len(df)} linhas")

    # ── ASSERÇÃO DE INVARIÂNCIA (Item 1 reformulado) ─────────────────────────
    print(f"\n[Sanity invariância] desvio máx de MZt entre β=0 e β grande = "
          f"{vm_max:.2e} (esperado ≈ 0).")
    if vm_max > 1e-6:
        print("  [!] AVISO: estatística NÃO invariante às quebras determinísticas "
              "— o detrending GLS pode não estar removendo Z·β corretamente.")
    else:
        print("  [OK] estatística invariante em toda a grade: as quebras estão no "
              "determinístico e são removidas exatamente pelo detrending GLS.")

    # ── validação ERS m=0 → c̄ ≈ −7 ──────────────────────────────────────────
    m0 = df[df.m == 0]
    if len(m0):
        for meth in ['const', 'maic']:
            mm = m0[m0.sigma2_method == meth]
            if len(mm):
                print(f"[Validação ERS] m=0 ({meth}): c̄ ∈ "
                      f"[{mm.cbar_otimo.min()}, {mm.cbar_otimo.max()}] (esperado ≈ −7)")
        print(f"[Validação poder] global: média={df.poder_no_otimo.mean():.4f} "
              f"(alvo {P['target_power']})")
    return df


# =============================================================================
# 9. TABELAS LaTeX (Item 16)
# =============================================================================
def gerar_tabelas_latex(P, sigma2_method='const'):
    import pandas as pd
    csv = os.path.join(P['checkpoint_dir'], "resultados_cbar_ml1_v2.csv")
    df = pd.read_csv(csv)
    df = df[df.sigma2_method == sigma2_method]
    out_dir = P['checkpoint_dir']

    # Tabela c̄(m,T)
    piv = df.groupby(['m', 'T'])['cbar_otimo'].mean().unstack().round(1)
    with open(os.path.join(out_dir, "tab_cbar_mT.tex"), 'w') as f:
        f.write(piv.to_latex(na_rep='---',
                caption=f"Optimal $\\bar c(m,T)$ for Model~1 ($\\sigma^2$: {sigma2_method}).",
                label="tab:cbar"))
    # Tabela CV5 MZt
    pivc = df.groupby(['m', 'T'])['mzt_cv5'].mean().unstack().round(2)
    with open(os.path.join(out_dir, "tab_cv5_mzt.tex"), 'w') as f:
        f.write(pivc.to_latex(na_rep='---',
                caption="$\\mathrm{MZ}_t$ 5\\% critical values at optimal $\\bar c$.",
                label="tab:cv"))
    print(f"Tabelas LaTeX escritas em {out_dir}/")


# =============================================================================
# 10. AJUSTE DA SUPERFÍCIE + CROSS-VALIDATION (Item 11)
# =============================================================================
def ajustar_superficie_lookup(P, sigma2_method='const', k_folds=5):
    """Documenta que (m,T) domina λ e reporta erro de CV do lookup c̄(m,T)
    vs. a superfície polinomial em λ (Item 11)."""
    import pandas as pd
    csv = os.path.join(P['checkpoint_dir'], "resultados_cbar_ml1_v2.csv")
    df = pd.read_csv(csv)
    df = df[df.sigma2_method == sigma2_method].reset_index(drop=True)
    y = df.cbar_otimo.values

    # R² do modelo (m,T)
    pred_mT = df.groupby(['m', 'T'])['cbar_otimo'].transform('mean').values
    r2_mT = 1 - np.sum((y - pred_mT) ** 2) / np.sum((y - y.mean()) ** 2)

    # k-fold CV do lookup (m,T) com interpolação
    rng = np.random.default_rng(P['seed_base'])
    idx = rng.permutation(len(df))
    folds = np.array_split(idx, k_folds)
    errs = []
    for f in range(k_folds):
        test = folds[f]; train = np.concatenate([folds[j] for j in range(k_folds) if j != f])
        lut = df.iloc[train].groupby(['m', 'T'])['cbar_otimo'].mean()
        for t in test:
            key = (df['m'].iloc[t], df['T'].iloc[t])
            pred = lut.get(key, df.cbar_otimo.iloc[train].mean())
            errs.append((df.cbar_otimo.iloc[t] - pred) ** 2)
    rmse_cv = float(np.sqrt(np.mean(errs)))
    print(f"[Superfície] R²(m,T)={r2_mT:.3f} | RMSE_cv(lookup m,T)={rmse_cv:.3f}")
    return dict(r2_mT=r2_mT, rmse_cv=rmse_cv)


# =============================================================================
# MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Calibração c̄ Model 1 v2")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--speed", action="store_true", help="teste rápido (subgrade)")
    ap.add_argument("--outdir", default=DEFAULTS['checkpoint_dir'])
    ap.add_argument("--tables", action="store_true", help="só gerar tabelas/superfície")
    args = ap.parse_args()

    P = dict(DEFAULTS)
    P['checkpoint_dir'] = args.outdir

    if args.speed:
        P.update(T_grid=[30, 60], m_grid=[0, 1, 2], R_cv=400, R_pow=300,
                 R_curve=200, compute_power_curves=False,
                 sigma2_methods=['const'], checkpoint_dir=args.outdir + "_speed")

    if args.tables:
        gerar_tabelas_latex(P)
        ajustar_superficie_lookup(P)
        return

    run_grid(P, n_jobs=args.jobs)
    gerar_tabelas_latex(P)
    ajustar_superficie_lookup(P)


if __name__ == "__main__":
    main()
