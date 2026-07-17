# RELEASE v1.2.0 — checklist e processo de commit

Processo idêntico ao precedente v1.1.9 (commit -> tag -> push -> Zenodo ->
registrar DOI), com os conteúdos desta versão. Manuscrito pareado: v73.

## 0. O que mudou nesta versão (para o CHANGELOG)

- `boot_ppp_cbar.py`:
  * `pac1` 0.4 (legado de diagnóstico de 4 lags, fora de toda a faixa
    empírica) -> **0.27** (mediana do coeficiente ADF de 1ª ordem das 8
    séries; CLI `--pac1`, guarda de sobrescrita);
  * bloco empírico: CV config-faithful agora **sieve-own** por default
    (φ̂ ADF da própria moeda, fixo entre réplicas, estacionariedade
    verificada); `--common-nuisance` reproduz o design de família;
  * coluna `pvalue_cf` (p-valor empírico) e `nuisance` (proveniência φ̂)
    em `ppp_empirical.csv`;
  * `--bp` (célula λ-exata no modo `--grid`), `--seed-base` (jitter),
    `--pac-hw`; pac1/pac_hw gravados em `meta.json`.
- NOVO `pac_diagnostic.py` -> `ppp_pac_diagnostic.csv` (proveniência do
  0.27: PACF e γ ADF por moeda).
- `dependence_bound.py` / `dependence_count_pmf.py`: poderes 0.30/0.31 ->
  0.48/0.45; P(0|H1): indep 0.055 -> 0.0063; dependência 0.21-0.23 ->
  0.086-0.098 (moda k=3, não mais zero).
- Artefatos `boot_out/` regenerados (superfície pac1=0.27 + empírico
  sieve-own com p-valores). Células p=0 reproduzem v1.1.9 bit a bit
  (validação: são pac1-free).

## 1. Sensibilidades a arquivar (o manuscrito v73 cita "archived in the
##    replication package" — estes artefatos SÃO a citação)

Criar `boot_out/sensitivity/` e mover para lá os CSVs empíricos de:
- `boot_sens_common/empirical/`  -> `ppp_empirical_common027.csv`
  (design de família comum 0.27; sustenta o contraste AUD vs grupo-1985
  do parágrafo 1 de sec:pppverdicts: -2.12 vs -2.00)
- `boot_pac_cad/empirical/`      -> `ppp_empirical_family037.csv`
- `boot_out_v119.../` (zip ref)  -> `ppp_empirical_legacy040.csv`
  (ou regenerar com --common-nuisance --pac1 0.4 --out proprio)
- `boot_j1/j2/j3/empirical/`     -> `ppp_empirical_seed<base>.csv`
  (jitter: sustenta a nota-† do CAD: spread do CV ~0.02, verdito binário
  alterna, p estável 0.049-0.050)
- células λ-exatas (--grid --bp 12 19): gravar o output de tela em
  `lambda_exact_bp12_19.md` (cbar* -10.839/-12.000, cv -2.112/-2.232,
  powF 0.450/0.410) — sustenta a nota 3b de sec:pppcalib.

Depois: remover os diretórios temporários `boot_smoke/`, `boot_j*/`,
`boot_sens_common/`, `boot_pac_cad/` (conteúdo já copiado).

## 2. Metadados de versão

- `CITATION.cff`: version 1.2.0; manter concept DOI 10.5281/zenodo.21229773.
- `.zenodo.json`: version 1.2.0; descrição curta das mudanças.
- `CHANGELOG.md`: entrada v1.2.0 (seção 0 acima).
- `README.md`: pac1=0.27 + proveniência; sieve-own default e
  --common-nuisance; pac_diagnostic.py; coluna pvalue_cf; --bp/--seed-base.
- `MC_vs_BOOTSTRAP.md`: Procedure 2 agora de fato "per currency" no CV
  (o código passou a cumprir o que o texto já prometia) — 1 parágrafo.

## 3. Comandos (precedente v1.1.9; executar do clone)

```powershell
cd C:\Users\SAMSUNG\OneDrive\Documents\Paper\Articles\SOLOW2\Mod_Teorico\rgs-model-LB-replication

git add -A
git status                       # conferir: sem temporarios, sem .venv
git commit -m "Release v1.2.0: pac1 recalibration (0.4->0.27, documented provenance), sieve-own config-faithful CVs, empirical p-values, lambda-exact cell mode, dependence-count update"
git tag v1.2.0
git push origin main --tags
```

Zenodo (webhook do GitHub minta a versão automaticamente no concept DOI
10.5281/zenodo.21229773):
1. Conferir o novo version DOI no registro Zenodo.
2. Registrar o version DOI em `CITATION.cff` (identifiers) e
   `CHANGELOG.md` ("Record v1.2.0 Zenodo version DOI"); commit + push
   (precedente: commit 36c6019 para v1.1.9).

## 4. Pós-release (manuscrito)

- O Data statement do manuscrito cita o CONCEPT DOI (auto-resolve) — sem
  edição obrigatória; opcional: atualizar a linha "the exact snapshot
  underlying this manuscript is release v1.1.9/DOI" para v1.2.0 + novo
  version DOI (1 edição no .tex, Statements and declarations).
- Gate final do v73 já verificado: 49 pp, 0 undefined, 0 overfull,
  0 multiply-defined.
