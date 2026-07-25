# RELEASE v1.3.0 — checklist e processo de commit

Processo idêntico ao precedente v1.2.0/v1.2.1 (commit -> tag -> push -> Zenodo ->
registrar DOI), com os conteúdos desta versão. Diferença em relação aos
precedentes: esta versão adiciona seis arquivos que **não existem no repositório
GitHub atual** (última tag `v1.2.1`) — vinham sendo desenvolvidos fora do
pacote e chegam aqui pela primeira vez.

## 0. O que mudou nesta versão (para o CHANGELOG)

Já escrito em `CHANGELOG.md`, seção `## [1.3.0] — 2026-07-25`. Resumo:

- **Novos** (não existem em v1.2.1): `ppp_two_axis_columns.py`,
  `modulus_numerics.py` (+ `modulus_table*.csv`), `overdetrend_power.py`
  (-> `Figure_1.pdf`), `materiality_figure.py` (-> `Figure_2.pdf`),
  `materiality_c7_vs_surface.py`, `assumption1_numerics.py`.
- **Renomeação de figuras** para casar com a numeração que o próprio LaTeX
  atribui: `Figure_overdetrend.pdf`->`Figure_1`, `Figure_size.pdf`->`Figure_2`,
  `Figure_4.pdf`(rer-series)->`Figure_3`, `Figure_1.pdf`(limiting
  density)->`Figure_S1`, `Figure_5.pdf`(HL forest)->`Figure_S2`. Duas figuras
  legadas (não citadas pelo manuscrito atual) arquivadas em `legacy_figures/`
  em vez de apagadas.
- **Tradução integral para inglês US** de `ppp_two_axis_columns.py` (quase
  todo em português) e `modulus_numerics.py` (inteiramente em português) —
  sem alteração de lógica, seed ou constante.
- **Bug real corrigido**: comparação de BIC em `fit_short_run` usava janelas
  amostrais de comprimento diferente por candidato `p` — corrigido para
  janela comum fixada em `pmax`.
- **Causa raiz do mismatch de `--validate` identificada e corrigida**: o `p`
  publicado vem de `ppp_ar_diagnostic.csv`'s `k_bic_cq` (produzido por
  `select_ar_order.py`, regressão estilo ADF com termo de nível), não de
  nenhuma recomputação interna. `ppp_two_axis_columns.py` agora lê `p` dessa
  coluna por padrão (`--ar-diagnostic`, casando com `boot_ppp_cbar.py`);
  `--recompute-p` restaura a busca de BIC independente antiga como modo de
  diagnóstico explícito. `--validate` agora reproduz o `p` publicado em 8 de
  8 moedas (antes 0 de 8).
- **Achado novo, ainda aberto**: o valor crítico simulado (`cv (III)`)
  continua sistematicamente mais negativo que o publicado nas 8 moedas
  (confirmado a `--nrep 15000` que não é ruído de Monte Carlo). Registrado
  como "Known issue" no CHANGELOG; **B-4 (coluna N em `tab:ppp`) continua
  bloqueado** até isso se resolver — agora por um motivo mais estreito e
  melhor caracterizado do que antes desta versão.

## 1. O que NÃO mudou (checar antes de assumir que mudou)

- Nenhum dado bruto (`ppp_panel.csv`, `exog_dates.csv`) foi alterado —
  conferido byte-a-byte (após normalizar CRLF/LF) contra a cópia usada na
  última calibração de produção arquivada em `Empirico\boot` (hash SHA1
  truncado `8821489e7079`, igual ao registrado em `meta.json` daquela rodada).
- `boot_out/`, `cbar_surface.csv` e as demais Tabelas 1-9 não foram tocados
  nesta versão — o achado de `--validate` é isolado a
  `ppp_two_axis_columns.py`, um script de verificação independente que não é
  importado por nenhum outro módulo da produção.
- `CITATION.cff`/`.zenodo.json` foram primeiro **sincronizados** com o estado
  real do GitHub (estavam presos em v1.2.0 nesta cópia de trabalho, faltando
  o identifier DOI da v1.2.1) antes de aplicar o bump para 1.3.0 — conferir
  o diff antes de commitar para não perder o identifier de v1.2.1.

## 2. Metadados de versão

- `CITATION.cff`: version 1.3.0, date-released 2026-07-25; identifiers
  já incluem o DOI de v1.2.1 (sincronizado do GitHub); **não** adicionar
  identifier de v1.3.0 ainda — isso é o passo 5 (pós-DOI).
- `.zenodo.json`: version 1.3.0; descrição estendida para citar
  `ppp_two_axis_columns.py` e os módulos do §S2.
- `CHANGELOG.md`: entrada `[1.3.0]` já escrita (seção 0 acima) — sem menção
  de DOI ainda (mesma convenção de v1.2.0: o DOI entra no commit "Record").
- `README.md`: linha da tabela de `ppp_two_axis_columns.py` e o box de
  `--validate` atualizados para o estado pós-conserto (8/8 em `p`, `cv`
  ainda aberto); seção "Figures" e "Data & shipped artifacts" já refletiam a
  nova numeração de figuras de uma passada anterior desta mesma sessão de
  trabalho.
- `DEPENDENCY_GRAPH.md`: linha de `ppp_ar_diagnostic.csv` e o edge de
  `ppp_two_axis_columns.py` atualizados — o arquivo passou a ser uma
  dependência de DADOS (não de código) desse script.

## 3. Antes de commitar: arquivos novos vs. modificados

Como `ppp_two_axis_columns.py` e os outros cinco scripts não existem no
GitHub, `git add -A` vai marcá-los como **novos** (não modificados) — normal,
não é um sinal de erro. Conferir com `git status` que a lista de novos
arquivos bate exatamente com a seção 0 (6 `.py` + 3 `.csv` de
`modulus_numerics.py`), sem nenhum arquivo temporário (`__pycache__/`,
`.nbc`/`.nbi`, `boot_*/` de rascunho) — o `.gitignore` deste pacote já cobre
esses padrões, mas vale conferir com `git status` antes do `add`.

## 4. Comandos (precedente v1.2.0/v1.2.1; executar do clone)

```powershell
cd C:\Users\SAMSUNG\OneDrive\Documents\Paper\Articles\SOLOW2\Mod_Teorico\rgs-model-LB-replication

git status                       # conferir ANTES do add: novos vs. modificados
                                  # bate com a seção 0? nada temporário?
git add -A
git status                       # conferir de novo: sem temporarios, sem .venv
git commit -m "Release v1.3.0: add PPP two-axis decomposition and supplement S2 numerics, fix ppp_two_axis_columns.py --validate AR-order mismatch, sequential figure renaming"
git tag v1.3.0
git push origin main --tags
```

Zenodo (webhook do GitHub monta a versão automaticamente no concept DOI
10.5281/zenodo.21229773):
1. Conferir o novo version DOI no registro Zenodo.
2. Registrar o version DOI em `CITATION.cff` (identifiers) e
   `CHANGELOG.md` ("Record v1.3.0 Zenodo version DOI"); commit + push
   (precedente: commit `0a715fa` para v1.2.0, commit `d92723a` para v1.2.1).

## 5. Pós-release (manuscrito)

- **Nota (não automática desta vez):** o Data statement do manuscrito
  (`silva_levelbreaks_jtsa.tex`, por volta da l. 2044-2046) ainda cita
  release **v1.2.0** como "the exact snapshot underlying this manuscript"
  — já defasado em relação ao GitHub (que está em v1.2.1) mesmo antes desta
  release. Decidir: (a) atualizar a linha para v1.3.0 + o novo version DOI
  assim que ele existir, pulando v1.2.1 (a v1.2.1 só mudou a aparência das
  figuras, não os números); ou (b) manter a citação em v1.2.0 se o
  manuscrito realmente foi escrito contra aquele snapshot e as mudanças de
  v1.2.1/v1.3.0 são posteriores ao texto atual. Isso é uma decisão do autor,
  não uma correção mecânica — os números que o manuscrito cita (Tabelas 1-9)
  não mudaram nesta release, mas os SCRIPTS que ficam arquivados como "the
  replication package" mudaram bastante.
- Se a decisão for (a): 1 edição no `.tex`, Statements and declarations.
- **B-4 continua bloqueado.** Não inserir a coluna (N) em `tab:ppp` até o
  "Known issue" desta versão (o valor crítico simulado) reconciliar — ver
  `CHANGELOG.md`.
