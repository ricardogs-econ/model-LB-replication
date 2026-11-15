# Data sources and terms

The data files under `inputs/` are derived from public sources or compiled from
published records. No raw source file is redistributed. Each section below gives the
source, the download, the terms and, for the files built from raw downloads, the sha256
of the raw files they were built from.

## Real exchange rates (`inputs/ppp_panel.csv`)

The panel is a derived series — log real exchange rates of nine currencies against
the US dollar, 1973–2024 — computed from two public sources by
`mlb/build_ppp_panel.py`. The raw files are read from a local cache
folder (`--cache`); `--fetch` downloads them into it from
`https://data.bis.org/static/bulk/WS_XRU_csv_flat.zip` (annual frequency, collection A)
and `https://api.worldbank.org/v2/country/all/indicator/FP.CPI.TOTL?format=json&per_page=20000&date=1972:2024`.
The shipped panel was built from raw files with these sha256: `WS_XRU_csv_flat.zip`
fdf001f693e6312207ee80c1eeea854bc654bf7d808c403f054ac88fba9258c8, its annual extract
`bis_xru_annual_A.csv` 1b137e3a092de6e2389fba1c393df5b45a70a09aaca8340bc4fe2f39904275d3,
`wb_cpi.json` 64b8ec8def0156b2d1d8116dbe4746f126c8dd9e8b145d35402ccda9820dc130. The same raw files
feed the admission screen `mlb/admissible_universe.py`, whose
outputs are in `inputs/ppp_sweep/`. The files used hold a BIS data file dated
2026-08-12 and a World Bank payload last updated 2026-07-13; later downloads may
differ.

**Nominal exchange rates.** Bank for International Settlements, bilateral US
dollar exchange rates (bulk export, `data.bis.org`). The BIS terms for
statistics place no restriction on use provided the BIS is cited as the source,
which this package and the accompanying article do. Nothing here implies
endorsement by, or affiliation with, the BIS.
https://www.bis.org/terms_statistics.htm

**Consumer prices.** World Bank, World Development Indicators, series
`FP.CPI.TOTL` (2010 = 100), distributed under Creative Commons Attribution 4.0.
The series is used with attribution and has been transformed — the computation
of real exchange rates, annual alignment, common-sample trimming — and those
changes are indicated here as the license requires.

## Crisis years (`inputs/jst_crises.csv`)

The crisis design S-JST adds to the break dates the systemic banking crises of each
currency's issuing country, taken from the Jordà–Schularick–Taylor Macrohistory
Database, release 6 (https://www.macrohistory.net/database/). The database is
distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International License (CC BY-NC-SA 4.0), and the dataset is not redistributed here.
`inputs/jst_crises.csv` holds only the crisis years: the years 1972–2024 in which the
database's variable `crisisJST` marks the first year of a systemic banking crisis, for
the issuers of the Australian and Canadian dollars, the Swiss franc, the Danish
krone, the pound sterling, the yen and the Norwegian and Swedish kronor (New Zealand
is not in the database, and the database ends in 2020).
`mlb/build_jst_crises.py` wrote it from the file `JSTdatasetR6.dta`,
downloaded from that page on 2026-10-02 (sha256
b0ebb74a8d1b5b1bc9033fc46a6dcc578736afff8ce1e1086b7840f2649e79b3).
`python mlb/build_jst_crises.py --cache <folder> --check` rebuilds the years from a copy
of `JSTdatasetR6.dta` in that folder and compares them with the shipped file.

The crisis years are reproduced with attribution to the database and, to the extent
that its license applies to them, under its terms (non-commercial use, share-alike),
not under the license of the code. Please cite:

> Jordà, Ò., Schularick, M. and Taylor, A.M. (2017). Macrofinancial history and the
> new business cycle facts. *NBER Macroeconomics Annual* 31, 213–263.

## OECD membership (`inputs/oecd_1973.csv`)

The OECD members by 31 December 1973 and the date the Convention on the
Organisation for Economic Co-operation and Development entered into force for each
(admission condition C0), compiled from the OECD's list of members
(https://www.oecd.org/en/about/legal.html) and the record of the Convention in the
United Nations Treaty Collection (UNTS vol. 888, p. 179, registration no. 12736);
each row gives its sources.

## Regime events (`inputs/regime_events.csv`)

The currency-regime events of rule E, one row per event, each with its date and the
primary source it was read from (central bank and government publications, G-5 and
G-7 communiqués), cited in the row; compiled for this package.

## Critical values of the robust level-shift tests (`inputs/level_shift_cv/`)

Eight tables of asymptotic critical values and constants of the tests of
Carrion-i-Silvestre and Gadea (2023), copied byte for byte from the authors'
replication files, and read by the Python port of their procedure,
`mlb/level_shifts.py`. The critical values of the tests of Harvey, Leybourne and
Taylor (2010) that the port uses are written in its source, as they are in the MATLAB
code of Carrion-i-Silvestre and Gadea.
`inputs/level_shift_cv/README.md` gives the provenance and sha256 of each table and
the references to cite. The tables are the authors' work, distributed with this
attribution; they are not covered by the license of the code.

## Code

Code in this repository is released under the terms of `LICENSE`; the sections
above govern the data and the third-party tables, not the code.
