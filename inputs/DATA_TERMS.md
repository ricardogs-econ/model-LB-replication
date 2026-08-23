# Data sources and terms

The panel distributed with this package is a derived series — log real exchange
rates against the US dollar — computed from two public sources. No raw source
file is redistributed, and the `--fetch` path rebuilds the panel from the
sources.

**Nominal exchange rates.** Bank for International Settlements, bilateral US
dollar exchange rates (bulk export, `data.bis.org`). The BIS terms for
statistics place no restriction on use provided the BIS is cited as the source,
which this package and the accompanying article do. Nothing here implies
endorsement by, or affiliation with, the BIS.
https://www.bis.org/terms_statistics.htm

**Consumer prices.** World Bank, World Development Indicators, series
`FP.CPI.TOTL` (2010 = 100), distributed under Creative Commons Attribution 4.0.
The series is used with attribution and has been transformed — real exchange
rate construction, annual alignment, common-sample trimming — and those changes
are indicated here as the licence requires.

Code in this repository is released under the terms of `LICENSE`; the
paragraphs above govern the data, not the code.
