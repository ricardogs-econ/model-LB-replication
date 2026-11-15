# Critical values of the robust level-shift tests

The eight tables in this folder hold asymptotic critical values and kappa constants of
the tests for multiple level shifts of Carrion-i-Silvestre and Gadea (2023), computed by
those authors. They are read by `mlb/level_shifts.py`, which
`mlb/robust_level_shift.py` runs for Section S5.6 of the supplement
(`ssub:app-robust`).

## Provenance

Every table is a byte-for-byte copy of the file of the same name in the authors'
replication files: the archive `jae_empirical_web.zip` distributed from M. Dolores
Gadea's web page (sha256
`8e08a72f60e4974aedee9dc49f20c910da2fcb4d8fa151197d9f3aafa61341fc`), folder
`JAE_empirical_web/functions_empirical`. The tables are included whole, as the authors
distribute them. No license accompanies the authors' files; their readme
asks users to cite the authors and the web page. The tables are distributed here with
that attribution; they are the authors' work and are not covered by the license of
this package's code.

| file (archive: `functions_empirical/<file>`) | content | read by | sha256 |
|---|---|---|---|
| `cv_BP_1_Ftest_0_m_5.dat` | BP_1 (I(1) noise) tests of 0 breaks: 4 rows (10, 5, 2.5, 1%) x 7 columns (m = 1..5, UDmax, WDmax) | `ftest_0_m_max` | `b26709f5fe3256177e96ae9d5932737a4ec154ef0c1a5c26967e45e62e6e25ea` |
| `cv_BP_0_Ftest_0_m_5.dat` | the same for BP_0 (I(0) noise) | `ftest_0_m_max` | `3a41533985d05d8512ff19548e601deab94fbf8df5d4e52d073a23b5c000ba11` |
| `kappa_Ftest_0_m_5.dat` | kappa constants of the union of BP_1 and BP_0: 4 rows x 7 columns (m = 1..5, UDmax, WDmax) | `ftest_0_m_max`, `ftest_union_decisions` | `5abeccd1c792dde0eab3ae62d65d636e45f70f1154243bdf37a20fcc8685f810` |
| `cv_F_0.dat` | sequential BP_0 test of l against l + 1 breaks: rows (trimming, quantile, l = 0..9) | `multiple_breaks_inference` | `aa83acf5bf20d58a300e3c3fa3ab7d4fedb6d15c18f1e1667774d51cfca0d9d7` |
| `cv_F_1_a.dat` | the same for the first modified BP_1 statistic | `multiple_breaks_inference` | `f8a5495891c63eaedc55db950f3f6c15194a03f8dd4b0f179f69826a00a407be` |
| `cv_F_1_b.dat` | the same for the second modified BP_1 statistic | `multiple_breaks_inference` | `099775b0aa6085b2c746810e75fad6ce5c8551548aa6d69001f153172710012d` |
| `mat_kappa_U_a.dat` | kappa constants of the union of the first modified BP_1 and BP_0 (same layout) | `multiple_breaks_inference` | `dafff6ccbb93901979517b094ed316e1411748780160d8e1b50ef130478791c6` |
| `mat_kappa_U_b.dat` | kappa constants of the union of the second modified BP_1 and BP_0 (same layout) | `multiple_breaks_inference` | `d1423ed09f15ebe736eacbecaef32ef621e95256b8e614d7eb1e8b5c770e4566` |

The last five tables hold the trimmings 0.05, 0.15 and 0.20 and the quantiles 0.90, 0.95,
0.975 and 0.99; the port selects the rows whose first column equals the trimming, as the
MATLAB code does. The sha256 values are of the files as shipped, which have LF line
endings, as the authors' files do; `mlb/robust_level_shift.py` records
them in its run record.

Two sets of constants are not in this folder because the authors' code writes them in
its source, and the port does the same: the critical values and kappa of Harvey,
Leybourne and Taylor (2010) for the windows 0.10 and 0.15 that the package uses (from
the authors' file `HLT_test_empirical.m`) and the response-surface coefficients of the
sequential BP_1 test for the trimming 0.15 (from `RS_BP_1.m`).

## The Python code

`mlb/level_shifts.py` is this package's own code, released with the package under
the terms of `LICENSE`: a Python port that follows the authors' MATLAB functions one
function per file. The MATLAB code itself is not included.

## Please cite

If you use these tables or the tests, please cite:

> Harvey, D.I., Leybourne, S.J. and Taylor, A.M.R. (2010). Robust methods for detecting
> multiple level breaks in autocorrelated time series. *Journal of Econometrics* 157,
> 342–358.

> Carrion-i-Silvestre, J.L. and Gadea, M.D. (2023). Testing for multiple level shifts
> with an integrated or stationary noise component. *Journal of Applied Econometrics*
> 38(6), 801–819.
