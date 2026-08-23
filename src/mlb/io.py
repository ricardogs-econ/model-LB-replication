"""Surface CSV schema and table rendering (brief s.4 target tree).

The calibration surface schema (cbar_surface.csv, written by the kernel's
aggregate(); read in-session from the shipped file's header):

    config_key, T, m, sigma2_method, cbar_otimo, poder_no_otimo, se_poder,
    se_cbar, fallback_used, refined, inv_dev, cfg_seed,
    lambda1..lambda5,
    {mza,msb,mzt,pt,mpt}_cv{1,2_5,5,10} and ..._se  (5 stats x 4 levels + SE)

inv_dev is the package's own Lemma-invariance diagnostic (~1e-10, the ridge
scale). LaTeX table rendering for the (m,T) tables lives in the
kernel's make_latex_tables (ex-src/mlb/kernel_numba.py:1083); the aggregation convention
it implies is the named function in aggregate.py.

Facade (numerical content unchanged):
  cbar_from_csv(csv_path, m, T, method) -> (cbar, cv5) lookup
  make_latex_tables(P, df, intercepts, diag, sigma2_method)
"""

from src.mlb._compat import numba_kernel as _nk

_k = _nk()
cbar_from_csv = _k.cbar_from_csv
make_latex_tables = _k.make_latex_tables
SURFACE_COLUMNS_DOC = __doc__
