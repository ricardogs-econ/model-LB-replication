"""Deterministic design for Model LB (paper eq. (1), label eq:Zmodel1).

Symbols: Z = [1, DU_1, ..., DU_m] with DU_{j,t} = 1{t > tau_j},
tau_j = floor(lambda_j * T); m break dummies plus a constant; breaks are
KNOWN and IMPOSED — there is NO break-date estimator in this paper
(brief s.4) and none is provided.

Facade over the production kernel (numerical content unchanged):
  build_z(nt, break_pos)             -> kernel build_z_nb
  break_pos_from_lambdas(T, lambdas) -> kernel break_pos_from_lambdas
  make_lambdas(m, trim, min_spacing, n_grid) -> the lambda simplex grid
"""

from src.mlb._compat import numba_kernel as _nk

_k = _nk()
build_z = _k.build_z_nb
break_pos_from_lambdas = _k.break_pos_from_lambdas
make_lambdas = _k.make_lambdas
config_seed = _k.config_seed
enumerate_configs = _k.enumerate_configs
