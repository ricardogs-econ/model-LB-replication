"""Model LB for a user's own series: the paper's procedure as a command and as
functions.

Command (from any directory once the package is installed with `pip install -e .`
in the repository):

    python -m model_lb --csv FILE --date-col year --col q --breaks 1986,1993

Functions (model_lb/api.py): prepare, statistic, sieve, null_distribution,
summarize_null, column_ii, half_power_point, column_iii, model_lb_test, half_life,
holm, variance_profile_size, design_positions.
"""

from model_lb.api import (
    CBAR,
    CV_ASY,
    K_SET,
    KMAX_BIC,
    KMAX_MAIC,
    R_DEFAULT,
    SCALE_BAND,
    SEED_DEFAULT,
    T_GRID_MIN,
    column_ii,
    column_iii,
    design_positions,
    half_life,
    half_power_point,
    holm,
    model_lb_test,
    null_distribution,
    prepare,
    sieve,
    statistic,
    summarize_null,
    variance_profile_size,
)

__all__ = [
    "CBAR",
    "CV_ASY",
    "K_SET",
    "KMAX_BIC",
    "KMAX_MAIC",
    "R_DEFAULT",
    "SCALE_BAND",
    "SEED_DEFAULT",
    "T_GRID_MIN",
    "column_ii",
    "column_iii",
    "design_positions",
    "half_life",
    "half_power_point",
    "holm",
    "model_lb_test",
    "null_distribution",
    "prepare",
    "sieve",
    "statistic",
    "summarize_null",
    "variance_profile_size",
]
