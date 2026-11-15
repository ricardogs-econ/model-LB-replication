"""mlb: the Model LB replication package, every module (paper symbols in each module's
docstring). Run a module as `python mlb/<module>.py ...` from the package root.

numba-only: the kernel is `kernel_numba`, compiled by numba (`requirements.txt`
lists its version); without numba the import fails.

The kernel and the shared functions:

    kernel_numba            the numba kernel: design, detrending, M statistics, DGP
    break_dates             the break dates tau_j = floor(lambda_j T), computed exactly
    envelope_exact          the exact power envelope and half-power point (finite T)
    envelope_limit          the limiting power envelope
    mzt_exact_linear        the exact-linear MZ_t core (pure numpy)
    level_shifts            the robust level-shift tests (port of the authors' MATLAB)
    paths                   the package root, its folders and the paths of shared files
    _figstyle               the palette and typography of the figures

The computations, theory and calibration:

    limit_envelope          the limiting power envelope (characteristic function)
    lemma1_bounds           Lemma 1 in exact rational arithmetic on 3,004 designs
    exact_surface_producer  Table 1: the exact half-power points
    propflat_exact          Proposition 1 on the exact envelope
    regularity_m_plus_1     the (m+1) regularity of the half-power points
    surface_summaries       the summaries and trimming lattices of the exact surface
    envelope_monotonicity   the strict decrease of the finite-sample envelope
    critical_values_blocks  Table 2: the block critical values of the M statistics
    prose_numbers_grid      the grid summaries the prose prints
    ckp_cbar_surface        the CKP (2009) response surface for cbar (the -21.7)
    fixed_index             the fixed index that maximizes power, full sample and blocks
    rule_comparison         Section 4.2: following the surface against -7
    oracle_sigma2_ladder    the null quantile of MZ_t with the true long-run variance
    maic_lag_zero           the MAIC lag-zero shares under the calibration design
    overdetrend_power_precise  the power of the over-detrending shortcuts (Figure 2)
    overdetrend_figure      Figure 2, drawn from overdetrend_power.json
    size_figure             Figure 1: the true size at cbar = -7 against T

The computations, serial dependence:

    ar1_size_power          Table S1: size and power under AR(1) errors
    ar1_recalibration       Table S2: the half-power point under AR(1) errors, exact

The computations, the PPP application (Section 6):

    regime_events           the regime events: dates -> tau -> break positions
    sj_events               the dates of the crisis design SJ (inputs/sj_dates.csv)
    app_seeds               the seed registry of the application's Monte Carlo
    ppp_application         the unit-root tests, band power, count law, sensitivity sets
    ppp_ar_null             the AR(p) null (numba kernel)
    ppp_sieve               the level-break residual, the sieve and the AR in levels
    ppp_half_lives          Tables S7-S8: the grid-bootstrap half-lives
    ppp_sj                  the crisis design SJ: design E plus the banking crises
    ppp_variance_scope      Table S9: the unit-root tests under the variance profile
    pesaran_cd              Pesaran's cross-sectional dependence statistic
    robust_level_shift      the robust level-shift tests on the nine series
    c6_critical_values      the simulated critical values of the admission test C6
    gate_sizes              the size and power of the admission tests C4 and C6
    ppp_tables              Tables 3 and S3-S8, rendered from reference/application/

The data building (raw downloads; run by neither the quick nor the full mode; the
files they write ship in inputs/):

    build_ppp_panel         the real exchange rate panel (inputs/ppp_panel.csv)
    build_jst_crises        the banking crisis years (inputs/jst_crises.csv)
    admissible_universe     the admissible universe, C0-C6 (inputs/ppp_sweep/)

The runner and the steps:

    runner                  runs the steps and compares what they write with reference/
    steps                   the steps: each module's arguments and what is compared
"""
