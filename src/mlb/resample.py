"""The package's resampling procedures — taxonomy from MC_vs_BOOTSTRAP.md.

MC_vs_BOOTSTRAP.md (repository root) is the authoritative vocabulary note
(brief s.4.1): the package runs THREE distinct resampling/simulation
procedures at three stages on three different data-generating assumptions,
plus a robustness stage (Section 5) that re-uses Procedure 1's machinery
under deliberately misspecified nulls. Its one-line summary, reproduced:

| # | Name | Stage | Null | Resampling | Driver | Count |
|---|---|---|---|---|---|---|
| 1 | Monte Carlo calibration | s.3-4 | parametric RW (assumed) | draw N(0,1) innovations | replicate_section3_4.py | R |
| - | Robustness (MC, misspecified null) | s.5 | AR(1) / true-omega2 oracle / alt. trimming (assumed) | draw from the varied null | replicate_section5.py | R |
| 2 | Sieve-AR(p) bootstrap | s.6 calib | AR(p)+unit root (estimated) | redraw sieve innovations | replicate_section6.py boot | B=9999 |
| 3 | Wild bootstrap | s.6 half-life | AR + heteroskedastic (estimated) | sign-randomize residuals | replicate_section6.py hl | B=9999 |

Verified by grep (Phase 3): the sieve machinery lives in boot_ppp_cbar.py
(whose module docstring carries the EQUIVALENCE CONTRACT — kernels imported
from src.mlb.kernel_numba, never reimplemented; Option-B/ERS-envelope tangency on the
ORACLE PT; R_cv/R_pow/kmax conventions — that contract is the specification
and is preserved untouched in that module); the wild scheme lives in
hl_median_unbiased.py (Rademacher sign-randomization for the half-life
grid-t intervals; NOTE its legacy np.random.seed at :181/:261/:290,
deliberately NOT migrated). On-the-fly critical values (run_model_lb.py) are
Procedure 1 invoked pointwise, not a fourth procedure.

This module is deliberately a map, not a re-implementation: re-implementing
a resampler here would fork the numerics the contract forbids forking.
"""

PROCEDURES = {
    1: dict(
        name="Monte Carlo calibration",
        stage="sections 3-4",
        null="parametric driftless RW, breaks imposed (assumed)",
        resampling="draw i.i.d. N(0,1) innovations",
        driver="replicate_section3_4.py",
        produces="cbar_surface.csv",
    ),
    "5R": dict(
        name="Robustness (MC, misspecified null)",
        stage="section 5",
        null="AR(1) / true-omega2 oracle / alternative trimming",
        resampling="draw from the varied parametric null",
        driver="replicate_section5.py",
        produces="robustness_out/ tables, power_comparison.csv",
    ),
    2: dict(
        name="Sieve-AR(p) bootstrap",
        stage="section 6 calibration",
        null="AR(p) + unit root, estimated per currency",
        resampling="redraw sieve innovations under the fitted filter",
        driver="replicate_section6.py boot (boot_ppp_cbar.py)",
        produces="boot_out/calib/surface_ppp_boot.csv, "
        "boot_out/empirical/ppp_empirical.csv",
    ),
    3: dict(
        name="Wild bootstrap",
        stage="section 6 half-lives",
        null="fitted AR with heteroskedastic innovations",
        resampling="Rademacher sign-randomization of residuals",
        driver="replicate_section6.py hl (hl_median_unbiased.py)",
        produces="half-life grid-t intervals (tab:hl, Figure_S2)",
    ),
}
