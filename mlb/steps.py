"""The steps of the section scripts, each defined once (sections share steps).

A step runs one module of mlb/ from the package root:

  argv       the script and its arguments ("{jobs}" is replaced by --jobs);
  compare    (regenerated file, reference file) pairs that the runner compares
             (TOLERANCE); a third element Rows(k, keys) compares only the rows the
             module wrote (a subset of the reference's rows, matched on their first k
             fields), and requires the rows `keys`;
  ignore_keys  fields of a JSON file (or columns of a CSV) left out of the comparison;
  info_rc    exit statuses of the module that report only its own byte comparison with
             the reference: the runner then compares the files itself (`compare`) and
             its comparison decides; any other non-zero status is a failure;
  check      how the step decides, when the module compares its outputs itself (its
             tolerance and why); otherwise the step is decided by `compare`;
  source     'recomputed' (the step computes the result again and compares it) or
             'reference' (the step shows a result from reference/: it draws or renders
             it, or rebuilds it from other reference files);
  regenerated_by  for a 'reference' step: the --full steps (and scripts) that
             regenerate the results it shows;
  needs      steps whose outputs the step reads (run first, once);
  setup      (reference file, destination) copies made before the step (a module that
             reads the registered values from its output folder);
  products   the tables and figures the step writes (printed after the step);
  echo       prefixes of lines of the module's console that the runner prints after the
             step, as information (the result of a check that does not decide the step,
             e.g. Table S1's agreement checks);
  fresh      folders of resumable stores removed before the step, so that it recomputes
             (the long regenerations keep theirs: an interrupted --full run resumes);
  seconds    the wall time at 8 worker processes (TIME_BASIS).
"""

from dataclasses import dataclass

# How the runner compares a regenerated file with its reference file. Line endings are
# normalized to LF; identical bytes are reported 'byte-identical'. Otherwise the two
# files are compared token by token (JSON: as parsed documents): text and integers
# exactly, floating-point numbers to |a - b| <= REL_TOL max(1, |b|), and the step
# reports 'equal within 1e-9 (largest relative difference X)'. The floating-point
# results of the package (least squares, eigensolvers, sums in numpy) can differ in
# their last digits between numpy, LAPACK and BLAS builds and between platforms; 1e-9
# is far below every precision the paper prints and far above those differences (about
# 1e-16 relative).
REL_TOL = 1e-9
TOLERANCE = ("byte-identical after LF normalization, or token by token: text and "
             "integers exactly, floats to |a - b| <= 1e-9 max(1, |b|) (numpy, LAPACK "
             "and BLAS builds move the last digits of floating-point results; 1e-9 is "
             "far below every printed precision)")

TIME_BASIS = ("wall time at --jobs 8; the quick-mode times were measured on a Windows "
              "machine with 12 logical processors, nothing else running and an empty "
              "numba cache; the --full regenerations are estimated from their CPU time "
              "divided by 8, or from the wall time measured on that machine where it "
              "differed from that estimate by more than 30%; the over-detrending run "
              "(full_overdetrend) was never run whole on this version, so its time is "
              "the CPU estimate")

PY = "mlb/"
APP = "output/replicate/steps/application"
APP_WORK = "output/replicate/steps/application_work"
VAR = "output/replicate/steps/variance_scope"
CVO = "output/critical_values"

RT = "reference/theory"
RC = "reference/calibration"
RS = "reference/serial_dependence"
RF = "reference/figures"
RM = "reference/application/main"
RCD = "reference/application/crisis_design"
RV = "reference/application/variance"
RR = "reference/application/robust_tests"
RTB = "reference/application/tables"
RAD = "reference/application/admission"


@dataclass(frozen=True)
class Rows:
    """A comparison by rows: every row the module wrote equals the reference's row with
    the same first `key_cols` fields, and the rows `keys` (tuples of those fields, as
    written) are among them."""
    key_cols: int
    keys: tuple = ()


@dataclass(frozen=True)
class Step:
    key: str
    title: str
    argv: tuple
    seconds: float
    compare: tuple = ()
    ignore_keys: tuple = ()
    info_rc: tuple = ()
    check: str = ""
    source: str = "recomputed"
    regenerated_by: str = ""
    needs: tuple = ()
    setup: tuple = ()
    products: tuple = ()
    fresh: tuple = ()
    echo: tuple = ()


def _same(names, new_dir, ref_dir):
    """compare pairs: the files of one folder against the same names in a reference
    folder."""
    return tuple((f"{new_dir}/{n}", f"{ref_dir}/{n}") for n in names)


_SURFSUM = ("surface_cells.csv", "surface_summaries.csv", "surface_trimming.csv",
            "surface_trimming_configs.csv", "surface_trimming_range.csv")
_SURFSUM_OUT = f"{CVO}/surface_summaries"
_RECAL_OUT = "output/robustness/ar1_recalibration"
_RULE = ("rule_fixed_index.csv", "rule_configs.csv", "rule_levels.csv")
_RULE_OUT = f"{CVO}/rule_comparison"
_AR1 = ("ar1_size_power.csv", "ar1_maic_cv.csv")
_AR1_OUT = "output/robustness/ar1_size_power"
_MAIC_OUT = f"{CVO}/maic_lag_zero"
_MAIC_FILES = ("maic_k_by_cell.csv", "maic_k0_share_table.csv")
_TABLES_TEX = ("tab_ppp.tex", "stab_app_dates.tex", "stab_app_power.tex",
               "stab_app_count.tex", "stab_app_sets.tex", "stab_app_hl.tex",
               "stab_app_corr.tex")
_VAR_SETUP = ((RM, ("ppp_null.csv", "ppp_ar_diagnostic.csv", "readings.csv",
                    "cbar_half_configs.csv")),
              (RCD, ("sj_null.csv", "sj_statistics.csv", "sj_readings.csv",
                     "sj_cbar_half_configs.csv")))
# exit status 4 of surface_summaries.py, ar1_recalibration.py, rule_comparison.py,
# ar1_size_power.py and maic_lag_zero.py --check: their own byte comparison with the
# reference found a difference (information; see info_rc)
_BYTE_RC = (4,)


def _app_stage(key, stage, names, seconds, needs, work=True):
    argv = (PY + "ppp_application.py", "--stage", stage, "--out-dir", APP)
    if work:
        argv += ("--work", APP_WORK)
    return Step(key, f"the application, stage {stage}: regenerated and compared",
                argv + ("--jobs", "{jobs}"), seconds, compare=_same(names, APP, RM),
                needs=needs)


def _sj_stage(key, stage, names, seconds, needs, work=True, title=None):
    argv = (PY + "ppp_sj.py", "--stage", stage, "--out-dir", APP)
    if work:
        argv += ("--work", APP_WORK)
    return Step(key, title or (f"the crisis design S-JST, stage {stage}: regenerated "
                               "and compared"),
                argv + ("--jobs", "{jobs}"), seconds, compare=_same(names, APP, RCD),
                needs=needs)


_steps = [
    # ---- the quick mode: exact computations, small simulations, and results shown
    #      from reference/ -----------------------------------------------------------
    Step("limit_envelope",
         "the limit envelope (inversion of the characteristic function), recomputed",
         (PY + "limit_envelope.py", "--check"), 76.9,
         check="limit_envelope.py --check: limit_envelope.json byte-identical, or "
               "every number within 1e-9 (1 + |b|) (complex transcendentals and a "
               "symmetric eigensolver move the last digits between machines)"),
    Step("lemma1_bounds",
         "Lemma 1 in exact rational arithmetic on 3,004 designs, recomputed",
         (PY + "lemma1_bounds.py", "--out-dir", "output/replicate/steps/lemma1_bounds",
          "--check"), 3.1,
         check="lemma1_bounds.py --check: lemma1_bounds.json byte-identical (exact "
               "rational arithmetic)"),
    Step("regularity",
         "the (m+1) regularity of the cell means, recomputed",
         (PY + "regularity_m_plus_1.py", "--out-dir",
          "output/replicate/steps/regularity"), 0.1,
         compare=_same(("regularity_m_plus_1.json",),
                       "output/replicate/steps/regularity",
                       f"{RC}/regularity_m_plus_1")),
    Step("monotonicity_reference",
         "strict decrease of the envelope: the verdicts and totals of "
         "reference/calibration/envelope_monotonicity/",
         (PY + "envelope_monotonicity.py", "--show-reference"), 0.5,
         source="reference",
         regenerated_by="full_monotonicity (replicate_sec_4.py --full)",
         check="envelope_monotonicity.py --show-reference: prints the verdict and the "
               "totals of each of the four summaries and exits 1 unless every one "
               "reports strict decrease"),
    Step("tabcv_from_reference",
         "Table 2: the block means and the table rebuilt from the 20 blocks of the "
         "reference file",
         (PY + "critical_values_blocks.py", "--aggregate", "--from-reference",
          "--table"), 0.7,
         compare=((f"{CVO}/critical_values_blocks/critical_values_blocks.json",
                   f"{RC}/critical_values_blocks/critical_values_blocks.json"),
                  (f"{CVO}/critical_values_blocks/tabcv_full_percentiles.csv",
                   f"{RC}/tabcv_full_percentiles.csv")),
         source="reference",
         regenerated_by=("full_tabcv_blocks (replicate_sec_4.py or "
                        "replicate_sup_S4.py --full)"),
         products=(f"{CVO}/critical_values_blocks/tabcv_full_percentiles.csv",)),
    Step("figure_size",
         "Figure 1 drawn from its reference data, its points checked",
         (PY + "size_figure.py", "--from-reference", "--check", "--out-dir",
          "output/replicate/steps/figure_size"), 1.3,
         source="reference",
         regenerated_by="full_figure_size (replicate_sec_4.py --full)",
         check="size_figure.py --from-reference --check: the drawn points equal the "
               "reference data Figure_1_data.csv exactly",
         products=("output/replicate/steps/figure_size/Figure_1.pdf",)),
    Step("figure_overdetrend",
         "Figure 2 drawn from reference/calibration/"
         "overdetrend_power.json, its points checked",
         (PY + "overdetrend_figure.py", "--out-dir",
          "output/replicate/steps/figure_overdetrend", "--check"), 1.9,
         source="reference",
         regenerated_by=("full_overdetrend (replicate_sec_4.py or "
                        "replicate_sup_S4.py --full)"),
         check="overdetrend_figure.py --check: the points read from "
               "overdetrend_power.json and the drawn points equal the reference data "
               "Figure_2_data.csv (byte-identical, or text and integers exactly and "
               "floats within 1e-9 max(1, |b|))",
         products=("output/replicate/steps/figure_overdetrend/Figure_2.pdf",)),
    Step("prose_grid",
         "the grid summaries that the prose prints, re-derived from reference/",
         (PY + "prose_numbers_grid.py",), 0.7,
         compare=_same(("prose_numbers_grid.json",),
                       "output/critical_values/prose_numbers_grid",
                       f"{RC}/prose_numbers_grid"),
         source="reference",
         regenerated_by=("the --full steps of replicate_sec_4.py and "
                         "replicate_sec_5.py, which regenerate the files it reads")),
    Step("ckp_table1",
         "the CKP (2009) response surface at the 45 cells of its Table 1, recomputed",
         (PY + "ckp_cbar_surface.py", "--table1-json",
          "output/replicate/steps/ckp_table1/ckp_table1_grid.json"), 0.1,
         compare=(("output/replicate/steps/ckp_table1/ckp_table1_grid.json",
                   f"{RC}/ckp_table1_grid.json"),)),
    Step("exact_surface",
         "Table 1: the 404 exact half-power points, recomputed; the table rendered",
         (PY + "exact_surface_producer.py", "--jobs", "{jobs}", "--out-dir",
          "output/replicate/steps/exact_surface", "--check"), 65.0,
         check="exact_surface_producer.py --check: cbar_surface_exact.csv "
               "byte-identical, or every number within the module's 1e-6 (the "
               "half-power points are roots found to xtol = 1e-6)",
         products=("output/replicate/steps/exact_surface/tab_cbar_mT_exact.tex",)),
    Step("propflat",
         "Proposition 1 on the exact envelope (displacements, power gap), recomputed",
         (PY + "propflat_exact.py", "--jobs", "{jobs}", "--out-dir",
          "output/replicate/steps/propflat", "--check"), 23.6,
         check="propflat_exact.py --check: both CSV files byte-identical, or every "
               "number within the module's 1e-7 (1 + |b|) (half-power points found to "
               "xtol = 1e-9)"),
    Step("surface_summaries",
         "the surface summaries and the trimming lattices (exact), recomputed",
         (PY + "surface_summaries.py", "--jobs", "{jobs}"), 32.0,
         compare=_same(_SURFSUM, _SURFSUM_OUT, f"{RC}/surface_summaries"),
         info_rc=_BYTE_RC),
    Step("ar1_recalibration",
         "Table S2: the half-power point under AR(1) errors (exact), recomputed; the "
         "table rendered",
         (PY + "ar1_recalibration.py", "--jobs", "{jobs}"), 6.0,
         compare=_same(("ar1_recalibration_envelope.csv",), _RECAL_OUT,
                       f"{RS}/ar1_recalibration"),
         info_rc=_BYTE_RC, products=(f"{_RECAL_OUT}/stab_recal.tex",)),
    Step("fixed_index",
         "the best fixed index at (2, 60) and its block argmaxes, recomputed",
         (PY + "fixed_index.py", "--out-dir",
          "output/replicate/steps/fixed_index", "--check"), 43.7,
         check="fixed_index.py --check: fixed_index.json byte-identical, or its floats "
               "within the module's 1e-9 max(1, |a|, |b|) (critical values estimated "
               "from simulated draws can move in their last digits)"),
    Step("rule_agreement",
         "the rule comparison's engine against the kernel, and against the power "
         "of reference/calibration/overdetrend_power.json",
         (PY + "rule_comparison.py", "--part", "agreement"), 22.0,
         check="rule_comparison.py --part agreement: the engine against the kernel "
               "(relative difference below 1e-7, draw for draw) and Pi(-7) against "
               "overdetrend_power.json (within 2 se)"),
    Step("oracle_cells",
         "the oracle null quantile ladder at T = 300 and 1,200 (8 of its 12 cells), "
         "recomputed",
         (PY + "oracle_sigma2_ladder.py", "--cells", "0:300", "1:300", "2:300", "3:300",
          "0:1200", "1:1200", "2:1200", "3:1200", "--jobs", "{jobs}", "--out-dir",
          "output/replicate/steps/oracle_cells", "--check"), 13.7,
         check="oracle_sigma2_ladder.py --check: the rows of the 8 cells "
               "byte-identical, or every number within 1e-9 (1 + |b|)"),
    Step("maic_cells",
         "the MAIC lag-zero shares of the two cells that bound the printed range, "
         "recomputed",
         (PY + "maic_lag_zero.py", "--cells", "0:300", "5:80", "--check", "--jobs",
          "{jobs}"), 6.8,
         compare=tuple((f"{_MAIC_OUT}/{n}", f"{RC}/maic_lag_zero/{n}",
                        Rows(2, (("0", "300"), ("5", "80"))))
                       for n in _MAIC_FILES),
         info_rc=_BYTE_RC),
    Step("robust_tests",
         "the robust level-shift tests (HLT 2010, CISG 2023) on the nine series, "
         "recomputed",
         (PY + "robust_level_shift.py", "--out-dir",
          "output/replicate/steps/robust_tests", "--check"), 1.8,
         check="robust_level_shift.py --check: tokens identical except floats, which "
               "agree to a relative 1e-9 (of max(|a|, |b|))"),
    Step("ppp_tables",
         "Tables 3 and S3-S8, rendered from reference/application/",
         (PY + "ppp_tables.py",), 0.3,
         compare=_same(("readings_tables.csv",), "output/bootstrap_empirical/tables",
                       RTB),
         source="reference",
         regenerated_by=("the --full steps of replicate_sec_6.py and "
                         "replicate_sup_S5.py, which regenerate the files it reads"),
         products=tuple(f"output/bootstrap_empirical/tables/{n}" for n in _TABLES_TEX)),
    Step("figure_series",
         "Figure 3: the nine series with the break dates of design E and "
         "the regime means, and the facts of Section 6.1, recomputed",
         (PY + "series_figure.py", "--out-dir", "output/replicate/steps/figure_series"),
         2.0,
         compare=(("output/replicate/steps/figure_series/Figure_3_data.csv",
                   f"{RF}/Figure_3_data.csv"),
                  ("output/replicate/steps/figure_series/series_summary.json",
                   f"{RM}/series_summary.json")),
         products=("output/replicate/steps/figure_series/Figure_3.pdf",)),
    Step("regime_events",
         "the break dates of rule E: f and tau regenerated from the event dates",
         (PY + "regime_events.py", "--check"), 0.1,
         check="regime_events.py --check: f and tau of every event equal those "
               "generated from its date (text and integers, exactly)"),
    Step("sj_events",
         "the crisis design's dates (inputs/sj_dates.csv) regenerated",
         (PY + "sj_events.py", "--check"), 0.1,
         check="sj_events.py --check: inputs/sj_dates.csv regenerated byte for byte "
               "(dates and integers)"),
    Step("app_seeds",
         "the seed registry of the application, printed and compared",
         (PY + "app_seeds.py", "--out",
          "output/replicate/steps/app_seeds/app_seeds.csv"), 0.2,
         compare=(("output/replicate/steps/app_seeds/app_seeds.csv",
                   f"{RM}/app_seeds.csv"),)),
    _app_stage("app_design", "design", ("cbar_half_configs.csv",), 4.6, (),
               work=False),
    _app_stage("app_data", "data", ("ppp_ar_diagnostic.csv", "ppp_statistics.csv",
                                    "band_bounds.csv", "rhat_sample.csv",
                                    "pesaran_cd.csv"), 1.9, ("app_design",),
               work=False),
    _sj_stage("sj_design", "design", ("sj_cbar_half_configs.csv",), 5.0, ("app_data",),
              work=False),
    _sj_stage("sj_data", "data", ("sj_statistics.csv",), 1.9, ("sj_design",),
              work=False),
    Step("half_lives",
         "Tables S7-S8: the grid-bootstrap half-lives (B = 19,999), regenerated and "
         "compared",
         (PY + "ppp_half_lives.py", "--out-dir", APP, "--work", APP_WORK, "--jobs",
          "{jobs}"), 123.2,
         compare=_same(("hl_cells.csv", "hl_nodes.csv"), APP, RM),
         needs=("app_data",), fresh=(APP_WORK,)),
    Step("variance_scope",
         "Table S9: the unit-root tests under the variance profile, regenerated and "
         "compared",
         (PY + "ppp_variance_scope.py", "--out-dir", VAR, "--work",
          "output/replicate/steps/variance_scope_work", "--jobs", "{jobs}"), 132.8,
         compare=_same(("ppp_variance_scope.csv",), VAR, RV),
         setup=tuple((f"{g}/{n}", f"{VAR}/{n}") for g, ns in _VAR_SETUP for n in ns),
         products=(f"{VAR}/stab_app_var.tex",),
         fresh=("output/replicate/steps/variance_scope_work",)),
    Step("table_s1_reference",
         "Table S1 rendered from "
         "reference/serial_dependence/ar1_size_power/",
         (PY + "ar1_size_power.py", "--table-from-reference", "--out-dir",
          "output/replicate/steps/table_s1"), 0.5,
         source="reference",
         regenerated_by=("full_ar1_size_power (replicate_sec_5.py or "
                         "replicate_sup_S4.py --full)"),
         check="ar1_size_power.py --table-from-reference: Table S1 rendered from the "
               "reference data by the function that renders it after the simulation; "
               "nothing is simulated or compared",
         products=("output/replicate/steps/table_s1/stab_ar1.tex",)),
    # ---- the full mode: the simulations regenerated from the recorded seeds ---------
    Step("full_tabcv_blocks",
         "Table 2: the 20 blocks of 165,000 draws of each of its 46 cells",
         (PY + "critical_values_blocks.py", "--blocks", "20", "--r-block", "165000",
          "--jobs", "{jobs}", "--name", "replicate_full"),
         600.0),
    Step("full_tabcv_table",
         "Table 2: the block means and the table rebuilt from the regenerated blocks, "
         "compared",
         (PY + "critical_values_blocks.py", "--aggregate", "--table", "--name",
          "replicate_full"), 0.9,
         compare=(("output/critical_values/replicate_full/critical_values_blocks.json",
                   f"{RC}/critical_values_blocks/critical_values_blocks.json"),
                  ("output/critical_values/replicate_full/tabcv_full_percentiles.csv",
                   f"{RC}/tabcv_full_percentiles.csv")),
         needs=("full_tabcv_blocks",),
         products=(
             "output/critical_values/replicate_full/tabcv_full_percentiles.csv",)),
    Step("full_overdetrend",
         "the power of the over-detrending shortcuts: 428,000,000 replications (the "
         "data of Figure 2 and of Section 4.4)",
         (PY + "overdetrend_power_precise.py", "--seed", "520260724", "--n-null",
          "320000000", "--n-printed", "80000000", "--n-figure", "2000000", "--jobs",
          "{jobs}", "--out",
          "output/replicate/steps/full_overdetrend/overdetrend_power.json"),
         8350.5,
         compare=(("output/replicate/steps/full_overdetrend/overdetrend_power.json",
                   f"{RC}/overdetrend_power.json"),)),
    Step("full_figure_overdetrend",
         "Figure 2 redrawn from the regenerated "
         "overdetrend_power.json, its points checked",
         (PY + "overdetrend_figure.py", "--power-data",
          "output/replicate/steps/full_overdetrend/overdetrend_power.json", "--out-dir",
          "output/replicate/steps/full_figure_overdetrend", "--check"), 2.2,
         check="overdetrend_figure.py --power-data <the regenerated file> --check: the "
               "points read from the regenerated overdetrend_power.json and the drawn "
               "points equal the reference data Figure_2_data.csv (byte-identical, or "
               "text and integers exactly and floats within 1e-9 max(1, |b|))",
         needs=("full_overdetrend",),
         products=("output/replicate/steps/full_figure_overdetrend/Figure_2.pdf",)),
    Step("full_oracle",
         "the oracle null quantile ladder, all 12 cells (T = 300, 1,200 and 9,600), "
         "recomputed",
         (PY + "oracle_sigma2_ladder.py", "--jobs", "{jobs}", "--out-dir",
          "output/replicate/steps/full_oracle", "--check"), 157.6,
         check="oracle_sigma2_ladder.py --check: the CSV byte-identical, or every "
               "number within 1e-9 (1 + |b|)"),
    Step("full_figure_size",
         "Figure 1: the 18 cells at N = 200,000, regenerated",
         (PY + "size_figure.py", "--jobs", "{jobs}", "--check", "--out-dir",
          "output/replicate/steps/full_figure_size"), 65.0,
         check="size_figure.py --check: the data file byte-identical, or its counts "
               "exactly and its critical values within 1e-9 (1 + |b|)",
         products=("output/replicate/steps/full_figure_size/Figure_1.pdf",)),
    Step("full_monotonicity",
         "the strict decrease of the envelope on the 637 configurations of Section 4, "
         "recomputed",
         (PY + "envelope_monotonicity.py", "--jobs", "{jobs}", "--out-dir",
          "output/replicate/steps/full_monotonicity", "--check"), 437.0,
         check="envelope_monotonicity.py --check: the four summaries byte-identical, "
               "or every number within 1e-9 (1 + |b|)"),
    Step("full_rule_comparison",
         "the rule comparison (78 configurations, R = 200,000), regenerated and "
         "compared",
         (PY + "rule_comparison.py", "--part", "all", "--state-dir",
          "output/replicate/steps/full_rule_comparison/state", "--jobs", "{jobs}"),
         392.7,
         compare=_same(_RULE, _RULE_OUT, f"{RC}/rule_comparison"), info_rc=_BYTE_RC),
    Step("full_maic_T45_100",
         "the MAIC lag-zero shares, part T45_100 (T = 45-100), compared row by row",
         (PY + "maic_lag_zero.py", "--part", "T45_100", "--check", "--jobs",
          "{jobs}"), 116.6,
         compare=tuple((f"{_MAIC_OUT}/{n}", f"{RC}/maic_lag_zero/{n}", Rows(2))
                       for n in _MAIC_FILES),
         info_rc=_BYTE_RC),
    Step("full_maic_T150_300",
         "the MAIC lag-zero shares, part T150_300 (T = 150-300), compared; then the "
         "whole table of the two parts",
         (PY + "maic_lag_zero.py", "--part", "T150_300", "--check", "--jobs",
          "{jobs}"), 70.7,
         compare=_same(_MAIC_FILES, _MAIC_OUT, f"{RC}/maic_lag_zero"), info_rc=_BYTE_RC,
         needs=("full_maic_T45_100",)),
    Step("full_ar1_size_power",
         "Table S1: size and power under AR(1) errors (R = 200,000 per entry), "
         "regenerated and compared",
         (PY + "ar1_size_power.py", "--part", "all", "--state-dir",
          "output/replicate/steps/full_ar1_size_power/state", "--jobs", "{jobs}",
          "--proceed-after-agreement-failure"), 424.4,
         compare=_same(_AR1, _AR1_OUT, f"{RS}/ar1_size_power"), info_rc=_BYTE_RC,
         products=(f"{_AR1_OUT}/stab_ar1.tex",),
         # the agreement checks of the const arm (z per cell, the cells beyond 2 se, the
         # joint chi-square): they fail and the run proceeds
         # (--proceed-after-agreement-failure); printed as information, the comparison
         # decides the step
         echo=("[ar1_size_power] cv_agreement", "[ar1_size_power] agreement checks")),
    _app_stage("full_app_null", "null", ("ppp_null.csv",), 357.5, ("app_data",)),
    _app_stage("full_app_bb", "bb", ("ppp_berger_boos_summary.csv",), 96.8,
               ("full_app_null",)),
    _app_stage("full_app_power", "power", ("ppp_power.csv", "ppp_power_band.csv"),
               632.8, ("full_app_null",)),
    _app_stage("full_app_joint", "joint", ("ppp_count_law.csv",), 754.1,
               ("full_app_power",)),
    Step("full_half_lives",
         "Tables S7-S8: the grid-bootstrap half-lives, regenerated and compared",
         (PY + "ppp_half_lives.py", "--out-dir", APP, "--work", APP_WORK, "--jobs",
          "{jobs}"), 143.1,
         compare=_same(("hl_cells.csv", "hl_nodes.csv"), APP, RM),
         needs=("app_data",)),
    _app_stage("full_app_sets", "sets", ("ppp_sets.csv",), 2.2,
               ("full_app_null", "full_half_lives"), work=False),
    _app_stage("full_app_readings", "readings", ("readings.csv",), 2.3,
               ("full_app_joint", "full_app_sets"), work=False),
    _sj_stage("full_sj_null", "null", ("sj_null.csv",), 69.4,
              ("sj_data", "full_app_null")),
    _sj_stage("full_sj_hl", "hl", (), 170.3, ("sj_data", "full_half_lives"),
              title="the crisis design S-JST, stage hl: the half-lives that the sets "
                    "of Table S6 read, regenerated"),
    _sj_stage("full_sj_sets", "sets", ("sj_sets.csv",), 1.5,
              ("full_sj_null", "full_sj_hl", "full_app_sets"), work=False),
    _sj_stage("full_sj_readings", "readings", ("sj_readings.csv",), 1.4,
              ("full_sj_sets",), work=False),
    Step("full_variance_scope",
         "Table S9: the variance profile on the regenerated application, compared",
         (PY + "ppp_variance_scope.py", "--out-dir", APP, "--work",
          "output/replicate/steps/full_variance_work", "--jobs", "{jobs}"),
         122.6,
         compare=_same(("ppp_variance_scope.csv",), APP, RV),
         needs=("full_app_readings", "full_sj_readings"),
         products=(f"{APP}/stab_app_var.tex",)),
    Step("full_c6",
         "the simulated critical values of the admission test C6 (Table S3), "
         "regenerated and compared",
         (PY + "c6_critical_values.py", "--out",
          "output/replicate/steps/full_c6/c6_critical_values.csv", "--jobs", "{jobs}"),
         144.2,
         compare=_same(("c6_critical_values.csv",), "output/replicate/steps/full_c6",
                       RAD)),
    Step("full_gate_sizes",
         "the size and power of the admission tests C4 and C6 (Table S3), regenerated "
         "and compared",
         (PY + "gate_sizes.py", "--out",
          "output/replicate/steps/full_gate_sizes/gate_sizes.csv", "--jobs", "{jobs}"),
         100.0,
         compare=_same(("gate_sizes.csv",), "output/replicate/steps/full_gate_sizes",
                       RAD)),
]
STEPS = {s.key: s for s in _steps}
assert len(STEPS) == len(_steps), "duplicate step key"
for _s in _steps:
    assert not _s.info_rc or _s.compare, f"{_s.key}: info_rc without compare"
    assert all(len(c) == 2 or isinstance(c[2], Rows) for c in _s.compare), _s.key
    assert all(c[1].startswith("reference/") for c in _s.compare), _s.key
    assert _s.source in ("recomputed", "reference"), _s.key
    assert (_s.source == "reference") == bool(_s.regenerated_by), _s.key
    assert all(n in STEPS for n in _s.needs), _s.key


def plan(keys):
    """The steps to run for the step keys given, in order: a step's prerequisites
    first, each step once."""
    out = []

    def add(k):
        if k in out:
            return
        for n in STEPS[k].needs:
            add(n)
        out.append(k)

    for k in keys:
        add(k)
    return out


def seconds(keys):
    """The estimated wall time (8 workers) of the plan of the step keys given."""
    return sum(STEPS[k].seconds for k in plan(keys))
