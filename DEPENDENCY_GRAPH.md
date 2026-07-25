# Dependency graph — model-LB replication package

This file documents how the modules and data artifacts depend on one another, so
a replicator can see at a glance what produces what and in which order to run.

The package separates **compute** from **figures** for three of the current
manuscript's five figures: the compute modules run the Monte Carlo /
estimation and write CSV artifacts; a single figure module,
`generate_figures.py`, reads those CSVs and renders `Figure_3.pdf`,
`Figure_S1.pdf`, and `Figure_S2.pdf` (plus two archived `legacy_figures/`
outputs). The other two, `Figure_1.pdf` and `Figure_2.pdf`, are each a short,
self-contained Monte Carlo + plot in their own script
(`overdetrend_power.py`, `materiality_figure.py`) — fast enough (under a
minute) that splitting compute from plotting was not worth the extra moving
part for them. See `README.md`'s "Figures" section for the full map.

## Execution order (topological)

```
replicate_section3_4.py  →  cbar_surface.csv  →  replicate_section5.py
      (calibration)             (bridge)             (robustness)
            │                       │
            │                       └───────────────→  replicate_section6.py
            │                                              (PPP application)
            │
            └── (--limiting-density) → limiting_density.csv
                                                 │
   cbar_surface.csv, limiting_density.csv,       ▼
   power_comparison.csv, ppp_panel.csv,   generate_figures.py  →  Figure_3, Figure_S1,
   exog_dates.csv, hl_results*.csv    ───────►  (reads CSVs only)   Figure_S2 + legacy/

materiality_c7_vs_surface.py  →  overdetrend_power.py     →  Figure_1.pdf
        (shared core)         →  materiality_figure.py    →  Figure_2.pdf
                                  (each self-contained: own Monte Carlo, own plot)
```

Run Section 3–4 first; it writes `cbar_surface.csv`, the input to Sections 5 and
6 and to the CSV-backed figures. Run `generate_figures.py` last, after the CSVs
exist. `overdetrend_power.py` and `materiality_figure.py` have no CSV
dependency and can run at any time.

## Module dependency graph

```
                          ┌──────────────────┐
                          │    mlb_core.py   │   kernel library (numba GLS,
                          │   (the kernel)   │   calibration, run_test, gates)
                          └─────────┬────────┘
        imported by                │
   ┌───────────────┬───────────────┼───────────────┬──────────────────┐
   │               │               │               │                  │
   ▼               ▼               ▼               ▼                  ▼
replicate_     replicate_     replicate_     run_model_lb.py   size_power_cbar_
section3_4     section5       section6       (user CLI)        comparison.py
   │               │               │                           (§5 power data)
   │               │ runpy         │ runpy
   │               ▼               ▼
   │        robustness.py    boot_ppp_cbar.py ──┐
   │        size_power_...    hl_median_unbiased │ (fallback: pure-Python
   │                          ppp_sweep_bis.py   │  kernel if numba absent)
   │                                             │
   └──── writes ─────────────────────────────────
              cbar_checkpoints/cbar_surface.csv
              (copied to ./cbar_surface.csv)

generate_figures.py  →  imports numpy + matplotlib ONLY (no mlb_core, no numba).
                        Reads the CSV artifacts and writes Figure_3/S1/S2 (+ legacy/).

materiality_c7_vs_surface.py  →  imports numpy only (no mlb_core, no numba;
                                  standalone re-implementation, deliberately
                                  not sharing code with the kernel).
        imported by
   ┌─────────────────────┬─────────────────────┐
   ▼                     ▼
overdetrend_power.py   materiality_figure.py
(→ Figure_1.pdf)       (→ Figure_2.pdf)

ppp_two_axis_columns.py  →  imports numpy (+ pandas for --data) ONLY. Also
                            standalone (no mlb_core import) by design, so that
                            reproducing the published Table `tab:ppp` columns
                            is an independent check, not a tautology.
```

## Data flow

| Artifact | Produced by | Consumed by | Distributed |
|---|---|---|---|
| `cbar_surface.csv` | `replicate_section3_4.py --full` | §5, §6, `mlb_core`, `run_model_lb`, `size_power_cbar_comparison`, `generate_figures` (legacy `Figure_legacy_cbar_surface.pdf`) | yes (full surface: 427 configs, 46 cells) |
| `limiting_density.csv` | `replicate_section3_4.py --limiting-density` | `generate_figures` (Fig S1) | generated (.gitignore) |
| `power_comparison.csv` | `replicate_section5.py power` (`size_power_cbar_comparison`) | `generate_figures` (legacy `Figure_legacy_power_comparison.pdf`) | generated (.gitignore) |
| `ppp_panel.csv` | BIS + World Bank build | §6 (`boot`, `hl`); `generate_figures` (Fig 3); `ppp_two_axis_columns.py` | yes |
| `exog_dates.csv` | primary central-bank sources | §6 (`sweep`, `boot`, `hl`); `generate_figures` (Fig 3) | yes |
| `ppp_ar_diagnostic.csv` | `select_ar_order.py` | §6 (`boot`, `hl`); `ppp_two_axis_columns.py` (`p` source by default as of v1.3.0, `k_bic_cq` column; `--recompute-p` drops this dependency) | yes |
| `hl_results.csv`, `hl_results_wild.csv` | `replicate_section6.py hl` | `generate_figures` (Fig S2) | generated (.gitignore) |
| `boot_out/calib/surface_ppp_boot.csv` | `boot_ppp_cbar.py --full --empirical` | Table `pppsurface` | yes (bundled) |
| `boot_out/empirical/ppp_empirical.csv` | (same run) | Table `ppp` | yes (bundled) |
| `modulus_table.csv`, `modulus_table_T52.csv`, `modulus_table_small.csv` | `modulus_numerics.py` | Supplement §S2, Theorem S2.1's observable implication | yes (bundled) |

## Import edges (verified)

- `replicate_section3_4.py`  → `import mlb_core` (calls `run_grid`, `aggregate`, `warm_up_numba`; and the kernels `build_z_nb`/`gen_dgp_nb`/`mstats_nb`/`break_pos_from_lambdas` for the Figure S1 null law)
- `replicate_section5.py`    → `runpy` of `robustness`, `size_power_cbar_comparison`
- `replicate_section6.py`    → `runpy` of `ppp_sweep_bis`, `boot_ppp_cbar`, `hl_median_unbiased`
- `generate_figures.py`      → `import numpy, matplotlib` only (reads CSVs; no `mlb_core`, no numba)
- `run_model_lb.py`          → `import mlb_core` (calls `run_test`)
- `size_power_cbar_comparison.py` → `import mlb_core` (build_z_nb, mstats_nb, gen_dgp_nb)
- `boot_ppp_cbar.py`         → `import mlb_core` (numba kernels); pure-Python fallback if absent
- `robustness.py`            → `import mlb_core` (kernels; never reimplemented)
- `boot_ppp_cbar.py`         → fallback `import mlb_kernel` (pure-Python) when numba absent
- `mlb_kernel.py`            → `import numpy` only (no local deps; the fallback leaf)
- `overdetrend_power.py`     → `import materiality_c7_vs_surface` (detrender, Lmat, mzt) + `numpy, matplotlib`; no `mlb_core`
- `materiality_figure.py`    → `import materiality_c7_vs_surface` (detrender, Lmat, mzt, CV_ASY) + `numpy, matplotlib`; no `mlb_core`
- `materiality_c7_vs_surface.py` → `import numpy` only (standalone re-implementation; no `mlb_core`, by design)
- `modulus_numerics.py`      → `import numpy, scipy.signal.lfilter`; `--selftest` additionally imports `assumption1_numerics` (A_general) for a cross-check
- `assumption1_numerics.py`  → `import numpy` only
- `ppp_two_axis_columns.py`  → `import numpy` (+ `pandas` inside `load_panel`/`load_ar_diagnostic`, for `--data`/`--ar-diagnostic`); no `mlb_core` (no CODE import from the rest of the package, so the recomputed MZt/cv/p-value are an independent check). DOES read `ppp_ar_diagnostic.csv` as a DATA input by default as of v1.3.0 (see H3 in its module docstring) -- `--recompute-p` removes even that, at the cost of no longer matching the published `p`.

All edges above were checked to resolve to existing modules; no `import` or
`runpy` target is missing.
