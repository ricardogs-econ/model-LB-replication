# Dependency graph — model-LB replication package

This file documents how the modules and data artifacts depend on one another, so
a replicator can see at a glance what produces what and in which order to run.

The package separates **compute** from **figures**: the compute modules run the
Monte Carlo / estimation and write CSV artifacts; a single figure module,
`generate_figures.py`, reads those CSVs and renders every figure. No figure is
drawn inside a compute module.

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
   power_comparison.csv, ppp_panel.csv,   generate_figures.py  →  the 5 figure PDFs
   exog_dates.csv, hl_results*.csv    ───────►  (reads CSVs only)
```

Run Section 3–4 first; it writes `cbar_surface.csv`, the input to Sections 5 and
6 and to the figures. Run `generate_figures.py` last, after the CSVs exist.

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
                        Reads the CSV artifacts and writes the figure PDFs.
```

## Data flow

| Artifact | Produced by | Consumed by | Distributed |
|---|---|---|---|
| `cbar_surface.csv` | `replicate_section3_4.py --full` | §5, §6, `mlb_core`, `run_model_lb`, `size_power_cbar_comparison`, `generate_figures` (Fig 2) | yes (full surface: 427 configs, 46 cells) |
| `limiting_density.csv` | `replicate_section3_4.py --limiting-density` | `generate_figures` (Fig 1) | generated (.gitignore) |
| `power_comparison.csv` | `replicate_section5.py power` (`size_power_cbar_comparison`) | `generate_figures` (Fig 3) | generated (.gitignore) |
| `ppp_panel.csv` | BIS + World Bank build | §6 (`boot`, `hl`); `generate_figures` (Fig 4) | yes |
| `exog_dates.csv` | primary central-bank sources | §6 (`sweep`, `boot`, `hl`); `generate_figures` (Fig 4) | yes |
| `ppp_ar_diagnostic.csv` | AR(p) selection on the panel | §6 (`boot`, `hl`) | yes |
| `hl_results.csv`, `hl_results_wild.csv` | `replicate_section6.py hl` | `generate_figures` (Fig 5) | generated (.gitignore) |
| `boot_out/calib/surface_ppp_boot.csv` | `boot_ppp_cbar.py --full --empirical` | Table `pppsurface` | yes (bundled) |
| `boot_out/empirical/ppp_empirical.csv` | (same run) | Table `ppp` | yes (bundled) |

## Import edges (verified)

- `replicate_section3_4.py`  → `import mlb_core` (calls `run_grid`, `aggregate`, `warm_up_numba`; and the kernels `build_z_nb`/`gen_dgp_nb`/`mstats_nb`/`break_pos_from_lambdas` for the Figure-1 null law)
- `replicate_section5.py`    → `runpy` of `robustness`, `size_power_cbar_comparison`
- `replicate_section6.py`    → `runpy` of `ppp_sweep_bis`, `boot_ppp_cbar`, `hl_median_unbiased`
- `generate_figures.py`      → `import numpy, matplotlib` only (reads CSVs; no `mlb_core`, no numba)
- `run_model_lb.py`          → `import mlb_core` (calls `run_test`)
- `size_power_cbar_comparison.py` → `import mlb_core` (build_z_nb, mstats_nb, gen_dgp_nb)
- `boot_ppp_cbar.py`         → `import mlb_core` (numba kernels); pure-Python fallback if absent
- `robustness.py`            → `import mlb_core` (kernels; never reimplemented)
- `boot_ppp_cbar.py`         → fallback `import mlb_kernel` (pure-Python) when numba absent
- `mlb_kernel.py`            → `import numpy` only (no local deps; the fallback leaf)

All edges above were checked to resolve to existing modules; no `import` or
`runpy` target is missing.
