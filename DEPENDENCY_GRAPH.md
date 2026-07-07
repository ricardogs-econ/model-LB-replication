# Dependency graph — model-LB replication package

This file documents how the modules and data artifacts depend on one another, so
a replicator can see at a glance what produces what and in which order to run.

## Execution order (topological)

```
replicate_section3_4.py   →   cbar_surface.csv   →   replicate_section5.py
        (calibration)              (bridge)              (robustness)
                                       │
                                       └──────────────→   replicate_section6.py
                                                              (PPP application)
```

Run Section 3–4 first; it writes `cbar_surface.csv`, the input to Sections 5 and 6.

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
   │               │               │                           (§5 power table)
   │               │ runpy         │ runpy
   │               ▼               ▼
   │        robustness.py    boot_ppp_cbar.py ──┐
   │        size_power_...    hl_median_unbiased │ (fallback: pure-Python
   │                          figs_ppp.py        │  kernel if numba absent)
   │                          ppp_sweep_bis.py   │
   │                                             │
   └──── writes ─────────────────────────────────
              cbar_checkpoints/cbar_surface.csv
              (copied to ./cbar_surface.csv)
```

## Data flow

| Artifact | Produced by | Consumed by | Distributed |
|---|---|---|---|
| `cbar_surface.csv` | `replicate_section3_4.py --full` | §5, §6, `mlb_core`, `run_model_lb`, `size_power_cbar_comparison` | yes (3-cell placeholder; regenerate) |
| `ppp_panel.csv` | BIS + World Bank build | §6 (`boot`, `hl`, `figures`) | yes |
| `exog_dates.csv` | primary central-bank sources | §6 (`sweep`, `boot`, `hl`, `figures`) | yes |
| `ppp_ar_diagnostic.csv` | AR(p) selection on the panel | §6 (`boot`, `hl`) | yes |
| `power_comparison.csv` | `replicate_section5.py power` | `size_power_cbar_comparison` figure | generated |
| `hl_results.csv`, `hl_results_wild.csv` | `replicate_section6.py hl` | `figures` stage | generated (.gitignore) |

## Import edges (verified)

- `replicate_section3_4.py`  → `import mlb_core` (calls `run_grid`, `aggregate`, `warm_up_numba`)
- `replicate_section5.py`    → `runpy` of `robustness`, `size_power_cbar_comparison`
- `replicate_section6.py`    → `runpy` of `ppp_sweep_bis`, `boot_ppp_cbar`, `hl_median_unbiased`, `figs_ppp`
- `run_model_lb.py`          → `import mlb_core` (calls `run_test`)
- `size_power_cbar_comparison.py` → `import mlb_core` (build_z_nb, mstats_nb, gen_dgp_nb)
- `boot_ppp_cbar.py`         → `import mlb_core` (numba kernels); pure-Python fallback if absent
- `robustness.py`            → `import mlb_core` (kernels; never reimplemented)
- `boot_ppp_cbar.py`         → fallback `import mlb_kernel` (pure-Python) when numba absent
- `mlb_kernel.py`            → `import numpy` only (no local deps; the fallback leaf)

All edges above were checked to resolve to existing modules; no `import` or
`runpy` target is missing.
