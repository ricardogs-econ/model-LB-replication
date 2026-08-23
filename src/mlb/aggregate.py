"""The lambda-aggregation rule (brief s.2.2), named and hard-coded.

RECOVERED AND RE-VERIFIED RULE (Phase 1, 2026-07-28): each printed (m,T)
cell of the checkpoint tables is the ARITHMETIC MEAN over the
lambda-configurations within that cell, ROUNDED AT THE TABLE'S NATIVE
PRECISION — 1 decimal for the cbar tables (tab_cbar_mT*.tex), 2 decimals for
the cv tables (tab_cv5_mzt*.tex). Verified 46/46 cells exact on all four
checkpoint tables (const and maic); the MEDIAN rule fails on all four
(21-33/46, worst deviation 0.020-0.44). Lambda-config counts per cell are
heterogeneous: {1, 5, 6, 7, 15, 20}.

    mean_over_lambda_at_native_precision(values, decimals)

is therefore THE aggregation convention of the published (m,T) tables and
the one the Phase-2b/3 regeneration applies when lambda-mode=grid.
"""

import csv
from collections import defaultdict

# NOTE: no `import statistics` here — src/mlb/statistics.py (the brief's
# target filename) would shadow the stdlib module if this file were executed
# as a script from inside the package directory. Run the CLI as
#     python -m src.mlb.aggregate ...
# from the repository root.


def mean_over_lambda_at_native_precision(values, decimals):
    """Arithmetic mean over lambda-configurations, rounded at the printed
    precision of the target table (1 dp for cbar tables, 2 dp for cv
    tables). `values`: the per-lambda-config cell values for one (m,T)."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), decimals)


def aggregate_surface(csv_path, value_col, method, decimals):
    """Aggregate a cbar_surface-style CSV to an {(m,T): value} table using
    the recovered rule. Rows are filtered to sigma2_method == method."""
    cells = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("sigma2_method") != method:
                continue
            cells[(int(row["m"]), int(row["T"]))].append(float(row[value_col]))
    return {
        k: mean_over_lambda_at_native_precision(v, decimals)
        for k, v in sorted(cells.items())
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True, help="cbar_surface-style CSV")
    ap.add_argument("--col", default="mzt_cv5")
    ap.add_argument("--method", default="const", choices=["const", "maic"])
    ap.add_argument("--decimals", type=int, default=2)
    a = ap.parse_args()
    table = aggregate_surface(a.csv, a.col, a.method, a.decimals)
    Ts = sorted({T for (_, T) in table})
    ms = sorted({m for (m, _) in table})
    print("m\\T  " + "  ".join(f"{T:>7}" for T in Ts))
    for m in ms:
        print(
            f"{m:>3}  "
            + "  ".join(
                f"{table.get((m, T), float('nan')) if table.get((m, T)) is not None else '---':>7}"
                for T in Ts
            )
        )
