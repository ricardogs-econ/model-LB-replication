"""The prose numbers read off the article's grid, T from 45 to 300 (the abstract and
Sections 1, 4, 5 and 7 of the article; supplement S4.1).

No simulation and no exact computation: this producer reads only reference files that
the package ships and writes one JSON with the numbers the text prints about the
grid of break configurations. Every value is written at full precision; the text
rounds them. Every input but Table 2's blocks holds the grid's T only; the blocks
also hold T = 30 for m = 0, 1, 2 (the cells of Table S1's agreement checks), and the
interpolation of Table 2 is computed on the T it prints (T >= T_MIN_TABLE_2 = 45).

Inputs (reference/):
  calibration/cbar_surface_exact/cbar_surface_exact.csv   exact half-power points
  calibration/surface_summaries/surface_cells.csv         within-cell dispersion
  calibration/rule_comparison/rule_configs.csv            the rule against -7
  calibration/rule_comparison/rule_levels.csv             power of -7, its se
  calibration/critical_values_blocks/critical_values_blocks.json
                                                          tab:cv, block se
  calibration/maic_lag_zero/maic_k0_share_table.csv       MAIC lag-zero shares
  serial_dependence/ar1_size_power/ar1_maic_cv.csv        MAIC critical values
  serial_dependence/ar1_recalibration/ar1_recalibration_envelope.csv

The advantage of -7 over the rule is the column delta of the rule_configs table (the
power of -7 minus the power of the rule, in percentage points here); the gain of the
rule is its negative, so a negative "largest gain" says that the rule is nowhere ahead
(c = -5, -10).

The extremes of the rule against -7, of the MAIC lag-zero shares and of the critical
values at T in {60, 100} carry the Monte Carlo standard error (se) of the cell that
attains each, in the units of the value: for the rule, se_delta_w4 or se_delta_c<c>
of rule_configs.csv (x 100) and, for the power of -7, se_power of rule_levels.csv (its
power checked equal to rule_configs.csv's); for the shares, se of
maic_k0_share_table.csv; for the MAIC critical values, se_quantile of
ar1_maic_cv.csv; for Table 2's, se_between_blocks of critical_values_blocks.json (its
mean checked equal to the six-decimal value of ar1_maic_cv.csv); for their distance,
the root of the sum of the two squared se (the two come from independent
simulations). The largest error of Table 2's interpolation in 1/T, leaving one T out,
is (1 - t) q_lo + t q_hi - q_T at the attaining cell, q_lo and q_hi the 5% values at
the two neighbors of T and t the weight of the upper one; its se is
sqrt(((1 - t) se_lo)^2 + (t se_hi)^2 + se_T^2), from se_between_blocks of
critical_values_blocks.json (the covariance of the three values, which share random
numbers within each block, is left out).

Output (output/critical_values/prose_numbers_grid/ unless --out-dir is given):
prose_numbers_grid.json and prose_numbers_grid.manifest.json (the sha256 of every input
and of the output; no time). Two runs give identical bytes. --check compares the JSON
with reference/calibration/prose_numbers_grid/ byte for byte (LF bytes), or every
number to a relative 1e-9; the exit code is non-zero on any difference.

    python mlb/prose_numbers_grid.py [--out-dir D] [--check]
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mlb import paths  # noqa: E402

SECTION, OBJ = "critical_values", "prose_numbers_grid"
T_MIN_TABLE_2 = 45  # the smallest T Table 2 prints
RC = paths.reference_dir("critical_values")  # reference/calibration/
RS = paths.reference_dir("calibration_robustness")  # reference/serial_dependence/
SOURCES = dict(
    surface=RC / "cbar_surface_exact" / "cbar_surface_exact.csv",
    surface_cells=RC / "surface_summaries" / "surface_cells.csv",
    rule_configs=RC / "rule_comparison" / "rule_configs.csv",
    rule_levels=RC / "rule_comparison" / "rule_levels.csv",
    blocks=RC / "critical_values_blocks" / "critical_values_blocks.json",
    k0=RC / "maic_lag_zero" / "maic_k0_share_table.csv",
    ar1_maic_cv=RS / "ar1_size_power" / "ar1_maic_cv.csv",
    ar1_recal=RS / "ar1_recalibration" / "ar1_recalibration_envelope.csv",
)
# the outputs whose reference file is in reference/ (what --check compares)
REFERENCE_OUTPUTS = (f"{OBJ}.json",)


def read_csv(path):
    """A CSV with every float parsed exactly (round trip), as the csv module would."""
    return pd.read_csv(path, float_precision="round_trip")


def f(x):
    return float(x)


def r_squared(y, fitted):
    return 1.0 - float(((y - fitted) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


# ----------------------------------------------------------------- the exact surface
def surface_numbers(s):
    lo = s.cbar_half_exact.idxmin()
    dev = (s.cbar_half_exact + 7.0).abs()
    far = dev.idxmax()
    cell_mean = s.groupby(["m", "T"])["cbar_half_exact"].transform("mean")
    return dict(
        n_configurations=int(len(s)),
        minimum=dict(
            value=f(s.cbar_half_exact[lo]), m=int(s.m[lo]), T=int(s["T"][lo])
        ),
        largest_displacement_from_minus7=dict(
            value=f(dev[far]), m=int(s.m[far]), T=int(s["T"][far])
        ),
        r_squared_of_the_cell_means=r_squared(
            s.cbar_half_exact.to_numpy(), cell_mean.to_numpy()
        ),
    )


# ------------------------------------------------------------ within-cell dispersion
def dispersion_numbers(cells, surface):
    multi = cells[cells.n_configs > 1].reset_index(drop=True)
    big, small = multi["range"].idxmax(), multi["range"].idxmin()
    # the cell means and ranges recomputed from the exact surface
    chk = (
        surface.groupby(["m", "T"])["cbar_half_exact"]
        .agg(mean_surface="mean", min_surface="min", max_surface="max",
             count="count")
        .reset_index()
        .merge(cells, on=["m", "T"])
    )
    return dict(
        n_cells=int(len(cells)),
        n_cells_with_several_configurations=int(len(multi)),
        mean_range=f(multi["range"].mean()),
        mean_sd_ddof1=f(multi["sd_ddof1"].mean()),
        largest_range=dict(
            value=f(multi["range"][big]), m=int(multi.m[big]), T=int(multi["T"][big])
        ),
        smallest_range=dict(
            value=f(multi["range"][small]),
            m=int(multi.m[small]),
            T=int(multi["T"][small]),
        ),
        agrees_with_the_surface=dict(
            max_abs_difference_of_means=f(
                (chk["mean_surface"] - chk["mean"]).abs().max()
            ),
            max_abs_difference_of_ranges=f(
                ((chk["max_surface"] - chk["min_surface"]) - chk["range"]).abs().max()
            ),
            counts_equal=bool((chk["count"] == chk["n_configs"]).all()),
        ),
    )


# ----------------------------------------------------------------- the rule against -7
def power_se(levels, row, c):
    """se_power of rule_levels.csv for the power of -7 at c in the configuration of the
    rule_configs.csv row `row`, whose power_minus7_c<c> it must equal."""
    hit = levels[
        (levels.config == f"{row['config_key']}__{row['kind']}")
        & (levels.node_role == "-7")
        & (levels.c == float(c))
    ]
    if len(hit) != 1 or f(hit.power.iloc[0]) != f(row[f"power_minus7_c{c}"]):
        raise ValueError(
            f"rule_levels.csv: no single power of -7 at c = {c} equal to that of "
            f"{row['config_key']} in rule_configs.csv"
        )
    return f(hit.se_power.iloc[0])


def rule_numbers(rule_configs, levels):
    a = rule_configs[rule_configs.X == "cbarbar_half(m,T)"].reset_index(drop=True)
    gain = -a.delta_w4
    best = gain.idxmax()
    worst = a.delta_w4.idxmax()
    out = dict(
        n_cells=int(len(a)),
        kinds={k: int(v) for k, v in sorted(a.kind.value_counts().items())},
        classes={k: int(v) for k, v in
                 sorted(a.classification.value_counts().items())},
        largest_gain_of_the_rule_pp=dict(
            value=f(gain[best] * 100),
            se=f(a.se_delta_w4[best] * 100),
            m=int(a.m[best]),
            T=int(a["T"][best]),
            kind=str(a.kind[best]),
        ),
        largest_advantage_of_minus7_pp=dict(
            value=f(a.delta_w4[worst] * 100),
            se=f(a.se_delta_w4[worst] * 100),
            m=int(a.m[worst]),
            T=int(a["T"][worst]),
            kind=str(a.kind[worst]),
        ),
    )
    for c in (-5, -10, -15, -20):
        d, se = a[f"delta_c{c}"], a[f"se_delta_c{c}"]
        z = d / se
        hi, lo = d.idxmax(), d.idxmin()
        power = a[f"power_minus7_c{c}"]
        low = power.idxmin()
        out[f"c{c}"] = dict(
            minus7_ahead=int((z > 2).sum()),
            indistinguishable=int((z.abs() <= 2).sum()),
            rule_ahead=int((z < -2).sum()),
            largest_advantage_of_minus7_pp=dict(
                value=f(d[hi] * 100),
                se=f(se[hi] * 100),
                m=int(a.m[hi]),
                T=int(a["T"][hi]),
            ),
            largest_gain_of_the_rule_pp=dict(
                value=f(-d[lo] * 100),
                se=f(se[lo] * 100),
                m=int(a.m[lo]),
                T=int(a["T"][lo]),
            ),
            smallest_power_of_minus7=dict(
                value=f(power[low]),
                se=power_se(levels, a.loc[low], c),
                m=int(a.m[low]),
                T=int(a["T"][low]),
            ),
        )
    return out


# ----------------------------------------------------------------- tab:cv, block se
def leave_one_out(table, tmin):
    """Linear interpolation in 1/T between the neighbors of each interior T, by m:
    the error (interpolated minus tabulated value) and its se from the se of the three
    values, sqrt(((1 - t) se_lo)^2 + (t se_hi)^2 + se_T^2)."""
    rows = []
    for m, g in table.groupby("m"):
        g = g[g["T"] >= tmin].sort_values("T")
        Ts, cv, se = g["T"].to_numpy(), g.cv.to_numpy(), g.se.to_numpy()
        for i in range(1, len(Ts) - 1):
            x0, x1, x = 1 / Ts[i - 1], 1 / Ts[i + 1], 1 / Ts[i]
            pred = cv[i - 1] + (cv[i + 1] - cv[i - 1]) * (x - x0) / (x1 - x0)
            t = (x - x0) / (x1 - x0)
            se_error = np.sqrt(
                ((1 - t) * se[i - 1]) ** 2 + (t * se[i + 1]) ** 2 + se[i] ** 2
            )
            rows.append(
                {
                    "m": int(m),
                    "T": int(Ts[i]),
                    "error": f(pred - cv[i]),
                    "se": f(se_error),
                }
            )
    return pd.DataFrame(rows)


def blocks_numbers(blocks):
    rows = []
    for key, cell in blocks["cells"].items():
        m, T, _const = key.split("_")
        rows.append(
            dict(
                m=int(m[1:]),
                T=int(T[1:]),
                cv=cell["mzt_5.0%"]["mean"],
                se=cell["mzt_5.0%"]["se_between_blocks"],
            )
        )
    out = dict()
    loo = leave_one_out(pd.DataFrame(rows), T_MIN_TABLE_2)
    k = loo.error.abs().idxmax()
    out["interpolation_in_1_over_T_leaving_one_T_out"] = dict(
        largest_abs_error=f(loo.error.abs()[k]),
        largest_abs_error_se=f(loo.se[k]),
        m=int(loo.m[k]),
        T=int(loo["T"][k]),
        n=int(len(loo)),
    )
    return out


# ----------------------------------------------------------------- MAIC lag-zero shares
def k0_numbers(k0):
    a = k0
    lo, hi = a.share_k0.idxmin(), a.share_k0.idxmax()
    return dict(
        n_cells=int(len(a)),
        minimum=dict(
            value=f(a.share_k0[lo]),
            se=f(a.se[lo]),
            m=int(a.m[lo]),
            T=int(a["T"][lo]),
        ),
        maximum=dict(
            value=f(a.share_k0[hi]),
            se=f(a.se[hi]),
            m=int(a.m[hi]),
            T=int(a["T"][hi]),
        ),
    )


# ----------------------------------------------------------------- AR(1) robustness
def extremes_se(values, ses):
    """[se at the minimum, se at the maximum] of `values`: the se of the attaining
    cell (the largest when cells tie)."""
    return [f(ses[values == values.min()].max()), f(ses[values == values.max()].max())]


def ar1_numbers(cv, blocks):
    a = cv[cv["T"].isin([60, 100]) & (cv.m <= 2)].copy()
    a["distance"] = (a.cv_const_tabcv - a.cv_maic).abs()
    tab = [blocks["cells"][f"m{m}_T{T}_const"]["mzt_5.0%"] for m, T in zip(a.m, a["T"])]
    for t, v, m, T in zip(tab, a.cv_const_tabcv, a.m, a["T"]):
        if abs(t["mean"] - v) > 5e-7:
            raise ValueError(f"tab:cv at m = {m}, T = {T}: {v} is not the block mean")
    a["se_tabcv"] = [t["se_between_blocks"] for t in tab]
    a["se_distance"] = np.sqrt(a.se_quantile**2 + a.se_tabcv**2)
    return dict(
        maic_critical_values_T60_100_m_le_2=dict(
            n=int(len(a)),
            maic=[f(a.cv_maic.min()), f(a.cv_maic.max())],
            tabcv=[f(a.cv_const_tabcv.min()), f(a.cv_const_tabcv.max())],
            distance=[f(a.distance.min()), f(a.distance.max())],
            maic_se=extremes_se(a.cv_maic, a.se_quantile),
            tabcv_se=extremes_se(a.cv_const_tabcv, a.se_tabcv),
            distance_se=extremes_se(a.distance, a.se_distance),
        ),
    )


def recal_numbers(en):
    e = en[en["T"].isin([60, 100])]
    half = e[e.rho == 0.5]
    far = half.delta.abs().idxmax()
    return dict(
        rho_half=dict(
            n=int(len(half)),
            median_abs_delta=f(half.delta.abs().median()),
            largest_abs_delta=dict(
                value=f(half.delta.abs()[far]),
                T=int(half["T"][far]),
                m=int(half.m[far]),
            ),
        ),
    )


def sha256_lf(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def numbers():
    surface = read_csv(SOURCES["surface"])
    with open(SOURCES["blocks"], encoding="utf-8") as fh:
        blocks = json.load(fh)
    res = dict(
        surface=surface_numbers(surface),
        within_cell_dispersion=dispersion_numbers(read_csv(SOURCES["surface_cells"]),
                                                  surface),
        rule_against_minus7=rule_numbers(
            read_csv(SOURCES["rule_configs"]), read_csv(SOURCES["rule_levels"])
        ),
        tabcv_blocks=blocks_numbers(blocks),
        maic_lag_zero=k0_numbers(read_csv(SOURCES["k0"])),
        ar1_robustness=ar1_numbers(read_csv(SOURCES["ar1_maic_cv"]), blocks),
        ar1_recalibration=recal_numbers(read_csv(SOURCES["ar1_recal"])),
    )
    return res


RTOL = 1e-9


def _same_json(x, y):
    """Structure, strings and integers equal; floats to a relative RTOL."""
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same_json(x[k], y[k]) for k in x)
    if isinstance(x, list) and isinstance(y, list):
        return len(x) == len(y) and all(map(_same_json, x, y))
    if isinstance(x, float) or isinstance(y, float):
        ok = isinstance(x, (int, float)) and isinstance(y, (int, float))
        return ok and abs(x - y) <= RTOL * max(1.0, abs(x), abs(y))
    return type(x) is type(y) and x == y


def _lf(data):
    return data.replace(b"\r\n", b"\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="output folder (default output/critical_values/prose_numbers_grid/)",
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare the JSON with the reference (LF bytes, or floats to 1e-9)",
    )
    a = ap.parse_args(argv)
    res = numbers()
    out = Path(a.out_dir) if a.out_dir else paths.out_dir(SECTION) / OBJ
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{OBJ}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
        fh.write("\n")
    man = dict(
        output=path.name,
        output_sha256=sha256_lf(path),
        inputs_sha256_lf={k: sha256_lf(p) for k, p in sorted(SOURCES.items())},
    )
    with open(out / f"{OBJ}.manifest.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(man, fh, indent=1)
        fh.write("\n")
    print(json.dumps(res, indent=1, sort_keys=True))
    if not a.check:
        return 0
    ref_dir = paths.reference_dir(SECTION) / OBJ
    ok = True
    for name in REFERENCE_OUTPUTS:
        r = ref_dir / name
        if not r.is_file():
            word = "MISSING"
        elif _lf(r.read_bytes()) == _lf((out / name).read_bytes()):
            word = "identical"
        elif _same_json(json.loads(r.read_bytes()),
                        json.loads((out / name).read_bytes())):
            word = "equal within 1e-9"
        else:
            word = "DIFFERENT"
        print(f"--check: {name}: {word}")
        ok = ok and word in ("identical", "equal within 1e-9")
    print(f"--check: {'PASS' if ok else 'FAIL'} against {ref_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
