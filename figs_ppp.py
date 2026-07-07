#!/usr/bin/env python3
# =============================================================================
# figs_ppp.py -- production figures for the PPP application (Section 6).
# -----------------------------------------------------------------------------
# Figure 1 (fig_rer_series.pdf): the eight real exchange rates q_it, 1973-2024,
#   with the H1 step deterministic  mu_i0 + sum_j delta_ij * DU_ij,t  (OLS at the
#   exogenous currency-regime break dates) overlaid.  The step is estimated on
#   the SAME window as the test (start_year onward), so the displayed
#   deterministic matches the one the test detrends against.
#
# upper limits mapped by the includes-one flags (not alpha_ci_hi);
#        xmax=80 so the largest finite upper (CAD, 69y) renders uncapped.
# Figure 2 (fig_hl_forest.pdf): forest plot of the 95% Hansen grid-t bootstrap
#   half-life confidence intervals, constant-mean (MP) vs exogenous-level-break
#   (LB) specification.  The LB interval is the heteroskedasticity-robust WILD
#   (Rademacher) bootstrap -- the inference reported in Table 6; MP is the
#   recursive baseline (its infinite upper limit is scheme-invariant).  An arrow
#   marks an infinite upper limit; currencies whose interval collapses from
#   unbounded (MP) to bounded (LB) are flagged.
#
# EQUIVALENCE / PROVENANCE.  The window is an explicit argument (--start-year,
# default 1973), applied to the panel before the OLS step estimation, so the
# figure is reproducible from the same convention as boot_ppp_cbar_production.py
# and hl_median_unbiased.py.  The break-date design matrix Z = [1, DU_1, ...] is
# built exactly as there (indicator on the exogenous years within the window).
#
# Inputs (all read from --data-dir, default '.'):
#   ppp_panel.csv        currency, year, q            (real exchange rate)
#   exog_dates.csv       currency, break_year, ...    (exogenous regime dates)
#   hl_results.csv       recursive grid-t CIs         (MP baseline)
#   hl_results_wild.csv  wild grid-t CIs              (LB, preferred)
#
# Usage:
#   python figs_ppp.py --start-year 1973 --data-dir . --out-dir .
# =============================================================================
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

CURS = ["AUD", "CAD", "CHF", "GBP", "JPY", "NOK", "NZD", "SEK"]
ROGOFF_LO, ROGOFF_HI = 3.0, 5.0        # Rogoff (1996) consensus half-life band


# ---------------------------------------------------------------------------
# I/O helpers (robust: explicit errors instead of silent failure)
# ---------------------------------------------------------------------------
def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"required input not found: {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"input is empty: {path}")
    return rows


def load_inputs(data_dir: Path):
    panel: dict[str, list[tuple[int, float]]] = {}
    for r in _read_csv(data_dir / "ppp_panel.csv"):
        panel.setdefault(r["currency"], []).append(
            (int(r["year"]), float(r["q"])))
    dates: dict[str, list[int]] = {}
    for r in _read_csv(data_dir / "exog_dates.csv"):
        dates.setdefault(r["currency"], []).append(int(r["break_year"]))
    hl = {r["currency"]: r for r in _read_csv(data_dir / "hl_results.csv")}
    wild_path = data_dir / "hl_results_wild.csv"
    if wild_path.exists():
        hlw = {r["currency"]: r for r in _read_csv(wild_path)}
    else:
        print(f"[warn] {wild_path.name} not found; forest plot LB interval "
              f"falls back to the recursive bootstrap.", file=sys.stderr)
        hlw = hl
    return panel, dates, hl, hlw


# ---------------------------------------------------------------------------
# design matrix (identical convention to the test kernels)
# ---------------------------------------------------------------------------
def build_Z(years: np.ndarray, break_years: list[int]) -> np.ndarray:
    cols = [np.ones(len(years))]
    for b in sorted(break_years):
        if years[0] < b <= years[-1]:
            cols.append((years >= b).astype(float))
    return np.column_stack(cols)


def scalar_hl(alpha: float) -> float:
    """Half-life ln(0.5)/ln(alpha); +inf if alpha >= 1 (or non-positive)."""
    return float(np.log(0.5) / np.log(alpha)) if 0.0 < alpha < 1.0 else np.inf


# ---------------------------------------------------------------------------
# Figure 1: real exchange rates with the H1 step deterministic
# ---------------------------------------------------------------------------
def fig_rer_series(panel, dates, start_year, out_path):
    fig, axes = plt.subplots(4, 2, figsize=(6.6, 8.2), sharex=True)
    for ax, cur in zip(axes.flat, CURS):
        obs = sorted(o for o in panel[cur] if o[0] >= start_year)   # WINDOW
        yrs = np.array([o[0] for o in obs])
        q = np.array([o[1] for o in obs])
        bry = [b for b in sorted(dates.get(cur, [])) if yrs[0] < b <= yrs[-1]]
        Z = build_Z(yrs, bry)
        beta, *_ = np.linalg.lstsq(Z, q, rcond=None)   # OLS on the SAME window
        step = Z @ beta
        ax.plot(yrs, q, color="black", lw=0.9)
        ax.plot(yrs, step, color="0.45", lw=2.0)
        for b in bry:
            ax.axvline(b, color="0.6", lw=0.7, ls="--")
        ax.set_title(f"{cur}  ($m={len(bry)}$: {', '.join(map(str, bry))})",
                     fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("year", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: forest plot of half-life confidence intervals, with collapse flags
# ---------------------------------------------------------------------------
def fig_hl_forest(hl, hlw, out_path, xmax=80.0):
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    yticks, ylabels = [], []
    for y, cur in enumerate(CURS[::-1]):
        r, rw = hl[cur], hlw[cur]
        # LB from the wild (heteroskedasticity-robust) bootstrap; MP recursive.
        lb_lo = scalar_hl(float(rw["alpha_ci_lo_LB"]))
        # The infinite/finite classification of the upper limit is the
        # includes-one FLAG (grid point alpha=1.00 accepted), not the largest
        # sub-unity grid point stored in alpha_ci_hi: for GBP under the wild
        # scheme the flag is 1 while alpha_ci_hi=0.995 is finite, and mapping
        # the column instead of the flag closes an interval the paper reports
        # as [1.2, inf). Same rule applied to the MP (recursive) side.
        lb_hi = (np.inf if int(float(rw.get("LB_includes_one", 0) or 0)) == 1
                 else scalar_hl(float(rw["alpha_ci_hi_LB"])))
        mp_lo = scalar_hl(float(r["alpha_ci_lo_MP"]))
        mp_hi = (np.inf if int(float(r.get("MP_includes_one", 0) or 0)) == 1
                 else scalar_hl(float(r["alpha_ci_hi_MP"])))
        lb_pt = (float(rw["HL_scalar_LB"])
                 if rw["HL_scalar_LB"] != "inf" else np.inf)
        mp_pt = (float(r["HL_scalar_MP"])
                 if r["HL_scalar_MP"] != "inf" else np.inf)
        collapse = int(rw.get("collapse", 0)) == 1
        for tag, lo, hi, pt, col, off in (
                ("LB", lb_lo, lb_hi, lb_pt, "black", +0.18),
                ("MP", mp_lo, mp_hi, mp_pt, "0.55", -0.18)):
            yy = y + off
            ax.plot([lo, min(hi, xmax)], [yy, yy], color=col, lw=1.7,
                    solid_capstyle="butt")
            if not np.isfinite(hi):
                ax.annotate("", xy=(xmax * 1.18, yy), xytext=(xmax, yy),
                            arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
            if np.isfinite(pt):
                ax.plot([pt], [yy], marker="o", ms=4, color=col)
        # flag the collapse cases (unbounded under MP -> bounded under LB)
        if collapse:
            ax.annotate("collapse", xy=(xmax * 1.30, y), fontsize=7,
                        style="italic", color="black", va="center")
        yticks.append(y)
        ylabels.append(cur + (r"$^{\ast}$" if collapse else ""))
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xscale("log")
    ax.set_xlim(0.7, xmax * 1.7)
    ax.set_xticks([1, 2, 3, 5, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("half-life (years, log scale)")
    ax.axvspan(ROGOFF_LO, ROGOFF_HI, color="0.92", zorder=0)   # Rogoff band
    ax.legend(handles=[
        Line2D([0], [0], color="0.55", lw=1.7, marker="o", ms=4,
               label="constant mean (MP)"),
        Line2D([0], [0], color="black", lw=1.7, marker="o", ms=4,
               label="level breaks (LB), wild bootstrap")],
        loc="lower right", fontsize=8, frameon=False)
    ax.text(0.012, 0.02, r"$^{\ast}$ CI collapses to bounded under LB",
            transform=ax.transAxes, fontsize=7, style="italic")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start-year", type=int, default=1973,
                    help="first year of the window (default 1973, the "
                         "post-Bretton-Woods float)")
    ap.add_argument("--data-dir", default=".",
                    help="directory holding the input CSVs")
    ap.add_argument("--out-dir", default=".",
                    help="directory for the output PDFs")
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 9, "font.family": "serif",
                         "axes.linewidth": 0.6})
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        panel, dates, hl, hlw = load_inputs(data_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    missing = [c for c in CURS if c not in panel]
    if missing:
        print(f"[error] currencies absent from panel: {missing}",
              file=sys.stderr)
        sys.exit(1)

    f1 = out_dir / "fig_rer_series.pdf"
    f2 = out_dir / "fig_hl_forest.pdf"
    fig_rer_series(panel, dates, args.start_year, f1)
    fig_hl_forest(hl, hlw, f2)
    n_collapse = sum(int(hlw[c].get("collapse", 0)) == 1 for c in CURS)
    print(f"figures written to {out_dir}/:")
    print(f"  {f1.name}  (window {args.start_year}-latest)")
    print(f"  {f2.name}  ({n_collapse}/8 collapse cases flagged)")


if __name__ == "__main__":
    main()
