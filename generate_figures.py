#!/usr/bin/env python3
# =============================================================================
# generate_figures.py -- the single figure entry point for the Model LB paper.
# =============================================================================
#
# This module is READ-ONLY with respect to the science: it never runs a Monte
# Carlo and never estimates a model. It reads the CSV artifacts written by the
# compute modules and renders the five publication figures. Keeping all plotting
# in one place (and all computation in the compute modules) is what makes the
# figures reproducible from the shipped data alone -- a referee can regenerate
# every figure in seconds without re-running any simulation.
#
# All figures are grayscale (color-blind / print safe): series are distinguished
# by line style + marker + grey level, never by hue.
#
# FIGURE -> INPUT CSV -> PRODUCER
#   Fig 1  limiting_density.pdf   <- limiting_density.csv   (replicate_section3_4.py --limiting-density)
#   Fig 2  cbar_surface.pdf       <- cbar_surface.csv       (replicate_section3_4.py --full)
#   Fig 3  fig_power.pdf          <- power_comparison.csv   (size_power_cbar_comparison.py)
#   Fig 4  fig_rer_series.pdf     <- ppp_panel.csv + exog_dates.csv
#   Fig 5  fig_hl_forest.pdf      <- hl_results.csv + hl_results_wild.csv  (hl_median_unbiased.py)
#
# USAGE
#   python generate_figures.py                       # all available figures
#   python generate_figures.py --only fig5           # one figure
#   python generate_figures.py --data-dir . --out-dir .
#   python generate_figures.py --start-year 1973     # window for Fig 4
# A figure whose input CSV is absent is skipped with a clear message, so the
# script is safe to run at any point in the reproduction pipeline.
#
# REQUIRES
#   numpy, matplotlib. No numba, no pandas (I/O via the csv standard library).
# =============================================================================
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# --- shared configuration -----------------------------------------------------
CURS = ["AUD", "CAD", "CHF", "GBP", "JPY", "NOK", "NZD", "SEK"]
ROGOFF_LO, ROGOFF_HI = 3.0, 5.0     # Rogoff (1996) consensus half-life band

# Grayscale style per break count m: distinct WITHOUT color (grey + line + marker).
M_STYLE = {
    0: dict(color="0.00", ls="-",  marker="o", ms=4.5),
    1: dict(color="0.30", ls="--", marker="s", ms=4.5),
    2: dict(color="0.00", ls="-.", marker="^", ms=5.0),
    3: dict(color="0.45", ls=":",  marker="D", ms=4.0),
    4: dict(color="0.20", ls="--", marker="v", ms=5.0),
    5: dict(color="0.55", ls="-.", marker="P", ms=5.0),
}

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "legend.fontsize": 8, "savefig.dpi": 300, "axes.linewidth": 0.8,
})


# --- I/O helper ---------------------------------------------------------------
def _read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"required input not found: {path}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"input is empty: {path}")
    return rows


# =============================================================================
# Figure 1 -- limiting null law of MZ_t across m
#   panels: (a) density at T=300, (b) rejection-tail ECDF at T=300,
#           (c) density at T=60.  Data: limiting_density.csv [panel,T,m,x,y].
# =============================================================================
def fig_limiting_density(data_dir, out_path):
    rows = _read_csv(os.path.join(data_dir, "limiting_density.csv"))
    curves = defaultdict(list)      # (panel, T, m) -> list[(x, y)]
    for r in rows:
        curves[(r["panel"], int(r["T"]), int(r["m"]))].append(
            (float(r["x"]), float(r["y"])))
    ms = sorted({m for (_, _, m) in curves})

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.2))

    def _draw(ax, panel, T):
        for m in ms:
            pts = sorted(curves.get((panel, T, m), []))
            if not pts:
                continue
            x = [p[0] for p in pts]; y = [p[1] for p in pts]
            st = M_STYLE[m % len(M_STYLE)]
            ax.plot(x, y, color=st["color"], ls=st["ls"], lw=1.5, label=fr"$m={m}$")

    _draw(axA, "density", 300)
    axA.axvline(-2.0, ls=(0, (1, 1)), color="0.6", lw=0.7)
    axA.set_xlabel(r"$\mathrm{MZ}_t$"); axA.set_ylabel("density")
    axA.set_title(r"(a) Null law at $T=300$: coincides across $m$")
    axA.legend(frameon=False)

    _draw(axB, "ecdf", 300)
    axB.axhline(0.05, ls=":", color="0.5", lw=0.8)
    axB.axvline(-2.0, ls=(0, (1, 1)), color="0.6", lw=0.7)
    axB.set_xlim(-3.3, -1.2); axB.set_ylim(0, 0.20)
    axB.set_xlabel(r"$\mathrm{MZ}_t$"); axB.set_ylabel("CDF (rejection tail)")
    axB.set_title(r"(b) Rejection tail at $T=300$: 5\% quantiles align")
    axB.legend(frameon=False, loc="upper left")

    _draw(axC, "density", 60)
    axC.axvline(-2.0, ls=(0, (1, 1)), color="0.6", lw=0.7)
    axC.set_xlabel(r"$\mathrm{MZ}_t$"); axC.set_ylabel("density")
    axC.set_title(r"(c) $T=60$: laws separate by $O((m{+}1)/T)$")
    axC.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Figure 2 -- the finite-sample surface c-bar(m,T)
#   (a) c-bar vs 1/T by m; (b) slope a(m) of the 1/T fit vs m+1.
#   Data: cbar_surface.csv (aggregated over lambda configs, const method).
# =============================================================================
def _surface_cells(path, method="const"):
    buckets = defaultdict(list)
    for r in _read_csv(path):
        if r.get("sigma2_method") != method:
            continue
        buckets[(int(r["m"]), int(r["T"]))].append(
            (float(r["cbar_otimo"]), float(r["se_cbar"])))
    agg = {}
    for k, vs in buckets.items():
        v = np.array([x[0] for x in vs])
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else vs[0][1]
        agg[k] = (float(v.mean()), float(se))
    return agg


def _wls_fit(points):
    """Weighted least squares of c-bar on 1/T (weights = 1/se^2).
    Returns (intercept, slope, se_slope) with se_slope from (X'WX)^-1."""
    T = np.array([p[0] for p in points], float)
    y = np.array([p[1] for p in points], float)
    w = 1.0 / np.array([p[2] for p in points], float) ** 2
    X = np.column_stack([np.ones_like(T), 1.0 / T])
    XtWX_inv = np.linalg.inv((X.T * w) @ X)
    beta = XtWX_inv @ ((X.T * w) @ y)
    return float(beta[0]), float(beta[1]), float(np.sqrt(XtWX_inv[1, 1]))


def fig_cbar_surface(data_dir, out_path):
    cells = _surface_cells(os.path.join(data_dir, "cbar_surface.csv"))
    ms = sorted({m for (m, _) in cells})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    slopes, slope_se, handles = {}, {}, {}
    invT_max = max(1.0 / T for (m, T) in cells)
    for m in ms:
        pts = [(T, *cells[(m, T)]) for (mm, T) in sorted(cells) if mm == m]
        if len(pts) < 2:
            continue
        Ts = np.array([p[0] for p in pts]); cs = np.array([p[1] for p in pts])
        st = M_STYLE[m % len(M_STYLE)]; col = st["color"]
        # open (hollow) markers for the cell means; fit line per-m style + grey
        ax1.plot(1.0 / Ts, cs, ls="none", marker=st["marker"], ms=4.0,
                 color=col, markerfacecolor="white", markeredgecolor=col)
        if len(pts) >= 3:
            c_inf, a, se_a = _wls_fit(pts); slopes[m] = a; slope_se[m] = se_a
            x_hi = float((1.0 / Ts).max())          # this m's own data range
            xx = np.array([0.0, x_hi])
            ax1.plot(xx, c_inf + a * xx, ls=st["ls"], color=col, lw=1.4)
        handles[m] = Line2D([0], [0], color=col, ls=st["ls"], lw=1.4,
                            marker=st["marker"], markerfacecolor="white",
                            markeredgecolor=col, ms=4.0, label=fr"$m={m}$")
    ax1.axhline(-7.0, ls=":", color="0.55", lw=1)
    ax1.text(invT_max * 1.05, -6.99, r"$-7$ (ERS)", fontsize=8, color="0.4",
             va="bottom", ha="right")
    ax1.set_xlim(-0.001, invT_max * 1.08)
    ax1.set_xlabel(r"$1/T$"); ax1.set_ylabel(r"$\bar c(m,T)$")
    ax1.set_title(r"(a) $\bar c$ vs $1/T$: common intercept $\approx -7$", fontsize=10)
    # column-major legend layout (m=0..2 | m=3..5) to match the design
    order = [0, 1, 2, 3, 4, 5]  # matplotlib fills column-major -> m0..2 | m3..5
    hlist = [handles[m] for m in order if m in handles]
    ax1.legend(handles=hlist, loc="lower left", ncol=2, frameon=False, fontsize=8)

    # panel (b): slope a(m) vs m+1 with 95% CI, plus the origin ray for m>=2
    ax2.axhline(0.0, ls="-", color="0.75", lw=0.8)
    mm = np.array(sorted(slopes))
    aa = np.array([slopes[m] for m in mm])
    ee = np.array([1.96 * slope_se[m] for m in mm])
    ax2.errorbar(mm + 1, aa, yerr=ee, ls="none", marker="o", ms=5,
                 markerfacecolor="white", markeredgecolor="black",
                 ecolor="0.35", elinewidth=1.0, capsize=3, label=r"$a(m)$ (95% CI)")
    big = mm[mm >= 2]
    if len(big) >= 2:
        sl = float(np.sum((big + 1) * np.array([slopes[m] for m in big])) /
                   np.sum((big + 1) ** 2))
        xr = np.array([0.0, float(big.max()) + 1.0])
        ax2.plot(xr, sl * xr, ls="--", lw=1.2, color="black",
                 label=fr"origin line, slope $\approx {sl:.0f}$ ($m\geq 2$)")
    ax2.set_xlabel(r"$m+1$ (orthogonal degrees of freedom)")
    ax2.set_ylabel(r"$a(m)$: coefficient of $1/T$")
    ax2.set_title(r"(b) $a(m)\propto(m+1)$: the $O((m{+}1)/T)$ rate", fontsize=10)
    ax2.legend(frameon=False, fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)


# =============================================================================
# Figure 3 -- power of MZ_t under three c-bar choices
#   Data: power_comparison.csv [spec, cbar, cv5, alpha, c_alt, c, power].
# =============================================================================
def fig_power(data_dir, out_path):
    rows = _read_csv(os.path.join(data_dir, "power_comparison.csv"))
    curve = defaultdict(list)       # spec -> list[(|c|, power)]
    cbar = {}
    for r in rows:
        spec = r["spec"]
        curve[spec].append((abs(float(r["c"])), float(r["power"])))
        cbar[spec] = float(r["cbar"])
    alpha = float(rows[0]["alpha"]); c_alt = float(rows[0]["c_alt"])

    # Monochrome palette: the paper's proposal is solid black with a filled
    # marker; the two dominated shortcuts are grey with open markers.
    style = {
        "Calibrated Model LB":  dict(color="black", marker="o", ls="-",  mfc="black"),
        "Linear break-count":   dict(color="0.35",  marker="s", ls="--", mfc="white"),
        "Trend-break surface":  dict(color="0.55",  marker="^", ls=":",  mfc="white"),
    }
    order = [s for s in style if s in curve] + [s for s in curve if s not in style]

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for spec in order:
        pts = sorted(curve[spec])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        st = style.get(spec, dict(color="0.5", marker=".", ls="-", mfc="0.5"))
        ax.plot(xs, ys, marker=st["marker"], color=st["color"], ls=st["ls"],
                ms=4.5, lw=1.4, markerfacecolor=st["mfc"], markeredgecolor=st["color"],
                label=fr"{spec} ($\bar c={cbar[spec]:+.1f}$)")
    ax.axhline(alpha, ls=":", color="0.5", lw=1)
    ax.text(0.3, alpha + 0.012, fr"nominal size ${alpha:g}$", fontsize=8, color="0.4")
    ax.axvline(abs(c_alt), ls="--", color="0.7", lw=0.9)
    ax.text(abs(c_alt) + 0.3, 0.95, f"col.~(iii)\n$c={c_alt:.0f}$", fontsize=7.5,
            color="0.5", va="top")
    ax.set_xlabel(r"local alternative $|c|$  (root $=1-|c|/T$)")
    ax.set_ylabel(r"rejection rate (power)")
    ax.set_title(r"Power of $\mathrm{MZ}_t$ under three $\bar c$ choices, $T=60$, $m=2$",
                 fontsize=10)
    ax.set_ylim(0, 1.02); ax.set_xlim(-0.5, 30.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Figure 4 -- the eight real exchange rates with the H1 step deterministic
#   Data: ppp_panel.csv (currency, year, q) + exog_dates.csv.
# =============================================================================
def _build_Z(years, break_years):
    cols = [np.ones(len(years))]
    for b in sorted(break_years):
        if years[0] < b <= years[-1]:
            cols.append((years >= b).astype(float))
    return np.column_stack(cols)


def fig_rer_series(data_dir, out_path, start_year):
    panel = defaultdict(list)
    for r in _read_csv(os.path.join(data_dir, "ppp_panel.csv")):
        panel[r["currency"]].append((int(r["year"]), float(r["q"])))
    dates = defaultdict(list)
    for r in _read_csv(os.path.join(data_dir, "exog_dates.csv")):
        dates[r["currency"]].append(int(r["break_year"]))

    fig, axes = plt.subplots(4, 2, figsize=(6.6, 8.2), sharex=True)
    for ax, cur in zip(axes.flat, CURS):
        obs = sorted(o for o in panel[cur] if o[0] >= start_year)   # WINDOW
        yrs = np.array([o[0] for o in obs]); q = np.array([o[1] for o in obs])
        bry = [b for b in sorted(dates.get(cur, [])) if yrs[0] < b <= yrs[-1]]
        Z = _build_Z(yrs, bry)
        beta, *_ = np.linalg.lstsq(Z, q, rcond=None)     # OLS on the SAME window
        step = Z @ beta
        ax.plot(yrs, q, color="black", lw=0.9)
        ax.plot(yrs, step, color="0.45", lw=2.0)
        for b in bry:
            ax.axvline(b, color="0.6", lw=0.7, ls="--")
        ax.set_title(f"{cur}  ($m={len(bry)}$: {', '.join(map(str, bry))})", fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("year", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Figure 5 -- forest plot of median-unbiased half-life confidence intervals
#   MP (constant mean, recursive) vs LB (level breaks, wild bootstrap).
#   Data: hl_results.csv + hl_results_wild.csv.
# =============================================================================
def _scalar_hl(alpha):
    a = float(alpha)
    return float(np.log(0.5) / np.log(a)) if 0.0 < a < 1.0 else np.inf


def fig_hl_forest(data_dir, out_path, xmax=80.0):
    hl = {r["currency"]: r for r in _read_csv(os.path.join(data_dir, "hl_results.csv"))}
    wild_path = os.path.join(data_dir, "hl_results_wild.csv")
    if os.path.exists(wild_path):
        hlw = {r["currency"]: r for r in _read_csv(wild_path)}
    else:
        print(f"[warn] {os.path.basename(wild_path)} not found; LB interval falls "
              f"back to the recursive bootstrap.", file=sys.stderr)
        hlw = hl

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    yticks, ylabels = [], []
    for y, cur in enumerate(CURS[::-1]):
        r, rw = hl[cur], hlw[cur]
        # The infinite/finite classification of the upper limit is the includes-one
        # FLAG, not the largest sub-unity grid point stored in alpha_ci_hi.
        lb_lo = _scalar_hl(float(rw["alpha_ci_lo_LB"]))
        lb_hi = (np.inf if int(float(rw.get("LB_includes_one", 0) or 0)) == 1
                 else _scalar_hl(float(rw["alpha_ci_hi_LB"])))
        mp_lo = _scalar_hl(float(r["alpha_ci_lo_MP"]))
        mp_hi = (np.inf if int(float(r.get("MP_includes_one", 0) or 0)) == 1
                 else _scalar_hl(float(r["alpha_ci_hi_MP"])))
        lb_pt = float(rw["HL_scalar_LB"]) if rw["HL_scalar_LB"] != "inf" else np.inf
        mp_pt = float(r["HL_scalar_MP"]) if r["HL_scalar_MP"] != "inf" else np.inf
        collapse = int(float(rw.get("collapse", 0) or 0)) == 1
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
        if collapse:
            ax.annotate("collapse", xy=(xmax * 1.30, y), xytext=(-11.34, 0),
                        textcoords="offset points", fontsize=7, style="italic",
                        color="black", va="center")
        yticks.append(y)
        ylabels.append(cur + (r"$^{\ast}$" if collapse else ""))
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels)
    ax.set_xscale("log"); ax.set_xlim(0.7, xmax * 1.7)
    ax.set_xticks([1, 2, 3, 5, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("half-life (years, log scale)")
    ax.axvspan(ROGOFF_LO, ROGOFF_HI, color="0.92", zorder=0)     # Rogoff band
    # Opaque, on-top legend box so the bottom currency's line cannot bleed
    # through it (previously the light-grey MP line for SEK crossed the legend).
    leg = ax.legend(handles=[
        Line2D([0], [0], color="0.55", lw=1.7, marker="o", ms=4,
               label="constant mean (MP)"),
        Line2D([0], [0], color="black", lw=1.7, marker="o", ms=4,
               label="level breaks (LB), wild bootstrap")],
        loc="lower right", fontsize=8, frameon=True, facecolor="white",
        framealpha=1.0, edgecolor="0.85")
    leg.set_zorder(6)
    ax.text(0.012, 0.02, r"$^{\ast}$ CI collapses to bounded under LB",
            transform=ax.transAxes, fontsize=7, style="italic")
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)     # reserve room for the collapse labels
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
FIGURES = {
    "fig1": ("limiting_density.pdf", lambda d, o, a: fig_limiting_density(d, o)),
    "fig2": ("cbar_surface.pdf",     lambda d, o, a: fig_cbar_surface(d, o)),
    "fig3": ("fig_power.pdf",        lambda d, o, a: fig_power(d, o)),
    "fig4": ("fig_rer_series.pdf",   lambda d, o, a: fig_rer_series(d, o, a.start_year)),
    "fig5": ("fig_hl_forest.pdf",    lambda d, o, a: fig_hl_forest(d, o)),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(FIGURES), default=None,
                    help="render a single figure (default: all available)")
    ap.add_argument("--data-dir", default=".", help="directory with the input CSVs")
    ap.add_argument("--out-dir", default=".", help="directory for the figure PDFs")
    ap.add_argument("--start-year", type=int, default=1973,
                    help="first year of the estimation window for Figure 4")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    keys = [args.only] if args.only else list(FIGURES)
    made, skipped = 0, 0
    for key in keys:
        name, fn = FIGURES[key]
        out_path = os.path.join(args.out_dir, name)
        try:
            fn(args.data_dir, out_path, args)
            print(f"[{key}] wrote {out_path}")
            made += 1
        except FileNotFoundError as e:
            print(f"[{key}] skipped -- {e}", file=sys.stderr)
            skipped += 1
    print(f"\n[done] {made} figure(s) written, {skipped} skipped (missing input).")


if __name__ == "__main__":
    main()
