#!/usr/bin/env python3
"""
materiality_figure.py

Companion to materiality_c7_vs_surface.py.
(1) prints the finite-sample 5% cv of MZ_t at c-bar=-7 vs the asymptotic -1.98,
    to show the O((m+1)/T) leftward shift that drives the over-rejection;
(2) produces Figure_2.pdf: empirical size of the "-7 + asymptotic cv" recipe
    as a function of T, for m=0,1,2 -- the practical case for finite-sample
    critical values.

Usage:
    python materiality_figure.py                  # table + Figure_2.pdf
    python materiality_figure.py --nrep 50000      # override the replication count
    python materiality_figure.py --no-figure       # table only, skip plotting
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from materiality_c7_vs_surface import detrender, Lmat, mzt, CV_ASY

SEED = 20260720
NREP = 200_000


def null_mzt(T, m, cbar, E):
    """MZ_t under the null (c=0), detrended at `cbar`, for each column of `E`."""
    M = detrender(T, m, cbar)
    U = Lmat(T, 0.0) @ E
    return mzt(M @ U, T)


def main():
    global NREP
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nrep", type=int, default=NREP,
                    help="Monte Carlo replications per (m, T) cell")
    ap.add_argument("--out", default="Figure_2.pdf",
                    help="output path for the empirical-size figure")
    ap.add_argument("--no-figure", action="store_true",
                    help="print the table only; skip the figure")
    args = ap.parse_args()
    NREP = args.nrep

    rng = np.random.default_rng(SEED)

    print("Finite-sample 5% cv of MZ_t at c-bar=-7 vs asymptotic (-1.98):")
    print(f"{'m':>2} {'T':>4} {'cv_FS(-7)':>10} {'size@-1.98':>11}")
    for m in (0, 1, 2):
        for T in (30, 60, 100):
            E = rng.standard_normal((T, NREP))
            mz = null_mzt(T, m, -7.0, E)
            cv = np.quantile(mz, 0.05)
            sz = np.mean(mz < CV_ASY)
            print(f"{m:>2} {T:>4} {cv:>10.3f} {sz*100:>10.1f}%")

    if args.no_figure:
        return

    # Figure: empirical size vs T, for m = 0, 1, 2
    Tgrid = [20, 30, 40, 50, 60, 80, 100, 150, 200]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = {0: "#4269d0", 1: "#efb118", 2: "#e15759"}
    for m in (0, 1, 2):
        sizes = []
        for T in Tgrid:
            E = rng.standard_normal((T, NREP))
            mz = null_mzt(T, m, -7.0, E)
            sizes.append(np.mean(mz < CV_ASY) * 100)
        ax.plot(Tgrid, sizes, "o-", color=colors[m], lw=1.8, ms=4.5,
                label=f"$m={m}$")
        print(f"m={m}: sizes={[round(float(s), 1) for s in sizes]}")
    ax.axhline(5.0, color="grey", ls="--", lw=1, zorder=0)
    ax.text(200, 5.6, "nominal 5%", ha="right", va="bottom", fontsize=8,
            color="grey")
    ax.set_xlabel("sample size $T$")
    ax.set_ylabel(r"empirical size of the $5\%$ test (%)")
    ax.set_title(r"Over-rejection of $\mathrm{MZ}_t$ at $\bar c=-7$ with the "
                 r"asymptotic critical value ($-1.98$)", fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
