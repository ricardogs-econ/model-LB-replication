#!/usr/bin/env python3
"""
overdetrend_power.py

Does Model LB correct over-detrending? The relevant contrast is NOT
cbar*(m,T) vs -7 (both correct, ~equal power) but the CORRECT no-trend
value (-7) vs the OVER-detrending a practitioner incurs by reusing a
trend-break device:
   -13.5  ERS/CKP linear-trend detrending value (trend vs no-trend confusion)
   -17    linear break-count scaling shortcut
   -21.7  reused CKP trend-break response surface (T=60, m=2 cell)
All tests are size-corrected (each cbar uses its OWN finite-sample 5% cv),
so differences are pure power. Model LB DGP, demeaned MZ_t, iid N(0,1).

Produces Figure_1.pdf: size-corrected power of the demeaned MZ_t test, at the
default cell (T=60, m=2), for the correct value and the three over-detrending
shortcuts above.

Usage:
    python overdetrend_power.py                  # table + Figure_1.pdf
    python overdetrend_power.py --nrep 50000      # override the replication count
    python overdetrend_power.py --no-figure       # table only, skip plotting
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from materiality_c7_vs_surface import detrender, Lmat, mzt

SEED = 20260724
NREP = 200_000
CBARS = [-7.0, -13.5, -17.0, -21.7]

# Qualitative palette (Observable10-style), one color per cbar; matches the
# blue/orange/red family already used in materiality_figure.py, extended
# with a fourth (purple) entry for the additional curve on this figure.
CBAR_STYLE = {
    -7.0:   dict(color="#4269d0", label=r"$\bar c = -7$ (Model LB, correct)"),
    -13.5:  dict(color="#efb118", label=r"$-13.5$ (trend value)"),
    -17.0:  dict(color="#e15759", label=r"$-17$ (linear scaling)"),
    -21.7:  dict(color="#a463f2", label=r"$-21.7$ (reused trend surface)"),
}


def power_curve(T, m, cbars, c_grid, rng):
    """Size-corrected power of MZ_t at each cbar in `cbars`, over `c_grid`.

    Each cbar is size-corrected against its OWN finite-sample 5% critical
    value (estimated under the null c=0), so the power differences across
    cbar are attributable to detrending alone, not to a shared, possibly
    miscalibrated critical value.

    Returns
    -------
    (cv, out) : cv is {cbar: finite-sample 5% cv}; out is {cbar: [power(c)
    for c in c_grid]}.
    """
    E0 = rng.standard_normal((T, NREP))
    Ms = {cb: detrender(T, m, cb) for cb in cbars}
    U0 = Lmat(T, 0.0) @ E0
    cv = {cb: np.quantile(mzt(Ms[cb] @ U0, T), 0.05) for cb in cbars}
    out = {cb: [] for cb in cbars}
    for c in c_grid:
        E1 = rng.standard_normal((T, NREP))
        U1 = Lmat(T, c) @ E1
        for cb in cbars:
            out[cb].append(np.mean(mzt(Ms[cb] @ U1, T) < cv[cb]))
    return cv, out


def plot_power_curve(cv, out, c_grid, T, m, out_path):
    """Render the size-corrected power comparison and save it to `out_path`."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for cb in CBARS:
        st = CBAR_STYLE[cb]
        ax.plot(c_grid, [100.0 * p for p in out[cb]], "o-",
                color=st["color"], lw=1.8, ms=4.5, label=st["label"])
    ax.axvline(10, ls=":", color="0.6", lw=1, zorder=0)
    ax.set_xlabel("local alternative $|c|$")
    ax.set_ylabel("size-corrected power (%)")
    ax.set_title(rf"Over-detrending loses power at moderate alternatives "
                 rf"($T={T}$, $m={m}$, nominal $5\%$)", fontsize=9)
    ax.set_xlim(min(c_grid) - 1, max(c_grid) + 1)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    global NREP
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nrep", type=int, default=NREP,
                    help="Monte Carlo replications per cbar/alternative cell")
    ap.add_argument("--out", default="Figure_1.pdf",
                    help="output path for the power-comparison figure")
    ap.add_argument("--no-figure", action="store_true",
                    help="print the table only; skip the figure")
    args = ap.parse_args()

    NREP = args.nrep

    rng = np.random.default_rng(SEED)
    T, m = 60, 2
    c_grid = [-5, -7, -10, -13.5, -17, -21.7, -25]
    print(f"Model LB, T={T}, m={m}, demeaned MZ_t, size-corrected (own FS 5% cv), "
          f"N={NREP:,}")
    cv, out = power_curve(T, m, CBARS, c_grid, rng)
    print("  cbar   FScv | " + "  ".join(f"c={c:>5}" for c in c_grid))
    for cb in CBARS:
        row = "  ".join(f"{100*out[cb][i]:6.1f}" for i in range(len(c_grid)))
        tag = " (correct)" if cb == -7.0 else " (over-detrend)"
        print(f"{cb:6.1f} {cv[cb]:6.2f} | {row}{tag}")

    # headline: power loss of the shortcuts vs -7 at the moderate alternative c=-10
    j = c_grid.index(-10)
    base = out[-7.0][j]
    print(f"\nPower at c=-10 (size-corrected): correct -7 = {100*base:.1f}%")
    for cb in CBARS[1:]:
        print(f"   over-detrend {cb}: {100*out[cb][j]:.1f}%  "
              f"(loss {100*(base-out[cb][j]):.1f} pp)")

    # also a short-sample application-like cell
    print("\n--- application-like cell T=52, m=2 ---")
    cv2, out2 = power_curve(52, 2, CBARS, [-7, -10, -13.5], rng)
    for c_i, c in enumerate([-7, -10, -13.5]):
        line = "  ".join(f"{cb}:{100*out2[cb][c_i]:5.1f}%" for cb in CBARS)
        print(f"  c={c:>6}: {line}")

    if not args.no_figure:
        # A denser grid than the printed table's, purely for a smooth curve;
        # 0 is included so all four size-corrected curves visibly meet at the
        # nominal 5% level at the null, which the sparse table grid above
        # (which starts at |c|=5) does not show.
        plot_c_grid = [-2.5 * k for k in range(11)]        # 0, -2.5, ..., -25
        _, out_plot = power_curve(T, m, CBARS, plot_c_grid, rng)
        plot_power_curve(cv, out_plot, [abs(c) for c in plot_c_grid], T, m, args.out)
        print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
