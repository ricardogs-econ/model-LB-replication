#!/usr/bin/env python3
"""Figure 3 (fig:series): the nine real exchange rates of the application with the
break dates of the registered design E and the regime means; and the facts of
Section 6.1 that the figure shows. No simulation.

Data: q of inputs/ppp_panel.csv, 1973-2024 (T = 52), for the nine candidates in the
order AUD, CAD, DKK, GBP, JPY, NOK, NZD, SEK, CHF. Break dates: the dates tau of
design E, generated from inputs/regime_events.csv by mlb/regime_events.py (the dates
of Table 3); the first regime runs from 1973 to tau_1 - 1, the next from tau_1, and
so on. The eight admitted currencies are those of the admission screen
(inputs/ppp_sweep/sweep_gates.csv, through mlb/ppp_application.universes); the
Swiss franc, not admitted, carries a dagger.

Each panel: q minus its sample mean (black); a dashed vertical line half a year
before each break date; a horizontal segment at the mean of each regime.

The facts (Section 6.1; series_summary.json):
  shift      |difference of consecutive regime means| at each break;
  sd_regime  the standard deviation of q around its regime means, divisor
             T - (m + 1);
  z_drift    the mean of Delta q over its standard deviation divided by
             sqrt(T - 1) (a drift in conventional standard errors);
and, over the eight admitted currencies, the smallest and largest shift and
sd_regime, the number of breaks and the number whose shift is smaller than
sd_regime; over the nine, the largest |z_drift|; and the exact half-power point of
the constant alone at T = 52 (Section 6.2: envelope_exact.half_power_point(52, ()),
the bracket (-25, -1) and xtol = 1e-9 of the application's designs).

Outputs (in output/figures/series_figure/ unless --out-dir is given):
  Figure_3_data.csv    one row per currency and year: currency, year, q,
                       q_demeaned, regime (0, 1, ...), regime_mean_demeaned
                       (floats at full precision, repr);
  series_summary.json  the facts above, per currency and over the sets;
  Figure_3.pdf         drawn from the data file (without a creation date).
The step figure_series of replicate_sec_6.py compares the first two with
reference/figures/Figure_3_data.csv and reference/application/main/
series_summary.json.

    python mlb/series_figure.py [--out-dir DIR]
"""

import argparse
import csv
import io
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from mlb import envelope_exact as EE  # noqa: E402
from mlb import paths  # noqa: E402
from mlb import ppp_application as P  # noqa: E402
from mlb import regime_events as RE  # noqa: E402
from mlb._figstyle import BLACK, ORDINAL_RAMP, apply_house_rcparams  # noqa: E402

ORDER = ("AUD", "CAD", "DKK", "GBP", "JPY", "NOK", "NZD", "SEK", "CHF")
NAMES = {"AUD": "Australian dollar", "CAD": "Canadian dollar", "DKK": "Danish krone",
         "GBP": "Pound sterling", "JPY": "Yen", "NOK": "Norwegian krone",
         "NZD": "New Zealand dollar", "SEK": "Swedish krona", "CHF": "Swiss franc"}
PANEL = paths.INPUTS / "ppp_panel.csv"
DESIGN = "E"
T_CONST = 52
DATA = "Figure_3_data.csv"
SUMMARY = "series_summary.json"
PDF = "Figure_3.pdf"
COLS = ("currency", "year", "q", "q_demeaned", "regime", "regime_mean_demeaned")


def load_panel(path=PANEL):
    """{currency: (years, q)} of the panel, years increasing."""
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["currency"], {})[int(r["year"])] = float(r["q"])
    return {c: (np.array(sorted(v)), np.array([v[t] for t in sorted(v)]))
            for c, v in out.items()}


def series(years, q, taus):
    """The regimes and the facts of one series."""
    edges = [int(years[0])] + list(taus) + [int(years[-1]) + 1]
    regime = np.zeros(len(years), dtype=int)
    means = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        sel = (years >= a) & (years < b)
        regime[sel] = i
        means.append(float(q[sel].mean()))
    fit = np.array([means[i] for i in regime])
    m = len(taus)
    t = len(q)
    sd = float(np.sqrt(((q - fit) ** 2).sum() / (t - (m + 1))))
    dq = np.diff(q)
    z = float(dq.mean() / (dq.std(ddof=1) / np.sqrt(len(dq))))
    shifts = [abs(means[i + 1] - means[i]) for i in range(m)]
    return dict(regime=regime, means=means, sd_regime=sd, z_drift=z, shifts=shifts,
                m=m, T=t, edges=edges)


def compute():
    panel = load_panel()
    rows = RE.load()
    n8, _ = P.universes(False)
    per = {}
    for c in ORDER:
        taus = RE.configuration(rows, c, DESIGN)[0]
        years, q = panel[c]
        s = series(years, q, taus)
        s.update(taus=list(taus), years=years, q=q, admitted=c in n8)
        per[c] = s
    adm = [per[c] for c in ORDER if per[c]["admitted"]]
    shifts = [(x, c, tau) for c in ORDER if per[c]["admitted"]
              for x, tau in zip(per[c]["shifts"], per[c]["taus"])]
    sds = [(per[c]["sd_regime"], c) for c in ORDER if per[c]["admitted"]]
    below = sum(1 for c in ORDER if per[c]["admitted"]
                for x in per[c]["shifts"] if x < per[c]["sd_regime"])
    zmax = max(((abs(per[c]["z_drift"]), c) for c in ORDER))
    cbar, b_t = EE.half_power_point(T_CONST, (), xtol=1e-9)
    summary = {
        "design": DESIGN,
        "currencies": {c: {
            "admitted": per[c]["admitted"], "m": per[c]["m"], "T": per[c]["T"],
            "tau": per[c]["taus"], "regime_means": per[c]["means"],
            "shifts": per[c]["shifts"], "sd_regime": per[c]["sd_regime"],
            "z_drift": per[c]["z_drift"]} for c in ORDER},
        "admitted": [c for c in ORDER if per[c]["admitted"]],
        "admitted_summary": {
            "n_series": len(adm),
            "n_breaks": len(shifts),
            "shift_min": dict(zip(("value", "currency", "tau"), min(shifts))),
            "shift_max": dict(zip(("value", "currency", "tau"), max(shifts))),
            "sd_regime_min": dict(zip(("value", "currency"), min(sds))),
            "sd_regime_max": dict(zip(("value", "currency"), max(sds))),
            "n_shift_below_sd_regime": below},
        "all_nine": {
            "abs_z_drift_max": dict(zip(("value", "currency"), zmax)),
            "n_by_m": {str(m): sum(1 for c in ORDER if per[c]["m"] == m)
                       for m in sorted({per[c]["m"] for c in ORDER})}},
        "constant_only": {"T": T_CONST, "cbar_half": cbar, "b_T": b_t,
                          "bracket": [-25.0, -1.0], "xtol": 1e-9},
    }
    return per, summary


def data_text(per):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLS)
    for c in ORDER:
        s = per[c]
        mean = float(s["q"].mean())
        for y, v, r in zip(s["years"], s["q"], s["regime"]):
            w.writerow([c, int(y), repr(float(v)), repr(float(v - mean)), int(r),
                        repr(float(s["means"][r] - mean))])
    return buf.getvalue()


def draw(path, data_path):
    """Figure 3 from the data file."""
    rows = {}
    with open(data_path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            rows.setdefault(r["currency"], []).append(r)
    summary_taus = {}
    for c, rs in rows.items():
        reg = [int(r["regime"]) for r in rs]
        summary_taus[c] = [int(rs[i]["year"]) for i in range(1, len(rs))
                           if reg[i] != reg[i - 1]]
    n8, _ = P.universes(False)
    apply_house_rcparams()
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.0), sharex=True)
    for ax, c in zip(axes.ravel(), ORDER):
        rs = rows[c]
        yrs = np.array([int(r["year"]) for r in rs])
        y = np.array([float(r["q_demeaned"]) for r in rs])
        reg = np.array([int(r["regime"]) for r in rs])
        ax.plot(yrs, y, color=BLACK, lw=1.0)
        for i in sorted(set(reg)):
            sel = reg == i
            level = float(rs[int(np.argmax(sel))]["regime_mean_demeaned"])
            ax.plot([yrs[sel][0], yrs[sel][-1]], [level, level],
                    color=ORDINAL_RAMP[2]["color"], lw=1.6, solid_capstyle="butt")
        for tau in summary_taus[c]:
            ax.axvline(tau - 0.5, color=ORDINAL_RAMP[3]["color"], lw=0.8, ls="--")
        name = NAMES[c] + ("" if c in n8 else r"$^{\dagger}$")
        ax.set_title(f"{name}: {', '.join(map(str, summary_taus[c]))}", fontsize=8)
        ax.set_xlim(1972, 2025)
        ax.set_xticks([1980, 2000, 2020])
        ax.tick_params(labelsize=7.5, length=2.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("$q_{it}$, demeaned", fontsize=8)
    fig.tight_layout(h_pad=0.9, w_pad=0.7)
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(paths.OUTPUT / "figures" / "series_figure"),
                    help="output folder (default output/figures/series_figure/)")
    a = ap.parse_args(argv)
    per, summary = compute()
    os.makedirs(a.out_dir, exist_ok=True)
    data_path = os.path.join(a.out_dir, DATA)
    with open(data_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(data_text(per))
    with open(os.path.join(a.out_dir, SUMMARY), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(summary, fh, indent=1)
        fh.write("\n")
    draw(os.path.join(a.out_dir, PDF), data_path)
    s = summary["admitted_summary"]
    print(f"[series_figure] admitted {len(summary['admitted'])}: shifts "
          f"{s['shift_min']['value']:.6f} ({s['shift_min']['currency']} "
          f"{s['shift_min']['tau']}) to {s['shift_max']['value']:.6f} "
          f"({s['shift_max']['currency']} {s['shift_max']['tau']}); sd_regime "
          f"{s['sd_regime_min']['value']:.6f} ({s['sd_regime_min']['currency']}) to "
          f"{s['sd_regime_max']['value']:.6f} ({s['sd_regime_max']['currency']}); "
          f"{s['n_shift_below_sd_regime']} of {s['n_breaks']} shifts below sd_regime")
    z = summary["all_nine"]["abs_z_drift_max"]
    print(f"[series_figure] largest |z_drift| {z['value']:.6f} ({z['currency']}); "
          f"constant only at T = {T_CONST}: c-bar_1/2 = "
          f"{summary['constant_only']['cbar_half']:.6f}")
    print(f"[series_figure] wrote {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
