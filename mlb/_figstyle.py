"""Shared figure palette and typography.

The ordinal ramp (black -> lighter blue, luminance strictly increasing) used by
the figures that index by an ordinal axis -- the break count m of Figure 1
(fig:size, size_figure.py) and the |c-bar| severity of Figure 2
(fig:overdetrend, the CBAR_STYLE of overdetrend_figure.py). Defined here once,
for both producers.
"""
BLACK = "#000000"

ORDINAL_RAMP = [
    dict(color=BLACK,      ls="-",  marker="o", ms=4.5),
    dict(color="#08306B",  ls="--", marker="s", ms=4.5),
    dict(color="#2171B5",  ls="-.", marker="^", ms=4.5),
    dict(color="#6BAED6",  ls=":",  marker="D", ms=4.0),
]

# --- house rcParams ------------------------------------------------------------
# The typography of the printed Figures 1 and 2 (9 pt, TrueType fonts), applied
# by size_figure.py and overdetrend_figure.py before they draw. Nothing
# is applied on import.
HOUSE_RCPARAMS = {
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 7.5, "savefig.dpi": 300, "axes.linewidth": 0.7,
    "pdf.fonttype": 42, "ps.fonttype": 42,   # CID TrueType, not Type-3 bitmap
}


def apply_house_rcparams():
    """Apply HOUSE_RCPARAMS to matplotlib (call before drawing)."""
    import matplotlib
    matplotlib.rcParams.update(HOUSE_RCPARAMS)
