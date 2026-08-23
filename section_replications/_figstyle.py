"""Shared figure palette.

The ordinal ramp (black -> lighter blue, luminance strictly increasing) used by
every figure that indexes by an ordinal axis -- the |c-bar| severity of Figure 1,
the break count m of Figure 2 and the supplement figures. Consolidated here so
door 1 (fig:overdetrend, fig:size) and door 4 (generate_figures) share ONE
definition rather than one importing the other across doors.
"""
BLACK, BLUE, VERMILLION = "#000000", "#0072B2", "#D55E00"

ORDINAL_RAMP = [
    dict(color=BLACK,      ls="-",  marker="o", ms=4.5),
    dict(color="#08306B",  ls="--", marker="s", ms=4.5),
    dict(color="#2171B5",  ls="-.", marker="^", ms=4.5),
    dict(color="#6BAED6",  ls=":",  marker="D", ms=4.0),
]
