"""Shared helpers for the doors. Everything resolves through src/mlb/paths.py;
no door computes a path itself. Read-only against inputs/ and output/golden/.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mlb import paths


def present(key, *, golden=False):
    """Return the resolved path if the shipped artefact exists, else raise —
    loudly, naming the key — so a missing shipped file is never silent."""
    p = paths.resolve(key)
    if not p.exists():
        raise FileNotFoundError(f"[{key}] shipped artefact missing: {p}")
    return p


def render_tab_cbar_from_surface():
    """tab:cbar --quick: re-derive the printed table from the frozen surface
    (inputs/cbar_surface.csv) by the producer's convention — const arm,
    groupby(m,T) mean of cbar_otimo, one decimal. Cheap (no Monte Carlo).
    Returns the LaTeX string."""
    import csv
    from collections import defaultdict
    import numpy as np
    buckets = defaultdict(list)
    with open(paths.resolve("cbar_surface"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["sigma2_method"] != "const":
                continue
            buckets[(int(r["m"]), int(r["T"]))].append(float(r["cbar_otimo"]))
    import pandas as pd
    rows = [dict(m=m, T=T, cbar=round(float(np.mean(v)), 1))
            for (m, T), v in buckets.items()]
    piv = pd.DataFrame(rows).groupby(["m", "T"])["cbar"].mean().unstack().round(1)
    return piv.to_latex(na_rep="---", float_format="%.1f",
                        caption="Optimal $\\bar c(m,T)$ for Model~LB ($\\sigma^2$: const).",
                        label="tab:cbar")


def _norm_tex(s):
    return "\n".join(ln.rstrip() for ln in s.strip().splitlines())


def diff_against_golden_tab_cbar():
    """Return (ok, detail): does the --quick re-derivation match the shipped
    golden tab_cbar_mT.tex body (the numeric cells), ignoring caption/label?"""
    import re
    got = render_tab_cbar_from_surface()
    gold = open(paths.resolve("tab_cbar"), encoding="utf-8").read()

    def cells(s):
        return re.findall(r"-?\d+\.\d|---", s)
    a, b = cells(got), cells(gold)
    return (a == b, f"{sum(x == y for x, y in zip(a, b))}/{len(b)} cells match")
