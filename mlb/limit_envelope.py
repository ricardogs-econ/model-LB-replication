"""The limiting power envelope (article section 2; supplement S1.2).

Exact numbers, no simulation: the characteristic function of the limit of the
point-optimal statistic is inverted (mlb/envelope_limit.py).

  * at cbar = -7: b_inf, the 5% null quantile of L, and pi_inf = P(L <= b_inf)
    on the diagonal c = cbar;
  * cbar_inf, the root of pi_inf = 1/2, the gap -7 - cbar_inf, and the slope
    pi_inf'(cbar_inf) by a central difference (h = 1e-3);
  * pi_inf and b_inf on the grid of step 0.5 over [-20, -3], and whether pi_inf
    decreases strictly on it;
  * stability: every number recomputed with twice the integration range and
    half the panel width (scale 2), and the largest absolute change.

The JSON holds values only (full precision), so a rerun is compared with the
reference byte for byte; the versions go to the manifest beside it.

    python mlb/limit_envelope.py            # write
    python mlb/limit_envelope.py --check    # and compare (bytes, or every
                                            # number to 1e-9)
"""

import argparse
import hashlib
import json
import os
import platform
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402
import scipy  # noqa: E402

from mlb import envelope_exact as EX  # noqa: E402
from mlb import envelope_limit as EL  # noqa: E402
from mlb import paths  # noqa: E402

SECTION, OBJ = "supplement_appendix", "limit_envelope"
CBAR = -7.0
H = 1e-3


def _grid():
    """cbar from -20 to -3 in steps of 0.5 (35 points)."""
    return [float(v) for v in np.linspace(-20.0, -3.0, 35)]


def _headline(scale):
    b = EL.b_inf(CBAR, scale=scale)
    p = EL.cdf(b, CBAR, CBAR, scale)
    root = EL.cbar_inf(scale=scale)
    grid = _grid()
    bs = [EL.b_inf(cb, scale=scale) for cb in grid]
    env = [EL.cdf(b_, cb, cb, scale) for b_, cb in zip(bs, grid)]
    return dict(
        b_inf_m7=b,
        pi_inf_m7=p,
        cbar_inf=root,
        gap_m7_above=CBAR - root,
        slope_at_cbar_inf=EL.slope(root, h=H, scale=scale),
        grid=[dict(cbar=cb, b_inf=b_, pi_inf=e) for cb, b_, e in zip(grid, bs, env)],
        strictly_decreasing=all(a > b_ for a, b_ in zip(env, env[1:])),
    )


def _close(a, b, tol=1e-9):
    """Equal in structure and in every number to `tol` (absolute and relative): the
    exact laws are computed by complex transcendentals and a symmetric eigensolver,
    whose last digits can differ between machines."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return abs(a - b) <= tol * (1.0 + abs(b))
    return a == b


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare the JSON with the reference, byte for byte",
    )
    a = ap.parse_args()
    base = _headline(1.0)
    wide = _headline(2.0)
    keys = ("b_inf_m7", "pi_inf_m7", "cbar_inf", "slope_at_cbar_inf")
    change = {k: abs(base[k] - wide[k]) for k in keys}
    change["grid_pi_inf"] = max(
        abs(u["pi_inf"] - v["pi_inf"]) for u, v in zip(base["grid"], wide["grid"])
    )
    change["grid_b_inf"] = max(
        abs(u["b_inf"] - v["b_inf"]) for u, v in zip(base["grid"], wide["grid"])
    )
    res = dict(
        size=EL.SIZE,
        cbar=CBAR,
        slope_step=H,
        limit=base,
        stability=dict(
            scale=2.0,
            largest_change=change,
            wide_strictly_decreasing=wide["strictly_decreasing"],
        ),
        conventions=dict(
            envelope_limit=dict(gl_nodes=EL.GL_NODES), envelope_exact=EX.conventions()
        ),
    )
    out = paths.out_dir(SECTION) / OBJ
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{OBJ}.json"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
        fh.write("\n")
    data = open(path, "rb").read()
    modules = {}
    for m in ("envelope_limit.py", "envelope_exact.py"):
        raw = open(os.path.join(_ROOT, "mlb", m), "rb").read()
        modules[m] = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    manifest = dict(
        output=path.name,
        output_sha256=hashlib.sha256(data).hexdigest(),
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        modules_sha256_lf=modules,
    )
    with open(out / f"{OBJ}.manifest.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
    shown = keys + ("gap_m7_above", "strictly_decreasing")
    print(json.dumps({k: base[k] for k in shown}, indent=1))
    print("largest change at scale 2:", max(change.values()))
    if a.check:
        ref = paths.reference_dir(SECTION) / OBJ / f"{OBJ}.json"
        raw = open(ref, "rb").read().replace(b"\r\n", b"\n") if ref.exists() else None
        same = raw == data
        close = same or (raw is not None and _close(json.loads(data), json.loads(raw)))
        word = (
            "identical to"
            if same
            else "equal within 1e-9 to"
            if close
            else "DIFFERS from"
        )
        print(f"--check: {word} {ref}")
        return 0 if close else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
