"""Figure 2 (fig:overdetrend), drawn from overdetrend_power.json: no simulation.

The figure shows the size-corrected power of the demeaned MZ_t test (nominal 5%,
T = 60, m = 2) against the local alternative |c| when the series is detrended at
cbar = -7 (the correct value), -13.5 (the ERS linear-trend value) and -21.7 (the CKP
trend-break response surface). Every point is read from
reference/calibration/overdetrend_power.json, the data of Figure 2
and of the power losses printed in Section 4.4, so the figure and the text carry the
same numbers.

  curves     the cbars of overdetrend_power_precise.CBARS: -7, -13.5 and -21.7;
  points     the reference's grid, |c| in {0, 2.5, 5, 7, 7.5, 10, 12.5, 13.5, 15, 17,
             17.5, 20, 21.7, 22.5, 25}: 15 points per curve; at |c| = 10 the cells
             printed in Section 4.4 (N = 80,000,000), at the other 14 the figure grid
             (N = 2,000,000);
  CRN        the run evaluates every cbar on the same draw of each replication
             (overdetrend_power_precise._draw_mzt), so the curves share draws at each
             |c|, and each alternative has its own disjoint stream;
  se         per point, the reference's se_total (binomial and critical-value terms);
             for a loss against -7, the reference's se_loss, which pairs the binomial
             part but treats the two critical values as independent although they
             come from the same null draws (a conservative, upper se);
  style      CBAR_STYLE below and the house rcParams of _figstyle.py.

Outputs (output/critical_values/overdetrend_figure/ unless --out-dir is given):
  Figure_2_data.csv            one row per drawn point (45: 15 values of |c| for each
                               of the 3 curves), in the order of the reference's grid
                               sorted by |c|: abs_c, c, cbar, power, power_pct
                               (= 100 power, the plotted value), se_total,
                               se_binomial, nrep, n_ok, source, loss_vs_ref,
                               se_loss, seed; floats at full precision (repr), LF;
  Figure_2_data.manifest.json  the source reference and its sha256, the design, the
                               columns, per curve the |c| where its loss against -7
                               changes sign on the grid (chord and shape-preserving
                               cubic, with the delta-method se of the root) and the
                               largest |c| where the loss is above 2 se, and the data
                               file's sha256; nothing time- or machine-dependent;
  Figure_2.pdf                 drawn from the same points (SOURCE_DATE_EPOCH = 0, so
                               the PDF carries no run date). The PDF is not compared:
                               its bytes depend on the matplotlib version.

--power-data PATH reads the points from another copy of the power document, e.g.
the overdetrend_power.json that a --full run of
overdetrend_power_precise.py writes, so that the figure is redrawn from the
regenerated file; the manifest then names PATH (relative to the repository root
when it lies inside it) and its sha256.

--check compares the data file, i.e. the points read from the power document, with
reference/figures/Figure_2_data.csv (the only output with a reference file): byte
for byte (LF bytes); failing that, text and integers exactly and every float to
|a - b| <= 1e-9 max(1, |b|). It also checks that the points drawn are the power_pct
values of the data file. The
exit code is non-zero on any difference, and 2 when the source is not a document of
overdetrend_power_precise.py with the curves of CBARS. A few seconds.

    python mlb/overdetrend_figure.py [--out-dir D] [--power-data PATH]
                                                      [--check]
"""

import argparse
import csv
import hashlib
import importlib
import io
import itertools
import json
import math
import os
import sys

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

matplotlib = importlib.import_module("matplotlib")
matplotlib.use("Agg")
plt = importlib.import_module("matplotlib.pyplot")
paths = importlib.import_module("mlb.paths")
_figstyle = importlib.import_module("mlb._figstyle")
_precise = importlib.import_module("mlb.overdetrend_power_precise")

CBARS, REF, C_PRINTED = _precise.CBARS, _precise.REF, _precise.C_PRINTED
# |c-bar| = 7, 13.5, 21.7 is an ORDINAL severity axis (how far the
# practitioner's detrending value has drifted from the correct -7), not three
# unrelated categories, so this indexes the sequential ORDINAL_RAMP (black ->
# lighter blue) by rank rather than reusing a qualitative palette -- see
# _figstyle.py for the ramp (luminance strictly increasing with |c-bar|).
CBAR_STYLE = {
    -7.0: dict(**_figstyle.ORDINAL_RAMP[0],
               label=r"$\bar c = -7$ (Model LB, correct)"),
    -13.5: dict(**_figstyle.ORDINAL_RAMP[1], label=r"$-13.5$ (trend value)"),
    _precise.CBAR_CKP: dict(**_figstyle.ORDINAL_RAMP[3],
                            label=r"$-21.7$ (reused trend surface)"),
}

SECTION, OBJ = "critical_values", "overdetrend_figure"
FIGURE = "Figure_2.pdf"
DATA = "Figure_2_data.csv"
MANIFEST = "Figure_2_data.manifest.json"
REFERENCE_DIR = paths.reference_dir("figures")
# the outputs compared by --check: those with a file in reference/figures/
REFERENCE_OUTPUTS = (DATA,)
SOURCE = "reference/calibration/overdetrend_power.json"
COLUMNS = ["abs_c", "c", "cbar", "power", "power_pct", "se_total", "se_binomial",
           "nrep", "n_ok", "source", "loss_vs_ref", "se_loss", "seed"]
TOL = 1e-9


def _lf(data):
    return data.replace(b"\r\n", b"\n")


def _sha(data):
    return hashlib.sha256(_lf(data)).hexdigest()


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _write(path, data):
    with open(path, "wb") as fh:
        fh.write(data)


def source_name(path):
    """The source's path relative to the repository root (forward slashes), or its
    file name when it lies outside the repository."""
    p, root = os.path.abspath(path), os.path.abspath(str(paths.REPO_ROOT))
    try:
        inside = os.path.commonpath([p, root]) == root
    except ValueError:  # another drive
        inside = False
    return os.path.relpath(p, root).replace(os.sep, "/") if inside else (
        os.path.basename(p))


def load(path=None):
    """(the power document, its bytes, the grid sorted by |c|, one row per point),
    from path (default the reference). A ValueError if the document is not one of
    overdetrend_power_precise.py with the curves of CBARS."""
    path = paths.resolve("overdetrend_power") if path is None else path
    raw = _read(path)
    g = json.loads(raw.decode("utf-8"))
    if g.get("reference_cbar") != REF or g.get("printed_alternative") != C_PRINTED:
        raise ValueError(
            f"{path}: reference_cbar {g.get('reference_cbar')!r} and "
            f"printed_alternative {g.get('printed_alternative')!r}, not {REF!r} and "
            f"{C_PRINTED!r}")
    missing = [cb for cb in CBARS if cb not in g.get("cbars", [])]
    if missing:
        raise ValueError(f"{path}: the curves {missing} are not among the cbars "
                         f"{g.get('cbars')}")
    grid = sorted(g["grid"], key=lambda c: abs(c))
    rows = []
    for c in grid:
        ent = g["by_c"][str(c)]
        for cb in CBARS:
            cell = ent["cells"][str(cb)]
            rows.append({
                "abs_c": abs(c), "c": c, "cbar": cb, "power": cell["power"],
                "power_pct": 100.0 * cell["power"], "se_total": cell["se_total"],
                "se_binomial": cell["se_binomial"], "nrep": ent["nrep"],
                "n_ok": cell["n_ok"],
                "source": ("printed cell of Section 4.4, N = 80,000,000"
                           if c == C_PRINTED
                           else "figure grid point, N = 2,000,000"),
                "loss_vs_ref": cell.get("loss_vs_ref", 0.0),
                "se_loss": cell.get("se_loss", 0.0), "seed": ent["seed"]})
    keys = [(r["c"], r["cbar"]) for r in rows]
    if not len(set(keys)) == len(rows) == len(grid) * len(CBARS):
        raise ValueError(f"{path}: {len(rows)} points, {len(set(keys))} distinct, "
                         f"for {len(grid)} values of c and {len(CBARS)} curves")
    return g, raw, grid, rows


def crossings(grid, rows):
    """Per curve: sign changes of loss(-7 - cb) along |c|, bracketing nodes and se.

    The root is given by the chord (linear) and by a shape-preserving cubic (pchip)
    through all grid nodes; the loss is convex near its zero, so the chord is biased
    outward. se of the pchip root by the delta method, se_loss(root) / |loss'(root)|,
    and the decimals it sustains (se <= 0.5 * 10^-d). A curve that does not cross is
    right-censored at the grid's last |c|.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq

    out = {}
    by = {(r["c"], r["cbar"]): r for r in rows}
    for cb in CBARS:
        if cb == REF:
            continue
        seq = [(abs(c), by[(c, cb)]["loss_vs_ref"], by[(c, cb)]["se_loss"])
               for c in grid if c != 0.0]
        xs, ls = [s[0] for s in seq], [s[1] for s in seq]
        f = PchipInterpolator(xs, ls)
        df = f.derivative()
        cr = []
        for (x0, l0, s0), (x1, l1, s1) in itertools.pairwise(seq):
            if l0 > 0 >= l1 or l0 < 0 <= l1:
                root = brentq(lambda x, f=f: float(f(x)), x0, x1, xtol=1e-12)
                se_at = s0 + (s1 - s0) * (root - x0) / (x1 - x0)
                se_root = se_at / abs(float(df(root)))
                cr.append({"abs_c_linear": x0 + (x1 - x0) * l0 / (l0 - l1),
                           "abs_c_pchip": root, "se_root_delta": se_root,
                           "d1_decimals": math.floor(-math.log10(2 * se_root)),
                           "left": {"abs_c": x0, "loss": l0, "se_loss": s0,
                                    "z": l0 / s0},
                           "right": {"abs_c": x1, "loss": l1, "se_loss": s1,
                                     "z": l1 / s1}})
        above = [x for x, loss, s in seq if loss > 2 * s]
        out[str(cb)] = {
            "crossings": cr, "crosses_within_grid": bool(cr),
            "grid_max_abs_c": max(xs),
            "largest_abs_c_loss_above_2se": max(above) if above else None,
            "right_censored": bool(above) and max(above) == max(xs),
            "losses": [{"abs_c": x, "loss": loss, "se_loss": s, "z": loss / s}
                       for x, loss, s in seq]}
    return out


def plot(grid, rows, out_path):
    """Draw the figure (6.2 x 3.8 in, |c| on the x axis, the printed alternative
    |c| = 10 dotted) for the curves in CBARS; return {cbar: (x, y)} as read back from
    the drawn curves."""
    by = {(r["c"], r["cbar"]): r for r in rows}
    x = [abs(c) for c in grid]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    drawn = {}
    for cb in CBARS:
        st = CBAR_STYLE[cb]
        (line,) = ax.plot(x, [by[(c, cb)]["power_pct"] for c in grid],
                          marker=st["marker"], ls=st["ls"], color=st["color"], lw=1.8,
                          ms=st["ms"], label=st["label"])
        drawn[cb] = ([float(v) for v in line.get_xdata()],
                     [float(v) for v in line.get_ydata()])
    ax.axvline(10, ls=":", color="0.6", lw=1, zorder=0)
    ax.set_xlabel("local alternative $|c|$")
    ax.set_ylabel("size-corrected power (%)")
    ax.set_title(r"Over-detrending loses power at moderate alternatives "
                 r"($T=60$, $m=2$, nominal $5\%$)", fontsize=9)
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return drawn


def data_bytes(rows):
    """The data file: COLUMNS, one row per point, floats as repr, LF."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


def drawn_problems(drawn, data):
    """Differences between the drawn points and the power_pct of the data file."""
    rows = list(csv.DictReader(io.StringIO(_lf(data).decode("utf-8"))))
    out = []
    for cb in CBARS:
        want = ([float(r["abs_c"]) for r in rows if float(r["cbar"]) == cb],
                [float(r["power_pct"]) for r in rows if float(r["cbar"]) == cb])
        if drawn.get(cb) != want:
            out.append(f"cbar {cb}: drawn {drawn.get(cb)} != data {want}")
    return out


def manifest_bytes(g, raw, grid, ana, data_sha, source=SOURCE):
    """The manifest of the data file (source: the name of the power document read)."""
    man = {
        "object": OBJ,
        "figure": f"{FIGURE} (fig:overdetrend)",
        "command": "python mlb/overdetrend_figure.py",
        "command_note": "--out-dir does not change the data file",
        "what": ("the points of Figure 2: the size-corrected power (%) of the "
                 "nominal-5% demeaned MZ_t test against |c| for detrending at each "
                 "cbar of the curves (T = 60, m = 2), read from the reference "
                 "overdetrend_power.json; no simulation"),
        "source": source,
        "source_sha256_lf": _sha(raw),
        "curves": CBARS,
        "n_abs_c_points": len(grid),
        "abs_c_points": [abs(c) for c in grid],
        "nrep": {"printed_cell": g["by_c"][str(C_PRINTED)]["nrep"],
                 "figure_grid": sorted({g["by_c"][str(c)]["nrep"] for c in grid
                                        if c != C_PRINTED})},
        "crn": ("yes: every cbar is evaluated on the same draw of each replication "
                "(overdetrend_power_precise._draw_mzt); each alternative has its own "
                "disjoint seed stream"),
        "se": "per point the reference's se_total; per loss the reference's paired se_loss",
        "columns": COLUMNS,
        "n_rows": len(grid) * len(CBARS),
        "power_pct": "100 * power in floating point, written with repr",
        "analysis": ana,
        "outputs_sha256_lf": {DATA: data_sha},
        "check": {"mode": ("LF bytes; otherwise text and integers exactly and floats "
                           "to |a - b| <= tol max(1, |b|)"), "tol": TOL},
    }
    return (json.dumps(man, indent=1) + "\n").encode("utf-8")


def _is_int(s):
    return s.lstrip("-").isdigit()


def _same_token(a, b):
    """Text and integers exactly; floats to TOL max(1, |b|); NaN equals NaN."""
    if a == b:
        return True
    if _is_int(a) or _is_int(b):
        return False
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return abs(x - y) <= TOL * max(1.0, abs(y))


def compare_data(new, ref):
    """('identical' | 'within tolerance' | 'DIFFERENT', [differences])."""
    if _lf(new) == _lf(ref):
        return "identical", []
    a = list(csv.reader(io.StringIO(_lf(new).decode("utf-8"))))
    b = list(csv.reader(io.StringIO(_lf(ref).decode("utf-8"))))
    if len(a) != len(b):
        return "DIFFERENT", [f"{len(a)} lines against {len(b)}"]
    diffs = [f"line {k + 1}: {ra} != {rb}" for k, (ra, rb) in enumerate(zip(a, b))
             if len(ra) != len(rb) or not all(map(_same_token, ra, rb))]
    return ("within tolerance" if not diffs else "DIFFERENT"), diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=None,
                    help="output folder (default output/critical_values/"
                         "overdetrend_figure/)")
    ap.add_argument("--power-data", default=None, metavar="PATH",
                    help="the power document to draw from (default the reference "
                         f"{SOURCE})")
    ap.add_argument("--check", action="store_true",
                    help="compare the data file with the reference")
    a = ap.parse_args(argv)
    src = (str(paths.resolve("overdetrend_power")) if a.power_data is None
           else a.power_data)
    name = source_name(src)
    try:
        g, raw, grid, rows = load(src)
    except (OSError, ValueError, KeyError) as e:
        print(f"[{OBJ}] error: cannot read the power document: {e}", file=sys.stderr)
        return 2
    out = a.out_dir or str(paths.out_dir(SECTION) / OBJ)
    os.makedirs(out, exist_ok=True)
    _figstyle.apply_house_rcparams()
    data = data_bytes(rows)
    ana = crossings(grid, rows)
    man = manifest_bytes(g, raw, grid, ana, _sha(data), source=name)
    _write(os.path.join(out, DATA), data)
    _write(os.path.join(out, MANIFEST), man)
    drawn = plot(grid, rows, os.path.join(out, FIGURE))
    problems = drawn_problems(drawn, _read(os.path.join(out, DATA)))
    for cb, v in ana.items():
        cr = [(c["abs_c_linear"], c["abs_c_pchip"], c["se_root_delta"])
              for c in v["crossings"]]
        censored = " (right-censored at the grid edge)" if v["right_censored"] else ""
        print(f"[{OBJ}] {cb}: crossings (linear, pchip, se) {cr}; largest |c| with "
              f"loss > 2 se: {v['largest_abs_c_loss_above_2se']}{censored}")
    n_pts = sum(len(x) for x, _ in drawn.values())
    print(f"[{OBJ}] wrote {DATA}, {MANIFEST} and {FIGURE} ({n_pts} points drawn) "
          f"in {out}")
    for p in problems:
        print(f"[{OBJ}] {p}")
    if not a.check:
        return 0 if not problems else 1
    new = {DATA: data}
    ok = not problems
    for fname in REFERENCE_OUTPUTS:
        rpath = REFERENCE_DIR / fname
        word, diffs = (("MISSING", []) if not rpath.is_file()
                       else compare_data(new[fname], _read(rpath)))
        print(f"--check: {fname}: {word}")
        for d in diffs[:20]:
            print(f"  {d}")
        ok = ok and word in ("identical", "within tolerance")
    print(f"--check: {'PASS' if ok else 'FAIL'} against {REFERENCE_DIR}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
