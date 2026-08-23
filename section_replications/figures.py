"""Door: figures (consolidates all figure rendering; no simulation).

  FIGURE           DATA SOURCE                                   RENDERED BY
  fig:rerseries    inputs/ppp_panel.csv + inputs/exog_dates.csv  generate_figures --only fig3  -> Figure_3.pdf
  sfig:density     inputs/limiting_density.csv (frozen input)    generate_figures --only figS1 -> Figure_S1.pdf
  sfig:forest      output/golden/.../hl_results_wild.csv (+hl_results.csv) generate_figures --only figS2 -> Figure_S2.pdf
  fig:overdetrend  Figure_1.pdf -- the critical_values door's object (replicate.py cv --full)
  fig:size         Figure_2.pdf -- the critical_values door's object (replicate.py cv --full)

generate_figures reads CSV/artefact data and renders PDFs -- NO simulation. The shared palette
(ORDINAL_RAMP) is imported from section_replications/_figstyle.py, not duplicated.

--quick: confirm the data sources for every figure are present in-tree.
--full: render Figure_3 / Figure_S1 / Figure_S2 into output/figures/ from the shipped data and
confirm each PDF is non-empty. matplotlib PDFs are NOT byte-deterministic, so the DATA is what is
certified: Figure_3 from shipped inputs, Figure_S2 from the golden hl table, Figure_S1 from the
frozen limiting_density.csv.
"""
import os
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mlb import paths

DOOR = "figures"
OBJECTS = ["fig:rerseries", "sfig:density", "sfig:forest"]

# figure key -> (object label, data keys it reads, output pdf name)
_FIGS = {
    "fig3":  ("fig:rerseries", ["ppp_panel", "exog_dates"], "Figure_3.pdf"),
    "figS1": ("sfig:density", ["limiting_density"], "Figure_S1.pdf"),
    "figS2": ("sfig:forest", ["hl_results_wild", "hl_results"], "Figure_S2.pdf"),
}


def run(full=False):
    print(f"\n[{DOOR}] {'--full' if full else '--quick'}")
    all_ok = True
    for _fig, (obj, data_keys, _pdf) in _FIGS.items():
        ok = all(paths.resolve(dk).exists() for dk in data_keys)
        all_ok = all_ok and ok
        print(f"  {obj:<14} data: {', '.join(data_keys)}  {'OK' if ok else 'MISSING'}")
    print("  fig:overdetrend / fig:size (Figure_1/2): the critical_values door's objects")
    if not full:
        return all_ok

    outdir = paths.out_dir("figures")
    data_dir = outdir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for dk in ("ppp_panel", "exog_dates", "limiting_density", "hl_results_wild", "hl_results"):
        src = paths.resolve(dk)
        shutil.copy(src, data_dir / src.name)
    prod = os.path.join(paths.REPO_ROOT, "section_replications", "generate_figures.py")
    for fig, (obj, _dk, pdf_name) in _FIGS.items():
        subprocess.run(
            [sys.executable, prod, "--only", fig, "--data-dir", str(data_dir),
             "--out-dir", str(outdir), "--start-year", "1973"],
            check=True, capture_output=True, text=True)
        pdf = outdir / pdf_name
        size = pdf.stat().st_size if pdf.exists() else 0
        ok = size > 1000
        all_ok = all_ok and ok
        print(f"  {obj:<14} -> {pdf_name}  ({size} bytes, {'OK' if ok else 'EMPTY/MISSING'})")
    return all_ok
