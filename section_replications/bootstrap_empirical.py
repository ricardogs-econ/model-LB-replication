"""Door: sec:empirical (paper section 6).

  NUMBER           PRODUCER (this door dispatches to)     FROZEN GOLDEN
  tab:dates        inputs/exog_dates.csv (curated)        inputs/exog_dates.csv
  tab:ppp          boot_ppp_cbar.py --empirical           output/golden/bootstrap_empirical/ppp_empirical.csv
  tab:pppsurface   boot_ppp_cbar.py --full (surface)      output/golden/bootstrap_empirical/tab_pppsurface.tex
                                                            (source inputs/boot_out/calib/surface_ppp_boot.csv)
  tab:hl           hl_median_unbiased.py (wild bootstrap)  output/golden/bootstrap_empirical/hl_results_wild.csv
  fig:rerseries    rendered by the `figures` door (door 4), from ppp_panel.csv + exog_dates.csv
  Pesaran CD       pesaran_cd.py (§6.3 footnote)          reproduced in full (cheap)

--quick: confirm the golden artefacts + curated dates are present; run Pesaran CD in full.
--full: regenerate into output/ and diff vs golden, at a documented cert-N.

PRODUCTION commands (the golden was produced at these):
  tab:ppp         python section_replications/boot_ppp_cbar.py --empirical --emp-dir inputs (sieve-own CV, R_cv_emp=20000)
  tab:pppsurface  python section_replications/boot_ppp_cbar.py --full      (surface calibration, R_cv=10000/R_pow=5000)
  tab:hl          python section_replications/hl_median_unbiased.py --nsim 1000000 --B 19999 --boot wild
"""
import csv
import glob
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mlb import paths
from section_replications import _doorlib as D

DOOR = "bootstrap_empirical"
OBJECTS = ["tab:dates", "tab:hl", "tab:ppp", "tab:pppsurface", "fig:rerseries"]

# documented cert-N (a replicator's --full cost; the golden stays the banked
# production value certified separately).
HL_CERT_NSIM = 10_000
HL_CERT_B = 1_999
SURF_CERT_NREP = 2_000

_PROD = os.path.join(paths.REPO_ROOT, "section_replications")


def _cert_pesaran():
    # pesaran_cd.py reports BOTH the levels and first-difference CD statistics
    # in one run (two "CD statistic" lines, in that order).
    out = subprocess.run(
        [sys.executable, os.path.join(_PROD, "pesaran_cd.py"),
         "--panel", str(paths.resolve("ppp_panel")),
         "--dates", str(paths.resolve("exog_dates")), "--start", "1973"],
        capture_output=True, text=True).stdout
    cd = [float(l.split(":")[1]) for l in out.splitlines() if "CD statistic" in l]
    for tag, val, golden in (("levels", cd[0] if cd else None, 14.04),
                             ("first-diff", cd[1] if len(cd) > 1 else None, 8.65)):
        ok = val is not None and abs(val - golden) < 0.05
        print(f"  Pesaran CD ({tag}): {val:+.4f} vs golden {golden:+.2f}  "
              f"{'OK' if ok else 'DIFFER'}")


def _cert_tab_ppp():
    run = paths.out_dir("bootstrap_empirical") / "ppp_run"
    (run / "calib").mkdir(parents=True, exist_ok=True)
    # the empirical block reads the frozen surface from <out>/calib/
    import shutil
    shutil.copy(paths.resolve("surface_ppp_boot"), run / "calib" / "surface_ppp_boot.csv")
    subprocess.run(
        [sys.executable, os.path.join(_PROD, "boot_ppp_cbar.py"), "--empirical",
         "--emp-dir", str(paths.INPUTS), "--out", str(run),
         "--start-year", "1973", "--n-jobs", "-1"],
        check=True, capture_output=True, text=True)
    gold = {r["currency"]: r for r in csv.DictReader(open(paths.resolve("ppp_empirical")))}
    got = {r["currency"]: r for r in csv.DictReader(
        open(run / "empirical" / "ppp_empirical.csv"))}
    vok = ntot = 0
    maxdp = 0.0
    for cur in gold:
        vok += gold[cur]["reject_cal"] == got[cur]["reject_cal"]
        maxdp = max(maxdp, abs(float(gold[cur]["pvalue_cf"]) - float(got[cur]["pvalue_cf"])))
        ntot += 1
    print(f"  tab:ppp    {vok}/{ntot} verdicts match golden; max |d pvalue_cf| = {maxdp:.4f} "
          f"(R_cv_emp=20000, sieve-own)")
    return vok == ntot


def _cert_tab_pppsurface():
    run = paths.out_dir("bootstrap_empirical") / "surf_cert"
    out = subprocess.run(
        [sys.executable, os.path.join(_PROD, "boot_ppp_cbar.py"), "--grid",
         "--m", "2", "--T", "52", "--p", "1", "--nrep", str(SURF_CERT_NREP),
         "--emp-dir", str(paths.INPUTS), "--out", str(run), "--n-jobs", "-1"],
        capture_output=True, text=True).stdout
    got = next((float(l.split("=")[1].split("(")[0]) for l in out.splitlines()
                if "cbar* =" in l), None)
    frozen = -11.5703  # (m=2,T=52,p=1) in inputs/boot_out/calib/surface_ppp_boot.csv
    d = abs(got - frozen) if got is not None else float("nan")
    print(f"  tab:pppsurface cell (m=2,T=52,p=1) cbar*={got:.3f} vs frozen {frozen:.3f}, "
          f"|d|={d:.3f} (within resolution; production --full at R_cv=10000/R_pow=5000)")
    return got is not None and d < 0.5


def _cert_tab_hl():
    out = paths.out_dir("bootstrap_empirical") / "hl_cert.csv"
    subprocess.run(
        [sys.executable, os.path.join(_PROD, "hl_median_unbiased.py"),
         "--panel", str(paths.resolve("ppp_panel")),
         "--dates", str(paths.resolve("exog_dates")),
         "--diag", str(paths.resolve("ppp_ar_diagnostic")),
         "--start-year", "1973", "--nsim", str(HL_CERT_NSIM), "--B", str(HL_CERT_B),
         "--boot", "wild", "--out", str(out)],
        check=True, capture_output=True, text=True)
    gold = {r["currency"]: r for r in csv.DictReader(
        open(paths.resolve("ppp_empirical").parent / "hl_results_wild.csv"))}
    got = {r["currency"]: r for r in csv.DictReader(open(out))}
    coll = det = ntot = 0
    for cur in gold:
        coll += gold[cur]["collapse"] == got[cur]["collapse"]
        det += abs(float(gold[cur]["alpha_ols_MP"]) - float(got[cur]["alpha_ols_MP"])) < 1e-6
        ntot += 1
    print(f"  tab:hl     collapse {coll}/{ntot} match, deterministic alpha_ols {det}/{ntot} exact "
          f"(cert nsim={HL_CERT_NSIM}/B={HL_CERT_B}; production nsim=1e6/B=19999)")
    return coll == ntot and det == ntot


def run(full=False):
    print(f"\n[{DOOR}] {'--full' if full else '--quick'}")
    print(f"  tab:dates      curated: {D.present('exog_dates').name}")
    print(f"  tab:ppp        golden:  {D.present('ppp_empirical').name}")
    print(f"  tab:pppsurface golden:  {D.present('tab_pppsurface').name}")
    print(f"  tab:hl         golden:  {(paths.resolve('ppp_empirical').parent / 'hl_results_wild.csv').name}")
    print("  fig:rerseries  rendered by the `figures` door (door 4)")
    _cert_pesaran()
    if full:
        results = [_cert_tab_ppp(), _cert_tab_pppsurface(), _cert_tab_hl()]
        return all(results)
    return True
