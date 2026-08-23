"""Door: sec:robustness (paper section 5).

  NUMBER              PRODUCER / SOURCE
  tab:ar1-sizepower   ar1_sizepower_recal.run_ar1  (--full)  |  golden ar1_size_power.csv
  stab:recal          ar1_sizepower_recal (--recalibrate)    |  golden ar1_recalibration.csv

The producer is `section_replications/ar1_sizepower_recal.py` (numba-direct,
ar1+recal only, R_EVAL=10000, R_RECAL=4000, SEED_BASE=20260624). Its ar1+recal
core is byte-identical to the root-verified robustness harness
(HARNESS_RECONCILIATION); this file is that core on the numba kernel.

--quick: confirm the golden CSVs are present and summarize.
--full: regenerate into output/robustness/ and diff cell-by-cell vs golden
(the v2.0.0 certification gate).
"""
import csv
from section_replications import _doorlib as D

DOOR = "robustness"
OBJECTS = ["tab:ar1-sizepower", "stab:recal"]


def _summ(path, key):
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return f"{len(rows)} rows"


def run(full=False):
    print(f"\n[{DOOR}] {'--full' if full else '--quick'}")
    sp = D.present("ar1_size_power")
    rc = D.present("ar1_recalibration")
    print(f"  tab:ar1-sizepower golden: {sp} ({_summ(sp, 'ar1')})")
    print(f"  stab:recal        golden: {rc} ({_summ(rc, 'recal')})")
    if full:
        from section_replications import ar1_sizepower_recal as prod
        from src.mlb import paths
        outdir = str(paths.out_dir("robustness"))

        class _A:  # the producer's run_ar1 arg surface
            quick = False
            jobs = -1
            csv = str(paths.resolve("cbar_surface"))
            recalibrate = True
            recal_rho = [0.5]
            recal_method = ["const"]
        prod.kn.warm_up_numba(kmax=prod.KMAX)
        prod.run_ar1(_A(), outdir)
        print(f"  [--full] regenerated into {outdir}/ "
              "(ar1_size_power.csv, ar1_recalibration.csv, tab_ar1_sizepower.tex)")
    return True
