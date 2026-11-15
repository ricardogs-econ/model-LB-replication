"""The size and the power of the admission tests C4 and C6 on each candidate's own
design.

For each currency with a rule-E design (inputs/regime_events.csv), Delta q_t is
simulated at T = 51 (t = 1974, ..., 2024) and the tests are computed exactly as on the
data (admissible_universe: the impulse residual at the event dates, kappa2, c6_test):

  size, R_SIZE = 100,000 draws of the null DGP, Delta q_t i.i.d. N(0, 1): the rejection
    rates of C4 (kappa2 > 1.358, > 1.628), of C6 with the asymptotic chi2(m) reference
    (p < 0.05, < 0.01) and of C6 with the simulated critical values of
    reference/application/admission/c6_critical_values.csv (W > cv_5, > cv_1);
  power, R_POWER = 20,000 draws of each of two alternatives, the standard deviation of
    Delta q doubled (a) from the first event date tau_1 on and (b) from 1999 on (a
    variance break away from the event dates): the rejection rates of C4 at 1.358 and of
    C6 with its simulated critical value at 5%.
Each rate is reported with its Monte Carlo se, sqrt(r (1 - r) / R).

Seeds: replication r of the size draws is np.random.default_rng(seed0("gate_size|<cur>")
+ r); alternative (a) uses seed0("gate_power|<cur>") + r and alternative (b)
seed0("gate_power|<cur>") + R_POWER + r (mlb/app_seeds.py, disjointness
asserted).

    python mlb/gate_sizes.py [--jobs 8] [--out <file.csv>]
        -> output/bootstrap_empirical/admission/gate_sizes.csv (default), and its run
           record gate_sizes.manifest.json beside it

The shipped result is reference/application/admission/gate_sizes.csv.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlb import admissible_universe as U  # noqa: E402
from mlb import app_seeds as A  # noqa: E402
from mlb import c6_critical_values as C6  # noqa: E402
from mlb import paths  # noqa: E402
from mlb import regime_events as R  # noqa: E402

R_SIZE = 100_000
R_POWER = 20_000
BREAK_B = 1999
YEARS = np.arange(1974, 2025)
T = len(YEARS)
OUT = str(paths.OUTPUT / "bootstrap_empirical" / "admission" / "gate_sizes.csv")
RATES = ["size_C4_5", "size_C4_1", "size_C6chi2_5", "size_C6chi2_1", "size_C6sim_5",
         "size_C6sim_1", "power_a_C4_5", "power_a_C6sim_5", "power_b_C4_5",
         "power_b_C6sim_5"]
COLUMNS = (["currency", "taus", "m", "R_size", "R_power", "seed0_size", "seed0_power"]
           + [c for r in RATES for c in (r, "se_" + r)])


def stats(z, taus):
    """(kappa2, W, chi2 p) of one simulated Delta q."""
    eps = U.impulse_residual(z, YEARS, taus)
    w, _, p = U.c6_test(eps, YEARS, taus)
    return U.kappa2(eps)[0], w, p


def one(cur, taus, cv5, cv1):
    size = np.array([stats(A.rng(f"gate_size|{cur}", r).standard_normal(T), taus)
                     for r in range(R_SIZE)])
    k2, w, p = size[:, 0], size[:, 1], size[:, 2]
    rec = dict(currency=cur, taus=";".join(map(str, taus)), m=len(taus), R_size=R_SIZE,
               R_power=R_POWER, seed0_size=A.seed0(f"gate_size|{cur}"),
               seed0_power=A.seed0(f"gate_power|{cur}"))
    rates = {"size_C4_5": ~(k2 <= U.KAPPA2_CV[0.05]),
             "size_C4_1": ~(k2 <= U.KAPPA2_CV[0.01]),
             "size_C6chi2_5": p < 0.05, "size_C6chi2_1": p < 0.01,
             "size_C6sim_5": w > cv5, "size_C6sim_1": w > cv1}
    for lab, start, offset in (("a", taus[0], 0), ("b", BREAK_B, R_POWER)):
        scale = np.where(YEARS >= start, 2.0, 1.0)
        pw = np.array([stats(scale * A.rng(f"gate_power|{cur}", offset + r)
                             .standard_normal(T), taus) for r in range(R_POWER)])
        rates[f"power_{lab}_C4_5"] = ~(pw[:, 0] <= U.KAPPA2_CV[0.05])
        rates[f"power_{lab}_C6sim_5"] = pw[:, 1] > cv5
    for key, hits in rates.items():
        r = float(np.mean(hits))
        rec[key] = r
        rec["se_" + key] = math.sqrt(r * (1.0 - r) / len(hits))
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    des = C6.designs(R.EVENTS)
    cv = C6.load(C6.CV_FILE)
    used = {f"gate_size|{c}": R_SIZE for c in des}
    used.update({f"gate_power|{c}": 2 * R_POWER for c in des})
    bound = A.assert_disjoint(used)
    args = [(c, t, float(cv[c]["cv_5"]), float(cv[c]["cv_1"]))
            for c, t in sorted(des.items())]
    if a.jobs > 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=a.jobs)(delayed(one)(*x) for x in args)
    else:
        recs = [one(*x) for x in args]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for rec in recs:
            w.writerow({k: (repr(v) if isinstance(v, float) else v)
                        for k, v in rec.items()})
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    man = dict(output=os.path.basename(a.out), output_sha256=sha,
               c6_cv_sha256_lf=hashlib.sha256(open(C6.CV_FILE, "rb").read()
                                              .replace(b"\r\n", b"\n")).hexdigest(),
               events_sha256_lf=hashlib.sha256(open(R.EVENTS, "rb").read()
                                               .replace(b"\r\n", b"\n")).hexdigest(),
               R_size=R_SIZE, R_power=R_POWER, break_b=BREAK_B,
               tabcv_seed_upper_bound=bound)
    json.dump(man, open(os.path.splitext(a.out)[0] + ".manifest.json", "w",
                        encoding="utf-8", newline="\n"), indent=1)
    for rec in recs:
        print(rec["currency"], {k: round(rec[k], 5) for k in RATES})
    print(f"sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
