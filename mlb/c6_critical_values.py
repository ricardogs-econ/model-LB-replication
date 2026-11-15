"""The critical values of the admission condition C6, simulated at T = 51 on each
currency's own design, and their validation.

C6 tests for a variance break at the event dates: W is the Wald statistic of the regime
indicators 1{t >= tau_j}, j = 1..m, in the OLS regression of eps_hat_t^2 on a constant
and them, with a Bartlett HAC covariance of lag floor(4 (T/100)^(2/9)) = 3
(admissible_universe.c6_test, the same function the data go through). At T = 51 the
asymptotic chi2(m) reference does not give a 5% test, so C6 uses critical values
simulated under the null:

  the null DGP: Delta q_t i.i.d. N(0, 1), t = 1974, ..., 2024 (T = 51 differences, no
  serial dependence);
  eps_hat = the OLS residual of Delta q on a constant and the impulses 1{t = tau_j} at
  the currency's event dates of rule E (inputs/regime_events.csv, merged); W computed
  exactly as on the data. W does not depend on the scale of Delta q.

For each currency, R_CAL = 100,000 draws give the critical values: cv_a = the draw of
rank ceil((1 - a) R_CAL) in increasing order; C6 rejects at level a when W > cv_a (the
verdict). The p-value #{draws >= W} / R_CAL agrees with that rule (W > cv_a if and only
if p <= a, draws without ties); its Monte Carlo se is sqrt(p (1 - p) / R_CAL). The se of
cv_a is half the distance between the order statistics of ranks
(1 - a) R_CAL +/- sqrt(R_CAL a (1 - a)).

A validation batch of R_VAL = 100,000 fresh draws (seeds disjoint from the
calibration's) measures the size of the test with cv_a. The file records the deviation
|size - a| in two standard errors: se = sqrt(a (1 - a) (1/R_CAL + 1/R_VAL)), which
counts the calibration's error and the validation's (column check_5, the check this
module applies), and se_val = sqrt(a (1 - a) / R_VAL), the validation's alone (column
check_5_val).

Seeds: replication r of item "c6_cal|<currency>" or "c6_val|<currency>" draws
np.random.default_rng(seed0(item) + r) (mlb/app_seeds.py, disjointness
asserted).

    python mlb/c6_critical_values.py [--jobs 8] [--out <file.csv>]
        -> output/bootstrap_empirical/admission/c6_critical_values.csv (default), and
           its run record c6_critical_values.manifest.json beside it

The values the other modules read (load(), admissible_universe, gate_sizes) are those
of reference/application/admission/c6_critical_values.csv (CV_FILE).
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys

import numpy as np
import scipy
from scipy import stats as st

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from mlb import admissible_universe as U  # noqa: E402
from mlb import app_seeds as A  # noqa: E402
from mlb import paths  # noqa: E402
from mlb import regime_events as R  # noqa: E402

R_CAL = 100_000
R_VAL = 100_000
HEAD = 1_000
LEVELS = (0.05, 0.01)
REL_TOL = 1e-9
YEARS = np.arange(1974, 2025)
T = len(YEARS)
CV_FILE = str(paths.resolve("c6_critical_values"))  # the values the modules read
OUT = str(paths.OUTPUT / "bootstrap_empirical" / "admission" / "c6_critical_values.csv")
COLUMNS = ["currency", "taus", "m", "T", "hac_lag", "R_cal", "seed0_cal", "R_val",
           "seed0_val", "cv_5", "cv_1", "se_cv_5", "se_cv_1", "chi2_cv_5", "chi2_cv_1",
           "val_size_5", "val_size_1", "se_5", "se_1", "se_val_5", "se_val_1", "z_5",
           "z_val_5", "check_5", "check_5_val", "draws_sha256", "draws_head_sha256"]


def null_draws(taus, item, n):
    """W under the null DGP, replications r = 0..n-1 of a work item."""
    out = np.empty(n)
    for r in range(n):
        z = A.rng(item, r).standard_normal(T)
        out[r] = U.c6_test(U.impulse_residual(z, YEARS, taus), YEARS, taus)[0]
    return out


def draws_sha256(w):
    return hashlib.sha256(np.ascontiguousarray(w, dtype="<f8").tobytes()).hexdigest()


def critical_value(w, a):
    s = np.sort(w)
    return float(s[int(math.ceil((1.0 - a) * len(s))) - 1])


def critical_value_se(w, a):
    """Half the distance between the order statistics of ranks
    (1 - a) R +/- sqrt(R a (1 - a))."""
    s = np.sort(w)
    n = len(s)
    h = math.sqrt(n * a * (1.0 - a))
    lo = int(math.ceil((1.0 - a) * n - h)) - 1
    hi = int(math.ceil((1.0 - a) * n + h)) - 1
    return float((s[min(hi, n - 1)] - s[max(lo, 0)]) / 2.0)


def p_value(w_obs, draws):
    return float(np.sum(draws >= w_obs)) / len(draws)


def p_value_se(p, n):
    return math.sqrt(p * (1.0 - p) / n)


def designs(events=R.EVENTS):
    rows = R.load(events)
    return {c: R.configuration(rows, c, "E")[0] for c in R.currencies(rows)
            if R.configuration(rows, c, "E")[0]}


def calibration(cur, row):
    """The calibration draws of a currency, regenerated and checked against its row:
    cv_5 and cv_1 to REL_TOL (relative); returns (draws, the draws' sha256 matches)."""
    taus = [int(t) for t in row["taus"].split(";")]
    draws = null_draws(taus, f"c6_cal|{cur}", int(row["R_cal"]))
    for a in LEVELS:
        tag = str(int(round(100 * a)))
        cv, ref = critical_value(draws, a), float(row[f"cv_{tag}"])
        assert abs(cv - ref) <= REL_TOL * abs(ref), (cur, tag, cv, ref)
    return draws, draws_sha256(draws) == row["draws_sha256"]


def one(cur, taus):
    cal = null_draws(taus, f"c6_cal|{cur}", R_CAL)
    val = null_draws(taus, f"c6_val|{cur}", R_VAL)
    m = len(taus)
    rec = dict(currency=cur, taus=";".join(map(str, taus)), m=m, T=T,
               hac_lag=U.hac_lag(T), R_cal=R_CAL, seed0_cal=A.seed0(f"c6_cal|{cur}"),
               R_val=R_VAL, seed0_val=A.seed0(f"c6_val|{cur}"),
               draws_sha256=draws_sha256(cal),
               draws_head_sha256=draws_sha256(cal[:HEAD]))
    for a in LEVELS:
        tag = str(int(round(100 * a)))
        cv = critical_value(cal, a)
        rec[f"cv_{tag}"] = cv
        rec[f"se_cv_{tag}"] = critical_value_se(cal, a)
        rec[f"chi2_cv_{tag}"] = float(st.chi2.ppf(1.0 - a, m))
        rec[f"val_size_{tag}"] = float(np.mean(val > cv))
        rec[f"se_{tag}"] = math.sqrt(a * (1.0 - a) * (1.0 / R_CAL + 1.0 / R_VAL))
        rec[f"se_val_{tag}"] = math.sqrt(a * (1.0 - a) / R_VAL)
    rec["z_5"] = (rec["val_size_5"] - 0.05) / rec["se_5"]
    rec["z_val_5"] = (rec["val_size_5"] - 0.05) / rec["se_val_5"]
    rec["check_5"] = "Y" if abs(rec["z_5"]) <= 2.0 else "N"
    rec["check_5_val"] = "Y" if abs(rec["z_val_5"]) <= 2.0 else "N"
    return rec


def load(path=CV_FILE):
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["currency"]: r for r in csv.DictReader(fh)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    des = designs(R.EVENTS)
    used = {f"c6_cal|{c}": R_CAL for c in des}
    used.update({f"c6_val|{c}": R_VAL for c in des})
    bound = A.assert_disjoint(used)
    if a.jobs > 1:
        from joblib import Parallel, delayed
        recs = Parallel(n_jobs=a.jobs)(delayed(one)(c, t)
                                       for c, t in sorted(des.items()))
    else:
        recs = [one(c, t) for c, t in sorted(des.items())]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for rec in recs:
            w.writerow({k: (repr(v) if isinstance(v, float) else v)
                        for k, v in rec.items()})
    events_sha = hashlib.sha256(
        open(R.EVENTS, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    out_sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    man = dict(output=os.path.basename(a.out), output_sha256=out_sha,
               events_sha256_lf=events_sha, T=T,
               R_cal=R_CAL, R_val=R_VAL, levels=list(LEVELS), hac_lag=U.hac_lag(T),
               null_dgp="Delta q_t i.i.d. N(0,1), t = 1974..2024 (T = 51), "
                        "no serial dependence; impulses and regime "
                        "indicators at each currency's rule-E dates",
               seeds="default_rng(seed0(item) + r); "
                     "items c6_cal|<cur>, c6_val|<cur> (app_seeds.py)",
               tabcv_seed_upper_bound=bound,
               check="check_5: |val_size_5 - 0.05| <= 2 se_5, "
                    "se_5 = sqrt(0.05*0.95*(1/R_cal + 1/R_val)); check_5_val: the same "
                    "with se_val_5 = sqrt(0.05*0.95/R_val)",
               reproduction="cv_5 and cv_1 regenerate to a relative 1e-9; the draws' "
                            "sha256 is recorded, not required",
               numpy=np.__version__, scipy=scipy.__version__,
               python=platform.python_version())
    json.dump(man, open(os.path.splitext(a.out)[0] + ".manifest.json", "w",
                        encoding="utf-8", newline="\n"),
              indent=1)
    for rec in recs:
        print(f"{rec['currency']} taus {rec['taus']}: "
              f"cv_5 {rec['cv_5']!r} cv_1 {rec['cv_1']!r} "
              f"val_size_5 {rec['val_size_5']!r} z {rec['z_5']:.3f} "
              f"(validation se alone: z {rec['z_val_5']:.3f}) "
              f"check {rec['check_5']} / {rec['check_5_val']}")
    print(f"sha256 {out_sha}")
    return 0 if all(r["check_5"] == "Y" for r in recs) else 1


if __name__ == "__main__":
    sys.exit(main())
