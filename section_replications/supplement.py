"""Door: supplement (online supplement §S2, §6.3).

  NUMBER            PRODUCER (this door dispatches to)     FROZEN GOLDEN
  stab:objects      modulus_feasible.py --csv-out          output/golden/supplement_appendix/modulus_feasible_applied.csv
  stab:disp         modulus_feasible.py --curves-out       output/golden/supplement_appendix/modulus_feasible_curves.csv
  stab:disp/L_T     power_cbar_star_vs_c7.run_cell         output/golden/supplement_appendix/g36_ladder.json
  §S2.1 assumption  assumption1_numerics.py                (deterministic identities, machine precision)
  §6.3 PMF          dependence_count_pmf.py                (exact enumeration)
  §6.3 bound        dependence_bound.py                    (one-factor Gaussian: 0.006 -> 0.09-0.10)
  §S2.1 modulus MC  modulus_numerics.py                    (M_T(h) across T; supporting)
  sfig:density / sfig:forest                               rendered by the `figures` door (door 4)

--quick: confirm the golden artefacts + run the deterministic numerics in full (cheap).
--full: additionally regenerate the Monte-Carlo modulus objects at a documented cert-N and
diff vs golden within resolution (the noise floor).

PRODUCTION commands:
  stab:objects/stab:disp  python section_replications/modulus_feasible.py --nrep 200000 --csv-out <a> --curves-out <c>
  stab:disp/L_T           power_cbar_star_vs_c7.run_cell at nrep=1,000,000 (g36 ladder)
"""
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mlb import paths

DOOR = "supplement"
OBJECTS = ["stab:objects", "stab:disp", "stab:disp/L_T", "sfig:density", "sfig:forest"]

MODULUS_CERT_NREP = 5_000      # golden is 200,000; the modulus is a supremum (upward-biased at low N)
GOLDEN_MODULUS_NREP = 200_000
LADDER_CERT_NREP = 50_000      # golden g36 is 1,000,000
_PROD = os.path.join(paths.REPO_ROOT, "section_replications")


def _run(mod, *args):
    return subprocess.run([sys.executable, os.path.join(_PROD, mod), *args],
                          capture_output=True, text=True)


def _cert_deterministic():
    # assumption1: machine-precision identities
    out = _run("assumption1_numerics.py").stdout
    errs = []
    for line in out.splitlines():
        for tok in line.replace(":", " ").split():
            try:
                v = float(tok)
                if 0 < abs(v) < 1e-3:
                    errs.append(abs(v))
            except ValueError:
                pass
    a_ok = bool(errs) and max(errs) < 1e-9
    print(f"  §S2 assumption1: closed-form vs general identities max err = "
          f"{max(errs):.1e}  {'OK (machine precision)' if a_ok else 'DIFFER'}")
    # dependence_bound + PMF: 0.006 independence -> 0.09-0.10 dependent
    out = _run("dependence_count_pmf.py").stdout
    p0 = {}
    for line in out.splitlines():
        if "independence" in line and "P(0)" in line:
            p0["ind"] = float(line.split("P(0)=")[1].split()[0])
        for rho in ("0.37", "0.41"):
            if line.startswith(f"rho={rho}"):
                p0[rho] = float(line.split("P(0)=")[1].split()[0])
    d_ok = (abs(p0.get("ind", 1) - 0.006) < 0.001
            and 0.08 < p0.get("0.37", 0) < 0.10 and 0.09 < p0.get("0.41", 0) < 0.11)
    print(f"  §6.3 dependence: P(0) independence={p0.get('ind')}, rho0.37={p0.get('0.37')}, "
          f"rho0.41={p0.get('0.41')}  {'OK (0.006 -> 0.09-0.10)' if d_ok else 'DIFFER'}")
    return a_ok and d_ok


def _cert_modulus():
    ad = paths.out_dir("supplement_appendix")
    a, c = str(ad / "modulus_applied.csv"), str(ad / "modulus_curves.csv")
    _run("modulus_feasible.py", "--nrep", str(MODULUS_CERT_NREP), "--csv-out", a, "--curves-out", c)
    # noise floor from the golden's own scope=noisefloor rows
    gold = list(csv.DictReader(open(paths.resolve("modulus_applied"))))
    floor = max(float(r["modulus"]) for r in gold if r["scope"] == "noisefloor")
    gkey = {(r["stat"], r["family"], r["scope"], r["T"], r["h"]): float(r["modulus"]) for r in gold}
    got = list(csv.DictReader(open(a)))
    worst = 0.0
    n = 0
    for r in got:
        k = (r["stat"], r["family"], r["scope"], r["T"], r["h"])
        if k in gkey:
            worst = max(worst, abs(float(r["modulus"]) - gkey[k]))
            n += 1
    # tolerance scaled to the cert-N's OWN MC resolution (not the production tol
    # loosened): the noise floor grows as 1/sqrt(nrep), so at cert-N the band is
    # golden_floor * sqrt(golden_nrep / cert_nrep).
    import math
    cert_floor = floor * math.sqrt(GOLDEN_MODULUS_NREP / MODULUS_CERT_NREP)
    tol = 2 * cert_floor
    print(f"  stab:objects modulus: {n} cells, max |d| = {worst:.4f} vs golden "
          f"(golden floor {floor:.4f}; cert-N floor {cert_floor:.4f}, tol 2x={tol:.4f})  "
          f"{'OK' if worst <= tol else 'EXCEEDS'} (cert nrep={MODULUS_CERT_NREP}; production 200000)")
    return worst <= tol


def _cert_ladder():
    import numpy as np
    from section_replications import power_cbar_star_vs_c7 as pc
    g = json.load(open(paths.resolve("g36_ladder")))
    cbars = sorted({float(r["cbar"]) for r in g["cells"]["60"]})
    rng = np.random.default_rng(g["seed"])
    res = pc.run_cell(60, 2, cbars, LADDER_CERT_NREP, 10_000, rng,
                      c_alt=g["c_alt"], verbose=False)
    gp = next(r["power_sc"] for r in g["cells"]["60"] if abs(float(r["cbar"]) + 7.0) < 1e-9)
    cp = float(res.power[-7.0])
    d = abs(gp - cp)
    print(f"  stab:disp/L_T ladder T=60, cbar=-7 power_sc: golden={gp:.4f} "
          f"cert={cp:.4f} |d|={d:.4f} (cert nrep={LADDER_CERT_NREP}; production 1e6)")
    return d < 0.01


def run(full=False):
    print(f"\n[{DOOR}] {'--full' if full else '--quick'}")
    for key, label in (("modulus_applied", "stab:objects"), ("modulus_curves", "stab:disp"),
                       ("g36_ladder", "stab:disp/L_T")):
        p = paths.resolve(key)
        print(f"  {label:<14} golden present: {p.exists()} ({p.name})")
    print("  sfig:density / sfig:forest: rendered by the `figures` door (door 4)")
    ok = _cert_deterministic()
    if full:
        ok = all([ok, _cert_modulus(), _cert_ladder()])
    return ok
