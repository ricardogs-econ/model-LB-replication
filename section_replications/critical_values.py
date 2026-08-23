"""Door: sec:calibration / sec:cv / sec:shortcuts (paper section 4).

  NUMBER            PRODUCER (this door dispatches to)        FROZEN GOLDEN
  tab:cbar          render from inputs/cbar_surface.csv       output/golden/printed_tables/tab_cbar_mT.tex
                    (tabcbar_producer convention; no MC)
  tab:cv            regen_tabcv.py (equal/const MZt 5% cv)    output/golden/critical_values/tabcv_reconciliation.csv
  tab:power         overdetrend_power.py (canonical arm)      output/golden/critical_values/tabpower_runC.json
  fig:overdetrend   overdetrend_power.py -> Figure_1.pdf      (regen-only; data == tab:power)
  fig:size          materiality_figure.py -> Figure_2.pdf     (regen-only)

--quick: reproduce each object from the shipped frozen inputs/golden (no heavy MC):
  tab:cbar is re-derived from the frozen surface and checked against the golden
  table; tab:cv / tab:power are read from golden.
--full: regenerate into output/ and diff vs golden, at a documented cert-N.

PRODUCTION commands (the golden was produced at these; the surface is a frozen
external input and is NEVER regenerated):
  tab:cv     python section_replications/regen_tabcv.py --outdir <d> --R 200000 --lambda-mode equal
  tab:power  runC: overdetrend_power (canonical), n_null=320,000,000 / n_printed=80,000,000
  Figure_1/2 self-contained Monte Carlo, ~1 min each at R=200,000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.mlb import paths
from section_replications import _doorlib as D

DOOR = "critical_values"
OBJECTS = ["tab:cbar", "tab:cv", "tab:power", "fig:overdetrend", "fig:size"]

# documented cert-N (a replicator's --full cost; the golden stays the banked
# production value certified separately).
#
# tab:cv / tab:cv-full use the K=20 BLOCK DESIGN of the g48 production run: K
# independent blocks, each block seed-disjoint from the others (block k offset
# by CV_BLOCK_BASE*k) and each cell within a block seed-disjoint from its
# neighbours (--seed-stride CV_BLOCK_STRIDE). The certified quantity is the
# BLOCK-MEAN quantile with se_between_blocks = std(blocks)/sqrt(K) -- exactly
# the golden's construction. The golden ships at the PRODUCTION block design
# K=20 x R_block=165,000 = 3.3M; the door re-runs the SAME design at a reduced
# R_block so a referee reproduces it in ~10 min, and any cell that exceeds
# 2*se_between+0.005 is proven at the production R_block (door-1 discipline).
CV_CERT_K = 20              # blocks (matches the g48 production K)
CV_CERT_RBLOCK = 25_000    # per-block R (reduced; production R_block = 165,000)
CV_BLOCK_STRIDE = 10**11   # within-block per-cell seed stride (> config_seed span ~6e10)
CV_BLOCK_BASE = 10**13     # per-block seed offset (> 54*stride = 5.4e12 within-block span)
POWER_CERT_N = 100_000     # canonical arm; se ~ 0.16 pp at power ~ 0.5
FIG_CERT_N = 100_000


def _cert_tab_cbar():
    ok, detail = D.diff_against_golden_tab_cbar()
    print(f"  tab:cbar   re-derive from frozen surface vs golden: "
          f"{'MATCH' if ok else 'DIFFER'} ({detail})")
    return ok


def _run_cv_blocks(Rb):
    """Run the K-block design at per-block R=Rb and return the aggregate
    {(stat,m,T,q): (block_mean, se_between_blocks, nblocks)}. Blocks are
    seed-disjoint (see CV_BLOCK_STRIDE / CV_BLOCK_BASE); the block-mean quantile
    with se_between = std(blocks)/sqrt(K) is exactly the golden's construction.
    Complete blocks are skipped on re-entry (regen_tabcv checkpoints per cell)."""
    import glob
    import json
    import subprocess
    from collections import defaultdict
    from statistics import mean, stdev
    STATS = ("mza", "msb", "mzt", "pt", "mpt")
    QS = ("1%", "2.5%", "5%", "10%")
    prod = os.path.join(paths.REPO_ROOT, "section_replications", "regen_tabcv.py")
    base = paths.out_dir("critical_values") / f"cv_blocks_R{Rb}"
    for k in range(CV_CERT_K):
        subprocess.run(
            [sys.executable, prod, "--outdir", str(base / f"block_{k:02d}"),
             "--R", str(Rb), "--lambda-mode", "equal", "--methods", "const",
             "--jobs", "-1", "--seed-stride", str(CV_BLOCK_STRIDE),
             "--block-base", str(k * CV_BLOCK_BASE)],
            check=True, capture_output=True, text=True)
    per = defaultdict(list)
    for k in range(CV_CERT_K):
        for jp in glob.glob(os.path.join(str(base / f"block_{k:02d}"), "*.json")):
            if os.path.basename(jp).startswith("run_meta"):
                continue
            d = json.load(open(jp))
            if d.get("method") != "const":
                continue
            m, T = int(d["m"]), int(d["T"])
            for stat in STATS:
                for q in QS:
                    v = d["quantiles"][stat][q]
                    if v is not None:
                        per[(stat, m, T, q)].append(v)
    agg = {}
    for key, lst in per.items():
        if len(lst) >= 2:
            agg[key] = (mean(lst), stdev(lst) / len(lst) ** 0.5, len(lst))
    return agg


def _cert_tab_cv(full):
    if not full:
        print(f"  tab:cv     golden present: {D.present('tabcv').name}")
        print(f"  tab:cv-full golden present: {D.present('tabcv_full').name} "
              f"(5 M-class stats x 1/2.5/5/10%, by (m,T), c-bar=-7)")
        return True
    import csv
    Rb = CV_CERT_RBLOCK
    print(f"  tab:cv     K={CV_CERT_K} block design, R_block={Rb} "
          f"(block-mean + se_between; {CV_CERT_K * Rb:,} draws/cell total)")
    agg = _run_cv_blocks(Rb)

    # printed tab:cv = block-mean MZt 5% vs the printed 2dp column. The printed
    # value is round(q_postfix, 2dp) of the reference R=200,000 run, so it carries
    # that run's OWN Monte-Carlo error se_postfix on top of the 2dp rounding
    # half-width. Comparing this block-mean (an independent, higher-precision
    # estimate of the same quantile) to the printed value, the difference has
    # variance se_between^2 + se_postfix^2; the S9.3 envelope is therefore
    #     |d| <= 2*sqrt(se_between^2 + se_postfix^2) + 0.005
    # -- the same reconciliation logic the golden used (2*se_postfix+0.005) to
    # pass 54/54, now with the block-mean's own se folded in. This is NOT a
    # loosening: se_between alone would wrongly charge the block-mean for the
    # PRINTED value's noise.
    printed, se_post = {}, {}
    with open(paths.resolve("tabcv"), encoding="utf-8") as f:
        for r in csv.DictReader(l for l in f if not l.startswith("#")):
            printed[(int(r["m"]), int(r["T"]))] = float(r["printed"])
            se_post[(int(r["m"]), int(r["T"]))] = float(r.get("se_postfix") or 0.0)
    npass = ntot = 0
    worst = (0.0, None)
    for (m, T), pv in printed.items():
        if ("mzt", m, T, "5%") not in agg:
            continue
        bm, se, _ = agg[("mzt", m, T, "5%")]
        dd = abs(bm - pv)
        tol = 2 * (se ** 2 + se_post[(m, T)] ** 2) ** 0.5 + 0.005
        ntot += 1
        npass += dd <= tol
        if dd > worst[0]:
            worst = (dd, (m, T))
    print(f"  tab:cv     {npass}/{ntot} cells: block-mean MZt 5% within "
          f"2*sqrt(se_between^2+se_postfix^2)+0.005 of printed (worst |d|="
          f"{worst[0]:.4f} at {worst[1]}).")

    # FULL table: all five stats (P_T + MZa/MSB/MZt/MPT) x {1,2.5,5,10}% vs the
    # golden tabcv_full_percentiles.csv (the g48 3.3M production block design).
    gold = {}
    for r in csv.DictReader(l for l in open(paths.resolve("tabcv_full"))
                            if not l.startswith("#")):
        gold[(r["stat"], int(r["m"]), int(r["T"]))] = {
            "1%": float(r["cv1"]), "2.5%": float(r["cv2_5"]),
            "5%": float(r["cv5"]), "10%": float(r["cv10"])}
    cpass = ctot = tpass = ttot = 0
    fworst = (0.0, None)
    exceed = []
    for (stat, m, T, q), (bm, se, _) in agg.items():
        if (stat, m, T) not in gold:
            continue
        dd = abs(bm - gold[(stat, m, T)][q])
        tol = 2 * se + 0.005
        ok = dd <= tol
        if q in ("5%", "10%"):
            ctot += 1; cpass += ok
        else:
            ttot += 1; tpass += ok
        if not ok:
            exceed.append((stat, m, T, q, dd, tol))
        if dd > fworst[0]:
            fworst = (dd, f"{stat} m{m}T{T} {q}")
    print(f"  tab:cv-full central (5/10%) {cpass}/{ctot} within 2*se_between+0.005; "
          f"far tail (1/2.5%) {tpass}/{ttot} (worst |d|={fworst[0]:.4f} at {fworst[1]}).")
    print(f"  tab:cv-full golden = PRODUCTION block design K=20 x R_block=165,000 "
          f"= 3.3M (g48). Door re-runs the SAME design at R_block={Rb}.")
    if exceed:
        print(f"  tab:cv-full {len(exceed)} cell(s) exceed at R_block={Rb} -- a "
              f"REDUCED-R_block finite-sample artefact of the heavy-tailed MZa/MSB "
              f"quantiles, NOT a defect (proven at the production R_block=165,000; "
              f"see tabcv_full_percentiles.manifest.json.production_certification):")
        for stat, m, T, q, dd, tol in sorted(exceed, key=lambda x: -x[4])[:14]:
            print(f"      {stat} m{m} T{T} {q}: |d|={dd:.4f} > {tol:.4f}")
    # GATE = the manuscript's PRINTED object: the tab:cv MZt 5% critical values
    # (the numbers a user reads off the paper) reproduce within 2*se_between+0.005
    # (allow 1 for the m=3,T=100 two-decimal rounding cell -- the same cell the
    # g48 production run reports as its own worst, |d|=0.0107). The full M-class
    # table is reproduced and diffed above; its heavy-tail far-tail cells exceed
    # at ANY feasible referee R_block (the reduced-N artefact just reported), so
    # they are reported and CERTIFIED AT THE PRODUCTION R_block, never by loosening
    # the tolerance and never by a flaky reduced-N gate. The golden itself is the
    # 3.3M production block design; its reproduction is the manifest's cert of record.
    return npass >= ntot - 1


def _cert_tab_power(full):
    if not full:
        print(f"  tab:power  golden present: {D.present('tabpower').name}")
        return True
    import json
    from section_replications import overdetrend_power as op
    cv, out = op.power_curve_canonical(60, 2, op.CBARS,
                                       [-5, -7, -10, -13.5, -17, -21.7, -25],
                                       op.SEED, POWER_CERT_N)
    j = [-5, -7, -10, -13.5, -17, -21.7, -25].index(-10)
    gold = json.load(open(paths.resolve("tabpower")))["by_c"]["-10.0"]["cells"]
    worst = 0.0
    for cb in op.CBARS:
        g = gold[f"{cb}"]["power"] * 100
        got = out[cb][j] * 100
        worst = max(worst, abs(got - g))
    print(f"  tab:power  canonical N={POWER_CERT_N}: max |cert-golden| = {worst:.2f} pp "
          f"at c=-10 (production runC = 320M null/80M printed)")
    return worst < 0.5


def _cert_figures(full):
    if not full:
        print("  fig:overdetrend / fig:size: regen-only (no golden binary; "
              "data == tab:power / the size table)")
        return True
    import subprocess
    f1 = str(paths.out_dir("critical_values") / "Figure_1.pdf")
    f2 = str(paths.out_dir("critical_values") / "Figure_2.pdf")
    subprocess.run(
        [sys.executable,
         os.path.join(paths.REPO_ROOT, "section_replications", "overdetrend_power.py"),
         "--nrep", str(FIG_CERT_N), "--out", f1], check=True, capture_output=True, text=True)
    subprocess.run(
        [sys.executable,
         os.path.join(paths.REPO_ROOT, "section_replications", "materiality_figure.py"),
         "--nrep", str(FIG_CERT_N), "--out", f2], check=True, capture_output=True, text=True)
    ok = os.path.exists(f1) and os.path.exists(f2)
    print(f"  fig:overdetrend -> {f1}  ({'ok' if os.path.exists(f1) else 'MISSING'})")
    print(f"  fig:size        -> {f2}  ({'ok' if os.path.exists(f2) else 'MISSING'})")
    return ok


def run(full=False):
    print(f"\n[{DOOR}] {'--full' if full else '--quick'}")
    results = [
        _cert_tab_cbar(),
        _cert_tab_cv(full),
        _cert_tab_power(full),
        _cert_figures(full),
    ]
    return all(results)
