"""The seeds of the PPP application's Monte Carlo: one disjoint integer interval per
work item.

Every simulated quantity of the application draws replication r of work item w from
np.random.default_rng(seed0(w) + r), r = 0, ..., R_w - 1 (the kernel's per-replication
recipe). seed0 is a pure function of the work item:
seed0(w) = BASE + STRIDE * (the index of w in ITEMS). ITEMS is an ordered list; an
item added at its end leaves the seeds of every item before it unchanged.

  BASE = 10**16, STRIDE = 10**9, R_w <= STRIDE: the intervals [seed0(w), seed0(w) + R_w)
    of two items are disjoint.

assert_disjoint() asserts, in code, that the intervals of the items used are pairwise
disjoint and lie above every seed of the calibration the application reads, the
critical-value table tab:cv. tabcv_seed_upper_bound() is the largest of those seeds:
in critical_values_blocks.py, block b and replication r draw the seed
SEED_BASE + b * STRIDE + r with SEED_BASE = 22,240,601, STRIDE = 10**9, b < K_BLOCKS =
20 and r < R_BLOCK = 165,000, so the largest is 19,022,405,600.
"""
import numpy as np

BASE = 10 ** 16
STRIDE = 10 ** 9
CANDIDATES = ("AUD", "CAD", "CHF", "DKK", "GBP", "JPY", "NOK", "NZD", "SEK")

# The ordered list of work items.
ITEMS = (
    # C6: the calibration of the critical values, per currency
    [f"c6_cal|{c}" for c in CANDIDATES]
    # C6: the validation batch, per currency
    + [f"c6_val|{c}" for c in CANDIDATES]
    # C4 and C6: their size under the null, per currency (gate_sizes.py)
    + [f"gate_size|{c}" for c in CANDIDATES]
    # C4 and C6: their power under a variance break, per currency (gate_sizes.py)
    + [f"gate_power|{c}" for c in CANDIDATES]
)

# ------------------------------------------------------------------------------------
# The unit-root tests, the band's power, the count, the half-lives and the sensitivity
# sets (ppp_application.py, ppp_half_lives.py). The enumerations are literals: the
# designs that regime_events.py gives from inputs/regime_events.csv.
# ------------------------------------------------------------------------------------
SETS = ("E", "S1", "S2", "S3", "S4", "S5")
# every distinct (currency, break_pos) design of E and S1-S5, first occurrence in
# (CANDIDATES, SETS) order; DKK has no design under S1
CELLS = (
    ("AUD", (11, 13)), ("AUD", (10,)), ("AUD", (4, 11, 13)), ("AUD", (13,)),
    ("AUD", (10, 12)), ("CAD", (13,)), ("CAD", (12,)), ("CHF", (13, 42)),
    ("CHF", (12,)), ("CHF", (13, 39, 42)), ("CHF", (13,)), ("CHF", (12, 42)),
    ("DKK", (13,)), ("DKK", (13, 26)), ("DKK", (12,)), ("GBP", (13, 20)),
    ("GBP", (12, 19)), ("GBP", (13, 18, 20)), ("GBP", (13,)), ("JPY", (13,)),
    ("JPY", (12,)), ("NOK", (13, 20)), ("NOK", (12, 19)), ("NOK", (13, 18, 20)),
    ("NOK", (13,)), ("NZD", (12,)), ("NZD", (3, 12)), ("NZD", (13,)),
    ("SEK", (13, 20)), ("SEK", (12, 19)), ("SEK", (13, 18, 20)),
    ("SEK", (5, 9, 13, 20)), ("SEK", (13,)),
)
# the design of each candidate under E
E_CELLS = (("AUD", (11, 13)), ("CAD", (13,)), ("CHF", (13, 42)), ("DKK", (13,)),
           ("GBP", (13, 20)), ("JPY", (13,)), ("NOK", (13, 20)), ("NZD", (12,)),
           ("SEK", (13, 20)))
# the half-lives' decomposition: the single-break designs of the two-break currencies
# under E that are not already in CELLS, in CANDIDATES order
HL_EXTRA = (("AUD", (11,)), ("CHF", (42,)), ("GBP", (20,)), ("NOK", (20,)),
            ("SEK", (20,)))
HL_CFGS = CELLS + HL_EXTRA
H_BAND = (3, 4, 5)
KAPPAS = ("0", "0.5", "1")
UNIVERSES = ("N8", "N9")
SCHEMES = ("rec", "wild")
P_RANGE = tuple(range(11))  # p = 0..10 lagged differences
BB_SLOTS = 33  # Berger-Boos points per currency: k = 0 is the center
# the synthetic panel and the known-case tests. Each name is used except unused1 and
# unused2, two positions without a consumer: a seed is a function of the position
# (seed0 below), so the positions hold the seeds of every later item, the null,
# half-life and variance items of the designs SJ and of the variance scope among them
SYNTH = ("panel", "level_sum", "lyapunov", "stationary_draws", "bb_geometry",
         "joint_k0", "joint_k1", "gridt_coverage", "hansen_mu", "gridt_sign",
         "gs_order", "draws", "jobs", "unused1", "unused2", "hl_ar", "hl_extension",
         "hl_cap")
SYNTH_UNUSED = ("unused1", "unused2")


def label(bp):
    """The label of a design: b11-13."""
    return "b" + "-".join(str(int(b)) for b in bp)


ITEMS += (
    # column (II): the sieve null at -7 per cell (and the i.i.d. null on the E cells)
    [f"null|{c}|{label(b)}" for c, b in CELLS]
    # column (III): the sieve null at cbar_half per cell
    + [f"null3|{c}|{label(b)}" for c, b in CELLS]
    # Berger-Boos: the points and the re-simulation of the maximizing point (E)
    + [f"bb|{c}|k{k}" for c in CANDIDATES for k in range(BB_SLOTS)]
    + [f"bbchk|{c}" for c in CANDIDATES]
    # the band: main, i.i.d. and stationary arms, the irf and unscaled arms, and the
    # power of (III) (E)
    + [f"alt|{c}|h{h}" for c in CANDIDATES for h in H_BAND]
    + [f"alt_irf|{c}|h{h}" for c in CANDIDATES for h in H_BAND]
    + [f"alt_unscaled|{c}|h{h}" for c in CANDIDATES for h in H_BAND]
    + [f"alt3|{c}|h{h}" for c in CANDIDATES for h in H_BAND]
    # the count: the joint law of K, under H0 and at each h, per universe and kappa
    + [f"joint|{u}|H0|k{k}" for u in UNIVERSES for k in KAPPAS]
    + [f"joint|{u}|h{h}|k{k}" for u in UNIVERSES for h in H_BAND for k in KAPPAS]
    # the half-lives: the grid-t nodes, one replication unit per node (r = k + 200)
    + [f"hl|{s}|LB|{c}|{label(b)}|p{p}" for s in SCHEMES for c, b in HL_CFGS
       for p in P_RANGE]
    + [f"hl|{s}|MP|{c}|p{p}" for s in SCHEMES for c in CANDIDATES for p in P_RANGE]
    # synthetic tests
    + [f"synth|{n}" for n in SYNTH]
)

# ------------------------------------------------------------------------------------
# The sensitivity design SJ (ppp_sj.py): E plus the systemic banking crises of the
# issuers. The designs of SJ that are not already in CELLS, in CANDIDATES order; CAD and
# NZD have the design of E and share its items. SJ_CELLS are the designs of
# inputs/sj_dates.csv. Their null items have the sieve arm only (the i.i.d. arm
# is on the E cells only).
# ------------------------------------------------------------------------------------
SJ_CELLS = (
    ("AUD", (11, 13, 17)), ("CHF", (13, 19, 36, 42)), ("DKK", (13, 15, 36)),
    ("GBP", (2, 13, 19, 35)), ("JPY", (13, 25)), ("NOK", (13, 16, 20)),
    ("SEK", (13, 19, 36)),
)
assert not set(SJ_CELLS) & set(CELLS), "an SJ design is already in CELLS"

ITEMS += (
    [f"null|{c}|{label(b)}" for c, b in SJ_CELLS]
    + [f"null3|{c}|{label(b)}" for c, b in SJ_CELLS]
    + [f"hl|{s}|LB|{c}|{label(b)}|p{p}" for s in SCHEMES for c, b in SJ_CELLS
       for p in P_RANGE]
)
# ------------------------------------------------------------------------------------
# The scope of the variance (ppp_variance_scope.py): the cells of E and of SJ that are
# simulated under a variance that changes at the regime dates, one null item and one
# wild bootstrap item each. CAD and NZD have the same cell under both designs. The null
# items carry two arms on the same draws (the registered null and the same draws scaled
# by the variance profile); the wild items have one.
# ------------------------------------------------------------------------------------
VAR_CELLS = E_CELLS + SJ_CELLS

ITEMS += (
    [f"vnull|{c}|{label(b)}" for c, b in VAR_CELLS]
    + [f"vwild|{c}|{label(b)}" for c, b in VAR_CELLS]
)
# the synthetic tests of the variance scope (each name is used)
SYNTH_VAR = ("vscope",)
ITEMS += [f"synth|{n}" for n in SYNTH_VAR]
_INDEX = {it: i for i, it in enumerate(ITEMS)}
assert len(_INDEX) == len(ITEMS), "ITEMS holds a duplicate"

# The replication count of each family (R_w <= STRIDE).
R_FAMILY = {"c6_cal": 100_000, "c6_val": 100_000, "gate_size": 100_000,
            "gate_power": 20_000, "null": 200_000, "null3": 200_000, "bb": 50_000,
            "bbchk": 50_000, "alt": 200_000, "alt_irf": 200_000,
            "alt_unscaled": 200_000, "alt3": 200_000, "joint": 100_000, "hl": 501,
            "synth": 10 ** 6,
            "vnull": 200_000, "vwild": 19_999}

# Common random numbers: the only items that more than one arm may read, and the arms
# allowed on them. Pair (i), sieve against i.i.d.: the null items of the E cells
# ("sieve", "iid") and the alt items ("main", "iid"). Pair (ii), u0 = 0 against the
# stationary law: the alt items ("main", "stationary") and the joint items at h
# ("u0zero", "stationary"). An item read by several arms must include the pair's anchor
# (sieve, main, u0zero, homo). The i.i.d. and stationary arms of an alt item share its
# draws only because both pair with the main arm; nothing compares them. Pair (iii),
# the registered null against the same draws scaled by a variance profile: the vnull
# items ("homo", "profile").
CRN_WHITELIST = {"null": ("sieve", "iid"), "alt": ("main", "iid", "stationary"),
                 "joint": ("u0zero", "stationary"), "vnull": ("homo", "profile")}
CRN_ANCHOR = {"null": "sieve", "alt": "main", "joint": "u0zero", "vnull": "homo"}
_E_NULL = {f"null|{c}|{label(b)}" for c, b in E_CELLS}

# tab:cv's stream (mlb/critical_values_blocks.py: SEED_BASE, STRIDE,
# K_BLOCKS, R_BLOCK): block b, replication r draws SEED_BASE + b * STRIDE + r
TABCV_SEED_BASE = 22_240_601
TABCV_STRIDE = 10 ** 9
TABCV_K_BLOCKS = 20
TABCV_R_BLOCK = 165_000


def seed0(item):
    """The first seed of a work item: a pure function of its position in ITEMS."""
    return BASE + STRIDE * _INDEX[item]


def rng(item, r):
    """The generator of replication r of a work item."""
    return np.random.default_rng(seed0(item) + r)


def tabcv_seed_upper_bound():
    """The largest seed tab:cv draws (see the module's docstring)."""
    return (TABCV_SEED_BASE + (TABCV_K_BLOCKS - 1) * TABCV_STRIDE + TABCV_R_BLOCK
            - 1)


def assert_disjoint(used):
    """used: {item: R}. Asserts the items are in ITEMS, their intervals pairwise
    disjoint and above tab:cv's; returns the tab:cv bound."""
    assert all(it in ITEMS for it in used), [it for it in used if it not in ITEMS]
    assert all(0 < R <= STRIDE for R in used.values()), used
    iv = sorted((seed0(it), seed0(it) + R, it) for it, R in used.items())
    for (a0, a1, ia), (b0, b1, ib) in zip(iv[:-1], iv[1:]):
        assert a1 <= b0, (ia, ib)
    bound = tabcv_seed_upper_bound()
    assert iv[0][0] > bound, (iv[0], bound)
    assert iv[-1][1] < 2 ** 63 - 1
    return bound


def family(item):
    return item.split("|", 1)[0]


def item_R(item):
    return R_FAMILY[family(item)]


def assert_consumers(ledger):
    """ledger: (item, stage, arm) triples, one per consumer. Asserts that every item is
    in ITEMS, consumed by one stage only, never twice by the same arm, and by more than
    one arm only on the whitelisted families and arms, with the family's anchor among
    them (the i.i.d. arm of a null item only on the E cells; the joint items under H0
    have one arm). Returns {item: sorted arms}."""
    stages, arms = {}, {}
    for item, stage, arm in ledger:
        assert item in _INDEX, item
        stages.setdefault(item, set()).add(stage)
        assert arm not in arms.setdefault(item, []), (item, arm)
        arms[item].append(arm)
    for item in stages:
        assert len(stages[item]) == 1, (item, sorted(stages[item]))
        a = arms[item]
        allowed = CRN_WHITELIST.get(family(item), ())
        if "|H0|" in item:
            allowed = ()
        if len(a) > 1:
            assert set(a) <= set(allowed), (item, a)
            assert CRN_ANCHOR[family(item)] in a, (item, a)
            if family(item) == "null":
                assert item in _E_NULL, (item, a)
    return {k: sorted(v) for k, v in arms.items()}


def table():
    """The registry as rows: (index, item, seed0, R, arms allowed on it)."""
    rows = []
    for i, it in enumerate(ITEMS):
        allowed = "" if "|H0|" in it else "+".join(CRN_WHITELIST.get(family(it), ()))
        if family(it) == "null" and it not in _E_NULL:
            allowed = "sieve"  # the i.i.d. arm only on the E cells
        rows.append((i, it, seed0(it), item_R(it), allowed))
    return rows


def main(argv=None):
    """[--out FILE]: print the application's seed list (to FILE)."""
    import argparse
    import csv
    import sys

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    fh = open(a.out, "w", encoding="utf-8", newline="") if a.out else sys.stdout
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["index", "item", "seed0", "R", "crn_arms"])
    w.writerows(table())
    if a.out:
        fh.close()
    return 0


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.exit(main())
