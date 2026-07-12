# -*- coding: utf-8 -*-
"""
================================================================================
dependence_bound.py -- Dependence-adjusted probability of observing ZERO
    rejections across the eight admissible currencies when the level-break
    alternative is true for every one of them (Section 6.3 of the paper).
================================================================================

WHAT IT COMPUTES
    The paper reports the independence benchmark P(zero rejections | H1) =
    prod_i (1 - power_i) and notes that the positive dollar-numeraire
    dependence raises it. This script quantifies "raises it" under the
    simplest structure consistent with the estimated dependence: a one-factor
    equicorrelated Gaussian copula for the eight test statistics, with
      - per-currency power at the applied AR order (Table pppsurface): 0.30
        for the five currencies at p=1 (AUD, CHF, GBP, JPY, NZD), 0.31 for
        the three at p=2 (CAD, NOK, SEK) -- no currency requires p>2, so
        these are the only two feasible powers that occur in the applied
        sample; and
      - latent correlation  rho in {0.37, 0.41}   (the signed and absolute
        mean pairwise correlations of the Model LB residuals, Section 6.3).

    P(zero) = E_F[ prod_i Phi( (z_i - sqrt(rho) F) / sqrt(1-rho) ) ],
    F ~ N(0,1), z_i = Phi^{-1}(1 - power_i) PER CURRENCY (heterogeneous,
    not a single power raised to the 8th power), evaluated by adaptive
    quadrature.

    This is an APPROXIMATION, not an estimate: the equicorrelated one-factor
    structure is an assumption standing in for the unknown joint law of the
    eight statistics. It is reported in the paper as such.

OUTPUT
    dependence_bound.csv (rho, prob_zero, independence) + a console table.
    Deterministic (pure quadrature, no simulation, no seeds).

REQUIRES
    numpy, scipy.
================================================================================
"""
import csv
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

# per-currency feasible power at the applied AR order (Table pppsurface):
# p=1 -> 0.30 (AUD, CHF, GBP, JPY, NZD); p=2 -> 0.31 (CAD, NOK, SEK)
CURRENCY_POWER = {
    "AUD": 0.30, "CHF": 0.30, "GBP": 0.30, "JPY": 0.30, "NZD": 0.30,
    "CAD": 0.31, "NOK": 0.31, "SEK": 0.31,
}
RHOS = (0.37, 0.41)            # signed / absolute mean pairwise correlation


def prob_zero(powers: dict, rho: float) -> float:
    """P(zero rejections) under the one-factor equicorrelated Gaussian,
    with each currency's OWN power (heterogeneous, not a single value^N)."""
    zs = [norm.ppf(1.0 - p) for p in powers.values()]
    def f(x):
        prod = 1.0
        for z in zs:
            prod *= norm.cdf((z - np.sqrt(rho) * x) / np.sqrt(1.0 - rho))
        return norm.pdf(x) * prod
    val, _ = quad(f, -10.0, 10.0)
    return float(val)


def main() -> None:
    rows = []
    indep = 1.0
    for p in CURRENCY_POWER.values():
        indep *= (1.0 - p)
    print(f"per-currency power: {CURRENCY_POWER}")
    print(f"independence P(zero|H1) = prod(1-power_i) = {indep:.4f}")
    print(f"{'rho':>6} {'P(zero|H1)':>12}   independence")
    for rho in RHOS:
        pz = prob_zero(CURRENCY_POWER, rho)
        rows.append(dict(rho=rho, prob_zero=round(pz, 4),
                         independence=round(indep, 4)))
        print(f"{rho:6.2f} {pz:12.4f}   {indep:.4f}")
    with open("dependence_bound.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("-> dependence_bound.csv")


if __name__ == "__main__":
    main()
