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
      - per-test power  p in {0.30, 0.325, 0.34}  (the feasible power at the
        tangency, Table 6 of the paper), and
      - latent correlation  rho in {0.37, 0.41}   (the signed and absolute
        mean pairwise correlations of the Model LB residuals, Section 6.3).

    P(zero) = E_F[ Phi( (z - sqrt(rho) F) / sqrt(1-rho) )^8 ],  F ~ N(0,1),
    z = Phi^{-1}(1 - p), evaluated by Gauss-Hermite-free adaptive quadrature.

    This is an APPROXIMATION, not an estimate: the equicorrelated one-factor
    structure is an assumption standing in for the unknown joint law of the
    eight statistics. It is reported in the paper as such.

OUTPUT
    dependence_bound.csv (p, rho, prob_zero) + a console table. Deterministic
    (pure quadrature, no simulation, no seeds).

REQUIRES
    numpy, scipy.
================================================================================
"""
import csv
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad

POWERS = (0.30, 0.325, 0.34)   # feasible power at the tangency (Table 6)
RHOS = (0.37, 0.41)            # signed / absolute mean pairwise correlation
N = 8                          # admissible currencies


def prob_zero(p: float, rho: float, n: int = N) -> float:
    """P(zero rejections) under the one-factor equicorrelated Gaussian."""
    z = norm.ppf(1.0 - p)
    f = lambda x: norm.pdf(x) * norm.cdf((z - np.sqrt(rho) * x)
                                         / np.sqrt(1.0 - rho)) ** n
    val, _ = quad(f, -10.0, 10.0)
    return float(val)


def main() -> None:
    rows = []
    print(f"{'power':>7} {'rho':>6} {'P(zero|H1)':>12}   independence")
    for p in POWERS:
        indep = (1.0 - p) ** N
        for rho in RHOS:
            pz = prob_zero(p, rho)
            rows.append(dict(power=p, rho=rho, prob_zero=round(pz, 4),
                             independence=round(indep, 4)))
            print(f"{p:7.3f} {rho:6.2f} {pz:12.3f}   {indep:.3f}")
    with open("dependence_bound.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("-> dependence_bound.csv")


if __name__ == "__main__":
    main()
