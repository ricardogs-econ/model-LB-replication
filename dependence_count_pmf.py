"""Count distribution of rejections under H1 with cross-sectional dependence.

Extension of dependence_bound.py for replication package v1.1.5. Computes the
FULL probability mass function of the number of rejections among the eight
currencies under the alternative, in the one-factor equicorrelated Gaussian
approximation used in Section 6.3, with the true heterogeneous per-currency
feasible power (five currencies at 0.30, p=1; three at 0.31, p=2).

Backs the manuscript claims (v63):
  - P(zero rejections | H1) ~ 0.21--0.23 at rho in {0.37, 0.41};
  - zero is the MODAL count under the documented dependence
    (P(0) = 0.215--0.234 > P(1) = 0.192--0.198 > ..., monotone decreasing);
  - under independence the modal count is two.

Deterministic: pure Gauss quadrature over the common factor plus exact
enumeration of the 2^8 rejection patterns. No seeds, no Monte Carlo.

Model. Currency i rejects iff its latent Gaussian Z_i exceeds z_i, where
z_i = Phi^{-1}(1 - power_i) and Z_i = sqrt(rho) F + sqrt(1-rho) e_i with
F, e_i iid N(0,1). Conditional on F the rejections are independent, so the
count pmf is a sum over subsets, integrated over F.
"""

import numpy as np
from itertools import combinations
from scipy.stats import norm
from scipy.integrate import quad

POWERS = [0.30] * 5 + [0.31] * 3   # AUD, CHF, GBP, JPY, NZD (p=1); CAD, NOK, SEK (p=2)
RHOS = (0.37, 0.41)                # mean pairwise correlations, Section 6.3


def count_pmf(powers, rho):
    """Exact pmf of the rejection count under the one-factor model."""
    z = np.array([norm.ppf(1.0 - p) for p in powers])
    n = len(powers)

    def integrand_for(subset):
        s = set(subset)

        def f(x):
            cond = norm.cdf((z - np.sqrt(rho) * x) / np.sqrt(1.0 - rho))
            probs = np.where([i in s for i in range(n)], 1.0 - cond, cond)
            return norm.pdf(x) * probs.prod()

        return f

    pmf = np.zeros(n + 1)
    for k in range(n + 1):
        for subset in combinations(range(n), k):
            pmf[k] += quad(integrand_for(subset), -8.0, 8.0, limit=200)[0]
    return pmf


def independence_pmf(powers):
    """Poisson-binomial pmf under independence (rho = 0), exact recursion."""
    pmf = np.array([1.0])
    for p in powers:
        pmf = np.convolve(pmf, [1.0 - p, p])
    return pmf


def _selftest():
    # 1. pmf sums to one at every rho
    for rho in RHOS:
        s = count_pmf(POWERS, rho).sum()
        assert abs(s - 1.0) < 1e-8, s
    # 2. rho -> 0 recovers the Poisson-binomial (independence)
    tiny = count_pmf(POWERS, 1e-10)
    indep = independence_pmf(POWERS)
    assert np.max(np.abs(tiny - indep)) < 1e-6
    # 3. P(0) under independence equals the closed-form product
    assert abs(indep[0] - np.prod([1 - p for p in POWERS])) < 1e-12
    print("selftest OK")


if __name__ == "__main__":
    _selftest()
    indep = independence_pmf(POWERS)
    print(f"\nindependence: P(0)={indep[0]:.4f}  E[rej]={sum(POWERS):.2f}  "
          f"mode k={int(np.argmax(indep))}")
    for rho in RHOS:
        pmf = count_pmf(POWERS, rho)
        head = "  ".join(f"P({k})={v:.3f}" for k, v in enumerate(pmf[:4]))
        print(f"rho={rho}: {head} ...  mode k={int(np.argmax(pmf))}")
