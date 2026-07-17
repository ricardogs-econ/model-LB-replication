# Lambda-exact tangency at the empirical two-break configuration

Runs: `boot_ppp_cbar.py --grid --T 52 --bp 12 19 --p {1,2} --nrep 10000`
(pac1=0.27 family; kernel bp check [12, 19]; lambdas (0.2404, 0.375),
empirical spacing 0.135 < 0.233 minimum of the averaging grid).

| p | cbar* (exact) | cbar* (surface) | cv5_MZt (exact) | powF (exact) | powF (surface) |
|---|---|---|---|---|---|
| 1 | -10.839 | -10.580 | -2.112 | 0.450 | 0.482 |
| 2 | -12.000 | -11.105 | -2.232 | 0.410 | 0.449 |

Backs the manuscript's lambda-exact caution (sec:pppcalib, v73): the
displacement is -0.26/-0.90 in cbar*, the feasible-power cost 0.03-0.04;
size is unaffected because every empirical critical value is simulated at
the exact configuration.
