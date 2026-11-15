"""The break dates tau_j = floor(lambda_j T), computed exactly.

The paper dates the j-th break at tau_j = floor(lambda_j T), clamped to [1, T - 1].
lambda_j T is evaluated in binary floating point, so a product that equals an integer
can fall just below it: 0.57 * 100 evaluates to 56.99999999999999, and a plain floor
dates the break one period early. The product is therefore rounded to three decimals
before the floor.

The rounding is exact for every fraction the package uses, because lambda_j T is then
either a multiple of 0.001 or farther than 0.0005 from an integer:

  * a three-decimal fraction (the lattices of the calibration grid): lambda_j T is a
    multiple of 0.001;
  * lambda_j = j / (m + 1), m <= 5: lambda_j T is a multiple of 1/(m + 1), so it is an
    integer or at least 1/6 from one;
  * lambda_j = (b + 1/2) / T, which dates a break at the integer b:
    lambda_j T = b + 1/2.

This is the one function that kernel_numba.break_pos_from_lambdas and
envelope_exact.break_positions call.
"""

import math

DECIMALS = 3


def break_dates(T, lambdas):
    """The list [clamp(floor(lambda_j T), 1, T - 1)] of integers, for a sample size T
    and break fractions lambdas (empty for no break)."""
    return [
        int(max(1, min(T - 1, math.floor(round(float(lam) * T, DECIMALS)))))
        for lam in lambdas
    ]
