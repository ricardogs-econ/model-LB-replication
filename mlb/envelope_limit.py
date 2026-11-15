"""The exact limiting power envelope of the point-optimal unit-root test.

J_c is the Ornstein-Uhlenbeck process dJ = c J dt + dW, J(0) = 0, on [0, 1]. The
limit of the test statistic is

    L = cbar^2 int_0^1 J_c(t)^2 dt - cbar J_c(1)^2,

under the null (c = 0) and on the diagonal c = cbar (the alternative the test is
built for). The joint moment generating function

    E exp{th1 int J^2 + th2 J(1)^2}
        = e^{-c/2} [cosh g - (c + 2 th2) sinh g / g]^{-1/2},  g = (c^2 - 2 th1)^{1/2},

at th1 = i s cbar^2, th2 = -i s cbar is the characteristic function phi(s) of L.
The distribution function is recovered by the Gil-Pelaez inversion

    F(x) = P(L <= x) = 1/2 - (1/pi) int_0^inf Im{e^{-isx} phi(s)} / s ds.

The root. cosh g and sinh g / g are even in g, so D(s) = cosh g - (c + 2 th2)
sinh g / g does not depend on the root taken for g, but D^{-1/2} must follow s
continuously from D(0) = e^{-c}. With g the principal root, g^2 = c^2 - 2 i s
cbar^2 has a non-negative real part and a negative imaginary part for s > 0, so
g(s) is continuous with Re g >= 0. Writing D = e^g B with

    B = (1 + e^{-2g}) / 2 - (c + 2 th2) (1 - e^{-2g}) / (2 g),

gives log D = g + log B, where arg B is unwrapped along increasing s (B = 1 at
s = 0). (1 - e^{-2g}) / (2g) is computed with expm1, which stays accurate as
g -> 0 (c = 0, s -> 0).

The quadrature. The integrand is analytic at s = 0 (it is E sin(s(L - x)) / s),
and |phi(s)| decays like exp(-|cbar| sqrt(s) / 2), since Re g ~ |cbar| sqrt(s).
The upper limit S is set where that bound reaches 1e-18, with a factor 2; the
rule is composite Gauss-Legendre, 16 nodes per panel, with the panel width at
most 0.5 / max(1, |x|, cbar^2 / 8): at most half a radian of the oscillation
e^{-isx} per panel, and the scale of phi near s = 0 resolved. The unwrapped
arg B must move by less than pi/4 between neighboring nodes (checked).
scale > 1 widens S and narrows the panels by that factor (a stability check).

    b_inf(cbar)    the 5% quantile of L under the null (the test rejects for
                   small L);
    pi_inf(cbar)   P(L <= b_inf(cbar)) on the diagonal: the power envelope;
    cbar_inf()     the root of pi_inf(cbar) = 1/2;
    slope(cbar)    pi_inf'(cbar) by a central difference (h = 1e-3).

The finite-sample statistic L_T of mlb/envelope_exact.py satisfies
L_T => L + cbar, so its 5% null quantile b_T and b_inf are related by
b_T - cbar -> b_inf(cbar).
"""
import numpy as np
from scipy.optimize import brentq

GL_NODES = 16
_GL_X, _GL_W = np.polynomial.legendre.leggauss(GL_NODES)
SIZE = 0.05


def _nodes(x, cbar, scale=1.0):
    """Composite Gauss-Legendre nodes and weights on [0, S], increasing in s."""
    width = 0.5 / max(1.0, abs(x), cbar * cbar / 8.0) / scale
    upper = scale * 2.0 * (2.0 * 18.0 * np.log(10.0) / abs(cbar)) ** 2
    n_panels = int(np.ceil(upper / width))
    left = np.arange(n_panels) * width
    s = (left[:, None] + (_GL_X[None, :] + 1.0) * width / 2.0).ravel()
    w = np.tile(_GL_W * width / 2.0, n_panels)
    return s, w


def phi(s, cbar, c):
    """The characteristic function of L at increasing points s > 0.

    Returns (phi, the largest step of the unwrapped arg B between points).
    """
    s = np.asarray(s, dtype=float)
    th1 = 1j * s * cbar * cbar
    th2 = -1j * s * cbar
    g = np.sqrt(c * c - 2.0 * th1 + 0j)
    tail = -np.expm1(-2.0 * g) / (2.0 * g)
    b = (1.0 + np.exp(-2.0 * g)) / 2.0 - (c + 2.0 * th2) * tail
    arg = np.unwrap(np.angle(b))
    step = float(np.max(np.abs(np.diff(arg)))) if s.size > 1 else 0.0
    log_b = np.log(np.abs(b)) + 1j * arg
    return np.exp(-c / 2.0 - g / 2.0 - log_b / 2.0), step


def cdf(x, cbar, c, scale=1.0):
    """P(L <= x) under the local-to-unity parameter c, by Gil-Pelaez."""
    s, w = _nodes(x, cbar, scale)
    f, step = phi(s, cbar, c)
    if step > np.pi / 4:
        raise RuntimeError(f"arg B moves by {step:.3f} between nodes: refine")
    integrand = np.imag(np.exp(-1j * s * x) * f) / s
    return 0.5 - float(np.sum(w * integrand)) / np.pi


def mean(cbar, c):
    """E L under c: E int J^2 and E J(1)^2 of the OU process in closed form."""
    if c == 0:
        e_int, e_end = 0.5, 1.0
    else:
        e_end = (np.exp(2.0 * c) - 1.0) / (2.0 * c)
        e_int = (np.exp(2.0 * c) - 1.0) / (4.0 * c * c) - 1.0 / (2.0 * c)
    return cbar * cbar * e_int - cbar * e_end


def b_inf(cbar, size=SIZE, scale=1.0):
    """The size quantile of L under the null (c = 0)."""
    return brentq(lambda b: cdf(b, cbar, 0.0, scale) - size, 1e-9,
                  mean(cbar, 0.0), xtol=1e-13, rtol=1e-15, maxiter=200)


def pi_inf(cbar, size=SIZE, scale=1.0):
    """The limiting power envelope at cbar: P(L <= b_inf(cbar)) at c = cbar."""
    return cdf(b_inf(cbar, size, scale), cbar, cbar, scale)


def cbar_inf(scale=1.0, lo=-8.0, hi=-6.0):
    """The cbar at which the limiting power envelope is one half."""
    return brentq(lambda cb: pi_inf(cb, scale=scale) - 0.5, lo, hi,
                  xtol=1e-12, rtol=1e-15, maxiter=200)


def slope(cbar, h=1e-3, scale=1.0):
    """pi_inf'(cbar) by a central difference with step h."""
    up, down = pi_inf(cbar + h, scale=scale), pi_inf(cbar - h, scale=scale)
    return (up - down) / (2.0 * h)
