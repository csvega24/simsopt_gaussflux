"""Independent numpy oracles for the field responses (write-up eq. 23 and 24).

Parameter t in [0, 1), quadrature (1/n_q) sum_q, prefactor mu0/4pi = 1e-7, as
SIMSOPT. Gamma' and Xi' are d/dt; the formulas are parametrization invariant.
"""

import numpy as np


def _r(X, gamma):
    r = X[:, None, :] - gamma[None, :, :]
    rn = np.linalg.norm(r, axis=-1)
    return r, rn


def b1_eq23_first(gamma, gammadash, current, X, xi, dxi):
    """First form: (Xi' x r - Gamma' x Xi)/|r|^3 + 3 (r.Xi) Gamma' x r / |r|^5."""
    r, rn = _r(X, gamma)
    term1 = (np.cross(dxi[None], r) - np.cross(gammadash, xi)[None]) / rn[..., None] ** 3
    term2 = 3 * np.einsum("ptk,tk->pt", r, xi)[..., None] * np.cross(gammadash[None], r) / rn[..., None] ** 5
    return 1e-7 * current / len(gamma) * np.sum(term1 + term2, axis=1)


def b1_eq23_second(gamma, gammadash, current, X, xi):
    """Integration-by-parts form (valid only when Xi' = dXi/dt):
    2 Xi x Gamma'/|r|^3 - 3 (r.Gamma') Xi x r/|r|^5 + 3 (r.Xi) Gamma' x r/|r|^5."""
    r, rn = _r(X, gamma)
    t1 = 2 * np.cross(xi, gammadash)[None] / rn[..., None] ** 3
    t2 = -3 * np.einsum("ptk,tk->pt", r, gammadash)[..., None] * np.cross(xi[None], r) / rn[..., None] ** 5
    t3 = 3 * np.einsum("ptk,tk->pt", r, xi)[..., None] * np.cross(gammadash[None], r) / rn[..., None] ** 5
    return 1e-7 * current / len(gamma) * np.sum(t1 + t2 + t3, axis=1)


def b2_eq24(gamma, gammadash, current, X, xi, dxi):
    """[(G' x r) P2 + (Xi' x r - G' x Xi) P1 - Xi' x Xi] / |r|^3 (Taylor coefficient)."""
    r, rn = _r(X, gamma)
    rx = np.einsum("ptk,tk->pt", r, xi)
    x2 = np.sum(xi * xi, axis=1)[None]
    P1 = 3 * rx / rn ** 2
    P2 = 7.5 * rx ** 2 / rn ** 4 - 1.5 * x2 / rn ** 2
    num = (np.cross(gammadash[None], r) * P2[..., None]
           + (np.cross(dxi[None], r) - np.cross(gammadash, xi)[None]) * P1[..., None]
           - np.cross(dxi, xi)[None])
    return 1e-7 * current / len(gamma) * np.sum(num / rn[..., None] ** 3, axis=1)


def gauss_hermite_nodes(order):
    """Nodes and weights for E[g(Z)], Z ~ N(0, 1) (probabilists' Hermite)."""
    x, w = np.polynomial.hermite_e.hermegauss(order)
    return x, w / np.sqrt(2 * np.pi)
