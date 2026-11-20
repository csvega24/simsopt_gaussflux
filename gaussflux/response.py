"""JAX Biot-Savart and the field responses B0, B1^a, B2^aa (PLAN.md §3.1).

Quadrature as SIMSOPT's (biot_savart_impl.h:44): B(x) = 1e-7 I / n_q *
sum_q gamma'_q x (x - gamma_q) / |x - gamma_q|^3, t in [0, 1). All in float64.
B1^a is one forward-mode JVP per mode on the coils the mode lives on; B2^aa is
the Taylor coefficient 1/2 d^2/de^2 B(gamma + e psi, gamma' + e psi') (eq. 24,
eq. 26 convention) by JVP-of-JVP, used only on the general path.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

MU0_OVER_4PI = 1e-7


def coil_field(gamma, gammadash, current, X):
    """Field of one coil at points X (P, 3)."""
    r = X[:, None, :] - gamma[None, :, :]
    rn3 = jnp.sum(r * r, axis=-1) ** 1.5
    cr = jnp.cross(gammadash[None, :, :], r)
    return MU0_OVER_4PI * current / gamma.shape[0] * jnp.sum(cr / rn3[..., None], axis=1)


_coil_field_jit = jax.jit(coil_field)


def total_field(gammas, gammadashs, currents, X):
    """Sum of coil fields; gammas (C, nq, 3)."""
    return jnp.sum(jax.vmap(coil_field, in_axes=(0, 0, 0, None))(gammas, gammadashs, currents, X), axis=0)


_total_field_jit = jax.jit(total_field)


def coil_arrays(coils):
    g = np.array([c.curve.gamma() for c in coils])
    gd = np.array([c.curve.gammadash() for c in coils])
    cur = np.array([c.current.get_value() for c in coils])
    return g, gd, cur


@jax.jit
def _b1_batch(gamma, gammadash, current, X, dg, dgd):
    f = lambda g, gd: coil_field(g, gd, current, X)
    _, lin = jax.linearize(f, gamma, gammadash)
    return jax.vmap(lin)(dg, dgd)


@jax.jit
def _b2_batch(gamma, gammadash, current, X, dg, dgd):
    def second(t, td):
        h = lambda e: coil_field(gamma + e * t, gammadash + e * td, current, X)
        return 0.5 * jax.jacfwd(jax.jacfwd(h))(0.0)
    return jax.vmap(second)(dg, dgd)


def responses(gammas, gammadashs, currents, X, modes, second=False, batch=32):
    """B0 (P,3), B1 (A,P,3) and, if second, B2 Taylor coefficients (A,P,3).

    Work is restricted to the coils each mode lives on: for coil j only the
    modes acting on j are differentiated (systematic modes sum their images).
    Returns also a work record (JVP count, batch count, batch size).
    """
    X = jnp.asarray(X)
    # per-coil sum: vmapping all coils at once materializes (C, P, nq, 3) intermediates
    B0 = sum(np.asarray(_coil_field_jit(jnp.asarray(gammas[j]), jnp.asarray(gammadashs[j]), float(currents[j]), X))
             for j in range(len(gammas)))
    A, P = modes.size, X.shape[0]
    B1 = np.zeros((A, P, 3))
    B2 = np.zeros((A, P, 3)) if second else None
    jvps, batches = 0, 0
    for j in range(len(gammas)):
        idx, dg, dgd = modes.tangents(j)
        if len(idx) == 0:
            continue
        g, gd, cur = jnp.asarray(gammas[j]), jnp.asarray(gammadashs[j]), float(currents[j])
        for s in range(0, len(idx), batch):
            sl = slice(s, s + batch)
            pad = batch - len(idx[sl])
            tdg = np.concatenate([dg[sl], np.zeros((pad,) + dg.shape[1:])]) if pad else dg[sl]
            tdgd = np.concatenate([dgd[sl], np.zeros((pad,) + dgd.shape[1:])]) if pad else dgd[sl]
            out = np.asarray(_b1_batch(g, gd, cur, X, jnp.asarray(tdg), jnp.asarray(tdgd)))
            np.add.at(B1, idx[sl], out[: len(idx[sl])])
            if second:
                out2 = np.asarray(_b2_batch(g, gd, cur, X, jnp.asarray(tdg), jnp.asarray(tdgd)))
                np.add.at(B2, idx[sl], out2[: len(idx[sl])])
            jvps += len(idx[sl])
            batches += 1
    return B0, B1, B2, {"single_coil_jvps": jvps, "batches": batches, "batch_size": batch,
                        "second_order": bool(second)}


# ----------------------------------------------------------------------
# Objective definitions (integral_BdotN.cpp:62-107) in JAX
# ----------------------------------------------------------------------

def surface_geometry(surface):
    n = surface.normal().reshape(-1, 3)
    absn = np.linalg.norm(n, axis=1)
    return surface.gamma().reshape(-1, 3), n / absn[:, None], absn


def objective(B, nhat, absn, target, definition):
    u = jnp.sum(B * nhat, axis=1) - target
    if definition == "quadratic flux":
        return 0.5 * jnp.sum(u * u * absn) / absn.shape[0]
    w = jnp.sum(B * B, axis=1)
    if definition == "normalized":
        return 0.5 * jnp.sum(u * u * absn) / jnp.sum(w * absn)
    if definition == "local":
        return 0.5 * jnp.sum(u * u / w * absn) / absn.shape[0]
    raise ValueError(definition)


def displaced_arrays(gammas, gammadashs, modes, xi):
    """Coil arrays displaced by sum_a xi_a psi_a (all coils), xi (A,)."""
    G = jnp.asarray(gammas)
    GD = jnp.asarray(gammadashs)
    P = np.zeros((modes.size, len(gammas), modes.nq, 3))
    DP = np.zeros_like(P)
    for a in range(modes.size):
        for j in range(len(gammas)):
            v = modes.on_coil(a, j)
            if v is not None:
                P[a, j], DP[a, j] = v
    dG = jnp.einsum("a,ajtd->jtd", xi, jnp.asarray(P))
    dGD = jnp.einsum("a,ajtd->jtd", xi, jnp.asarray(DP))
    return G + dG, GD + dGD


def scalar_objective(gammas, gammadashs, currents, X, nhat, absn, target, definition, modes):
    """f(xi) on the modally deformed coils (the validated unexpanded objective)."""
    cur = jnp.asarray(currents)
    def f(xi):
        G, GD = displaced_arrays(gammas, gammadashs, modes, xi)
        return objective(total_field(G, GD, cur, X), nhat, absn, target, definition)
    return f


# ----------------------------------------------------------------------
# Gradient pull-back (memory-bounded): cotangents of B0, B1, B2 -> (gamma, gamma', I) of one coil
# ----------------------------------------------------------------------

@jax.jit
def coil_vjp_b0(gamma, gammadash, current, X, cot):
    _, vjp = jax.vjp(lambda g, gd, i: coil_field(g, gd, i, X), gamma, gammadash, current)
    return vjp(cot)


@partial(jax.jit, static_argnames=("second",))
def coil_vjp_modes(gamma, gammadash, current, X, dg, dgd, c1, c2, second):
    """Gradient of sum_a <c1_a, B1_a> (+ <c2_a, B2_a>) over a padded batch of modes on one coil."""
    def h(g, gd, i):
        f = lambda gg, ggd: coil_field(gg, ggd, i, X)
        def one(t, td, ca, cb):
            val = jnp.sum(ca * jax.jvp(f, (g, gd), (t, td))[1])
            if second:
                hh = lambda e: coil_field(g + e * t, gd + e * td, i, X)
                val = val + jnp.sum(cb * 0.5 * jax.jacfwd(jax.jacfwd(hh))(0.0))
            return val
        return jnp.sum(jax.vmap(one)(dg, dgd, c1, c2))
    return jax.grad(h, argnums=(0, 1, 2))(gamma, gammadash, current)


def grad_batch_size(P, nq, budget=None):
    """Modes per reverse-mode batch so that batch * P * nq stays below budget (memory bound).

    Default budget (Phase 3, carried item 5): 2.5e6 x (1024 / P), the measured safe value at
    64x16 (records/phase1b_report_review_fable.md §4.1) scaled down with the surface size,
    since the reference's reverse-mode memory grows faster than batch * P * nq."""
    if budget is None:
        budget = 2.5e6 * 1024.0 / P
    return int(max(1, min(64, budget // (P * nq))))
