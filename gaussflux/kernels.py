"""Phase 1b kernel backend for Delta2 (records/phase1b_proposal_opus.md §2, §6).

For coil j with arrays (Gamma, Gamma', I) and points x_p, r_pq = x_p - Gamma_q,
the first-order field of a displacement (Xi, Xi') is (write-up eq. 23, first form)

    B1(x_p) = c_j sum_q [ K^Xi_pq Xi_q + K^Xi'_pq Xi'_q ],   c_j = 1e-7 I_j / n_q,
    K^Xi_pq  = ( -[Gamma'_q]_x + 3 (Gamma'_q x r_pq) r_pq^T / |r_pq|^2 ) / |r_pq|^3,
    K^Xi'_pq = -[r_pq]_x / |r_pq|^3,

with [a]_x b = a x b. The kernel matrix K_j (rows (p, k), columns (channel, q, i))
multiplies the stacked tangents of every mode acting on coil j in one matrix
product; product modes, vector modes and subsets are handled identically. B2 (the
Taylor coefficient, general path) comes from eq. (24) directly with each mode's own
(psi, psi'). The forward pass runs lax.map over coils; the gradient is an explicit
per-coil VJP accumulation inside a second lax.map (memory of one coil).
Unit 1b.1: the surface points are processed in static chunks of point_chunk
points in both passes; the pull-back sums per-chunk VJPs in a lax.scan.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

EPS = np.zeros((3, 3, 3))
EPS[0, 1, 2] = EPS[1, 2, 0] = EPS[2, 0, 1] = 1.0
EPS[0, 2, 1] = EPS[2, 1, 0] = EPS[1, 0, 2] = -1.0
EPS = jnp.asarray(EPS)


def coil_tables(modes):
    """Padded per-coil mode tables: idx (C, kmax) with -1 for padding, T, Td (C, kmax, nq, 3)."""
    tabs = [modes.tangents(j) for j in range(modes.ncoils)]
    kmax = max(1, max(len(t[0]) for t in tabs))
    C, nq = modes.ncoils, modes.nq
    idx = -np.ones((C, kmax), dtype=np.int64)
    T = np.zeros((C, kmax, nq, 3))
    Td = np.zeros((C, kmax, nq, 3))
    for j, (ix, dg, dgd) in enumerate(tabs):
        idx[j, :len(ix)] = ix
        T[j, :len(ix)] = dg
        Td[j, :len(ix)] = dgd
    return idx, T, Td, kmax


def _geometry(g, X):
    r = X[:, None, :] - g[None, :, :]                 # (P, nq, 3)
    r2 = jnp.sum(r * r, axis=-1)                       # (P, nq)
    inv3 = r2 ** -1.5
    return r, r2, inv3


def coil_b0(g, gd, cur, X):
    r, r2, inv3 = _geometry(g, X)
    return 1e-7 * cur / g.shape[0] * jnp.einsum("pqk,pq->pk", jnp.cross(gd[None], r), inv3)


def coil_b1(g, gd, cur, X, T, Td):
    """B1 rows (k, P, 3) of the modes with tangents T, Td (k, nq, 3) on this coil."""
    r, r2, inv3 = _geometry(g, X)
    P, nq = r2.shape
    # K^Xi = -[G']_x inv3  +  3 (G' x r) r^T inv5 ; K^Xi' = -[r]_x inv3
    gx = jnp.einsum("kmi,qm->qki", EPS, gd)                       # [G'_q]_x as (q, k, i): (G' x v)_k = eps_kmi G'_m v_i
    KA = -inv3[:, :, None, None] * gx[None]                        # (P, nq, 3, 3)
    c = jnp.cross(gd[None], r)                                     # (P, nq, 3)
    KB = 3.0 * (inv3 / r2)[:, :, None, None] * c[..., :, None] * r[..., None, :]
    rx = jnp.einsum("kmi,pqm->pqki", EPS, r)                       # [r]_x
    KC = -inv3[:, :, None, None] * rx
    K = jnp.concatenate([(KA + KB).transpose(0, 2, 1, 3).reshape(P * 3, nq * 3),
                         KC.transpose(0, 2, 1, 3).reshape(P * 3, nq * 3)], axis=1)   # (P*3, 2*nq*3)
    tang = jnp.concatenate([T.reshape(T.shape[0], nq * 3), Td.reshape(Td.shape[0], nq * 3)], axis=1)  # (k, 2*nq*3)
    out = (K @ tang.T).reshape(P, 3, -1)                          # (P, 3, k)
    return 1e-7 * cur / nq * out.transpose(2, 0, 1)


def coil_b2(g, gd, cur, X, T, Td, mode_chunk):
    """B2 Taylor coefficients (k, P, 3) from eq. (24), lax.map over mode chunks."""
    r, r2, inv3 = _geometry(g, X)
    nq = g.shape[0]
    cgr = jnp.cross(gd[None], r)                                   # (P, nq, 3)
    k = T.shape[0]
    pad = (-k) % mode_chunk
    Tp = jnp.concatenate([T, jnp.zeros((pad,) + T.shape[1:])]).reshape(-1, mode_chunk, nq, 3)
    Tdp = jnp.concatenate([Td, jnp.zeros((pad,) + Td.shape[1:])]).reshape(-1, mode_chunk, nq, 3)

    def chunk(args):
        t, td = args                                               # (m, nq, 3)
        rx = jnp.einsum("pqi,mqi->mpq", r, t)
        x2 = jnp.sum(t * t, axis=-1)                               # (m, nq)
        P1 = 3.0 * rx / r2[None]
        P2 = 7.5 * rx ** 2 / r2[None] ** 2 - 1.5 * x2[:, None, :] / r2[None]
        a = jnp.cross(td[:, None, :, :], r[None]) - jnp.cross(gd[None], t)[:, None, :, :]   # (m, P, nq, 3)
        num = cgr[None] * P2[..., None] + a * P1[..., None] - jnp.cross(td, t)[:, None, :, :]
        return jnp.einsum("mpqk,pq->mpk", num, inv3)
    out = jax.lax.map(chunk, (Tp, Tdp)).reshape(-1, X.shape[0], 3)[:k]
    return 1e-7 * cur / nq * out


def _chunk_fn(g, gd, cur, X, T, Td, general, mode_chunk):
    """(b0, b1 rows) of one coil at the points X, plus b2 rows on the general path only
    (Phase 2 item 1: no zero second-order arrays on the isotropic path)."""
    b0 = coil_b0(g, gd, cur, X)
    b1 = coil_b1(g, gd, cur, X, T, Td)
    if not general:
        return b0, b1
    return b0, b1, coil_b2(g, gd, cur, X, T, Td, mode_chunk)


def _split_points(X, point_chunk):
    """(nchunk, point_chunk, 3), the last chunk padded by repeating the last point (unit 1b.1)."""
    P = X.shape[0]
    n = -(-P // point_chunk)
    pad = n * point_chunk - P
    Xp = jnp.concatenate([X, jnp.repeat(X[-1:], pad, axis=0)]) if pad else X
    return Xp.reshape(n, point_chunk, 3), n, pad


def _pad_rows(c, pad, axis):
    if not pad:
        return c
    shape = list(c.shape); shape[axis] = pad
    return jnp.concatenate([c, jnp.zeros(shape)], axis=axis)


def _coil_fn(g, gd, cur, X, T, Td, general, mode_chunk, point_chunk):
    """Forward per coil; point chunks via lax.map when point_chunk < P (not differentiated)."""
    P = X.shape[0]
    if point_chunk >= P:
        return _chunk_fn(g, gd, cur, X, T, Td, general, mode_chunk)
    Xs, n, pad = _split_points(X, point_chunk)
    outs = jax.lax.map(lambda xc: _chunk_fn(g, gd, cur, xc, T, Td, general, mode_chunk), Xs)
    k = T.shape[0]
    b0 = outs[0].reshape(n * point_chunk, 3)[:P]
    rows = tuple(o.transpose(1, 0, 2, 3).reshape(k, n * point_chunk, 3)[:, :P] for o in outs[1:])
    return (b0,) + rows


@partial(jax.jit, static_argnames=("A", "general", "mode_chunk", "point_chunk"))
def forward(G, GD, I, X, T, Td, idx, A, general, mode_chunk=8, point_chunk=None):
    """B0 (P,3), B1 (A,P,3), and B2 (A,P,3) on the general path (None otherwise)."""
    pc = X.shape[0] if point_chunk is None else point_chunk
    outs = jax.lax.map(lambda a: _coil_fn(a[0], a[1], a[2], X, a[3], a[4], general, mode_chunk, pc),
                       (G, GD, I, T, Td))
    flat = jnp.where(idx.reshape(-1) < 0, A, idx.reshape(-1))
    P = X.shape[0]
    scatter = lambda b: jnp.zeros((A + 1, P, 3)).at[flat].add(b.reshape(-1, P, 3))[:A]
    B2 = scatter(outs[2]) if general else None
    return jnp.sum(outs[0], axis=0), scatter(outs[1]), B2


@partial(jax.jit, static_argnames=("general", "mode_chunk", "point_chunk"))
def pullback(G, GD, I, X, T, Td, idx, cB0, cB1, cB2, general, mode_chunk=8, point_chunk=None):
    """(v_G, v_GD, v_I) per coil from cotangents of B0, B1 (and B2 on the general path; cB2
    is ignored and may be None otherwise).

    Per coil (lax.map over coils), the coil's VJP is the sum of the VJPs of its
    point chunks, accumulated in a lax.scan whose body calls jax.vjp of one
    chunk (unit 1b.1): reverse-mode memory is that of one coil and one chunk.
    """
    A = cB1.shape[0]
    P = X.shape[0]
    pc = P if point_chunk is None else point_chunk
    pad1 = lambda c: jnp.concatenate([c, jnp.zeros((1,) + c.shape[1:])])
    cs = (pad1(cB1),) + ((pad1(cB2),) if general else ())
    gidx = jnp.where(idx < 0, A, idx)

    if pc >= P:
        def one(a):
            g, gd, cur, t, td, ix = a
            f = lambda gg, ggd, cc: _chunk_fn(gg, ggd, cc, X, t, td, general, mode_chunk)
            _, vjp = jax.vjp(f, g, gd, cur)
            return vjp((cB0,) + tuple(c[ix] for c in cs))
        return jax.lax.map(one, (G, GD, I, T, Td, gidx))

    Xs, n, pad = _split_points(X, pc)
    c0s = _pad_rows(cB0, pad, 0).reshape(n, pc, 3)

    def one(a):
        g, gd, cur, t, td, ix = a
        k = t.shape[0]
        crow = tuple(_pad_rows(c[ix], pad, 1).reshape(k, n, pc, 3).transpose(1, 0, 2, 3) for c in cs)

        def body(carry, xs):
            xc, c0 = xs[0], xs[1]
            f = lambda gg, ggd, cc: _chunk_fn(gg, ggd, cc, xc, t, td, general, mode_chunk)
            _, vjp = jax.vjp(f, g, gd, cur)
            vg, vgd, vi = vjp((c0,) + tuple(xs[2:]))
            return (carry[0] + vg, carry[1] + vgd, carry[2] + vi), None

        init = (jnp.zeros_like(g), jnp.zeros_like(gd), jnp.zeros_like(cur))
        out, _ = jax.lax.scan(body, init, (Xs, c0s) + crow)
        return out
    return jax.lax.map(one, (G, GD, I, T, Td, gidx))
