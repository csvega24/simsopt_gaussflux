"""Phase 2: mode sensitivities, a posteriori rigid-body correction, Schur prediction
(records/phase2_proposal_opus.md §3 and its §9 reconciliation).

Conventions
-----------
* Objective of the correction: SIMSOPT's quadratic flux J_B = 1/2 sum_p w^_p u_p^2,
  w^_p = |n_p|/P, u = B.n^ - target (all eight vacuum designs use it).
* Gram weights per definition with u0 dropped (proposal §3.1, includes the 1/2):
  quadratic flux 1/2 |n|/P; local 1/2 |n|/(P w0); normalized 1/2 |n|/D0, D0 = sum |n| w0
  (the form of gaussian_flux.delta2_from_responses).
* Correction map per physical coil j (proposal §3.2, re-based at q0 per §9 R1):
      Gamma^corr = c_j + R(dw)(G~ + X~ - c_j) + dt,   Gamma'^corr = R(dw)(G~' + X~'),
  G~ = c_j + R(w0)(G - c_j) + t0, G~' = R(w0) G', X~ = R(w0) X; R(w) = expm([w]x).
  q = (t_x, t_y, t_z, w_x, w_y, w_z) per coil, coil-major.
* Scaled coordinates: translations in m, rotations times rho_j (RMS distance of the
  coil from c_j), so one unit moves the coil by about 1 m RMS.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
import scipy.optimize
import simsoptpp as sopp

from . import kernels as kn
from . import kl_modes as kl

jax.config.update("jax_enable_x64", True)

EYE = np.eye(3)


# ---------------------------------------------------------------------------
# geometry, weights, rotations
# ---------------------------------------------------------------------------

def gram_weights(absn, B0, nhat, definition):
    """omega_p of proposal §3.1 (with the objective's 1/2), u0 dropped."""
    P = absn.shape[0]
    w0 = np.sum(B0 * B0, axis=1)
    if definition == "quadratic flux":
        return 0.5 * absn / P
    if definition == "local":
        return 0.5 * absn / (P * w0)
    if definition == "normalized":
        return 0.5 * absn / np.sum(absn * w0)
    raise ValueError(definition)


def skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def rot(w):
    return scipy.linalg.expm(skew(np.asarray(w, float)))


def _jskew(w):
    z = jnp.zeros(())
    return jnp.array([[z, -w[2], w[1]], [w[2], z, -w[0]], [-w[1], w[0], z]])


def jrot(w):
    return jax.scipy.linalg.expm(_jskew(w))


def centroids(g, gd):
    """Arclength-weighted centroid c_j and RMS radius rho_j per coil."""
    w = np.linalg.norm(gd, axis=2)
    c = np.einsum("jq,jqk->jk", w, g) / w.sum(axis=1)[:, None]
    rho = np.sqrt(np.einsum("jq,jq->j", w, np.sum((g - c[:, None]) ** 2, axis=2)) / w.sum(axis=1))
    return c, rho


def q_scaling(rho):
    """Diagonal S with q = S q^ (q^ scaled): translations 1, rotations 1/rho_j."""
    return np.concatenate([np.r_[1.0, 1.0, 1.0, 1 / r, 1 / r, 1 / r] for r in rho])


def apply_rigid(g, gd, c, q):
    """Gamma^corr for rigid parameters q (C*6) about centres c (no error term)."""
    q = np.asarray(q).reshape(-1, 6)
    G, GD = np.empty_like(g), np.empty_like(gd)
    for j in range(len(g)):
        R = rot(q[j, 3:])
        G[j] = c[j] + (g[j] - c[j]) @ R.T + q[j, :3]
        GD[j] = gd[j] @ R.T
    return G, GD, [rot(q[j, 3:]) for j in range(len(g))]


def rebase_modes(modes, Rs):
    """Rotate a ModeSet with its coils (proposal §9 R1): systematic image maps
    A_j -> R_j A_j, statistical (lab-frame) modes psi -> R_owner psi."""
    psi, dpsi = np.array(modes.base_psi), np.array(modes.base_dpsi)
    for a in range(modes.size):
        if modes.level[a] == 1:
            R = Rs[int(modes.owner[a])]
            psi[a] = psi[a] @ R.T
            dpsi[a] = dpsi[a] @ R.T
    out = modes._copy(base_psi=psi, base_dpsi=dpsi, direction=np.full(modes.size, -1),
                      info=dict(modes.info, rebased=True))
    out.maps = np.array([Rs[j] @ modes.maps[j] for j in range(modes.ncoils)])
    return out


def rigid_modes(g, gd, c, nfp, stellsym, nbase):
    """Six statistical vector modes per coil at the configuration (g, gd): translations
    e_d (psi' = 0) and rotations e_d x (Gamma - c) (psi' = e_d x Gamma'), lam = 1."""
    C, nq = g.shape[:2]
    psi, dpsi, owner = [], [], []
    for j in range(C):
        v = g[j] - c[j]
        for d in range(3):
            psi.append(np.broadcast_to(EYE[d], (nq, 3)).copy()); dpsi.append(np.zeros((nq, 3))); owner.append(j)
        for d in range(3):
            psi.append(np.cross(EYE[d], v)); dpsi.append(np.cross(EYE[d], gd[j])); owner.append(j)
    n = len(psi)
    return kl.ModeSet("rigid", nq, nfp, stellsym, nbase, np.ones(n), np.zeros((n, nq)), np.zeros((n, nq)),
                      np.full(n, -1), np.ones(n, int), np.array(owner), {"kind": "rigid", "order": "t,t,t,w,w,w"},
                      base_psi=np.array(psi), base_dpsi=np.array(dpsi))


def fourier_m(modes):
    """Angular wavenumber m of each Fourier KL mode (block order [1, cos1, sin1, cos2, ...])."""
    m = np.zeros(modes.size, int)
    cut = modes.info["cutoff"]
    a = 0
    while a < modes.size:
        lev = kl.LEVELS[int(modes.level[a])]
        nb = 2 * cut[lev] + 1
        m[a:a + nb] = (np.arange(nb) + 1) // 2
        a += nb
    return m


# ---------------------------------------------------------------------------
# responses through the kernels
# ---------------------------------------------------------------------------

def mode_u1(g, gd, cur, X, nhat, modes, point_chunk, Rs=None, batch=256):
    """u1^a = n.B1^a (A, P) and B0 (P, 3); modes re-based by Rs if given (after
    subsetting, since ModeSet copies rebuild the image maps)."""
    G, GD, I, XX = (jnp.asarray(x) for x in (g, gd, cur, X))
    out, B0 = [], None
    for s0 in range(0, modes.size, batch):
        sub = modes.subset(np.arange(s0, min(s0 + batch, modes.size)))
        if Rs is not None:
            sub = rebase_modes(sub, Rs)
        idx, T, Td, _ = kn.coil_tables(sub)
        b0, B1, _ = kn.forward(G, GD, I, XX, jnp.asarray(T), jnp.asarray(Td), jnp.asarray(idx), sub.size, False,
                               8, point_chunk)
        out.append(np.einsum("apk,pk->ap", np.asarray(B1), nhat))
        B0 = np.asarray(b0)
    return np.concatenate(out), B0


def rigid_u1(g, gd, cur, X, nhat, c, nfp, stellsym, nbase, point_chunk):
    rm = rigid_modes(g, gd, c, nfp, stellsym, nbase)
    return mode_u1(g, gd, cur, X, nhat, rm, point_chunk)


# ---------------------------------------------------------------------------
# projection, Gram diagonals, r(Lambda), sigma*
# ---------------------------------------------------------------------------

def projector(u1q, omega, tau=1e-10):
    """Rank-revealing weighted projection onto span{u1^q}: thin SVD of W^1/2 C."""
    M = (np.sqrt(omega)[:, None] * u1q.T)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    k = int(np.sum(s > tau * s[0]))
    return U[:, :k], s, k


def gram_diagonals(u1, omega, U):
    """Q_aa and Q_perp_aa for each row of u1 (A, P)."""
    v = u1 * np.sqrt(omega)[None, :]
    q = np.sum(v * v, axis=1)
    vp = v - (v @ U) @ U.T
    return q, np.sum(vp * vp, axis=1)


def r_lambda(lam, q, qperp):
    den = float(np.sum(lam * q))
    if den == 0.0:
        return {"r": None, "status": "undefined (zero denominator)"}
    return {"r": 1.0 - float(np.sum(lam * qperp)) / den, "status": "ok"}


def sigma_star(f0, d2hat):
    """sigma* with the degenerate cases of PLAN §4a."""
    if f0 > 0 and d2hat > 0:
        return {"sigma_star": float(np.sqrt(f0 / d2hat)), "case": "regular"}
    if f0 == 0 and d2hat > 0:
        return {"sigma_star": 0.0, "case": "f0 = 0, Delta2_hat > 0: sigma* = 0"}
    if d2hat == 0 and f0 > 0:
        return {"sigma_star": None, "case": "Delta2_hat = 0, f0 > 0: no finite scale"}
    if f0 == 0 and d2hat == 0:
        return {"sigma_star": None, "case": "f0 = Delta2_hat = 0: undefined"}
    return {"sigma_star": None, "case": "negative ratio: no real sigma*"}


# ---------------------------------------------------------------------------
# geometric decomposition T / R / S (tangential, normal)
# ---------------------------------------------------------------------------

def _spectral_derivative(f):
    """d/dt of a periodic sampled function on t in [0, 1) (rows along axis 0)."""
    n = f.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n)
    if n % 2 == 0:
        k[n // 2] = 0.0
    return np.real(np.fft.ifft(2j * np.pi * k[:, None] * np.fft.fft(f, axis=0), axis=0))


def decompose(modes, g, gd):
    """Four derived ModeSets (T, R, St, Sn) with T + R + St + Sn = the mode, per mode
    on its base curve (systematic) or owner coil (statistical), original geometry.
    Returns (dict of ModeSets, per-mode displacement norms dict)."""
    parts = {k: (np.zeros_like(modes.base_psi), np.zeros_like(modes.base_dpsi)) for k in ("T", "R", "St", "Sn")}
    norms = {k: np.zeros(modes.size) for k in ("T", "R", "St", "Sn", "all")}
    for a in range(modes.size):
        j = int(modes.owner[a])  # base index = coil index of the identity image (image_maps order)
        G, GD = g[j], gd[j]
        w = np.linalg.norm(GD, axis=1)
        c = (w @ G) / w.sum()
        v = G - c
        psi, dpsi = modes.base_psi[a], modes.base_dpsi[a]
        T = (w @ psi) / w.sum()
        d = psi - T
        M = np.einsum("q,qij->ij", w, np.einsum("qk,qk->q", v, v)[:, None, None] * EYE - np.einsum("qi,qj->qij", v, v))
        om = np.linalg.solve(M, np.einsum("q,qk->k", w, np.cross(v, d)))
        Rp, Rdp = np.cross(om, v), np.cross(om, GD)
        S, dS = d - Rp, dpsi - Rdp
        tau = GD / w[:, None]
        St = np.sum(S * tau, axis=1)[:, None] * tau
        dSt = _spectral_derivative(St)
        pieces = {"T": (np.broadcast_to(T, psi.shape), np.zeros_like(psi)), "R": (Rp, Rdp), "St": (St, dSt),
                  "Sn": (S - St, dS - dSt)}
        for k, (p_, dp_) in pieces.items():
            parts[k][0][a], parts[k][1][a] = p_, dp_
            norms[k][a] = float(w @ np.sum(p_ * p_, axis=1) / w.sum())
        norms["all"][a] = float(w @ np.sum(psi * psi, axis=1) / w.sum())
    out = {k: modes._copy(base_psi=parts[k][0], base_dpsi=parts[k][1], direction=np.full(modes.size, -1),
                          info=dict(modes.info, part=k)) for k in parts}
    return out, norms


# ---------------------------------------------------------------------------
# J_B, its gradient and Hessian blocks at the re-based configuration
# ---------------------------------------------------------------------------

def jb_value(B, nhat, absn, target):
    u = np.sum(B * nhat, axis=1) - target
    return 0.5 * float(np.sum(absn * u * u)) / len(absn)


@partial(jax.jit, static_argnames=("point_chunk",))
def _lin_contract(g, gd, cur, X, V, T, Td, point_chunk):
    """sum_p V_p . B1[T_a](p) for the tangents T (k, nq, 3) of one coil: (k,)."""
    Xs, n, pad = kn._split_points(X, point_chunk)
    Vs = kn._pad_rows(V, pad, 0).reshape(n, point_chunk, 3)

    def one(xv):
        xc, vc = xv
        return jnp.einsum("kpi,pi->k", kn.coil_b1(g, gd, cur, xc, T, Td), vc)
    return jnp.sum(jax.lax.map(one, (Xs, Vs)), axis=0)


@partial(jax.jit, static_argnames=("point_chunk",))
def _bil_contract(g, gd, cur, X, V, T, Td, Dg, Dgd, point_chunk):
    """sum_p V_p . d2B[T_a, D_m](p) for one coil: (k, m), by forward-mode derivative of
    coil_b1 (tangent table T fixed) along the coil-array directions D."""
    Xs, n, pad = kn._split_points(X, point_chunk)
    Vs = kn._pad_rows(V, pad, 0).reshape(n, point_chunk, 3)

    def one(xv):
        xc, vc = xv
        f = lambda gg, ggd: jnp.einsum("kpi,pi->k", kn.coil_b1(gg, ggd, cur, xc, T, Td), vc)
        d = lambda dg, dgd: jax.jvp(f, (g, gd), (dg, dgd))[1]
        return jax.vmap(d, out_axes=1)(Dg, Dgd)
    return jnp.sum(jax.lax.map(one, (Xs, Vs)), axis=0)


def rigid_tangents(g, gd, c):
    """(6, nq, 3) position and derivative tangents of the rigid parameters of one coil at dw = 0."""
    v = g - c
    T = np.array([np.broadcast_to(EYE[d], v.shape) for d in range(3)] + [np.cross(EYE[d], v) for d in range(3)])
    Td = np.array([np.zeros_like(v)] * 3 + [np.cross(EYE[d], gd) for d in range(3)])
    return T, Td


def rotation_curvature(g, gd, c):
    """d2 Gamma / d w_d d w_e at dw = 0, both channels (proposal §9 R3): dict (d, e) -> (pos, der)."""
    v = g - c
    out = {}
    for d in range(3):
        for e in range(d, 3):
            pos = 0.5 * (np.cross(EYE[d], np.cross(EYE[e], v)) + np.cross(EYE[e], np.cross(EYE[d], v)))
            der = 0.5 * (np.cross(EYE[d], np.cross(EYE[e], gd)) + np.cross(EYE[e], np.cross(EYE[d], gd)))
            out[(d, e)] = (pos, der)
    return out


def hessian_blocks(g, gd, cur, X, nhat, absn, target, c, modes_rb, u1m, u1q, B0, point_chunk):
    """H_qq (Nq, Nq), H_qxi (Nq, A), gradients at the re-based configuration (dq = 0, xi = 0)
    for J_B (proposal §3.4 with §9 R1, R3). modes_rb: modes already re-based."""
    P = len(absn)
    wh = absn / P
    u0 = np.sum(B0 * nhat, axis=1) - target
    V = jnp.asarray((wh * u0)[:, None] * nhat)          # grad_B f
    C = len(g)
    Nq, A = 6 * C, modes_rb.size
    Hqq = (u1q * wh[None]) @ u1q.T                      # <dB, grad^2 f dB>
    Hqx = (u1q * wh[None]) @ u1m.T
    gq = u1q @ (wh * u0)
    gx = u1m @ (wh * u0)
    XX = jnp.asarray(X)
    point_chunk = int(min(point_chunk, P))
    for j in range(C):
        gj, gdj, cj = jnp.asarray(g[j]), jnp.asarray(gd[j]), jnp.asarray(cur[j])
        T6, Td6 = rigid_tangents(g[j], gd[j], c[j])
        sl = slice(6 * j, 6 * j + 6)
        # rigid-rigid: d2B[tau_x, tau_y] and B1 of the map curvature
        Hqq[sl, sl] += np.asarray(_bil_contract(gj, gdj, cj, XX, V, jnp.asarray(T6), jnp.asarray(Td6),
                                                jnp.asarray(T6), jnp.asarray(Td6), point_chunk))
        curv = rotation_curvature(g[j], gd[j], c[j])
        keys = list(curv)
        lin = np.asarray(_lin_contract(gj, gdj, cj, XX, V, jnp.asarray(np.array([curv[k][0] for k in keys])),
                                       jnp.asarray(np.array([curv[k][1] for k in keys])), point_chunk))
        for (d, e), val in zip(keys, lin):
            Hqq[6 * j + 3 + d, 6 * j + 3 + e] += val
            if d != e:
                Hqq[6 * j + 3 + e, 6 * j + 3 + d] += val
        # mode-rigid on this coil: d2B[psi_a, tau_x] and B1 of e_d x psi_a (rotations)
        idx, dg, dgd = modes_rb.tangents(j)
        if len(idx) == 0:
            continue
        Hqx[sl, idx] += np.asarray(_bil_contract(gj, gdj, cj, XX, V, jnp.asarray(dg), jnp.asarray(dgd),
                                                 jnp.asarray(T6), jnp.asarray(Td6), point_chunk)).T
        for d in range(3):
            lin = np.asarray(_lin_contract(gj, gdj, cj, XX, V, jnp.asarray(np.cross(EYE[d], dg)),
                                           jnp.asarray(np.cross(EYE[d], dgd)), point_chunk))
            Hqx[6 * j + 3 + d, idx] += lin
    return {"Hqq": 0.5 * (Hqq + Hqq.T), "Hqx": Hqx, "grad_q": gq, "grad_xi": gx}



def hessian_xixi(g, gd, cur, X, nhat, absn, target, modes_rb, u1m, B0, point_chunk, batch=6):
    """H_xixi of J_B at the re-based configuration (dq = 0, xi = 0), unit P2V
    (records/phase2v_proposal_opus.md §1): Gauss-Newton part <dB, grad^2 f dB> and mismatch
    part grad_B f . d2B[psi_a, psi_b], returned separately; the latter is assembled in full
    (not mirrored) so its symmetry is a check. The error map is linear in xi and adds no
    map curvature; second derivatives of B have no cross-coil terms. The second set of
    mode directions is taken `batch` at a time (memory, as the H_qxi call of hessian_blocks)."""
    P = len(absn)
    wh = absn / P
    u0 = np.sum(B0 * nhat, axis=1) - target
    V = jnp.asarray((wh * u0)[:, None] * nhat)
    A = modes_rb.size
    Hgn = (u1m * wh[None]) @ u1m.T
    Hmm = np.zeros((A, A))
    XX = jnp.asarray(X)
    point_chunk = int(min(point_chunk, P))
    for j in range(len(g)):
        idx, dg, dgd = modes_rb.tangents(j)
        if len(idx) == 0:
            continue
        gj, gdj, cj = jnp.asarray(g[j]), jnp.asarray(gd[j]), jnp.asarray(cur[j])
        T, Td = jnp.asarray(dg), jnp.asarray(dgd)
        for b0 in range(0, len(idx), batch):
            b = slice(b0, b0 + batch)
            M = np.asarray(_bil_contract(gj, gdj, cj, XX, V, T, Td, jnp.asarray(dg[b]), jnp.asarray(dgd[b]), point_chunk))
            Hmm[np.ix_(idx, idx[b])] += M
    return {"H_GN": Hgn, "H_mm": Hmm}

def schur(lam, Hqq, Hqx, S, delta2_value):
    """Prediction (ii) with the conditions of proposal §3.4 / §9 R4 (scaled coordinates)."""
    Hs = S[:, None] * Hqq * S[None, :]
    hs = S[:, None] * Hqx
    mu, Vv = np.linalg.eigh(Hs)
    rec = {"eigenvalues": mu.tolist(), "mu_max": float(mu[-1]), "mu_min": float(mu[0])}
    rec["positive_definite"] = bool(mu[0] > 0)
    rec["cond"] = float(mu[-1] / mu[0]) if mu[0] > 0 else None
    proj = Vv.T @ hs                                     # (Nq, A)
    delta = 0.5 * (proj ** 2 @ lam) / np.where(mu > 0, mu, np.nan)
    rec["delta_i"] = delta.tolist()
    rec["cumulative_from_smallest"] = np.cumsum(delta).tolist()
    small = mu < 1e-4 * mu[-1]
    tot = float(np.nansum(delta))
    rec["fragile"] = bool(mu[0] > 0 and tot > 0 and float(np.nansum(delta[small])) > 0.5 * tot)
    if mu[0] <= 0:
        rec.update(available=False, reason="indefinite or singular H_qq")
    elif mu[0] < 1e-8 * mu[-1] or rec["cond"] > 1e8:
        rec.update(available=False, reason="ill-conditioned H_qq (eigenvalue below 1e-8 mu_max or cond > 1e8)")
    else:
        rec.update(available=True, reason="")
    rec["schur_term"] = tot
    rec["c_ii"] = (delta2_value - tot) if rec["available"] else None
    return rec


# ---------------------------------------------------------------------------
# nonlinear correction (Levenberg-Marquardt on the native residual)
# ---------------------------------------------------------------------------

def left_jacobian(w):
    """J_l(w) of SO(3): exp([w + d]x) = exp([J_l(w) d]x) exp([w]x) + O(d^2)."""
    w = np.asarray(w, float)
    th = np.linalg.norm(w)
    K = skew(w)
    if th < 1e-4:
        a, b = 0.5 - th ** 2 / 24.0, 1.0 / 6.0 - th ** 2 / 120.0
    else:
        a, b = (1 - np.cos(th)) / th ** 2, (th - np.sin(th)) / th ** 3
    return EYE + a * K + b * K @ K


def rigid_jac_tangents(base_g, base_gd, c, q):
    """d Gamma^corr / d q at q for the map c + R(w)(base - c) + t, per coil, both channels:
    translations e_d; rotations (J_l(w) e_d) x (R(w)(base - c)) and (J_l(w) e_d) x (R(w) base')."""
    q = np.asarray(q).reshape(-1, 6)
    C = len(base_g)
    T = np.zeros((C, 6) + base_g.shape[1:])
    Td = np.zeros_like(T)
    for j in range(C):
        R, Jl = rot(q[j, 3:]), left_jacobian(q[j, 3:])
        v, vd = (base_g[j] - c[j]) @ R.T, base_gd[j] @ R.T
        for d in range(3):
            T[j, d] = EYE[d]
            a = Jl[:, d]
            T[j, 3 + d] = np.cross(a, v)
            Td[j, 3 + d] = np.cross(a, vd)
    return T, Td


class RigidCorrection:
    """min_q J_B over rigid motions of each coil applied to the arrays (base_g, base_gd)
    (the re-based perturbed coils, or the nominal coils for the q0 solve)."""

    def __init__(self, base_g, base_gd, cur, X, nhat, absn, target, c, point_chunk):
        self.bg, self.bgd, self.cur = np.asarray(base_g), np.asarray(base_gd), np.asarray(cur)
        self.X, self.nhat, self.absn, self.target = X, nhat, absn, target
        self.c, self.pc = c, int(min(point_chunk, len(absn)))
        self.sw = np.sqrt(absn / len(absn))
        self.C = len(base_g)
        self.idx = np.arange(6 * self.C).reshape(self.C, 6)
        self.nfev = 0
        self.njev = 0

    def field(self, q):
        G, GD, _ = apply_rigid(self.bg, self.bgd, self.c, q) if np.any(q) else (self.bg, self.bgd, None)
        return sopp.biot_savart_B(np.ascontiguousarray(self.X), [np.ascontiguousarray(x) for x in G],
                                  [np.ascontiguousarray(x) for x in GD], [float(x) for x in self.cur]), G, GD

    def residual(self, q):
        self.nfev += 1
        B, _, _ = self.field(q)
        return self.sw * (np.sum(B * self.nhat, axis=1) - self.target)

    def jacobian(self, q):
        self.njev += 1
        T, Td = rigid_jac_tangents(self.bg, self.bgd, self.c, q)
        G, GD, _ = apply_rigid(self.bg, self.bgd, self.c, q)
        _, B1, _ = kn.forward(jnp.asarray(G), jnp.asarray(GD), jnp.asarray(self.cur), jnp.asarray(self.X),
                              jnp.asarray(T), jnp.asarray(Td), jnp.asarray(self.idx), 6 * self.C, False, 8, self.pc)
        return (self.sw[:, None] * np.einsum("apk,pk->pa", np.asarray(B1), self.nhat))

    def solve(self, q_start, gscale, S, gtol_rel=1e-10, max_nfev=100):
        """LM in scaled coordinates; converged when ||S grad_q J_B||_inf <= gtol_rel * gscale."""
        self.nfev = self.njev = 0
        fun = lambda qh: self.residual(S * qh)
        jac = lambda qh: self.jacobian(S * qh) * S[None, :]
        res = scipy.optimize.least_squares(fun, np.asarray(q_start) / S, jac=jac, method="lm", xtol=1e-15,
                                           ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
        qh = np.array(res.x)
        r = self.residual(S * qh)
        Jh = self.jacobian(S * qh) * S[None, :]
        grad = Jh.T @ r
        gmax = float(np.max(np.abs(grad)))
        lm_gmax = gmax
        # Gauss-Newton polish: LM stops on the f tolerance once the remaining decrease (~g^2/mu)
        # is below f's resolution; GN steps reduce the gradient itself (accepted only if it decreases).
        polish = 0
        while gmax > gtol_rel * gscale and polish < 25:
            step = -np.linalg.lstsq(Jh, r, rcond=None)[0]
            r_new = self.residual(S * (qh + step))
            J_new = self.jacobian(S * (qh + step)) * S[None, :]
            g_new = float(np.max(np.abs(J_new.T @ r_new)))
            polish += 1
            if g_new >= gmax:
                break
            qh, r, Jh, gmax = qh + step, r_new, J_new, g_new
        q = S * qh
        return {"q": q, "f": 0.5 * float(r @ r), "grad_scaled_max": gmax, "lm_grad_scaled_max": lm_gmax,
                "gn_polish_steps": polish, "converged": gmax <= gtol_rel * gscale, "status": int(res.status),
                "message": res.message, "nfev": self.nfev, "njev": self.njev}

    def solve_gn(self, q_start, gscale, S, gtol_rel=1e-10, ftol_rel=1e-15, max_iter=50):
        """Gauss-Newton with step halving (per-sample correction solves, proposal §3.5).

        Converged when ||S grad||_inf <= gtol_rel * gscale ("gradient"), or when the relative
        decrease of J_B stays below ftol_rel for 3 consecutive steps ("f-stationary"); the
        criterion met is recorded."""
        self.nfev = self.njev = 0
        qh = np.asarray(q_start, float) / S
        r = self.residual(S * qh)
        f = 0.5 * float(r @ r)
        small, it, reason = 0, 0, "max_iter"
        while it < max_iter:
            Jh = self.jacobian(S * qh) * S[None, :]
            gmax = float(np.max(np.abs(Jh.T @ r)))
            if gmax <= gtol_rel * gscale:
                reason = "gradient"
                break
            step = -np.linalg.lstsq(Jh, r, rcond=None)[0]
            a, accepted = 1.0, False
            for _ in range(30):
                r_new = self.residual(S * (qh + a * step))
                f_new = 0.5 * float(r_new @ r_new)
                if f_new <= f:
                    accepted = True
                    break
                a *= 0.5
            it += 1
            if not accepted:
                reason = "no decrease along the Gauss-Newton direction"
                break
            small = small + 1 if (f - f_new) <= ftol_rel * f else 0
            qh, r, f = qh + a * step, r_new, f_new
            if small >= 3:
                Jh = self.jacobian(S * qh) * S[None, :]
                gmax = float(np.max(np.abs(Jh.T @ r)))
                reason = "f-stationary"
                break
        return {"q": S * qh, "f": f, "grad_scaled_max": gmax, "converged": reason in ("gradient", "f-stationary"),
                "criterion": reason, "iterations": it, "nfev": self.nfev, "njev": self.njev,
                "lm_grad_scaled_max": None, "gn_polish_steps": 0}

    def solve_newton(self, q_start, gscale, S, H0, gtol_rel=1e-10, ftol_rel=1e-15, max_iter=30):
        """Newton iteration with the fixed full Hessian H0 = H_qq at (0, q0) (unscaled, from P2H),
        exact gradient J^T r at every iterate and step halving on f (per-sample solves, §3.5).
        For draws of sigma <= 1 mm the iteration matrix differs from the true Hessian by O(sigma),
        so the contraction is fast; Gauss-Newton (J^T J) omits the mismatch term and is slow in the
        softest directions. Convergence criteria and record as solve_gn."""
        self.nfev = self.njev = 0
        Hs = S[:, None] * H0 * S[None, :]
        L = np.linalg.cholesky(Hs)
        qh = np.asarray(q_start, float) / S
        r = self.residual(S * qh)
        f = 0.5 * float(r @ r)
        small, it, reason = 0, 0, "max_iter"
        while it < max_iter:
            Jh = self.jacobian(S * qh) * S[None, :]
            g = Jh.T @ r
            gmax = float(np.max(np.abs(g)))
            if gmax <= gtol_rel * gscale:
                reason = "gradient"
                break
            step = -np.linalg.solve(L.T, np.linalg.solve(L, g))
            a, accepted = 1.0, False
            for _ in range(30):
                r_new = self.residual(S * (qh + a * step))
                f_new = 0.5 * float(r_new @ r_new)
                if f_new <= f:
                    accepted = True
                    break
                a *= 0.5
            it += 1
            if not accepted:
                reason = "no decrease along the Newton direction"
                break
            small = small + 1 if (f - f_new) <= ftol_rel * f else 0
            qh, r, f = qh + a * step, r_new, f_new
            if small >= 3:
                Jh = self.jacobian(S * qh) * S[None, :]
                gmax = float(np.max(np.abs(Jh.T @ r)))
                reason = "f-stationary"
                break
        if reason == "no decrease along the Newton direction":
            reason = ("no decrease (f at roundoff; gradient <= 1e-8 x scale)" if gmax <= 1e-8 * gscale
                      else "no decrease along the Newton direction")
        return {"q": S * qh, "f": f, "grad_scaled_max": gmax,
                "converged": reason in ("gradient", "f-stationary", "no decrease (f at roundoff; gradient <= 1e-8 x scale)"),
                "criterion": reason, "iterations": it, "nfev": self.nfev, "njev": self.njev,
                "lm_grad_scaled_max": None, "gn_polish_steps": 0}
