"""Phase 1 tests: KL modes, field responses, Delta2 oracles (no solves, < 60 s)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from simsopt.field import BiotSavart, Current, coils_via_symmetries  # noqa: E402
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves  # noqa: E402
from simsopt.objectives import SquaredFlux  # noqa: E402

from gaussflux import kl_modes as kl  # noqa: E402
from gaussflux import mc_reference as mc  # noqa: E402
from gaussflux import oracles  # noqa: E402
from gaussflux import response as rsp  # noqa: E402
from gaussflux.gaussian_flux import DEFINITIONS, GaussianSquaredFlux, delta2_from_responses  # noqa: E402

NFP, NBASE, NQ, SIGMA, L = 2, 2, 48, 1e-3, 0.5


def setup(nq=NQ, nphi=12, ntheta=8):
    base = create_equally_spaced_curves(NBASE, NFP, stellsym=True, R0=1.0, R1=0.5, order=3, numquadpoints=nq)
    rng = np.random.default_rng(7)
    for c in base:
        c.x = c.x + 0.03 * rng.standard_normal(c.x.shape)
    coils = coils_via_symmetries(base, [Current(1e5), Current(-0.8e5)], NFP, True)
    s = SurfaceRZFourier(nfp=NFP, stellsym=True, mpol=2, ntor=2,
                         quadpoints_phi=np.linspace(0, 1, nphi, endpoint=False),
                         quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))
    s.set_rc(0, 0, 1.0); s.set_rc(1, 0, 0.15); s.set_zs(1, 0, 0.15); s.set_rc(1, 1, 0.03)
    target = 2e-3 * np.cos(2 * np.pi * s.quadpoints_theta)[None, :] * np.ones((nphi, 1))
    return base, coils, s, target


def blocks_err(R, K, n, M, pointvar):
    """Max block error of a (2,n,3,2,n,3) reconstruction against K (2n x 2n) x I3, PLAN §4a scaling."""
    worst = 0.0
    for a in range(2):
        for b in range(2):
            Rab = R[a, :, :, b, :, :]
            Kab = K[a * n:(a + 1) * n, b * n:(b + 1) * n]
            E = Rab - Kab[:, None, :, None] * np.eye(3)[None, :, None, :]
            worst = max(worst, np.max(np.abs(E)) / pointvar / max(1, 2 * np.pi * M) ** (a + b))
    return worst


# --------------------------------------------------------------------------
# G1(a)
# --------------------------------------------------------------------------

def test_fourier_modes_reconstruct_kernel_all_blocks():
    qp = np.linspace(0, 1, NQ, endpoint=False)
    modes = kl.fourier_modes(qp, NFP, True, NBASE, SIGMA, L)
    M = modes.info["cutoff"]["systematic"]
    assert M == 3
    K = mc.kernel_matrix(qp, SIGMA, L, 1)
    pv = SIGMA ** 2 * sum(np.exp(-j ** 2 / L ** 2) for j in range(-5, 6))
    for i in range(modes.ncoils):
        for j in range(modes.ncoils):
            R = modes.reconstruct(i, j)
            same_base = modes.image_base[i] == modes.image_base[j]
            # expected: statistical delta_ij K (x) I + systematic K (x) A_i A_j^T
            exp_map = (np.eye(3) if i == j else 0) + (modes.maps[i] @ modes.maps[j].T if same_base else 0)
            if not same_base and i != j:
                assert np.max(np.abs(R)) == 0
                continue
            Kt = K.reshape(2, NQ, 2, NQ)
            expected = np.einsum("ctds,pq->ctpdsq", Kt, exp_map)
            for a in range(2):
                for b in range(2):
                    err = np.max(np.abs(R[a, :, :, b] - expected[a, :, :, b])) / pv / max(1, 2 * np.pi * M) ** (a + b)
                    assert err <= 1e-10, (i, j, a, b, err)


def test_realized_modes_reconstruct_sampler_LLT():
    qp = np.linspace(0, 1, NQ, endpoint=False)
    sampler = mc.make_sampler(qp, SIGMA, L, 1, "simsopt_ldl")
    modes = kl.realized_modes(sampler, NFP, True, NBASE, levels=("statistical",))
    C = sampler.L @ sampler.L.T
    R = modes.reconstruct(3, 3)
    pv = SIGMA ** 2 * sum(np.exp(-j ** 2 / L ** 2) for j in range(-5, 6))
    for a in range(2):
        for b in range(2):
            Cab = C[a * NQ:(a + 1) * NQ, b * NQ:(b + 1) * NQ]
            E = R[a, :, :, b] - Cab[:, None, :, None] * np.eye(3)[None, :, None, :]
            assert np.max(np.abs(E)) / pv / (2 * np.pi * 3) ** (a + b) <= 1e-10


# --------------------------------------------------------------------------
# G1(b)
# --------------------------------------------------------------------------

def test_b0_b1_b2_against_native_and_oracles():
    # 96 points: the integration-by-parts form of eq. (23) agrees with the direct
    # form to 1e-15 from 96 points on (4.7e-8 at 48: trapezoid error of a derivative)
    base, coils, s, _ = setup(nq=96)
    X, nhat, absn = rsp.surface_geometry(s)
    g, gd, cur = rsp.coil_arrays(coils)
    qp = base[0].quadpoints
    for rep in ("fourier", "realized"):
        if rep == "fourier":
            modes = kl.fourier_modes(qp, NFP, True, NBASE, SIGMA, L)
        else:
            modes = kl.realized_modes(mc.make_sampler(qp, SIGMA, L, 1, "simsopt_ldl"), NFP, True, NBASE)
        B0, B1, B2, _ = rsp.responses(g, gd, cur, X, modes, second=True, batch=16)
        bs = BiotSavart(coils); bs.set_points(X)
        assert np.max(np.abs(B0 - bs.B())) <= 1e-12 * np.max(np.abs(B0))
        for a in list(range(0, modes.size, max(1, modes.size // 25))):
            o1 = np.zeros_like(B0); o1b = np.zeros_like(B0); o2 = np.zeros_like(B0)
            for j, _A in modes.support(a):
                psi, dpsi = modes.on_coil(a, j)
                o1 += oracles.b1_eq23_first(g[j], gd[j], cur[j], X, psi, dpsi)
                o1b += oracles.b1_eq23_second(g[j], gd[j], cur[j], X, psi)
                o2 += oracles.b2_eq24(g[j], gd[j], cur[j], X, psi, dpsi)
            tol1 = 1e-12 * np.max(np.abs(B1[a])) + 1e-14 * np.max(np.abs(B0))
            assert np.max(np.abs(B1[a] - o1)) <= tol1
            if rep == "fourier":
                assert np.max(np.abs(B1[a] - o1b)) <= tol1
            tol2 = 1e-10 * np.max(np.abs(B2[a])) + 1e-14 * np.max(np.abs(B0))
            assert np.max(np.abs(B2[a] - o2)) <= tol2


def test_structural_translation_tangential_rotation_current():
    base, coils, s, target = setup(nq=96)  # nulls resolved to the floor from 96 points (4.3e-11 at 64)
    X, nhat, absn = rsp.surface_geometry(s)
    g, gd, cur = rsp.coil_arrays(coils)
    bs = BiotSavart(coils); bs.set_points(X)
    B0 = bs.B()
    floor = 1e-14 * np.max(np.abs(B0))
    # translation of coil 5 along e_y equals -d_y B^(5)
    j, d = 5, 1
    one = BiotSavart([coils[j]]); one.set_points(X)
    e = np.zeros(3); e[d] = 1.0
    b1 = oracles.b1_eq23_first(g[j], gd[j], cur[j], X, np.tile(e, (len(g[j]), 1)), np.zeros_like(g[j]))
    assert np.max(np.abs(b1 - (-one.dB_by_dX()[:, d, :]))) <= 1e-12 * np.max(np.abs(b1)) + floor
    # tangential reparametrization Xi = eps Gamma' (Gamma'' from the curve) gives zero first-order response
    xi, dxi = coils[j].curve.gammadash(), coils[j].curve.gammadashdash()
    tb = oracles.b1_eq23_first(g[j], gd[j], cur[j], X, xi, dxi)
    assert np.max(np.abs(tb)) <= floor * np.max(np.abs(xi)), np.max(np.abs(tb)) / np.max(np.abs(xi))
    # circular coil rotated about its own axis: zero response
    circ = create_equally_spaced_curves(1, 1, stellsym=False, R0=1.0, R1=0.3, order=1, numquadpoints=96)[0]
    gc, gdc = circ.gamma(), circ.gammadash()
    centre = gc.mean(axis=0)
    axis = np.cross(gc[1] - centre, gc[0] - centre); axis /= np.linalg.norm(axis)
    rot, drot = np.cross(axis, gc - centre), np.cross(axis, gdc)
    rb = oracles.b1_eq23_first(gc, gdc, 1e5, X, rot, drot)
    bc = np.max(np.abs(np.asarray(rsp.coil_field(gc, gdc, 1e5, X))))
    assert np.max(np.abs(rb)) <= 1e-14 * bc * np.max(np.abs(rot)), np.max(np.abs(rb)) / bc
    # current uncertainty, eq. (53) with a nonzero target, exact for quadratic flux
    sigI = 3e3
    per_unit = []
    for c in coils:
        b = BiotSavart([c]); b.set_points(X)
        per_unit.append(b.B() / c.current.get_value())
    u = np.array([np.sum(p * nhat, axis=1) for p in per_unit])
    J = SquaredFlux(s, bs, target=target, definition="quadratic flux").J()
    expected = J + 0.5 * sigI ** 2 * np.sum(np.sum(u ** 2, axis=0) * absn) / len(absn)
    # module path: current modes as a pure first-order response with lam = sigI^2
    B1 = np.array(per_unit)
    d2 = float(delta2_from_responses(jnp.asarray(B0), jnp.asarray(B1), jnp.full(len(coils), sigI ** 2),
                                     None, jnp.asarray(nhat), jnp.asarray(absn),
                                     jnp.asarray(target.reshape(-1)), "quadratic flux"))
    assert abs(J + d2 - expected) <= 1e-12 * expected


# --------------------------------------------------------------------------
# G1(c0)
# --------------------------------------------------------------------------

def test_sigma_zero_equals_native_bitwise():
    base, coils, s, target = setup()
    modes = kl.fourier_modes(base[0].quadpoints, NFP, True, NBASE, 0.0, L)
    for d in DEFINITIONS:
        for tgt in (None, target):
            gf = GaussianSquaredFlux(s, coils, modes, definition=d, target=tgt)
            native = SquaredFlux(s, BiotSavart(coils), target=tgt, definition=d).J()
            assert gf.f0() == native and gf.delta2() == 0.0


def _oracle_case():
    base, coils, s, target = setup(nq=32, nphi=8, ntheta=6)
    qp = base[0].quadpoints
    full = kl.fourier_modes(qp, NFP, True, NBASE, 5e-3, L)
    # 12 modes: systematic translations of base 0 (x, y, z), base-1 m=1 cos y and sin z,
    # statistical m=0 x on coil 3, m=1 cos z on coil 6 (flipped image), m=1 sin x on coil 1,
    # systematic m=0 z on base 1, statistical m=0 y/z on coil 7, m=1 cos x on coil 2
    sel = []
    def find(level, owner, d, k):
        idx = [a for a in range(full.size) if full.level[a] == level and full.owner[a] == owner and full.direction[a] == d]
        return idx[k]
    sel += [find(0, 0, 0, 0), find(0, 0, 1, 0), find(0, 0, 2, 0), find(0, 1, 1, 1), find(0, 1, 2, 2),
            find(1, 3, 0, 0), find(1, 6, 2, 1), find(1, 1, 0, 2), find(0, 1, 2, 0), find(1, 7, 1, 0),
            find(1, 7, 2, 0), find(1, 2, 0, 1)]
    return coils, s, target, full.subset(sel)


@pytest.mark.parametrize("definition", DEFINITIONS)
def test_hessian_trace_oracle_general_path(definition):
    coils, s, target, modes = _oracle_case()
    assert not modes.certificate()["eligible"]  # anisotropic subset -> general path
    gf = GaussianSquaredFlux(s, coils, modes, definition=definition, target=target)
    assert gf.path == "general"
    g, gd, cur = rsp.coil_arrays(coils)
    X, nhat, absn = rsp.surface_geometry(s)
    f = rsp.scalar_objective(g, gd, cur, jnp.asarray(X), jnp.asarray(nhat), jnp.asarray(absn),
                             jnp.asarray(target.reshape(-1)), definition, modes)
    H = jax.hessian(f)(jnp.zeros(modes.size))
    oracle = 0.5 * float(jnp.sum(jnp.asarray(modes.lam) * jnp.diag(H)))
    assert abs(gf.delta2() - oracle) <= 1e-12 * abs(oracle)


def test_isotropic_shortcut_equals_general_path_and_certificate():
    base, coils, s, target = setup(nq=32, nphi=8, ntheta=6)
    modes = kl.fourier_modes(base[0].quadpoints, NFP, True, NBASE, 2e-3, L)
    cert = modes.certificate()
    assert cert["eligible"], cert["max_M0_dev"]
    for d in DEFINITIONS:
        a = GaussianSquaredFlux(s, coils, modes, definition=d, target=target, path="isotropic_shortcut").delta2()
        b = GaussianSquaredFlux(s, coils, modes, definition=d, target=target, path="general").delta2()
        assert abs(a - b) <= 1e-12 * abs(b)


def test_rotating_frame_negative_control():
    base, coils, s, target = setup(nq=32, nphi=8, ntheta=6)
    t = np.asarray(base[0].quadpoints)
    w = 2 * np.pi
    c, sn = np.cos(w * t), np.sin(w * t)
    z = np.zeros_like(t)
    psi = np.array([np.stack([c, sn, z], 1), np.stack([-sn, c, z], 1), np.stack([z, z, 1 + z], 1)])
    dpsi = np.array([np.stack([-w * sn, w * c, z], 1), np.stack([-w * c, -w * sn, z], 1), np.stack([z, z, z], 1)])
    k = 3
    modes = kl.ModeSet("rotating_frame", len(t), NFP, True, NBASE, np.full(k, 1e-5), np.zeros((k, len(t))),
                       np.zeros((k, len(t))), np.full(k, -1), np.ones(k, int), np.full(k, 4), {},
                       base_psi=psi, base_dpsi=dpsi)
    cert = modes.certificate()
    assert cert["max_M0_dev"] <= 1e-12 and cert["max_M1_dev"] > 1e-3 and not cert["eligible"]
    g, gd, cur = rsp.coil_arrays(coils)
    X, nhat, absn = rsp.surface_geometry(s)
    for d in DEFINITIONS:
        gen = GaussianSquaredFlux(s, coils, modes, definition=d, target=target)
        assert gen.path == "general"
        f = rsp.scalar_objective(g, gd, cur, jnp.asarray(X), jnp.asarray(nhat), jnp.asarray(absn),
                                 jnp.asarray(target.reshape(-1)), d, modes)
        oracle = 0.5 * float(jnp.sum(jnp.asarray(modes.lam) * jnp.diag(jax.hessian(f)(jnp.zeros(k)))))
        assert abs(gen.delta2() - oracle) <= 1e-12 * abs(oracle)
        B0, B1, _, _ = rsp.responses(g, gd, cur, X, modes)
        short = float(delta2_from_responses(jnp.asarray(B0), jnp.asarray(B1), jnp.asarray(modes.lam), None,
                                            jnp.asarray(nhat), jnp.asarray(absn), jnp.asarray(target.reshape(-1)), d))
        assert abs(short - oracle) > 1e-6 * abs(oracle)


def test_scaling_sigma_doubling_and_order4_unavailable():
    base, coils, s, target = setup(nq=32, nphi=8, ntheta=6)
    m1 = kl.fourier_modes(base[0].quadpoints, NFP, True, NBASE, 1e-3, L)
    m2 = kl.fourier_modes(base[0].quadpoints, NFP, True, NBASE, 2e-3, L)
    for d in DEFINITIONS:
        a = GaussianSquaredFlux(s, coils, m1, definition=d, target=target).delta2()
        b = GaussianSquaredFlux(s, coils, m2, definition=d, target=target).delta2()
        assert abs(b - 4 * a) <= 1e-13 * abs(b)
    with pytest.raises(NotImplementedError):
        GaussianSquaredFlux(s, coils, m1, order=4)


@pytest.mark.parametrize("general", [False, True])
def test_gradient_pullback_matches_full_reverse_mode_and_fd(general):
    """Smoke check of dJ (its gate is G3, Phase 3): batched pull-back equals full
    reverse-mode AD of the same Delta2, and a central difference of J."""
    if general:
        coils, s, target, modes = _oracle_case()
    else:
        base, coils, s, target = setup(nq=32, nphi=8, ntheta=6)
        modes = kl.fourier_modes(base[0].quadpoints, NFP, True, NBASE, 5e-3, L)
    gf = GaussianSquaredFlux(s, coils, modes, definition="normalized", target=target)
    assert (gf.path == "general") == general
    grad = gf.dJ()
    # reference: full reverse mode through the old all-at-once function
    g, gd, cur = rsp.coil_arrays(coils)
    vg, vgd, vI = jax.grad(gf._delta2_of_arrays(), argnums=(0, 1, 2))(jnp.asarray(g), jnp.asarray(gd), jnp.asarray(cur))
    d = gf.native.dJ(partials=True)
    for j, c in enumerate(coils):
        d = d + c.vjp(np.asarray(vg[j]), np.asarray(vgd[j]), np.array([float(vI[j])]))
    ref = d(gf)
    assert np.max(np.abs(grad - ref)) <= 1e-10 * np.max(np.abs(ref))
    # central difference along a random direction
    x0 = gf.x.copy()
    v = np.random.default_rng(3).standard_normal(x0.shape)
    v /= np.linalg.norm(v)
    h = 1e-6
    gf.x = x0 + h * v; jp = gf.J()
    gf.x = x0 - h * v; jm = gf.J()
    gf.x = x0
    fd = (jp - jm) / (2 * h)
    assert abs(fd - grad @ v) <= 1e-6 * max(abs(fd), abs(gf.J()))
