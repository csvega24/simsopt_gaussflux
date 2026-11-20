"""Phase 1b tests: kernel backend vs the Phase 1 JVP backend (G1b(i) on unit-test coils).

Delta2 within 1e-12 relative, dJ within 1e-10 relative to max|dJ|, for both
representations, the three definitions, with and without a target, the
isotropic and general paths, the anisotropic one-axis subset and the
rotating-frame vector modes.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent

import jax.numpy as jnp  # noqa: E402

from gaussflux import kernels as kn  # noqa: E402
from gaussflux import kl_modes as kl  # noqa: E402
from gaussflux import mc_reference as mc  # noqa: E402
from gaussflux import oracles  # noqa: E402
from gaussflux import response as rsp  # noqa: E402
from gaussflux.gaussian_flux import DEFINITIONS, GaussianSquaredFlux  # noqa: E402

from test_gaussian_flux import NBASE, NFP, L, _oracle_case, setup  # noqa: E402


def _rotating(base):
    t = np.asarray(base[0].quadpoints)
    w = 2 * np.pi
    c, sn = np.cos(w * t), np.sin(w * t)
    z = np.zeros_like(t)
    psi = np.array([np.stack([c, sn, z], 1), np.stack([-sn, c, z], 1), np.stack([z, z, 1 + z], 1)])
    dpsi = np.array([np.stack([-w * sn, w * c, z], 1), np.stack([-w * c, -w * sn, z], 1), np.stack([z, z, z], 1)])
    k = 3
    return kl.ModeSet("rotating_frame", len(t), NFP, True, NBASE, np.full(k, 1e-5), np.zeros((k, len(t))),
                      np.zeros((k, len(t))), np.full(k, -1), np.ones(k, int), np.full(k, 4), {},
                      base_psi=psi, base_dpsi=dpsi)


def _cases():
    base, coils, s, target = setup(nq=32, nphi=8, ntheta=6)
    qp = base[0].quadpoints
    fourier = kl.fourier_modes(qp, NFP, True, NBASE, 1e-3, L)
    sampler = mc.make_sampler(qp, 1e-3, L, 1, "simsopt_ldl")
    realized = kl.realized_modes(sampler, NFP, True, NBASE)
    oc, os_, ot, subset = _oracle_case()
    return {"fourier": (coils, s, target, fourier), "realized": (coils, s, target, realized),
            "subset": (oc, os_, ot, subset), "rotating": (coils, s, target, _rotating(base))}


CASES = _cases()


def _pair(name, definition, use_target, path):
    coils, s, target, modes = CASES[name]
    kw = dict(definition=definition, target=target if use_target else None, path=path)
    return (GaussianSquaredFlux(s, coils, modes, backend="kernel", **kw),
            GaussianSquaredFlux(s, coils, modes, backend="jvp", **kw))


def test_b1_b2_match_eq23_eq24_oracles_per_coil():
    coils, s, _, modes = CASES["fourier"]
    g, gd, cur = rsp.coil_arrays(coils)
    X, _, _ = rsp.surface_geometry(s)
    j = 3
    idx, dg, dgd = modes.tangents(j)
    b1 = np.asarray(kn.coil_b1(jnp.asarray(g[j]), jnp.asarray(gd[j]), jnp.asarray(cur[j]), jnp.asarray(X),
                               jnp.asarray(dg[:5]), jnp.asarray(dgd[:5])))
    for k in range(5):
        ref = oracles.b1_eq23_first(g[j], gd[j], cur[j], X, dg[k], dgd[k])
        assert np.max(np.abs(b1[k] - ref)) <= 1e-12 * np.max(np.abs(ref))
    b2 = np.asarray(kn.coil_b2(jnp.asarray(g[j]), jnp.asarray(gd[j]), jnp.asarray(cur[j]), jnp.asarray(X),
                               jnp.asarray(dg[:5]), jnp.asarray(dgd[:5]), 3))
    for k in range(5):
        ref = oracles.b2_eq24(g[j], gd[j], cur[j], X, dg[k], dgd[k])
        assert np.max(np.abs(b2[k] - ref)) <= 1e-12 * np.max(np.abs(ref))


@pytest.mark.parametrize("name,path", [("fourier", "isotropic_shortcut"), ("fourier", "general"),
                                       ("realized", "isotropic_shortcut"), ("realized", "general"),
                                       ("subset", "general"), ("rotating", "general")])
@pytest.mark.parametrize("use_target", [False, True])
def test_kernel_equals_jvp_delta2_and_responses(name, path, use_target):
    for d in DEFINITIONS:
        a, b = _pair(name, d, use_target, path)
        da, db = a.delta2(), b.delta2()
        assert abs(da - db) <= 1e-12 * abs(db), (d, da, db)
    Ba, Bb = a.responses(), b.responses()
    assert np.max(np.abs(Ba[0] - Bb[0])) <= 1e-13 * np.max(np.abs(Bb[0]))
    assert np.max(np.abs(Ba[1] - Bb[1])) <= 1e-12 * np.max(np.abs(Bb[1]))
    if path == "general":
        assert np.max(np.abs(Ba[2] - Bb[2])) <= 1e-10 * np.max(np.abs(Bb[2]))
    assert np.allclose(a.u1(), np.einsum("apk,pk->ap", Bb[1], a.nhat), rtol=0, atol=1e-12 * np.max(np.abs(Bb[1])))
    rec = a.certificate_record()
    assert rec["backend"] == "kernel" and rec["vector_modes"] == (3 if name == "rotating" else 0)


@pytest.mark.parametrize("name,path", [("fourier", "isotropic_shortcut"), ("realized", "general"),
                                       ("subset", "general"), ("rotating", "general")])
@pytest.mark.parametrize("definition", DEFINITIONS)
def test_kernel_equals_jvp_gradient(name, path, definition):
    a, b = _pair(name, definition, True, path)
    ga, gb = a.dJ(), b.dJ()
    assert np.max(np.abs(ga - gb)) <= 1e-10 * np.max(np.abs(gb))


# --------------------------------------------------------------------------
# Unit 1b.1: surface-point chunking (identity with the unchunked program)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("point_chunk", [16, 10])
@pytest.mark.parametrize("name,path", [("fourier", "isotropic_shortcut"), ("realized", "general"),
                                       ("subset", "general"), ("rotating", "general")])
def test_point_chunking_identity(name, path, point_chunk):
    coils, s, target, modes = CASES[name]
    P = len(s.quadpoints_phi) * len(s.quadpoints_theta)
    assert P % 16 == 0 and P % 10 != 0  # one dividing and one padded chunk size
    for d in DEFINITIONS:
        kw = dict(definition=d, target=target, path=path, backend="kernel")
        full = GaussianSquaredFlux(s, coils, modes, point_chunk=P, **kw)
        chunked = GaussianSquaredFlux(s, coils, modes, point_chunk=point_chunk, **kw)
        assert full.point_chunk == P and chunked.point_chunk == point_chunk
        a, b = chunked.delta2(), full.delta2()
        assert abs(a - b) <= 1e-13 * abs(b), (d, a, b)
        ga, gb = chunked.dJ(), full.dJ()
        assert np.max(np.abs(ga - gb)) <= 1e-12 * np.max(np.abs(gb))
    Ra, Rb = chunked.responses(), full.responses()
    for x, y in zip(Ra, Rb):
        if y is not None:
            assert np.max(np.abs(x - y)) <= 1e-13 * np.max(np.abs(y))
    assert chunked.certificate_record()["point_chunk"] == point_chunk


def test_point_chunk_default_and_clamp():
    coils, s, target, modes = CASES["fourier"]
    P = len(s.quadpoints_phi) * len(s.quadpoints_theta)
    assert GaussianSquaredFlux(s, coils, modes).point_chunk == P            # automatic: one chunk on small grids
    assert GaussianSquaredFlux(s, coils, modes, point_chunk=10 * P).point_chunk == P
    assert GaussianSquaredFlux(s, coils, modes, point_chunk=0).point_chunk == 1
