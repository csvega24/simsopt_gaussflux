"""Phase 3 tests (records/phase3_proposal_opus.md §3.3-3.4, §2 items 4-5): the independent
Hessian-trace gradient comparison by nested AD, the include_f0 identity, repeated .x updates
and fixed dofs, per-level realized samplers, the grid-aware reference budget."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent
TESTS = Path(__file__).resolve().parent
for p in (TESTS,):   # the installed package is imported; tests/ for test_gaussian_flux
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from simsopt.field import BiotSavart  # noqa: E402
from simsopt.objectives import SquaredFlux  # noqa: E402

from gaussflux import kl_modes as kl  # noqa: E402
from gaussflux import mc_reference as mc  # noqa: E402
from gaussflux import response as rsp  # noqa: E402
from gaussflux.gaussian_flux import DEFINITIONS, GaussianSquaredFlux  # noqa: E402
from test_gaussian_flux import NBASE, NFP, L, _oracle_case, setup  # noqa: E402


def _nested_ad_gradient(coils, s, target, modes, definition, gf):
    """d/dx [1/2 tr(Lambda Hess_xi f)] by jax.grad over the coil arrays of jax.hessian in xi
    of the validated unexpanded f, chained to the dofs by Coil.vjp."""
    g, gd, cur = rsp.coil_arrays(coils)
    X, nhat, absn = (jnp.asarray(a) for a in rsp.surface_geometry(s))
    tgt = jnp.asarray(np.zeros(len(absn)) if target is None else target.reshape(-1))
    lam = jnp.asarray(modes.lam)

    def half_trace(G, GD, I):
        f = rsp.scalar_objective(G, GD, I, X, nhat, absn, tgt, definition, modes)
        return 0.5 * jnp.sum(lam * jnp.diag(jax.hessian(f)(jnp.zeros(modes.size))))

    vg, vgd, vI = jax.grad(half_trace, argnums=(0, 1, 2))(jnp.asarray(g), jnp.asarray(gd), jnp.asarray(cur))
    d = None
    for j, c in enumerate(coils):
        v = c.vjp(np.asarray(vg[j]), np.asarray(vgd[j]), np.array([float(vI[j])]))
        d = v if d is None else d + v
    return d(gf)


@pytest.mark.parametrize("definition", DEFINITIONS)
def test_delta2_gradient_against_nested_ad_hessian_trace_general(definition):
    coils, s, target, modes = _oracle_case()           # anisotropic subset, systematic images, nonzero target
    gf = GaussianSquaredFlux(s, coils, modes, definition=definition, target=target, include_f0=False)
    assert gf.path == "general"
    ref = _nested_ad_gradient(coils, s, target, modes, definition, gf)
    assert np.max(np.abs(gf.dJ() - ref)) <= 1e-10 * np.max(np.abs(ref))


def test_delta2_gradient_against_nested_ad_hessian_trace_isotropic():
    base, coils, s, target = setup(nq=32, nphi=8, ntheta=6)
    modes = kl.fourier_modes(base[0].quadpoints, NFP, True, NBASE, 5e-3, L)
    gf = GaussianSquaredFlux(s, coils, modes, definition="quadratic flux", target=target, include_f0=False)
    assert gf.path == "isotropic_shortcut"
    ref = _nested_ad_gradient(coils, s, target, modes, "quadratic flux", gf)
    assert np.max(np.abs(gf.dJ() - ref)) <= 1e-10 * np.max(np.abs(ref))


def test_include_f0_identity_repeated_x_and_fixed_dofs():
    base, coils, s, target = setup(nq=32, nphi=8, ntheta=6)
    modes = kl.fourier_modes(base[0].quadpoints, NFP, True, NBASE, 5e-3, L)
    coils[0].current.fix_all()                          # a fixed dof, as the example's first current
    full = GaussianSquaredFlux(s, coils, modes, target=target)
    d2 = GaussianSquaredFlux(s, coils, modes, target=target, include_f0=False)
    native = SquaredFlux(s, BiotSavart(coils), target=target)
    # include_f0 identity
    assert abs((full.J() - d2.J()) - native.J()) <= 1e-15 * abs(full.J())
    gfull, gd2, gn = full.dJ(), d2.dJ(), native.dJ()
    assert np.max(np.abs((gfull - gd2) - gn)) <= 1e-13 * np.max(np.abs(gfull))
    # repeated .x updates: x1 -> x2 -> x1 is bit-identical
    x1 = full.x.copy()
    J1, G1 = full.J(), full.dJ()
    full.x = x1 + 1e-3 * np.random.default_rng(5).standard_normal(x1.shape)
    J2 = full.J()
    assert J2 != J1
    full.x = x1
    assert full.J() == J1 and np.array_equal(full.dJ(), G1)
    # fixed dof: no entry; unfixing gives the full vector whose free entries are identical
    nfree = len(x1)
    coils[0].current.unfix_all()
    Gu = full.dJ()
    assert len(Gu) == nfree + 1
    names_u = list(full.dof_names)
    coils[0].current.fix_all()
    names_f = list(full.dof_names)
    pos = [names_u.index(n) for n in names_f]
    assert np.array_equal(Gu[pos], G1)


def test_realized_modes_per_level_dict_identity():
    base, coils, s, target = setup(nq=32, nphi=8, ntheta=6)
    qp = base[0].quadpoints
    smp = mc.make_sampler(qp, 1e-3, L, 1, "simsopt_ldl")
    one = kl.realized_modes(smp, NFP, True, NBASE)
    two = kl.realized_modes({"systematic": smp, "statistical": smp}, NFP, True, NBASE)
    for a in ("lam", "phi", "dphi", "direction", "level", "owner"):
        assert np.array_equal(getattr(one, a), getattr(two, a)), a
    other = kl.realized_modes({"systematic": mc.make_sampler(qp, 2e-3, 0.25, 1, "simsopt_ldl"), "statistical": smp},
                              NFP, True, NBASE)
    st = other.level == kl.LEVELS.index("statistical")
    assert np.array_equal(other.lam[st], one.lam[one.level == kl.LEVELS.index("statistical")])
    assert not np.array_equal(other.lam[~st][:3], one.lam[one.level == kl.LEVELS.index("systematic")][:3])
    assert set(two.info["per_level"]) == {"systematic", "statistical"}


def test_grad_batch_size_default_is_grid_aware():
    assert rsp.grad_batch_size(1024, 360) == 6           # 2.5e6 / (1024 * 360)
    assert rsp.grad_batch_size(16384, 360) == 1          # 256x64: budget 1.56e5, floor 1
    assert rsp.grad_batch_size(1024, 360, budget=1.0e7) == 27   # explicit budgets unchanged
