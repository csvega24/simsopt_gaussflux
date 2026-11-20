"""Unit 4.3 tests (records/phase4_3_proposal_opus.md §11 and §16; gate G4.3). Fast tests on the brief's example case; the slow
clauses run as gate scripts in phase4_3/. Kernel: SIMSOPT exp(-d^2/l^2) periodized on [0,1); sigma in metres per component per level.
Written 2026-11-19 by Claude Opus 5.5 (developer); run first on the unchanged code (the record says which fail there)."""

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent

from simsopt.field import Current, coils_via_symmetries  # noqa: E402
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves  # noqa: E402

from gaussflux import kl_modes as kl  # noqa: E402
from gaussflux import mc_reference as mc  # noqa: E402
from gaussflux.gaussian_flux import GaussianSquaredFlux  # noqa: E402

NFP, NBASE = 2, 4
DEFS = ("quadratic flux", "normalized", "local")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def example(nq=75, last_nq=None, nfp=NFP, stellsym=True):
    """The brief's example: 4 base coils, nfp 2, stellsym, order 5, geometry seed 7, 32x16 full-torus surface."""
    base = create_equally_spaced_curves(NBASE, nfp, stellsym=stellsym, R0=1.0, R1=0.5, order=5, numquadpoints=nq)
    if last_nq is not None:
        base[-1] = create_equally_spaced_curves(NBASE, nfp, stellsym=stellsym, R0=1.0, R1=0.5, order=5, numquadpoints=last_nq)[-1]
    rng = np.random.default_rng(7)
    for c in base:
        c.x = c.x + 0.02 * rng.standard_normal(c.x.shape)
    currents = [Current(1.0) * 1e5 for _ in range(NBASE)]
    currents[0].fix_all()
    coils = coils_via_symmetries(base, currents, nfp, stellsym)
    s = SurfaceRZFourier(nfp=NFP, stellsym=True, mpol=2, ntor=2, quadpoints_phi=np.linspace(0, 1, 32, endpoint=False),
                         quadpoints_theta=np.linspace(0, 1, 16, endpoint=False))
    s.set_rc(0, 0, 1.0); s.set_rc(1, 0, 0.18); s.set_zs(1, 0, 0.18); s.set_rc(1, 1, 0.03); s.set_zs(1, 1, -0.03)
    return base, coils, s


def target_of(s):
    return 2e-3 * np.cos(2 * np.pi * np.asarray(s.quadpoints_theta))[None, :] * np.ones((len(s.quadpoints_phi), 1))


def two_call(s, coils, qp, sigma, l, levels, definition, target, builder="kernel_exact", rel_cut=1e-10):
    if builder == "kernel_exact":
        modes = kl.realized_modes(mc.make_sampler(qp, sigma, l, 1, "eigh"), NFP, True, NBASE, levels=levels, rel_cut=rel_cut)
    else:
        modes = kl.fourier_modes(qp, NFP, True, NBASE, sigma, l, levels=levels)
    return GaussianSquaredFlux(s, coils, modes, definition=definition, target=target)


# ---------------------------------------------------------------------------------------------------------------- clause (a3), R1
# ---------------------------------------------------------------------------------------------------------------- R2 (a)
# ---------------------------------------------------------------------------------------------------------------- clause (b)
@pytest.mark.parametrize("builder", ["kernel_exact", "fourier"])
@pytest.mark.parametrize("definition", DEFS)
def test_b2_constructor_equals_two_call(builder, definition):
    base, coils, s = example()
    qp = np.asarray(base[0].quadpoints)
    tgt = None if definition == "quadratic flux" else target_of(s)
    A = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, levels="both", definition=definition, target=tgt, builder=builder)
    B = two_call(s, coils, qp, 3e-3, 0.2, kl.LEVELS, definition, tgt, builder)
    assert A.J() == B.J()
    assert np.array_equal(np.asarray(A.dJ()), np.asarray(B.dJ()))


@pytest.mark.parametrize("definition", DEFS)
def test_b3_builders_agree(definition):
    base, coils, s = example()
    tgt = None if definition == "quadratic flux" else target_of(s)
    A = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, definition=definition, target=tgt, builder="kernel_exact")
    B = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, definition=definition, target=tgt, builder="fourier")
    assert abs(A.delta2() / B.delta2() - 1) <= 1e-10
    ga, gb = np.asarray(A.dJ()), np.asarray(B.dJ())
    assert np.max(np.abs(ga - gb)) <= 1e-10 * np.max(np.abs(gb))


def block_err(C, K, n, sigma, l):
    var = sigma ** 2 * sum(math.exp(-(j / l) ** 2) for j in range(-5, 6))
    sc = max(1.0, 2 * math.pi * kl.fourier_cutoff(l, 1e-12))
    return max(float(np.max(np.abs(C[a * n:(a + 1) * n, b * n:(b + 1) * n] - K[a * n:(a + 1) * n, b * n:(b + 1) * n]))) / var / sc ** (a + b)
               for a in range(2) for b in range(2))


@pytest.mark.parametrize("factor", ["simsopt_ldl", "eigh"])
@pytest.mark.parametrize("n_derivs", [1, 2])
def test_b5_sampler_modes(factor, n_derivs):
    """(b5): with a sampler the mode set is realized_modes of it, bit for bit; the rebuilt covariance against the sampler's block (b-2)."""
    base, coils, s = example()
    qp = np.asarray(base[0].quadpoints)
    smp = mc.make_sampler(qp, 0.01, 0.2, n_derivs, factor)
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, levels="both", sampler=smp, rel_cut=1e-13)
    ref = kl.realized_modes(smp, NFP, True, NBASE, rel_cut=1e-13)
    for name in ("lam", "base_psi", "base_dpsi", "direction", "level", "owner"):
        assert np.array_equal(getattr(G.modes, name), getattr(ref, name)), name
    n = len(qp)
    # R2 (b) as restated (records/phase4_3_review_fable.md §9, (b-2)): the rebuilt covariance against the sampler's block truncated at the
    # same rel_cut by an eigen-decomposition written here, <= 1e-12 of the block's largest entry
    stat = ref.subset(np.where(ref.level == 1)[0])
    R = stat.reconstruct(0, 0)
    block = smp.L[:2 * n] @ smp.L[:2 * n].T if n_derivs == 2 else smp.L @ smp.L.T
    w, V = np.linalg.eigh(block)
    keep = w > 1e-13 * w.max()
    Ct = (V[:, keep] * w[keep]) @ V[:, keep].T
    for d in range(3):
        C = R[:, :, d, :, :, d].reshape(2 * n, 2 * n)
        assert np.max(np.abs(C - Ct)) <= 1e-12 * np.max(np.abs(block))


def test_b6_rms_formula():
    base, coils, s = example()
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, {"systematic": 2e-3, "statistical": 4e-3},
                                             {"systematic": 0.3, "statistical": 0.15}, levels="both")
    em = G.error_model()
    f = lambda sg, l: sg * math.sqrt(sum(math.exp(-(j / l) ** 2) for j in range(-5, 6)))
    sys_, stat = f(2e-3, 0.3), f(4e-3, 0.15)
    assert abs(em["rms_formula_m"]["systematic"] / sys_ - 1) <= 1e-12
    assert abs(em["rms_formula_m"]["statistical"] / stat - 1) <= 1e-12
    assert abs(em["rms_formula_m"]["combined"] / math.hypot(sys_, stat) - 1) <= 1e-12
    assert "rms displacement" in str(G)


# ---------------------------------------------------------------------------------------------------------------- clause (c), guards
def test_c_swapped_coils_refused():
    base, coils, s = example()
    qp = np.asarray(base[0].quadpoints)
    modes = kl.realized_modes(mc.make_sampler(qp, 3e-3, 0.2, 1, "eigh"), NFP, True, NBASE, rel_cut=1e-10)
    sw = list(coils)
    sw[3], sw[4] = sw[4], sw[3]
    with pytest.raises(ValueError, match="symmetry image"):
        GaussianSquaredFlux(s, sw, modes)


def test_c_other_symmetry_refused():
    base, coils, s = example()
    qp = np.asarray(base[0].quadpoints)
    modes = kl.realized_modes(mc.make_sampler(qp, 3e-3, 0.2, 1, "eigh"), 4, False, NBASE, rel_cut=1e-10)
    assert modes.ncoils == len(coils)
    with pytest.raises(ValueError, match="symmetr"):
        GaussianSquaredFlux(s, coils, modes)


@pytest.mark.parametrize("call", ["J", "dJ", "f0", "delta2"])
def test_c_surface_change_refused(call):
    base, coils, s = example()
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2)
    G.J()
    s.set_rc(1, 0, 0.20); s.set_zs(1, 0, 0.20)
    with pytest.raises(RuntimeError, match="surface changed"):
        getattr(G, call)()


@pytest.mark.parametrize("sigma,l,match", [(3e-3, -0.2, "correlation length"), (3e-3, 0.0, "correlation length"),
                                           (3e-3, float("nan"), "correlation length"), (-3e-3, 0.2, "sigma"),
                                           (float("nan"), 0.2, "sigma")])
def test_c_numbers_refused(sigma, l, match):
    base, coils, s = example()
    with pytest.raises(ValueError, match=match):
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, sigma, l)
    with pytest.raises(ValueError, match=match):
        kl.fourier_modes(np.asarray(base[0].quadpoints), NFP, True, NBASE, sigma, l)


@pytest.mark.parametrize("levels,sigma", [("", 3e-3), ("random", 3e-3), (("systematic", "systematic"), 3e-3),
                                          ("both", {"systematic": 3e-3})])
def test_c_levels_refused(levels, sigma):
    base, coils, s = example()
    with pytest.raises(ValueError, match="level"):
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, sigma, 0.2, levels=levels)


def test_c_sigma_zero_valid():
    base, coils, s = example()
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 0.0, 0.2)
    assert G.delta2() == 0.0


def test_c_unequal_points_refused():
    base, coils, s = example(last_nq=90)
    with pytest.raises(ValueError, match="points"):
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2)


def test_c_modes_for_other_points_refused():
    base, coils, s = example()
    modes = kl.realized_modes(mc.make_sampler(np.linspace(0, 1, 60, endpoint=False), 3e-3, 0.2, 1, "eigh"), NFP, True, NBASE, rel_cut=1e-10)
    with pytest.raises(ValueError, match="points"):
        GaussianSquaredFlux(s, coils, modes)


def test_c_existing_refusals():
    base, coils, s = example()
    qp = np.asarray(base[0].quadpoints)
    modes = kl.realized_modes(mc.make_sampler(qp, 3e-3, 0.2, 1, "eigh"), NFP, True, NBASE, rel_cut=1e-10)
    with pytest.raises(ValueError, match="number of coils"):
        GaussianSquaredFlux(s, coils[:-1], modes)
    # planner's RR2 (records/phase4_3_review_fable.md §10.4): before RR2 SIMSOPT refused these at J() ("Btarget has wrong shape",
    # RuntimeError) while delta2() used them; since RR2 the constructor refuses them, naming both shapes
    for shape in ((16, 32), (32, 8), (512,)):
        with pytest.raises(ValueError, match=r"target has shape"):
            GaussianSquaredFlux(s, coils, modes, definition="normalized", target=np.zeros(shape))


def test_target_held_as_copy():
    base, coils, s = example()
    tgt = target_of(s)
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, definition="normalized", target=tgt)
    j = G.J()
    tgt *= 2.0
    assert G.J() == j
    assert not np.shares_memory(G.target, tgt) and not np.shares_memory(G.native.target, tgt)


def test_resolution_warning_and_ratio():
    base, coils, s = example()
    with pytest.warns(UserWarning, match="harmonic"):
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.04, builder="fourier")
    assert "kernel cutoff" in str(G)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2)
    em = G.error_model()
    assert 0 < em["ratio_rms_to_distance"] < 1 and "coil-surface distance" in str(G)
    with pytest.warns(UserWarning, match="metres"):
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3.0, 0.2)


def test_check_coils_off_reported():
    base, coils, s = example()
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, check_coils=False)
    assert "coil list not checked" in str(G)


# ---------------------------------------------------------------------------------------------------------------- clause (d)
def test_draw_levels_equals_draw_two_level():
    qp = np.linspace(0, 1, 75, endpoint=False)
    smp = mc.make_sampler(qp, 3e-3, 0.2, 1, "eigh")
    a_sys, a_stat = mc.draw_two_level(smp, NBASE, 16, mc.sample_rng(5, 2, 8))
    b_sys, b_stat = mc.draw_levels({"systematic": smp, "statistical": smp}, NBASE, 16, mc.sample_rng(5, 2, 8))
    assert np.array_equal(mc.sample_arrays(a_sys), mc.sample_arrays(b_sys))
    assert np.array_equal(mc.sample_arrays(a_stat), mc.sample_arrays(b_stat))


def test_d1_plain_values_are_reference_values():
    base, coils, s = example()
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2)
    res = G.accuracy_check(n_pairs=3, seed=11, estimates=("plain",), identity=False)
    smp = G.draw_samplers()
    for k in range(3):
        sy, st = mc.draw_levels(smp, NBASE, len(coils), mc.sample_rng(11, k, 3))
        for sgn, key in ((1.0, "f_plus"), (-1.0, "f_minus")):
            pc = mc.perturbed_coils_levels(coils, base, NFP, True,
                                           mc.samples_from_arrays(smp["systematic"], mc.sample_arrays(sy), sgn),
                                           mc.samples_from_arrays(smp["statistical"], mc.sample_arrays(st), sgn))
            assert mc.flux_values(s, pc, None, ("quadratic flux",))["quadratic flux"] == res["plain"][key][k]
    assert res["identity"] == "not checked"


def test_d5_seed_and_state():
    base, coils, s = example()
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2)
    x0, j0, d0 = np.array(G.x), G.J(), np.asarray(G.dJ())
    r1 = G.accuracy_check(n_pairs=4, seed=3, identity=True)
    r2 = G.accuracy_check(n_pairs=4, seed=3, identity=True)
    assert r1["plain"]["mean"] == r2["plain"]["mean"] and r1["reduced"]["mean"] == r2["reduced"]["mean"] and r1["seed"] == 3
    assert np.array_equal(np.array(G.x), x0) and G.J() == j0 and np.array_equal(np.asarray(G.dJ()), d0)
    assert r1["identity"]["rel_diff"] <= 1e-10


# ---------------------------------------------------------------------------------------------------------------- clause (e4)
# ---------------------------------------------------------------------------------------------------------------- planner's ruling on R2 (b) (iii)
def test_sampler_report_and_warnings():
    """With a sampler given, the summary describes the realized law beside the kernel; two warnings (not PSD to precision; realized
    value rms outside [1/2, 2] x the kernel's). records/phase4_3_review_fable.md §9."""
    base, coils, s = example(nq=128)
    qp = np.asarray(base[0].quadpoints)
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # SIMSOPT's GaussianSampler, 128 points, l 0.2, n_derivs 2: no sampler warning
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, sampler=mc.make_sampler(qp, 0.01, 0.2, 2, "simsopt_ldl"))
    r = G.error_model()["sampler_report"]["systematic"]
    assert 0.5 * r["kernel_rms_value_m"] <= r["realized_rms_value_m"][0] and r["realized_rms_value_m"][1] <= 2 * r["kernel_rms_value_m"]
    assert "block eigenvalues" in str(G)
    bad = mc.make_sampler(qp, 0.01, 0.2, 2, "eigh")
    bad.L = 3.0 * bad.L   # a sampler whose draws have three times the kernel's rms
    with pytest.warns(UserWarning, match="does not realize its kernel"):
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, sampler=bad)
    E = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, sampler=mc.make_sampler(qp, 0.01, 0.2, 1, "eigh"))
    e = E.error_model()["sampler_report"]["systematic"]
    for key, kern in (("realized_rms_value_m", "kernel_rms_value_m"), ("realized_rms_derivative_m", "kernel_rms_derivative_m")):
        assert abs(e[key][0] / e[kern] - 1) < 1e-9 and abs(e[key][1] / e[kern] - 1) < 1e-9


# ---------------------------------------------------------------------------------------------------------------- planner's RR2, RR3 (§10.4)
@pytest.mark.parametrize("shape", [(16, 32), (512,), (1, 512), (32, 8)])
@pytest.mark.parametrize("include_f0", [True, False])
def test_rr2_target_shape_refused(shape, include_f0):
    """A target with the surface grid's number of entries in another shape was used by delta2() (and by J, dJ with include_f0=False)."""
    base, coils, s = example()
    tgt = np.random.default_rng(3).standard_normal(int(np.prod(shape))).reshape(shape)
    import re
    with pytest.raises(ValueError, match=r"target has shape %s.*surface grid \(32, 16\)" % re.escape(str(tuple(shape)))):
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, definition="normalized", target=tgt,
                                                 include_f0=include_f0)
        G.delta2()                  # not reached after RR2; on the code before RR2 it returns a value


def test_rr2_right_shape_accepted():
    base, coils, s = example()
    tgt = target_of(s)
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, definition="normalized", target=tgt.tolist())
    assert G.target.shape == (32, 16) and np.array_equal(G.target, tgt)


@pytest.mark.parametrize("nfp,stellsym,nbase", [(NFP, True, NBASE), (NFP, False, 2 * NBASE), (1, False, 4 * NBASE)])
def test_rr3_stated_symmetry_reported(nfp, stellsym, nbase):
    """The stated (nfp, stellsym) decides which coils share a systematic error; error_model() and the summary print it."""
    base, coils, s = example()
    G = GaussianSquaredFlux.from_error_model(s, coils, nfp, stellsym, 3e-3, 0.2)
    em = G.error_model()
    assert (em["nfp"], em["stellsym"], em["base_coils"], em["coils"]) == (nfp, stellsym, nbase, len(coils))
    assert (em["surface_nfp"], em["surface_stellsym"]) == (NFP, True)
    text = str(G)
    assert f"stated symmetry: nfp {nfp}, stellsym {stellsym}; {nbase} base coils of {len(coils)}" in text
    assert ("differs from the surface's" in text) == ((nfp, stellsym) != (NFP, True))


def test_rn3_report_error_printed():
    from gaussflux.gaussian_flux import _summary_text
    em = {"definition": "quadratic flux", "levels": None, "note": "x", "modes": 0, "path": "general", "certificate_eligible": False,
          "max_M0_dev": 0.0, "max_M1_dev": 0.0, "coil_list_checked": True, "report_error": "KeyError: 'y'"}
    assert "report failed: KeyError: 'y'" in _summary_text(em)


# ================================================================================================ Amendment A (D21), items A1-A8
# records/phase4_3_proposal_opus.md §17-18; review records/phase4_3_review_fable.md §15-16; PLAN.md reading (ix): 1e-10 and 1e-6.
# The law residual r: per level, the retained modes' covariance on one coil against the stated law, reading (v)'s scales. The test
# computes the eleven-image kernel from its closed form here, independently of the package.

def _cutoff_true(l, tail=1e-12):
    m = np.arange(int(max(200, math.ceil(8.0 / (math.pi * l)))) + 1)
    c = np.exp(-np.pi ** 2 * m ** 2 * l ** 2)
    tails = 2 * (np.cumsum(c[::-1])[::-1] - c) / (c[0] + 2 * c[1:].sum())
    return int(np.argmax(tails <= tail))


def _kernel11(t, l, sigma=1.0):
    d = t[:, None] - t[None, :]
    k = np.zeros_like(d); k1 = np.zeros_like(d); k2 = np.zeros_like(d)
    for j in range(-5, 6):
        x = d + j
        e = np.exp(-x ** 2 / l ** 2)
        k += e; k1 += -2 * x / l ** 2 * e; k2 += (4 * x ** 2 / l ** 4 - 2 / l ** 2) * e
    n = len(t)
    K = np.zeros((2 * n, 2 * n))
    K[:n, :n], K[:n, n:], K[n:, :n], K[n:, n:] = k, -k1, k1, -k2
    return sigma ** 2 * K


def _fourier_full(t, l, sigma=1.0, M=200):
    d = t[:, None] - t[None, :]
    m = np.arange(1, M + 1)
    c0, c = l * np.sqrt(np.pi), l * np.sqrt(np.pi) * np.exp(-np.pi ** 2 * m ** 2 * l ** 2)
    w = 2 * np.pi * m
    arg = d[:, :, None] * w
    k = c0 + 2 * np.sum(c * np.cos(arg), axis=2)
    k1 = 2 * np.sum(-c * w * np.sin(arg), axis=2)
    k2 = 2 * np.sum(-c * w ** 2 * np.cos(arg), axis=2)
    n = len(t)
    K = np.zeros((2 * n, 2 * n))
    K[:n, :n], K[:n, n:], K[n:, :n], K[n:, n:] = k, -k1, k1, -k2
    return sigma ** 2 * K


def _r(C, K, n, l, sigma=1.0):
    var = sigma ** 2 * sum(math.exp(-(j / l) ** 2) for j in range(-5, 6))
    s = max(1.0, 2 * math.pi * _cutoff_true(l))
    E = np.abs(C - K)
    return max(E[:n, :n].max() / var, max(E[:n, n:].max(), E[n:, :n].max()) / (var * s), E[n:, n:].max() / (var * s * s))


def _law(modes, lev_id):
    sel = np.where((np.asarray(modes.level) == lev_id) & (np.asarray(modes.owner) == 0) & (np.asarray(modes.direction) == 0))[0]
    F = np.concatenate([np.asarray(modes.phi)[sel], np.asarray(modes.dphi)[sel]], axis=1)
    return (F.T * np.asarray(modes.lam)[sel]) @ F


def _law_warnings(rec):
    return [str(w.message) for w in rec if "law residual" in str(w.message)]


def test_a1_material_truncation_marked():
    base, coils, s = example()
    with pytest.warns(UserWarning, match="law residual"):
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, 0.2, rel_cut=0.5)
    em = G.error_model()
    assert em["retained_is_stated"] is False and em["law_residual"]["systematic"]["max"] > 1e-6
    res = G.accuracy_check(n_pairs=128, seed=20431)
    assert res["retained_is_stated"] is False and set(res["laws"]) == {"plain", "reduced", "identity"}
    assert res["reduced"]["speaks_for"] == res["identity"]["speaks_for"] == "retained mode set only"
    p, r = res["plain"], res["reduced"]
    assert abs(p["mean"] - r["mean"]) / math.hypot(p["se"], r["se"]) > 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        D = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, 0.2)
    assert D.error_model()["retained_is_stated"] is True and D.error_model()["law_residual"]["systematic"]["max"] <= 1e-10


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, 1.0, -1.0, "x", None, True, "1e-10", b"1e-10"])   # RA1: numeric strings too
def test_a2_rel_cut_refused(bad):
    base, coils, s = example()
    with pytest.raises(ValueError, match=r"rel_cut must be a real number in \[1e-15, 1\)"):
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, rel_cut=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), 0.0, 1.0, -1.0, "x", None, True, "1e-10", b"1e-10"])   # RA1: numeric strings too
def test_a2_tail_refused(bad):
    base, coils, s = example()
    with pytest.raises(ValueError, match=r"tail must be a real number in \[1e-15, 1\)"):
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, builder="fourier", tail=bad)


def test_a2_boundaries():
    base, coils, s = example()
    GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, rel_cut=1e-15)
    GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, builder="fourier", tail=1e-15)
    with pytest.raises(ValueError, match="rel_cut"):
        kl.realized_modes(mc.make_sampler(np.asarray(base[0].quadpoints), 3e-3, 0.2, 1, "eigh"), NFP, True, NBASE, rel_cut=1.0)
    with pytest.raises(ValueError, match="tail"):
        kl.fourier_modes(np.asarray(base[0].quadpoints), NFP, True, NBASE, 3e-3, 0.2, tail=1.0)


def test_a2_no_mode_refused(monkeypatch):
    base, coils, s = example()
    monkeypatch.setattr(kl, "_realized_functions", lambda sampler, rel_cut: ([], {"retained": 0, "rel_cut": rel_cut,
                        "sigma": float(sampler.sigma), "l": float(sampler.length_scale), "truncation_residual_rel": 1.0}))
    with pytest.raises(ValueError, match="no mode"):
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2)


A3_LS = (0.001, 0.005, 0.0084, 0.05, 0.1, 0.2, 0.5, 1.0, 1.05, 1.2, 1.5, 2.0, 5.0)


@pytest.mark.parametrize("nq", [75, 128])
@pytest.mark.parametrize("builder", ["kernel_exact", "fourier"])
@pytest.mark.parametrize("l", A3_LS)
def test_a3_law_certified(l, builder, nq):
    base, coils, s = example(nq=nq)
    t = np.asarray(base[0].quadpoints)
    K = _kernel11(t, l)
    if builder == "kernel_exact":
        w, V = np.linalg.eigh(K)
        r_builder, short = _r((V * np.maximum(w, 0)) @ V.T, K, nq, l), False
    else:
        r_builder, short = _r(_fourier_full(t, l), K, nq, l), _cutoff_true(l) > 200
    if short or r_builder > 1e-6:
        with pytest.raises(ValueError, match="Fourier table|not a covariance|infinite periodization"):
            GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1.0, l, levels="systematic", builder=builder)
        return
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1.0, l, levels="systematic", builder=builder)
    r = _r(_law(G.modes, 0), K, nq, l)
    got = G.error_model()["law_residual"]["systematic"]["max"]
    assert got == pytest.approx(r, rel=1e-6, abs=1e-14)
    assert bool(_law_warnings(rec)) == (r > 1e-10)
    assert G.error_model()["retained_is_stated"] is bool(r <= 1e-6)


def test_a3_fourier_short_table_refused():
    base, coils, s = example()
    with pytest.raises(ValueError, match="Fourier table"):
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.001, builder="fourier")
    assert kl.fourier_cutoff(0.1) == 16 and kl.fourier_cutoff(0.2) == 8 and kl.fourier_cutoff(0.5) == 3 and kl.fourier_cutoff(1.0) == 1


def test_a3_small_l_cutoff_warns():
    # RA3: the tight case is asserted with room (its residual, ~1e-10, is the factorization's roundoff and moves with sigma)
    base, coils, s = example(nq=360)
    with pytest.warns(UserWarning, match=r"law residual.*rel_cut"):
        G0 = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, 0.005, levels="systematic")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, 0.005, levels="systematic", rel_cut=1e-15)
    assert not any("rel_cut" in m for m in _law_warnings(rec))
    r0 = G0.error_model()["law_residual"]["systematic"]["max"]
    assert G.error_model()["law_residual"]["systematic"]["max"] <= r0 / 100


@pytest.mark.parametrize("rel_cut", [1e-10, 1e-15])
def test_ra2_precision_not_cutoff(rel_cut):
    # RA2: at 128 points and l 0.005 every mode is kept at both cutoffs; a residual above 1e-10 is the factorization's precision
    base, coils, s = example(nq=128)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, 0.005, levels="systematic", rel_cut=rel_cut)
    msgs = _law_warnings(rec)
    assert G.error_model()["law_residual_cause"].get("systematic") != "cutoff"
    assert not any("rel_cut" in m for m in msgs)
    assert all("precision" in m for m in msgs)


@pytest.mark.parametrize("nq", [60, 128])
def test_ra2_refused_builder_not_offered(nq):
    # RA2: at l 0.001 the Fourier builder is refused (it needs M = 1605), so the warning must not offer it
    base, coils, s = example(nq=nq)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, 0.001, levels="systematic")
    msgs = _law_warnings(rec)
    assert msgs and all("precision" in m and "remedy:" not in m and "fourier" not in m for m in msgs)   # no remedy offered


@pytest.mark.parametrize("builder", ["kernel_exact", "fourier"])
@pytest.mark.parametrize("l", [1.5, 2.0, 3.0, 5.0])
def test_a3_large_l_refused(l, builder):
    base, coils, s = example()
    with pytest.raises(ValueError, match="not a covariance|infinite periodization"):
        GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, l, builder=builder)


def test_a4_law_frozen():
    base, coils, s = example()
    qp = np.asarray(base[0].quadpoints)
    m = kl.realized_modes(mc.make_sampler(qp, 3e-3, 0.2, 1, "eigh"), NFP, True, NBASE, rel_cut=1e-10)
    G = GaussianSquaredFlux(s, coils, m)
    d2 = G.delta2()
    with pytest.raises(ValueError):
        G.modes.lam[0] *= 4
    m.lam *= 4
    assert G.delta2() == d2
    smp = mc.make_sampler(qp, 1e-3, 0.2, 1, "eigh")
    H = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, sampler=smp)
    L0 = np.array(H.draw_samplers()["systematic"].L, copy=True)
    H.draw_samplers()["systematic"].L *= 2
    smp.L *= 2
    assert np.array_equal(H.draw_samplers()["systematic"].L, L0)


def test_a4_scaled_on_frozen():
    base, coils, s = example()
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2)
    Gs = GaussianSquaredFlux(s, coils, G.modes.scaled(1.44), "quadratic flux")
    assert Gs.delta2() == pytest.approx(1.44 * G.delta2(), rel=1e-12)


def test_a5_negative_delta2():
    base, coils, s = example()
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, definition="normalized", target=np.ones((32, 16)))
    assert G.delta2() < 0
    res = G.accuracy_check(n_pairs=2, seed=20432, identity=False)
    assert res["plain"]["se_rel_delta2"] > 0 and res["reduced"]["se_rel_delta2"] > 0
    assert np.sign(res["plain"]["rel_delta2"]) == np.sign(res["plain"]["mean"])


def test_a6_summary_current():
    base, coils, s = example()
    G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2)
    d0 = G.error_model()["min_coil_surface_distance_m"]
    base[0].x = 0.85 * base[0].x   # coil 0 and its images shrink toward the surface (a +0.2 change of dof 1 left the minimum unchanged)
    G2 = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2)
    em = G.error_model()
    assert em["min_coil_surface_distance_m"] == pytest.approx(G2.error_model()["min_coil_surface_distance_m"], rel=1e-12)
    assert em["min_coil_surface_distance_m_at_construction"] == d0 != em["min_coil_surface_distance_m"]
    base2, coils2, s2 = example()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F = GaussianSquaredFlux.from_error_model(s2, coils2, NFP, True, 3e-3, 0.2, builder="fourier", tail=0.1)
    r = F.error_model()["resolution"]["systematic"]
    assert (r["retained_cutoff_M"], r["reference_cutoff_M"]) == (2, 8)
    assert "retained M 2" in str(F)


def test_r6_omitted_delta2():
    base, coils, s = example()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, 0.2, rel_cut=0.5)
    res = G.accuracy_check(n_pairs=128, seed=20431, identity=False)
    om = res["omitted_delta2"]
    p, r = res["plain"], res["reduced"]
    assert abs((p["mean"] - r["mean"]) - om["absolute"]) <= 3 * math.hypot(p["se"], r["se"])
    assert om["tightest"] == {"rel_cut": 1e-15}
    smp = mc.make_sampler(np.asarray(base[0].quadpoints), 1e-3, 0.2, 1, "eigh")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, sampler=smp, rel_cut=0.5)
    assert S.accuracy_check(n_pairs=2, seed=1, identity=False)["omitted_delta2"] == "not available for a supplied sampler"


def test_r7_uncertified_is_none():
    base, coils, s = example()
    qp = np.asarray(base[0].quadpoints)
    m = kl.realized_modes(mc.make_sampler(qp, 3e-3, 0.2, 1, "eigh"), NFP, True, NBASE, rel_cut=1e-10)
    G = GaussianSquaredFlux(s, coils, m)
    assert G.error_model()["retained_is_stated"] is None and "law not certified" in str(G)
    res = G.accuracy_check(n_pairs=2, seed=0, identity=False, samplers=GaussianSquaredFlux.from_error_model(
        s, coils, NFP, True, 3e-3, 0.2).draw_samplers())
    assert res["retained_is_stated"] is None and res["reduced"]["speaks_for"] == "law not certified"
    H = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2)
    assert H.accuracy_check(n_pairs=2, seed=0, identity=False, samplers=H.draw_samplers())["retained_is_stated"] is None


def test_rb1_large_l_kernel_named():
    # RB1: between l 1.1 and 1.5 the eleven-image kernel has a negative eigenvalue; the kernel_exact warning names that cause
    base, coils, s = example()
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 1.2, levels="systematic")
    msgs = _law_warnings(rec)
    assert msgs and all("not a covariance" in m and "precision" not in m for m in msgs)
    assert G.error_model()["law_residual_cause"]["systematic"] == "kernel"


# ------------------------------------------------------------------------------------- RC1-RC5 (records/phase4_3_review_fable.md §20.4)
from decimal import Decimal  # noqa: E402
from fractions import Fraction  # noqa: E402
import re  # noqa: E402

USAGE = PROJECT / "docs" / "USAGE.md"


def _quiet(**kw):
    base, coils, s = example(nq=kw.pop("nq", 75))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return GaussianSquaredFlux.from_error_model(s, coils, NFP, True, **kw), (base, coils, s)


@pytest.mark.parametrize("l", [0.2, 1.2])
def test_rc1_labels_name_what_was_drawn(l):
    # RC1: covariance agreement within 1e-6 is not equality of laws; the reduced estimate and the identity always name the retained
    # mode set, the plain estimate names the sampler's factor (at l 1.2 the kernel's nearest covariance, the kernel not being one)
    G, _ = _quiet(sigma=3e-3, correlation_length=l, levels="systematic")
    res = G.accuracy_check(n_pairs=2, seed=0)
    assert res["retained_is_stated"] is True
    for key in ("reduced", "identity"):
        sf = res[key]["speaks_for"]
        assert sf.startswith("retained mode set") and "within 1e-6" in sf and sf != "stated law"
    plain = res["laws"]["plain"]
    assert "factor" in plain and not plain.startswith("the stated law") and "nearest covariance" in plain
    assert ("not a covariance" in plain) == (l == 1.2)
    om = res["omitted_delta2"]
    assert isinstance(om, str) and "is the stated law" not in om and "not a bound" in om
    assert "= stated law" not in str(G) and "retained law = " not in str(G)


def test_rc1_material_warning_names_the_factor():
    G, _ = _quiet(sigma=1e-3, correlation_length=0.2, rel_cut=0.5)
    res = G.accuracy_check(n_pairs=2, seed=0, identity=False)
    assert res["reduced"]["speaks_for"] == "retained mode set only"
    assert "factor" in res["warning"] and "speaks for the stated law" not in res["warning"]


def test_rc1_page_and_docstring():
    page = USAGE.read_text().split("\n## Change log")[0]   # the body; the change log may quote removed wording
    doc = GaussianSquaredFlux.accuracy_check.__doc__
    for text in (page, doc):
        assert "same expectation" not in text
        assert "not a bound on Delta2" in text or "not a bound on Δ₂" in text
    assert "tighter cutoff" in page


RC2_ACCEPT = [1e-10, np.float64(1e-10), np.float32(1e-10), np.longdouble(1e-10), np.array(1e-10), np.array(1e-10, dtype=np.float32),
              Decimal("1e-10"), Fraction(1, 10 ** 10)]
RC2_REFUSE = [np.complex128(.5 + 1j), np.complex64(.5 + 1j), np.complex128(1e-10), complex(1e-10, 0), np.array(1e-10 + 0j),
              np.array("1e-10"), np.array(b"1e-10"), memoryview(b"1e-10"), np.str_("1e-10"), np.bytes_(b"1e-10"), bytearray(b"1e-10"),
              np.array(True), np.bool_(True), True, np.array([1e-10]), np.array(1e-10, dtype=object), [1e-10], (1e-10,), None]


@pytest.mark.parametrize("name", ["rel_cut", "tail"])
@pytest.mark.parametrize("v", RC2_ACCEPT, ids=repr)
def test_rc2_real_scalars_accepted(v, name):
    got = kl.check_control(name, v)
    assert type(got) is float and got == pytest.approx(1e-10, rel=1e-7)


@pytest.mark.parametrize("name", ["rel_cut", "tail"])
@pytest.mark.parametrize("v", RC2_REFUSE, ids=repr)
def test_rc2_refused_before_conversion(v, name):
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # no ComplexWarning: decided by type, before any conversion
        with pytest.raises(ValueError, match=rf"{name} must be a real number in \[1e-15, 1\)"):
            kl.check_control(name, v)


@pytest.mark.parametrize("route", ["exact/rel_cut", "exact/tail", "fourier/tail", "fourier/rel_cut", "sampler/rel_cut",
                                   "kl_realized/rel_cut", "kl_fourier/tail"])
@pytest.mark.parametrize("v", RC2_REFUSE, ids=repr)
def test_rc2_routes_refuse(v, route):
    base, coils, s = example()
    qp = np.asarray(base[0].quadpoints)
    how, name = route.split("/")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=rf"{name} must be a real number"):
            if how in ("exact", "fourier"):
                GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 3e-3, 0.2, builder="kernel_exact" if how == "exact" else "fourier",
                                                     **{name: v})
            elif how == "sampler":
                GaussianSquaredFlux.from_error_model(s, coils, NFP, True, sampler=mc.make_sampler(qp, 3e-3, 0.2, 1, "eigh"), rel_cut=v)
            elif how == "kl_realized":
                kl.realized_modes(mc.make_sampler(qp, 3e-3, 0.2, 1, "eigh"), NFP, True, NBASE, rel_cut=v)
            else:
                kl.fourier_modes(qp, NFP, True, NBASE, 3e-3, 0.2, tail=v)


@pytest.mark.parametrize("v", [np.array(1e-10), Decimal("1e-10"), Fraction(1, 10 ** 10), np.float32(1e-10)], ids=repr)
def test_rc2_constructor_accepts_real_scalars(v):
    G, _ = _quiet(sigma=3e-3, correlation_length=0.2, rel_cut=v)
    F, _ = _quiet(sigma=3e-3, correlation_length=0.2, builder="fourier", tail=v)
    assert type(G.error_model()["rel_cut"]) is float and type(F.error_model()["tail"]) is float


def _messages(builder, l, knob, value, nq=75):
    """(refused, messages) of one construction: the refusal's text, or the law warnings."""
    base, coils, s = example(nq=nq)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, l, levels="systematic", builder=builder, **{knob: value})
        except ValueError as err:
            return True, [str(err)]
    return False, _law_warnings(rec)


def _builds(builder, l, **kw):
    base, coils, s = example()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, l, levels="systematic", builder=builder, **kw)
        return True
    except ValueError:
        return False


RC3_LS = (0.001, 0.003, 0.005, 0.006, 0.0084, 0.02, 0.2, 1.0, 1.2, 1.45)


@pytest.mark.parametrize("value", [0.1, "default", 1e-15])
@pytest.mark.parametrize("builder", ["kernel_exact", "fourier"])
@pytest.mark.parametrize("l", RC3_LS)
def test_rc3_offered_values_accepted(l, builder, value):
    # RC3(b): every control value and builder a warning or refusal offers (or says it keeps) is one the constructor accepts
    knob = "rel_cut" if builder == "kernel_exact" else "tail"
    v = (1e-10 if knob == "rel_cut" else 1e-12) if value == "default" else value
    refused, msgs = _messages(builder, l, knob, v)
    for m in msgs:
        for k, num in re.findall(r"\b(rel_cut|tail) ([0-9][0-9.]*(?:e[+-]?[0-9]+)?)\b", m):
            if k == knob and float(num) == v:
                continue   # the caller's own value
            other = builder if k == knob else ("fourier" if k == "tail" else "kernel_exact")
            assert _builds(other, l, **{k: float(num)}), f"offered {k} {num} at l {l} ({builder}) is refused: {m}"
        for b in re.findall(r"builder='(\w+)'", m):
            assert _builds(b, l), f"offered builder {b!r} at l {l} is refused: {m}"


def test_rc3_fourier_table_named():
    # RC3(a): at l 0.005 with tail 0.1 the whole table (m <= 200) is short of the tail; the refusal names the table, not the periodization
    refused, msgs = _messages("fourier", 0.005, "tail", 0.1)
    assert refused and "m = 200" in msgs[0] and "infinite periodization" not in msgs[0]
    refused, msgs = _messages("fourier", 0.006, "tail", 0.1)
    assert not refused and msgs and all("whole table (m <= 200)" in m and "tail 1e-15" not in m for m in msgs)


def _zero_sigma_check():
    G, _ = _quiet(sigma=0.0, correlation_length=0.2)
    return G, G.accuracy_check(n_pairs=2, seed=0)


def test_rc4_zero_delta2_printing(capsys):
    G, acc = _zero_sigma_check()
    assert acc["plain"]["rel_delta2"] is None and acc["reduced"]["rel_delta2"] is None
    blocks = re.findall(r"```python\n(.*?)```", USAGE.read_text(), flags=re.S)
    block = next(b for b in blocks if "accuracy_check" in b)
    exec(block.replace("Jf.accuracy_check(n_pairs=8, seed=0)", "acc"), {"Jf": G, "acc": acc})
    assert capsys.readouterr().out.count("undefined (Delta2 = 0)") == 2
    ex = load_by_path("stage_two_gaussian_example_rc4", PROJECT / "examples" / "stage_two_gaussian_example.py")
    lines = ex.accuracy_lines(acc)
    assert sum("undefined (Delta2 = 0)" in x for x in lines) == 2 and all("R = " in x for x in lines)


def test_rc5_explicit_xi_is_the_callers():
    G, _ = _quiet(sigma=1e-3, correlation_length=0.2)
    res = G.accuracy_check(n_pairs=2, seed=0, estimates=("reduced",), identity=False, xi=np.zeros((2, G.modes.lam.size)))
    sf = res["reduced"]["speaks_for"]
    assert res["reduced"]["draws"] == "explicit" and "caller" in sf and "standard normal" in sf
    assert "caller" in GaussianSquaredFlux.accuracy_check.__doc__ and "caller" in USAGE.read_text().split("## 7.")[1].split("## 8.")[0]


# ------------------------------------------------------------------------------------- RD1 (records/phase4_3_review_fable.md §21.5)
@pytest.mark.parametrize("l", [0.006, 0.0084])
def test_rd1_fourier_omitted_at_tightest_accepted_tail(l):
    # RD1: with the Fourier builder at short l, omitted_delta2 compares with the smallest tail the constructor accepts there
    # (tail 1e-15 needs M > 200 and is refused); the figure equals the difference of Delta2 of the two constructions
    base, coils, s = example()
    tau = kl.fourier_whole_table_tail(l)
    assert tau is not None and tau > 1e-15
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        G = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, l, levels="systematic", builder="fourier", tail=0.1)
        T = GaussianSquaredFlux.from_error_model(s, coils, NFP, True, 1e-3, l, levels="systematic", builder="fourier", tail=tau)
    res = G.accuracy_check(n_pairs=2, seed=0, estimates=(), identity=False)
    om = res["omitted_delta2"]
    assert isinstance(om, dict), om
    assert om["tightest"] == {"tail": tau}
    assert om["absolute"] == T.delta2() - G.delta2() and om["delta2_tightest"] == T.delta2()


# ------------------------------------------------------------------------------- RE1-RE3 (records/phase4_3_review_fable.md §23.4)
def _page_section(n):
    body = USAGE.read_text().split("\n## Change log")[0]
    return body.split(f"\n## {n}. ")[1].split("\n## ")[0]


def test_re1_page_no_one_law():
    # RE1: no statement that the two methods use one law; the same requested kernel through separate factors
    s5 = _page_section(5)   # the public page has no section 9 (the driver); its assertions are dropped
    for s in (s5,):
        flat_text = " ".join(s.split()).replace("**", "")
        assert "one law" not in flat_text
        assert "same requested kernel" in flat_text
    assert "rel_cut" in s5


def test_re2_page_eigenvalue_kernel_exact_only():
    # RE2: the kernel's eigenvalue is reported for the kernel-exact builder only
    t7 = " ".join(_page_section(7).split())
    assert "kernel_min_eigenvalue" in t7 and "kernel-exact builder" in t7 and "Fourier builder" in t7


def test_re3_page_per_level_tail():
    # RE3: a warning's tail is that level's own limit; mixed l: the largest of the levels' values, omitted_delta2["tightest"]
    t4 = " ".join(_page_section(4).split())
    assert "level's own limit" in t4 and "largest of the levels'" in t4 and "tightest" in t4


# ------------------------------------------------------------------------------- RF1-RF2 (records/phase4_3_review_fable.md §24)
def test_rf1_page_measured_values_only():
    # RF1: §5 gives the factors' covariance difference at the three measured l, both blocks, not an unmeasured range
    s5 = _page_section(5)
    t5 = " ".join(s5.split())
    assert "l ≤ 1" not in t5 and "about 1e-11" not in t5
    for v in ("3.2e-11", "1.8e-12", "1.4e-11", "2.1e-10"):
        assert v in t5
    assert "`rel_cut`. the" not in t5
    assert max(len(line) for line in s5.splitlines()) <= 140   # the page's existing lines reach 134; RF1's line had ~600


def test_rf2_page_small_remainder_names_the_law():
    # RF2: a small |R/Delta2| speaks for the law that estimate is drawn from; for the reduced estimate it says nothing about the
    # truncation (pointer to "Does the truncation matter at this design?")
    t7 = " ".join(_page_section(7).split())
    assert "the expansion represents the expectation at this design and σ." not in t7
    assert "under the law that estimate is drawn from" in t7
    assert "says nothing about what the truncation leaves out" in t7 and "Does the truncation matter at this design?" in t7
