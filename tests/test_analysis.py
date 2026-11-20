"""Phase 2 tests: rigid correction map, projection null tests, decomposition, Hessian
assembly and the Schur prediction (records/phase2_proposal_opus.md §3, §9)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parent.parent

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from gaussflux import analysis as an  # noqa: E402
from gaussflux import kl_modes as kl  # noqa: E402
from gaussflux import response as rsp  # noqa: E402

from test_gaussian_flux import NBASE, NFP, L, _oracle_case, setup  # noqa: E402


def _case(nphi=16, ntheta=12, subset=True):
    base, coils, s, target = setup(nq=32, nphi=nphi, ntheta=ntheta)
    g, gd, cur = rsp.coil_arrays(coils)
    X, nhat, absn = rsp.surface_geometry(s)
    modes = kl.fourier_modes(base[0].quadpoints, NFP, True, NBASE, 5e-3, L)
    if subset:
        modes = modes.subset(np.arange(0, modes.size, max(1, modes.size // 12))[:12])
    return g, gd, cur, X, nhat, absn, target.reshape(-1), modes


def _near_optimal_case():
    """Target = n.B of the coils after a known small rigid motion, plus 0.3 x the Phase 1
    check pattern (not representable by rigid motions): q0 is small, nonzero and the
    residual u0 is nonzero, so the map-curvature (mismatch) terms are exercised."""
    import simsoptpp as sopp
    g, gd, cur, X, nhat, absn, tgt, modes = _case()
    c, rho = an.centroids(g, gd)
    rng = np.random.default_rng(11)
    q_true = np.concatenate([np.r_[5e-3 * rng.standard_normal(3), 2e-2 * rng.standard_normal(3)] for _ in range(len(g))])
    G, GD, _ = an.apply_rigid(g, gd, c, q_true)
    B = sopp.biot_savart_B(np.ascontiguousarray(X), [np.ascontiguousarray(x) for x in G],
                           [np.ascontiguousarray(x) for x in GD], [float(x) for x in cur])
    tgt2 = np.sum(B * nhat, axis=1) + 0.3 * tgt
    return g, gd, cur, X, nhat, absn, tgt2, modes, c, rho, q_true


def _solve_q0(g, gd, cur, X, nhat, absn, tgt, c, rho, point_chunk, require_converged=True):
    S = an.q_scaling(rho)
    rc = an.RigidCorrection(g, gd, cur, X, nhat, absn, tgt, c, point_chunk)
    r0 = rc.residual(np.zeros(6 * len(g)))
    gscale = float(np.max(np.abs((rc.jacobian(np.zeros(6 * len(g))) * S[None]).T @ r0)))
    sol = rc.solve(np.zeros(6 * len(g)), gscale=gscale, S=S, gtol_rel=1e-10, max_nfev=200)
    if require_converged:
        assert sol["converged"], sol
    return sol, S


def test_rotation_jacobian_and_identity_map():
    g, gd, cur, X, nhat, absn, tgt, _ = _case()
    c, rho = an.centroids(g, gd)
    q = 0.03 * np.random.default_rng(1).standard_normal(6 * len(g))
    T, Td = an.rigid_jac_tangents(g, gd, c, q)
    for j in (0, 5):
        fpos = lambda qq: c[j] + (jnp.asarray(g[j]) - c[j]) @ an.jrot(qq[3:]).T + qq[:3]
        fder = lambda qq: jnp.asarray(gd[j]) @ an.jrot(qq[3:]).T
        Jp = np.moveaxis(np.asarray(jax.jacfwd(fpos)(jnp.asarray(q[6 * j:6 * j + 6]))), -1, 0)
        Jd = np.moveaxis(np.asarray(jax.jacfwd(fder)(jnp.asarray(q[6 * j:6 * j + 6]))), -1, 0)
        assert np.max(np.abs(Jp - T[j])) <= 1e-12 * np.max(np.abs(Jp))
        assert np.max(np.abs(Jd - Td[j])) <= 1e-12 * np.max(np.abs(Jd))
    G0, GD0, _ = an.apply_rigid(g, gd, c, np.zeros(6 * len(g)))
    assert np.max(np.abs(G0 - g)) <= 1e-15 * np.max(np.abs(g)) * 4 and np.array_equal(GD0, gd)
    # at q = 0 the tangents are e_d and e_d x (Gamma - c)
    T0, Td0 = an.rigid_jac_tangents(g, gd, c, np.zeros(6 * len(g)))
    Tr, Tdr = an.rigid_tangents(g[2], gd[2], c[2])
    assert np.max(np.abs(T0[2] - Tr)) <= 1e-15 and np.max(np.abs(Td0[2] - Tdr)) <= 1e-15


def test_projection_null_tests_and_r_bounds():
    g, gd, cur, X, nhat, absn, tgt, modes = _case()
    c, _ = an.centroids(g, gd)
    u1q, B0 = an.rigid_u1(g, gd, cur, X, nhat, c, NFP, True, NBASE, len(X))
    om = an.gram_weights(absn, B0, nhat, "quadratic flux")
    U, s, k = an.projector(u1q, om, 1e-10)
    q, qp = an.gram_diagonals(u1q, om, U)
    assert np.max(qp) <= 1e-24 * np.max(q) or np.max(np.sqrt(qp / q)) <= 1e-12   # rigid responses annihilated
    u1m, _ = an.mode_u1(g, gd, cur, X, nhat, modes, len(X))
    qm, qpm = an.gram_diagonals(u1m, om, U)
    rng = np.random.default_rng(2)
    for _ in range(5):
        r = an.r_lambda(rng.random(modes.size), qm, qpm)["r"]
        assert -1e-14 <= r <= 1 + 1e-14
    assert an.r_lambda(np.zeros(modes.size), qm, qpm)["status"].startswith("undefined")


def test_sigma_star_degenerate_cases():
    assert an.sigma_star(4.0, 1.0)["sigma_star"] == 2.0
    assert an.sigma_star(0.0, 1.0)["sigma_star"] == 0.0
    assert an.sigma_star(1.0, 0.0)["sigma_star"] is None
    assert an.sigma_star(0.0, 0.0)["case"].endswith("undefined")
    assert an.sigma_star(-1.0, 1.0)["case"].startswith("negative")


def test_decomposition_identity_and_superposition():
    g, gd, cur, X, nhat, absn, tgt, _ = _case(subset=False)
    base = setup(nq=32, nphi=16, ntheta=12)[0]
    modes = kl.fourier_modes(base[0].quadpoints, NFP, True, NBASE, 5e-3, L)
    parts, norms = an.decompose(modes, g, gd)
    tot = sum(parts[k].base_psi for k in parts)
    totd = sum(parts[k].base_dpsi for k in parts)
    assert np.max(np.abs(tot - modes.base_psi)) <= 1e-14 and np.max(np.abs(totd - modes.base_dpsi)) <= 1e-12
    u1, _ = an.mode_u1(g, gd, cur, X, nhat, modes, len(X))
    usum = sum(an.mode_u1(g, gd, cur, X, nhat, parts[k], len(X))[0] for k in parts)
    assert np.max(np.abs(usum - u1)) <= 1e-12 * np.max(np.abs(u1))
    # translations (m = 0) are pure T; their R, St, Sn parts vanish
    m = an.fourier_m(modes)
    assert np.all(norms["R"][m == 0] <= 1e-28) and np.all(norms["Sn"][m == 0] <= 1e-28)


def _jax_f(g, gd, cur, X, nhat, absn, tgt, c, mode_tan):
    """f(xi, q) = J_B with errors sum_a xi_a psi_a then the rigid map, original coordinates."""
    C = len(g)
    P = len(absn)

    def f(xi, q):
        G = jnp.asarray(g) + jnp.einsum("a,ajqk->jqk", xi, mode_tan[0])
        GD = jnp.asarray(gd) + jnp.einsum("a,ajqk->jqk", xi, mode_tan[1])
        Gs, GDs = [], []
        for j in range(C):
            R = an.jrot(q[6 * j + 3:6 * j + 6])
            Gs.append(c[j] + (G[j] - c[j]) @ R.T + q[6 * j:6 * j + 3])
            GDs.append(GD[j] @ R.T)
        B = rsp.total_field(jnp.stack(Gs), jnp.stack(GDs), jnp.asarray(cur), jnp.asarray(X))
        u = jnp.sum(B * nhat, axis=1) - tgt
        return 0.5 * jnp.sum(absn * u * u) / P
    return f


def _mode_tangent_array(modes, C, nq):
    T = np.zeros((modes.size, C, nq, 3)); Td = np.zeros_like(T)
    for j in range(C):
        idx, dg, dgd = modes.tangents(j)
        T[idx, j], Td[idx, j] = dg, dgd
    return jnp.asarray(T), jnp.asarray(Td)


@pytest.mark.parametrize("pc", [None, 50])
def test_hessian_blocks_and_schur_against_jax(pc):
    g, gd, cur, X, nhat, absn, tgt, modes, c, rho, q_true = _near_optimal_case()
    C, nq, P = len(g), g.shape[1], len(X)
    point_chunk = P if pc is None else pc
    sol, S = _solve_q0(g, gd, cur, X, nhat, absn, tgt, c, rho, point_chunk)
    q0 = sol["q"]
    assert np.max(np.abs(q0[3::6])) > 1e-3                               # q0 != 0 (R1 test case)
    gt, gdt, Rs = an.apply_rigid(g, gd, c, q0)
    modes_rb = an.rebase_modes(modes, Rs)
    u1m, B0 = an.mode_u1(gt, gdt, cur, X, nhat, modes, point_chunk, Rs=Rs)
    u1q, _ = an.rigid_u1(gt, gdt, cur, X, nhat, c, NFP, True, NBASE, point_chunk)
    H = an.hessian_blocks(gt, gdt, cur, X, nhat, absn, tgt, c, modes_rb, u1m, u1q, B0, point_chunk)
    # (a) blocks against jax.hessian of the re-based f(xi, dq)
    mt_rb = _mode_tangent_array(modes_rb, C, nq)
    f_rb = _jax_f(gt, gdt, cur, X, nhat, absn, tgt, c, mt_rb)
    A = modes.size
    z0 = jnp.zeros(A + 6 * C)
    Hfull = np.asarray(jax.hessian(lambda z: f_rb(z[:A], z[A:]))(z0))
    Hxx, Hqx, Hqq = Hfull[:A, :A], Hfull[A:, :A], Hfull[A:, A:]
    assert np.max(np.abs(H["Hqq"] - Hqq)) <= 1e-10 * np.max(np.abs(Hqq))
    assert np.max(np.abs(H["Hqx"] - Hqx)) <= 1e-10 * np.max(np.abs(Hqx))
    gfull = np.asarray(jax.grad(lambda z: f_rb(z[:A], z[A:]))(z0))
    assert np.max(np.abs(H["grad_q"] - gfull[A:])) <= 1e-10 * np.max(np.abs(gfull[A:])) + 1e-18
    # production Delta2 on the isotropic or general path = 1/2 tr(Lambda Hxx)
    lam = modes.lam
    d2_jax = 0.5 * float(lam @ np.diag(Hxx))
    sch = an.schur(lam, H["Hqq"], H["Hqx"], S, d2_jax)
    assert sch["positive_definite"]
    # (b) Schur against jax.hessian in the ORIGINAL parametrization at (0, q0)
    mt = _mode_tangent_array(modes, C, nq)
    f0 = _jax_f(g, gd, cur, X, nhat, absn, tgt, c, mt)
    H0 = np.asarray(jax.hessian(lambda z: f0(z[:A], z[A:]))(jnp.concatenate([jnp.zeros(A), jnp.asarray(q0)])))
    Hxx0, Hqx0, Hqq0 = H0[:A, :A], H0[A:, :A], H0[A:, A:]
    schur0 = Hxx0 - Hqx0.T @ np.linalg.solve(Hqq0, Hqx0)
    c_orig = 0.5 * float(lam @ np.diag(schur0))
    c_rb = 0.5 * float(lam @ np.diag(Hxx - Hqx.T @ np.linalg.solve(Hqq, Hqx)))
    assert abs(c_rb - c_orig) <= 1e-10 * abs(c_orig)
    assert abs((d2_jax - sch["schur_term"]) - c_orig) <= 1e-10 * abs(c_orig)



@pytest.mark.parametrize("pc,batch", [(None, 6), (50, 5)])
def test_hessian_xixi_against_jax(pc, batch):
    """P2V: H_xixi = H_GN + H_mm against jax.hessian of the re-based f in xi at q0 != 0,
    u0 != 0 (near-optimal case), unchunked and chunked, with an uneven direction batch."""
    g, gd, cur, X, nhat, absn, tgt, modes, c, rho, q_true = _near_optimal_case()
    C, nq, P = len(g), g.shape[1], len(X)
    point_chunk = P if pc is None else pc
    sol, S = _solve_q0(g, gd, cur, X, nhat, absn, tgt, c, rho, point_chunk)
    gt, gdt, Rs = an.apply_rigid(g, gd, c, sol["q"])
    modes_rb = an.rebase_modes(modes, Rs)
    u1m, B0 = an.mode_u1(gt, gdt, cur, X, nhat, modes, point_chunk, Rs=Rs)
    H = an.hessian_xixi(gt, gdt, cur, X, nhat, absn, tgt, modes_rb, u1m, B0, point_chunk, batch=batch)
    f_rb = _jax_f(gt, gdt, cur, X, nhat, absn, tgt, c, _mode_tangent_array(modes_rb, C, nq))
    A = modes.size
    Hxx = np.asarray(jax.hessian(lambda xi: f_rb(xi, jnp.zeros(6 * C)))(jnp.zeros(A)))
    Hmine = H["H_GN"] + H["H_mm"]
    assert np.max(np.abs(H["H_GN"] - Hxx)) > 1e-6 * np.max(np.abs(Hxx))      # H_GN alone misses: H_mm is exercised
    assert np.max(np.abs(Hmine - Hxx)) <= 1e-10 * np.max(np.abs(Hxx))
    assert np.max(np.abs(H["H_mm"] - H["H_mm"].T)) <= 1e-12 * np.max(np.abs(H["H_mm"]))


def test_hessian_xixi_isotropic_trace_of_mismatch_part():
    """P2V check 2: on the full (isotropic) Fourier mode set the Lambda-trace of the mismatch
    part vanishes, 1/2 tr(Lambda H_mm) = 0, while 1/2 tr(Lambda H_GN) = Delta2 > 0."""
    g, gd, cur, X, nhat, absn, tgt, modes = _case(subset=False)
    P = len(X)
    u1m, B0 = an.mode_u1(g, gd, cur, X, nhat, modes, P)
    H = an.hessian_xixi(g, gd, cur, X, nhat, absn, tgt, modes, u1m, B0, P)
    tr_gn = 0.5 * float(modes.lam @ np.diag(H["H_GN"]))
    tr_mm = 0.5 * float(modes.lam @ np.diag(H["H_mm"]))
    lam_abs = 0.5 * float(modes.lam @ np.abs(np.diag(H["H_mm"])))
    assert tr_gn > 0 and lam_abs > 1e-6 * tr_gn                                # nontrivial entries
    assert abs(tr_mm) <= 1e-12 * lam_abs

def test_rotation_centre_independence_of_minimum():
    g, gd, cur, X, nhat, absn, tgt, modes, c, rho, q_true = _near_optimal_case()
    a, _ = _solve_q0(g, gd, cur, X, nhat, absn, tgt, c, rho, len(X))
    # the shifted centre only reparametrizes q; its LM stops at a roundoff gradient floor
    b, _ = _solve_q0(g, gd, cur, X, nhat, absn, tgt, c + 0.2, rho, len(X), require_converged=False)
    assert b["grad_scaled_max"] <= 1e-11
    assert abs(a["f"] - b["f"]) <= 1e-10 * a["f"]


def _perturbed_rebased_case():
    """Near-optimal case re-based at its q0, then displaced by a small error (~1 mm) along the
    first few mode tangents, so the per-sample solve starts away from its minimum."""
    g, gd, cur, X, nhat, absn, tgt, modes, c, rho, q_true = _near_optimal_case()
    sol, S = _solve_q0(g, gd, cur, X, nhat, absn, tgt, c, rho, len(X))
    gt, gdt, Rs = an.apply_rigid(g, gd, c, sol["q"])
    modes_rb = an.rebase_modes(modes, Rs)
    u1m, B0 = an.mode_u1(gt, gdt, cur, X, nhat, modes, len(X), Rs=Rs)
    u1q, _ = an.rigid_u1(gt, gdt, cur, X, nhat, c, NFP, True, NBASE, len(X))
    H = an.hessian_blocks(gt, gdt, cur, X, nhat, absn, tgt, c, modes_rb, u1m, u1q, B0, len(X))
    xi = np.zeros(modes.size); xi[:4] = 1e-3 / np.sqrt(np.maximum(modes.lam[:4], 1e-30)) * np.sqrt(modes.lam[:4])
    G, GD = gt.copy(), gdt.copy()
    for j in range(len(g)):
        idx, dg, dgd = modes_rb.tangents(j)
        if len(idx):
            G[j] += np.einsum("a,aqk->qk", xi[idx], dg); GD[j] += np.einsum("a,aqk->qk", xi[idx], dgd)
    return G, GD, cur, X, nhat, absn, tgt, c, S, H["Hqq"]


def test_solve_newton_same_minimum_as_lm_and_stop_rules():
    G, GD, cur, X, nhat, absn, tgt, c, S, Hqq = _perturbed_rebased_case()
    C = len(G)
    rc = an.RigidCorrection(G, GD, cur, X, nhat, absn, tgt, c, len(X))
    r0 = rc.residual(np.zeros(6 * C))
    gscale = float(np.max(np.abs((rc.jacobian(np.zeros(6 * C)) * S[None]).T @ r0)))
    lm = rc.solve(np.zeros(6 * C), gscale=gscale, S=S, gtol_rel=1e-10, max_nfev=200)
    nw = rc.solve_newton(np.zeros(6 * C), gscale=gscale, S=S, H0=Hqq, gtol_rel=1e-10, ftol_rel=1e-15, max_iter=30)
    assert lm["converged"] and nw["converged"], (lm, nw)
    assert abs(nw["f"] - lm["f"]) <= 1e-12 * lm["f"]
    assert np.linalg.norm((nw["q"] - lm["q"]) / S) <= 1e-8
    # stop rule "gradient": a loose gradient tolerance
    a = rc.solve_newton(np.zeros(6 * C), gscale=gscale, S=S, H0=Hqq, gtol_rel=1e-3, ftol_rel=0.0)
    assert a["criterion"] == "gradient" and a["converged"]
    # stop rule "f-stationary": an unreachable gradient tolerance and a loose f tolerance
    b = rc.solve_newton(np.zeros(6 * C), gscale=gscale, S=S, H0=Hqq, gtol_rel=0.0, ftol_rel=1e-2)
    assert b["criterion"] == "f-stationary" and b["converged"]

    # stop rule "no decrease": a residual that rises on any step away from the solve's start q_s
    class Rising(an.RigidCorrection):
        q_s = None

        def residual(self, q):
            r = super().residual(q)
            return np.concatenate([r, [np.sqrt(1e3 * np.linalg.norm((np.asarray(q) - self.q_s) / S))]])  # f rises linearly in the step

        def jacobian(self, q):
            J = super().jacobian(q)
            return np.vstack([J, np.zeros((1, J.shape[1]))])
    rr = Rising(G, GD, cur, X, nhat, absn, tgt, c, len(X))
    rr.q_s = lm["q"]                                  # tiny gradient there: "f at roundoff" label, converged
    cnv = rr.solve_newton(lm["q"], gscale=gscale, S=S, H0=Hqq, gtol_rel=0.0, ftol_rel=0.0)
    assert cnv["criterion"].startswith("no decrease (f at roundoff") and cnv["converged"], cnv["criterion"]
    rr.q_s = np.zeros(6 * C)                          # large gradient there: not converged
    far = rr.solve_newton(np.zeros(6 * C), gscale=gscale, S=S, H0=Hqq, gtol_rel=0.0, ftol_rel=0.0)
    assert far["criterion"] == "no decrease along the Newton direction" and not far["converged"]
